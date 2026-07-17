"""
gpu_scale_codebook.py  -- COLAB / GPU experiment
================================================

Scales the program's ONE surviving genuine-generativity result -- the amortized
NCA codebook (v6-v8, B1) -- to realistic size with GENUINELY RICH targets, to test
whether it holds and improves at scale. This directly attacks the honest limitation
flagged by the v9/v10 audit: everything so far was 8x8, low-rank blobs, single seed.

It runs the AUDIT-VALIDATED test (held-out realizability; NOT zlib Delta), at scale:

  (1) HELD-OUT REALIZABILITY: freeze a body trained on K_train targets, fit a small
      code to HELD-OUT targets (never trained on), vs a random-body control. This is
      B1's necessary+sufficient condition and the only measure that survived audit.
  (2) B1 SUFFICIENCY SIGNATURE at scale: held-out rises with K_train.
  (3) RICHNESS: targets are real images or high-eff-dim Gaussian-random-field
      textures (NOT low-rank blobs) -- so a pass is a genuine rich-target result.
  (4) LOW-I (scalar-ES) arm at scale + capacity: can a non-differentiable shaper
      also build/select on the big body (v8's claim at scale)?
  (5) AMORTIZED-INFERENCE ENCODER (the prompting-like step from architectures.md):
      train an encoder target->code; measure ZERO-SHOT held-out (no per-target opt).
  (6) dim_sel: effective # of distinct held-out targets realized.

USAGE (Colab):
    !pip -q install torch torchvision
    !python gpu_scale_codebook.py --targets mnist --grid 28 --hid 128 --channels 16
  or self-contained (no download):
    !python gpu_scale_codebook.py --targets grf --grid 32 --hid 128 --channels 16
  Flags: --ktrain 8,32,128  --held 16  --steps 48  --iters 2000  --device cuda

Reports metrics + saves gpu_scale_results.json. ~20-40 min on a T4.
"""

import argparse
import json
import math
import time
import numpy as np
import torch
import torch.nn.functional as F

# ----------------------------------------------------------------------------- targets


def grf_targets(n, L, seed=0, device="cpu"):
    """High-eff-dim Gaussian random field textures (power-law spectrum) -- genuinely
    rich (unlike low-rank blobs), zero-dependency, diverse."""
    g = torch.Generator().manual_seed(seed)
    ys, xs = torch.meshgrid(torch.fft.fftfreq(L), torch.fft.fftfreq(L), indexing="ij")
    r = torch.sqrt(xs ** 2 + ys ** 2) + 1e-3
    spec = r ** (-1.6)                                   # power-law -> rich but structured
    out = []
    for i in range(n):
        ph = torch.exp(2j * math.pi * torch.rand(L, L, generator=g))
        f = torch.fft.ifft2(spec * ph).real
        f = (f - f.min()) / (f.max() - f.min() + 1e-9)
        out.append(f)
    return torch.stack(out).to(device)


def image_targets(n_train_pool, n_held, L, which="mnist", device="cpu"):
    import torchvision
    import torchvision.transforms as TT
    tf = TT.Compose([TT.Grayscale(), TT.Resize((L, L)), TT.ToTensor()])
    ds = (torchvision.datasets.MNIST if which == "mnist" else torchvision.datasets.CIFAR10)
    d = ds(root="./data", train=True, download=True, transform=tf)
    idx = torch.randperm(len(d))[: n_train_pool + n_held]
    imgs = torch.stack([d[i][0][0] for i in idx])
    imgs = (imgs - imgs.amin(dim=(1, 2), keepdim=True)) / (
        imgs.amax(dim=(1, 2), keepdim=True) - imgs.amin(dim=(1, 2), keepdim=True) + 1e-9)
    return imgs.to(device)


def eff_dim(field):
    m = field - field.mean()
    s = torch.linalg.svdvals(m)
    p = s ** 2 / (s ** 2).sum()
    return float(1.0 / (p ** 2).sum())


# ----------------------------------------------------------------------------- model


class FiLMNCA(torch.nn.Module):
    def __init__(self, C, HID, D):
        super().__init__()
        self.C, self.HID, self.D = C, HID, D
        self.W1 = torch.nn.Parameter(0.1 * torch.randn(3 * C, HID))
        self.b1 = torch.nn.Parameter(torch.zeros(HID))
        self.W2 = torch.nn.Parameter(torch.zeros(HID, C))       # zero-init last layer
        self.b2 = torch.nn.Parameter(torch.zeros(C))
        self.Wg = torch.nn.Parameter(0.1 * torch.randn(D, HID))
        self.Wb = torch.nn.Parameter(0.1 * torch.randn(D, HID))
        sx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
        id3 = torch.zeros(3, 3); id3[1, 1] = 1.0
        self.register_buffer("kern", torch.stack([id3, sx, sx.t()]))

    def perceive(self, s):
        w = self.kern.view(3, 1, 3, 3).repeat(self.C, 1, 1, 1)
        return F.conv2d(F.pad(s, (1, 1, 1, 1), mode="circular"), w, groups=self.C)

    def grow(self, codes, L, steps):
        B = codes.shape[0]
        s = torch.zeros(B, self.C, L, L, device=codes.device)
        s[:, :, L // 2, L // 2] = 1.0
        gamma = 1 + codes @ self.Wg
        beta = codes @ self.Wb
        for _ in range(steps):
            p = self.perceive(s).permute(0, 2, 3, 1)
            h = torch.tanh(p @ self.W1 + self.b1)
            h = gamma[:, None, None, :] * h + beta[:, None, None, :]
            s = torch.clamp(s + (h @ self.W2 + self.b2).permute(0, 3, 1, 2), -3, 3)
        return torch.clamp(s[:, 0], 0, 1)


def corr(mm, T):
    a = mm.reshape(mm.shape[0], -1); b = T.reshape(T.shape[0], -1)
    a = a - a.mean(1, keepdim=True); b = b - b.mean(1, keepdim=True)
    return (a * b).sum(1) / (torch.sqrt((a * a).sum(1) * (b * b).sum(1)) + 1e-9)


# ----------------------------------------------------------------------------- routines


def train_body(net, targets, L, steps, iters, lr, dev):
    codes = torch.nn.Parameter(0.1 * torch.randn(len(targets), net.D, device=dev))
    opt = torch.optim.Adam(list(net.parameters()) + [codes], lr=lr)
    for it in range(iters):
        opt.zero_grad()
        loss = (1 - corr(net.grow(codes, L, steps), targets)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(net.parameters()) + [codes], 1.0)
        opt.step()
    with torch.no_grad():
        return float(corr(net.grow(codes, L, steps), targets).mean()), codes.detach()


def fit_code(net, tgt, L, steps, iters, dev):
    for p in net.parameters():
        p.requires_grad_(False)
    c = torch.nn.Parameter(0.1 * torch.randn(1, net.D, device=dev))
    opt = torch.optim.Adam([c], lr=0.03)
    for _ in range(iters):
        opt.zero_grad()
        loss = (1 - corr(net.grow(c, L, steps), tgt[None])).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        return float(corr(net.grow(c, L, steps), tgt[None])[0]), c.detach()


def es_fit_code(net, tgt, L, steps, dev, pop=48, gens=200):
    rng = np.random.default_rng(0); c = np.zeros(net.D); half = pop // 2
    with torch.no_grad():
        for g in range(gens):
            sigma = 0.3 * 0.3 ** (g / gens)
            eps = rng.normal(0, 1, (half, net.D)); eps = np.concatenate([eps, -eps], 0)
            cand = torch.tensor(c[None] + sigma * eps, dtype=torch.float32, device=dev)
            fit = corr(net.grow(cand, L, steps), tgt[None].expand(pop, -1, -1)).cpu().numpy()
            ranks = np.empty_like(fit); ranks[np.argsort(fit)] = np.arange(pop)
            c = c + 0.2 * (eps * (ranks / (pop - 1) - 0.5)[:, None]).mean(0) / sigma
        cc = torch.tensor(c[None], dtype=torch.float32, device=dev)
        return float(corr(net.grow(cc, L, steps), tgt[None])[0])


class Encoder(torch.nn.Module):
    """Amortized inference: target image -> code in ONE forward pass (prompting-like)."""
    def __init__(self, L, D):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 2, 1), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, 2, 1), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
            torch.nn.Linear(64, D))

    def forward(self, imgs):
        return self.net(imgs[:, None])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="grf", choices=["grf", "mnist", "cifar"])
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--code", type=int, default=16)
    ap.add_argument("--ktrain", default="8,32,128")
    ap.add_argument("--held", type=int, default=16)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = a.device
    L, C, HID, D = a.grid, a.channels, a.hid, a.code
    KT = [int(x) for x in a.ktrain.split(",")]
    t0 = time.time()
    print(f"device={dev}  grid={L}  C={C}  HID={HID}  code={D}  targets={a.targets}  steps={a.steps}")

    Kmax = max(KT)
    if a.targets == "grf":
        pool = grf_targets(Kmax, L, 0, dev); held = grf_targets(a.held, L, 999, dev)
    else:
        allt = image_targets(Kmax, a.held, L, a.targets, dev)
        pool, held = allt[:Kmax], allt[Kmax:]
    print(f"target richness (mean eff-dim): pool={np.mean([eff_dim(pool[i]) for i in range(min(8,len(pool)))]):.1f} "
          f"(blobs were ~2; higher = genuinely rich)")

    # control: random body held-out
    torch.manual_seed(1)
    rb = FiLMNCA(C, HID, D).to(dev)
    ctrl = np.mean([fit_code(rb, held[k], L, a.steps, 300, dev)[0] for k in range(a.held)])
    print(f"\ncontrol (random body) held-out corr = {ctrl:.3f}")

    res = {"config": vars(a), "control": float(ctrl), "richness_pool": float(np.mean([eff_dim(pool[i]) for i in range(min(8,len(pool)))])), "by_K": {}}
    for K in KT:
        torch.manual_seed(0)
        net = FiLMNCA(C, HID, D).to(dev)
        tr, codes = train_body(net, pool[:K], L, a.steps, a.iters, 2e-3, dev)
        bp_ho = [fit_code(net, held[k], L, a.steps, 400, dev)[0] for k in range(a.held)]
        es_ho = [es_fit_code(net, held[k], L, a.steps, dev) for k in range(min(6, a.held))]
        # dim_sel over held-out realized fields (fit_code needs grad; only the final forward is no_grad)
        hcodes = torch.stack([fit_code(net, held[k], L, a.steps, 300, dev)[1][0]
                              for k in range(a.held)])
        with torch.no_grad():
            fields = net.grow(hcodes, L, a.steps).reshape(a.held, -1).cpu().numpy()
        ds = eff_dim(torch.tensor(fields))
        print(f"K_train={K:4d}: train={tr:.3f}  held-out(backprop)={np.mean(bp_ho):.3f}  "
              f"held-out(scalarES)={np.mean(es_ho):.3f}  dim_sel={ds:.1f}  "
              f"[{time.time()-t0:.0f}s]")
        res["by_K"][K] = dict(train=tr, held_bp=float(np.mean(bp_ho)),
                              held_es=float(np.mean(es_ho)), dim_sel=float(ds))

    # amortized-inference encoder on the biggest body (zero-shot held-out)
    K = Kmax
    torch.manual_seed(0)
    net = FiLMNCA(C, HID, D).to(dev)
    tr, codes = train_body(net, pool[:K], L, a.steps, a.iters, 2e-3, dev)
    for p in net.parameters():
        p.requires_grad_(False)
    enc = Encoder(L, D).to(dev)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    for it in range(1500):
        opt.zero_grad()
        c = enc(pool[:K])
        loss = (1 - corr(net.grow(c, L, a.steps), pool[:K])).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        zshot = float(corr(net.grow(enc(held), L, a.steps), held).mean())
    print(f"\namortized-inference encoder: ZERO-SHOT held-out corr = {zshot:.3f} "
          f"(no per-target optimization -- the prompting-like regime)")
    res["encoder_zeroshot"] = zshot

    print("\nSUMMARY: held-out realizability is the audit-validated generativity test.")
    print(f"  control={ctrl:.3f}; " + "; ".join(
        f"K={K}:bp={res['by_K'][K]['held_bp']:.2f}" for K in KT))
    print(f"  B1 sufficiency (held-out rises with K_train): "
          f"{[round(res['by_K'][K]['held_bp'],3) for K in KT]}")
    print(f"  richest-target eff-dim (pool) = {res['richness_pool']:.1f}; encoder zero-shot={zshot:.2f}")
    json.dump(res, open("gpu_scale_results.json", "w"), indent=2)
    print(f"\n[done] {time.time()-t0:.0f}s  -> gpu_scale_results.json")


if __name__ == "__main__":
    main()
