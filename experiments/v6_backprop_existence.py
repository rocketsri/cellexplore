"""
v6_backprop_existence.py
========================

The EXISTENCE arm of the amortization test (the M0->M* ladder done honestly).

Two separate questions must not be conflated:
  (Q1, architecture) Does a shared-body + small-per-target-code solution with a
      POSITIVE amortized generativity gap even EXIST?  -> answered with backprop
      (high-bandwidth search), which is the right tool to certify existence.
  (Q2, low-I discoverability) Can a NON-differentiable, low-bandwidth shaper FIND
      that solution?  -> answered by the ES sweep in v6_amortization_curve.py.

Backprop here is used ONLY to certify existence (Q1). It is NOT a B-bio shaping
claim -- it delivers ~D bits/step and lives in the Claim-A corner by construction.
The point: if the generic-prior solution exists (Q1 yes) but ES cannot find it
(Q2 no), the residual gap is precisely "architecturally real but not low-I
discoverable" -- which sharpens I_c, exactly as promised.

Same FiLM-modulated NCA as the ES arm. Trains body+codes on K_train targets, then
FREEZES the body and gradient-fits a d-dim code to HELD-OUT targets. Reports
held-out correlation vs K_train and vs a random-frozen-body control.

Requires torch (CPU). ~3-5 min.
"""

import torch
import torch.nn.functional as Fnn
import numpy as np
import zlib
import json
import time

torch.manual_seed(0)
H = W = 8
C = 8
HID = 32          # capacity matters: at HID=12 the shared prior is too weak to
D = 12            # generalise; at HID=32 a genuine target-agnostic codebook emerges
STEPS = 24
dev = "cpu"

# fixed perception kernels (identity + Sobel x/y), depthwise over channels
_sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
_sy = _sx.t()
_id = torch.zeros(3, 3); _id[1, 1] = 1.0
_kern = torch.stack([_id, _sx, _sy])                       # (3,3,3)


def perceive(s):
    # s: (B,C,H,W) -> (B, 3C, H, W): identity, sobel-x, sobel-y per channel
    w = _kern.view(3, 1, 3, 3).repeat(C, 1, 1, 1)          # (3C,1,3,3)
    sp = Fnn.pad(s, (1, 1, 1, 1))
    out = Fnn.conv2d(sp, w, groups=C)                       # depthwise
    return out


class FiLMNCA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # proper growing-NCA init: last layer ZERO (identity start, du=0), so
        # gradients flow through the unrolled horizon instead of saturating.
        self.W1 = torch.nn.Parameter(0.2 * torch.randn(3 * C, HID))
        self.b1 = torch.nn.Parameter(torch.zeros(HID))
        self.W2 = torch.nn.Parameter(torch.zeros(HID, C))
        self.b2 = torch.nn.Parameter(torch.zeros(C))
        self.Wg = torch.nn.Parameter(0.2 * torch.randn(D, HID))
        self.Wb = torch.nn.Parameter(0.2 * torch.randn(D, HID))

    def body_numel(self):
        return sum(p.numel() for p in self.parameters())

    def grow(self, codes, steps=STEPS):
        B = codes.shape[0]
        s = torch.zeros(B, C, H, W)
        s[:, :, H // 2, W // 2] = 1.0                        # seed ALL channels at centre
        gamma = 1.0 + codes @ self.Wg                       # (B,HID) FiLM scale
        beta = codes @ self.Wb                              # FiLM shift
        for _ in range(steps):
            p = perceive(s).permute(0, 2, 3, 1)             # (B,H,W,3C)
            h = torch.tanh(p @ self.W1 + self.b1)
            h = gamma[:, None, None, :] * h + beta[:, None, None, :]
            du = h @ self.W2 + self.b2                       # (B,H,W,C)
            s = torch.clamp(s + du.permute(0, 3, 1, 2), -3, 3)
        return torch.clamp(s[:, 0], 0, 1)                    # (B,H,W) morphology


def corr_paired(mm, T):
    a = mm.reshape(mm.shape[0], -1); b = T.reshape(T.shape[0], -1)
    a = a - a.mean(1, keepdim=True); b = b - b.mean(1, keepdim=True)
    return (a * b).sum(1) / (torch.sqrt((a * a).sum(1) * (b * b).sum(1)) + 1e-9)


def target(seed, nb=4):
    r = np.random.default_rng(seed); yy, xx = np.mgrid[0:H, 0:W].astype(float); f = np.zeros((H, W))
    for _ in range(nb):
        cy, cx = r.uniform(0.5, H - 1.5), r.uniform(0.5, W - 1.5)
        sy, sx = r.uniform(0.7, 1.4), r.uniform(0.7, 1.4)
        a = r.uniform(0.5, 1) * (1 if r.random() < 0.7 else -1)
        f += a * np.exp(-((yy - cy) ** 2 / (2 * sy ** 2) + (xx - cx) ** 2 / (2 * sx ** 2)))
    f -= f.min(); f /= f.max() + 1e-9
    return torch.tensor(f, dtype=torch.float32)


def corr(mm, t):
    a = mm.reshape(mm.shape[0], -1); b = t.reshape(1, -1)
    a = a - a.mean(1, keepdim=True); b = b - b.mean()
    return (a * b).sum(1) / (torch.sqrt((a * a).sum(1) * (b * b).sum()) + 1e-9)


def dl_bits(t, q=16):
    qf = np.clip((t.numpy() * (q - 1)).round().astype(np.uint8), 0, q - 1)
    return 8 * len(zlib.compress(qf.tobytes(), 9))


def train_body(train_targets, iters=900):
    net = FiLMNCA()
    K = len(train_targets)
    codes = torch.nn.Parameter(0.1 * torch.randn(K, D))
    T = torch.stack(train_targets)
    params = list(net.parameters()) + [codes]
    opt = torch.optim.Adam(params, lr=3e-3)
    for it in range(iters):
        opt.zero_grad()
        loss = (1 - corr_paired(net.grow(codes), T)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    with torch.no_grad():
        tr = float(corr_paired(net.grow(codes), T).mean())
    return net, tr


def fit_code(net, tgt, iters=400):
    for p in net.parameters():
        p.requires_grad_(False)
    code = torch.nn.Parameter(0.1 * torch.randn(1, D))
    opt = torch.optim.Adam([code], lr=0.02)
    tt = tgt[None]
    for it in range(iters):
        opt.zero_grad()
        loss = (1 - corr_paired(net.grow(code), tt)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([code], 1.0)
        opt.step()
    with torch.no_grad():
        return float(corr_paired(net.grow(code), tt)[0])


def main():
    t0 = time.time()
    print("=" * 74)
    print("v6 / A-existence arm (BACKPROP): does a generic shared prior EXIST?")
    print("=" * 74)
    net0 = FiLMNCA()
    print(f"grid {H}x{W}, shared body |theta|={net0.body_numel()}, code dim d={D}, STEPS={STEPS}")
    print("(backprop = high-bandwidth; certifies architectural existence only, NOT a B-bio claim)")

    pool = [target(s) for s in range(30)]
    held = [target(200 + k) for k in range(4)]
    DL_held = float(np.mean([dl_bits(t) for t in held]))
    code_bits = D * 6

    rand_net = FiLMNCA()
    ctrl = float(np.mean([fit_code(rand_net, h) for h in held]))
    print(f"\ncontrol (random frozen body + code, backprop) held-out corr = {ctrl:.3f}")
    print(f"DL(held-out) ~ {DL_held:.0f} bits, marginal |code| ~ {code_bits} bits\n")

    rows = []
    for K in (2, 6, 12, 24):
        net, tr = train_body(pool[:K])
        ho = float(np.mean([fit_code(net, h) for h in held]))
        rows.append((K, tr, ho, ho - ctrl))
        print(f"K_train={K:3d}: train corr={tr:.3f}  held-out corr={ho:.3f}  "
              f"margin over control={ho-ctrl:+.3f}  [{time.time()-t0:.0f}s]")

    best = max(rows, key=lambda r: r[2])
    print(f"\nBest K_train={best[0]}: held-out {best[2]:.3f} (margin {best[3]:+.3f} over control)")
    print(f"held-out trend across K_train: {[round(r[2],3) for r in rows]}")
    if best[2] > 0.6 and best[3] > 0.1:
        print(f"=> EXISTENCE CONFIRMED (Q1 yes): a generic frozen body + {D}-param code reaches "
              f"held-out targets. Marginal Delta = {DL_held:.0f}-{code_bits} = "
              f"{DL_held-code_bits:+.0f} bits exists architecturally.")
    else:
        print("=> Existence NOT confirmed even with backprop: the FiLM-NCA prior itself is the "
              "limiter, not the shaper.")
    print(f"\n[done] runtime={time.time()-t0:.0f}s")
    print("JSON_BP_BEGIN")
    print(json.dumps(dict(control=ctrl, DL_held=DL_held, code_bits=code_bits,
                          rows=rows, best_K=best[0])))
    print("JSON_BP_END")


if __name__ == "__main__":
    main()
