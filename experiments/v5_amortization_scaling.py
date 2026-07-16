"""
v5_amortization_scaling.py
==========================

Two honest follow-ups demanded by the deflationary audit of O1:

(A) AMORTIZATION / MARGINAL GENERATIVITY.  The naive per-target generativity gap
    is ~0 under a consistent complexity measure: a pruned rule that reproduces the
    target IS a generator of it, so K(target) <= |rule|. The ONLY honest route to a
    positive gap is amortization, exactly as in biology: ONE shared rule (the
    conserved developmental toolkit / genome), paid once, plus a SMALL per-target
    code (the regulatory difference), realizes MANY rich targets. The marginal
    shaping cost of one more target is then |code|, while its standalone
    description length is DL >> |code|.

    Acid test (this is what makes it non-circular): after evolving the shared body
    on a TRAIN set of targets, FREEZE it and fit only a small code to a HELD-OUT
    target it was never shaped on. If a d-parameter code steers the frozen body to
    a new target, the body is a GENERIC morphogenetic prior (not a memorized
    target), and the marginal per-target information really is |code| << DL.
    If it fails, that failure bounds the generativity -- also an honest result.

(B) HONEST SCALING.  The Theta(N) claim requires target COMPLEXITY to scale with
    N. With a fixed blob count the target gets MORE compressible at higher res
    (DL/N falls), so DL = o(N) and the gap does NOT grow. Here we measure DL(N)
    and eff-dim(N) for a family whose blob count scales with N, to state exactly
    when Theta(N) holds and when it does not.

Pure numpy. Deterministic. ~3-4 min.
"""

import numpy as np
import zlib
import time
import json

# --------------------------------------------------------------------------- #
def sobel():
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float) / 8.0
    idg = np.zeros((3, 3)); idg[1, 1] = 1.0
    return [idg, sx, sx.T]


def build(H, W, C, HID):
    F = sobel(); PERC = C * 3
    GEN = PERC * HID + HID + HID * C + C

    def unpack(th):
        P = th.shape[0]; i = 0
        W1 = th[:, i:i + PERC * HID].reshape(P, PERC, HID); i += PERC * HID
        b1 = th[:, i:i + HID]; i += HID
        W2 = th[:, i:i + HID * C].reshape(P, HID, C); i += HID * C
        b2 = th[:, i:i + C]
        return W1, b1, W2, b2

    def dw(fld, k):
        P, h, w, c = fld.shape
        pad = np.pad(fld, ((0, 0), (1, 1), (1, 1), (0, 0))); o = np.zeros_like(fld)
        for dy in range(3):
            for dx in range(3):
                if k[dy, dx]:
                    o += k[dy, dx] * pad[:, dy:dy + h, dx:dx + w, :]
        return o

    def step(s, th):
        W1, b1, W2, b2 = unpack(th)
        p = np.concatenate([dw(s, f) for f in F], -1)
        hz = np.tanh(np.einsum('phwq,pqk->phwk', p, W1) + b1[:, None, None, :])
        du = np.einsum('phwk,pkc->phwc', hz, W2) + b2[:, None, None, :]
        return np.clip(s + du, -2, 2)

    def grow(th, codes, steps, dmg=None):
        """th: (P,GEN) shared body per member; codes: (P,CODE) seed hidden state."""
        P = th.shape[0]
        s = np.zeros((P, H, W, C))
        s[:, H // 2, W // 2, 0] = 1.0
        d = codes.shape[1]
        s[:, H // 2, W // 2, 1:1 + d] = codes            # code = seed's initial hidden state
        for t in range(steps):
            if dmg is not None and t == dmg:
                s[:, :, W // 2:, :] = 0.0
            s = step(s, th)
        return s

    return GEN, grow


def morph(s):
    return np.clip(s[..., 0], 0, 1)


def corr_batch(mm, t):
    a = mm.reshape(mm.shape[0], -1); b = t.ravel()[None, :]
    a = a - a.mean(1, keepdims=True); b = b - b.mean()
    return (a * b).sum(1) / (np.sqrt((a * a).sum(1) * (b * b).sum()) + 1e-9)


def rankn(x):
    r = np.empty_like(x); r[np.argsort(x)] = np.arange(len(x)); return r / (len(x) - 1) - 0.5


def target(H, W, seed, nb):
    r = np.random.default_rng(seed); yy, xx = np.mgrid[0:H, 0:W].astype(float); f = np.zeros((H, W))
    for _ in range(nb):
        cy, cx = r.uniform(0.5, H - 1.5), r.uniform(0.5, W - 1.5)
        sy, sx = r.uniform(0.7, 1.4), r.uniform(0.7, 1.4)
        a = r.uniform(0.5, 1) * (1 if r.random() < 0.7 else -1)
        f += a * np.exp(-((yy - cy) ** 2 / (2 * sy ** 2) + (xx - cx) ** 2 / (2 * sx ** 2)))
    f -= f.min(); f /= f.max() + 1e-9; return f


def dl_bits(field, q=16):
    qf = np.clip((field * (q - 1)).round().astype(np.uint8), 0, q - 1)
    return 8 * len(zlib.compress(qf.tobytes(), 9))


def eff_dim(field):
    m = field - field.mean(); s = np.linalg.svd(m, compute_uv=False)
    p = s ** 2 / (s ** 2).sum(); return 1.0 / (p ** 2).sum()


# --------------------------------------------------------------------------- #
def experiment_A():
    print("=" * 74)
    print("(A) AMORTIZATION: one shared rule + small per-target codes; held-out test")
    print("=" * 74)
    H = W = 8; C = 8; HID = 12; CODE = 6; STEPS = 18
    GEN, grow = build(H, W, C, HID)
    K = 3
    train = [target(H, W, 200 + k, nb=4) for k in range(K)]
    held = target(H, W, 777, nb=4)
    print(f"grid {H}x{W}, shared body |theta|={GEN}, code dim d={CODE}, "
          f"K_train={K}, STEPS={STEPS}")

    # ---- jointly evolve shared body + K codes on the train targets ----------
    dim = GEN + K * CODE
    rng = np.random.default_rng(0); x = rng.normal(0, 0.05, dim); pop = 48; half = pop // 2
    t0 = time.time()
    for g in range(320):
        sigma = 0.12 * 0.5 ** (g / 320); lr = 0.10 * 0.4 ** (g / 320)
        eps = rng.normal(0, 1, (half, dim)); eps = np.concatenate([eps, -eps], 0)
        cand = x[None, :] + sigma * eps
        body = cand[:, :GEN]
        fit = np.zeros(pop)
        for k in range(K):
            ck = cand[:, GEN + k * CODE:GEN + (k + 1) * CODE]
            fit += corr_batch(morph(grow(body, ck, STEPS)), train[k]) / K
        adv = rankn(fit); x = x + lr * (eps * adv[:, None]).mean(0) / sigma - 1e-4 * x
    body = x[None, :GEN]
    tr_fid = [float(corr_batch(morph(grow(body, x[None, GEN + k * CODE:GEN + (k + 1) * CODE], STEPS)),
                               train[k])[0]) for k in range(K)]
    print(f"  train fidelity per target (shared body, own code): "
          f"{[round(v,3) for v in tr_fid]}   ({time.time()-t0:.0f}s)")

    # ---- FREEZE body; fit only a d-dim code to a HELD-OUT target ------------
    body_frozen = np.broadcast_to(x[:GEN], (1, GEN))
    rng2 = np.random.default_rng(1); c = rng2.normal(0, 0.1, CODE); pop2 = 32; half2 = pop2 // 2
    for g in range(140):
        sigma = 0.15 * 0.4 ** (g / 140)
        eps = rng2.normal(0, 1, (half2, CODE)); eps = np.concatenate([eps, -eps], 0)
        cand = c[None, :] + sigma * eps
        bodyb = np.broadcast_to(x[:GEN], (pop2, GEN))
        fit = corr_batch(morph(grow(bodyb, cand, STEPS)), held)
        adv = rankn(fit); c = c + 0.1 * (eps * adv[:, None]).mean(0) / sigma
    held_fid = float(corr_batch(morph(grow(body_frozen, c[None], STEPS)), held)[0])
    # baseline: best code for held-out found WITHOUT the shared body being right?
    # control = a random frozen body (never trained) + code search:
    rngc = np.random.default_rng(9); rand_body = np.broadcast_to(rngc.normal(0, 0.5, GEN), (1, GEN))
    cc = rngc.normal(0, 0.1, CODE)
    for g in range(140):
        sigma = 0.15 * 0.4 ** (g / 140)
        eps = rngc.normal(0, 1, (half2, CODE)); eps = np.concatenate([eps, -eps], 0)
        cand = cc[None, :] + sigma * eps
        bodyb = np.broadcast_to(rand_body[0], (pop2, GEN))
        fit = corr_batch(morph(grow(bodyb, cand, STEPS)), held)
        adv = rankn(fit); cc = cc + 0.1 * (eps * adv[:, None]).mean(0) / sigma
    ctrl_fid = float(corr_batch(morph(grow(rand_body, cc[None], STEPS)), held)[0])

    DL_held = dl_bits(held)
    print(f"  HELD-OUT target (body FROZEN, only a {CODE}-param code fit):")
    print(f"    held-out fidelity (trained body + code) = {held_fid:.3f}")
    print(f"    control (random body + code)            = {ctrl_fid:.3f}")
    print(f"    => body generalises beyond its train set: {held_fid - ctrl_fid:+.3f} above control")
    print(f"  MARGINAL accounting for the held-out target:")
    print(f"    DL(held-out target)          = {DL_held} bits")
    code_bits = CODE * 6                          # ~6 effective bits per code coord
    print(f"    marginal shaping |code|      ~ {code_bits} bits ({CODE} params x ~6 bits)")
    valid = held_fid > ctrl_fid + 0.05
    print(f"    marginal generativity  Delta = DL - |code| = {DL_held - code_bits:+d} bits "
          f"-- but VOID unless the frozen body generalises: held_fid {held_fid:.2f} vs "
          f"control {ctrl_fid:.2f} -> {'VALID' if valid else 'VOID (body did not generalise)'}")
    print(f"  AMORTIZED family accounting (K={K} train + 1 held):")
    tot_ind = sum(dl_bits(t) for t in train) + DL_held
    tot_shared = GEN * 6 + (K + 1) * code_bits    # shared body once + codes
    print(f"    describe all {K+1} targets independently: sum DL   = {tot_ind} bits")
    print(f"    via shared body + {K+1} codes:            = {tot_shared} bits "
          f"(body ~{GEN*6}, codes ~{(K+1)*code_bits})")
    return dict(train_fid=tr_fid, held_fid=held_fid, ctrl_fid=ctrl_fid,
                DL_held=DL_held, code_bits=code_bits,
                marginal_delta=DL_held - code_bits)


def experiment_B():
    print("\n" + "=" * 74)
    print("(B) HONEST SCALING: does DL grow Theta(N)? Only if complexity scales.")
    print("=" * 74)
    print("  fixed blob count (audit's family): DL should be SUB-linear (DL/N falls)")
    for n in (10, 20, 40):
        t = target(n, n, 42, nb=12)
        print(f"    N={n*n:5d}  nb=12(fixed) DL={dl_bits(t):5d}  DL/N={dl_bits(t)/(n*n):.2f}  "
              f"effdim={eff_dim(t):.2f}")
    print("  complexity-scaled family (nb ~ N/8): DL should be ~LINEAR (DL/N ~ const)")
    res = []
    for n in (10, 20, 40):
        nb = max(4, (n * n) // 8)
        t = target(n, n, 42, nb=nb)
        d = dl_bits(t); res.append((n * n, d))
        print(f"    N={n*n:5d}  nb={nb:4d}(~N/8) DL={d:5d}  DL/N={d/(n*n):.2f}  "
              f"effdim={eff_dim(t):.2f}")
    print("  => Theta(N) generativity is available ONLY for complexity-scaled targets;")
    print("     with |theta| fixed in N, Delta=DL-I grows ~linearly for that family,")
    print("     PROVIDED a low-I shaper can still find the rule (the open O3 frontier).")
    return res


if __name__ == "__main__":
    t0 = time.time()
    A = experiment_A()
    B = experiment_B()
    print(f"\n[done] runtime = {time.time()-t0:.1f}s")
    print("JSON_AMORT_BEGIN"); print(json.dumps(dict(A=A, B=B))); print("JSON_AMORT_END")
