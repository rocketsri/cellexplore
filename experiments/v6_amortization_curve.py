"""
v6_amortization_curve.py
========================

The decisive follow-up to v5's negative amortization result.

v5 tested amortization at K_train=3 with a *seed-injected* code and it failed the
held-out acid test (frozen body generalised WORSE than a random body). The
hypothesis: amortized generativity should EMERGE as K_train grows -- more targets
FORCE the shared rule to become a generic morphogenetic prior rather than
memorising a few codes. This experiment tests that hypothesis with:

  (1) FiLM modulation instead of seed-injection. A per-target code c in R^d
      modulates every cell's shared rule (feature-wise scale+shift of the hidden
      layer) -- a GRN-like global regulatory setting, far more expressive than a
      seed. Body {W1,b1,W2,b2,Wg,Wb} is SHARED across all targets; only c differs.

  (2) A K_train sweep {2,6,12,24}. For each: evolve body + K codes on the train
      set (minibatched over targets), FREEZE the body, then fit only a d-parameter
      code to HELD-OUT targets it was never shaped on. Held-out fidelity vs
      K_train is the curve that decides whether a positive amortized gap emerges.

  (3) Controls: a random (never-trained) frozen body + code search, per held-out
      target -- the same control that beat the trained body in v5.

Consistent-measure accounting: the shared body is a fixed, reusable, target-
AGNOSTIC codebook (proven agnostic iff it generalises to held-out targets). Given
it, a new target costs |c| ~ d*6 bits; its standalone description is DL >> |c|.
So a POSITIVE held-out margin over control is the audit-surviving evidence v5
lacked. A negative result bounds the generativity and is reported as such.

Pure numpy. Deterministic. ~8-10 min.
"""

import numpy as np
import zlib
import time
import json

H = W = 8
C = 8
HID = 12
D = 8            # code dimension (per-target regulatory setting)
STEPS = 16

SX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float) / 8.0
SY = SX.T
IDG = np.zeros((3, 3)); IDG[1, 1] = 1.0
FILT = [IDG, SX, SY]
PERC = C * 3

_SIZES = [("W1", PERC * HID), ("b1", HID), ("W2", HID * C),
          ("b2", C), ("Wg", HID * D), ("Wb", HID * D)]
BODY = sum(s for _, s in _SIZES)          # shared-rule parameter count


def unpack_body(b):
    P = b.shape[0]; i = 0; o = {}
    for name, s in _SIZES:
        o[name] = b[:, i:i + s]; i += s
    return (o["W1"].reshape(P, PERC, HID), o["b1"],
            o["W2"].reshape(P, HID, C), o["b2"],
            o["Wg"].reshape(P, HID, D), o["Wb"].reshape(P, HID, D))


def dw(fld, k):
    P, h, w, c = fld.shape
    pad = np.pad(fld, ((0, 0), (1, 1), (1, 1), (0, 0))); out = np.zeros_like(fld)
    for dy in range(3):
        for dx in range(3):
            if k[dy, dx]:
                out += k[dy, dx] * pad[:, dy:dy + h, dx:dx + w, :]
    return out


def step(s, body, codes):
    W1, b1, W2, b2, Wg, Wb = unpack_body(body)
    gamma = 1.0 + np.einsum('phd,pd->ph', Wg, codes)     # FiLM scale  (P,HID)
    beta = np.einsum('phd,pd->ph', Wb, codes)            # FiLM shift  (P,HID)
    p = np.concatenate([dw(s, f) for f in FILT], -1)
    h = np.tanh(np.einsum('phwq,pqk->phwk', p, W1) + b1[:, None, None, :])
    h = gamma[:, None, None, :] * h + beta[:, None, None, :]
    du = np.einsum('phwk,pkc->phwc', h, W2) + b2[:, None, None, :]
    return np.clip(s + du, -2, 2)


def grow(body, codes, steps=STEPS, dmg=None):
    P = body.shape[0]
    s = np.zeros((P, H, W, C)); s[:, H // 2, W // 2, 0] = 1.0
    for t in range(steps):
        if dmg is not None and t == dmg:
            s[:, :, W // 2:, :] = 0.0
        s = step(s, body, codes)
    return s


def morph(s):
    return np.clip(s[..., 0], 0, 1)


def corr_batch(mm, t):
    a = mm.reshape(mm.shape[0], -1); b = t.ravel()[None, :]
    a = a - a.mean(1, keepdims=True); b = b - b.mean()
    return (a * b).sum(1) / (np.sqrt((a * a).sum(1) * (b * b).sum()) + 1e-9)


def rankn(x):
    r = np.empty_like(x); r[np.argsort(x)] = np.arange(len(x)); return r / (len(x) - 1) - 0.5


def target(seed, nb=4):
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


# --------------------------------------------------------------------------- #
def train_body(train_targets, generations=260, pop=36, M=5, seed=0):
    """Jointly evolve shared body + one code per train target (target-minibatched)."""
    K = len(train_targets)
    dim = BODY + K * D
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.05, dim)
    half = pop // 2
    for g in range(generations):
        sigma = 0.12 * 0.5 ** (g / generations); lr = 0.10 * 0.4 ** (g / generations)
        eps = rng.normal(0, 1, (half, dim)); eps = np.concatenate([eps, -eps], 0)
        cand = x[None, :] + sigma * eps
        body = cand[:, :BODY]
        codes_all = cand[:, BODY:].reshape(pop, K, D)
        idx = rng.choice(K, size=min(M, K), replace=False)
        fit = np.zeros(pop)
        for k in idx:
            fit += corr_batch(morph(grow(body, codes_all[:, k, :])), train_targets[k])
        fit /= len(idx)
        adv = rankn(fit)
        x = x + lr * (eps * adv[:, None]).mean(0) / sigma - 1e-4 * x
    body = x[:BODY]
    codes = x[BODY:].reshape(K, D)
    tr = [float(corr_batch(morph(grow(body[None], codes[k][None])), train_targets[k])[0])
          for k in range(K)]
    return body, np.mean(tr)


def fit_code(body, tgt, generations=120, pop=28, seed=1):
    """Freeze body; fit only a d-dim code to a target."""
    rng = np.random.default_rng(seed)
    c = rng.normal(0, 0.1, D); half = pop // 2
    bodyb = np.broadcast_to(body, (pop, BODY))
    for g in range(generations):
        sigma = 0.2 * 0.35 ** (g / generations)
        eps = rng.normal(0, 1, (half, D)); eps = np.concatenate([eps, -eps], 0)
        cand = c[None, :] + sigma * eps
        fit = corr_batch(morph(grow(bodyb, cand)), tgt)
        adv = rankn(fit); c = c + 0.15 * (eps * adv[:, None]).mean(0) / sigma
    return float(corr_batch(morph(grow(body[None], c[None])), tgt)[0])


def main():
    t0 = time.time()
    print("=" * 74)
    print("v6 / A: AMORTIZATION CURVE -- does held-out generativity emerge with K_train?")
    print("=" * 74)
    print(f"grid {H}x{W}, shared body |theta|={BODY}, code dim d={D}, STEPS={STEPS}")

    pool = [target(s) for s in range(30)]                 # train pool
    held = [target(200 + k) for k in range(4)]            # held-out (never trained)
    DL_held = np.mean([dl_bits(t) for t in held])
    code_bits = D * 6

    # control: random frozen body, fit code per held-out target
    rngb = np.random.default_rng(99)
    ctrl = np.mean([fit_code(rngb.normal(0, 0.5, BODY), h, seed=500 + i)
                    for i, h in enumerate(held)])
    print(f"\ncontrol (random frozen body + {D}-dim code) held-out corr = {ctrl:.3f}")
    print(f"DL(held-out) ~ {DL_held:.0f} bits, marginal |code| ~ {code_bits} bits\n")

    rows = []
    for K in (2, 6, 12, 24):
        body, tr = train_body(pool[:K], seed=0)
        ho = [fit_code(body, h, seed=300 + i) for i, h in enumerate(held)]
        ho_mean = float(np.mean(ho))
        margin = ho_mean - ctrl
        rows.append((K, tr, ho_mean, margin))
        print(f"K_train={K:3d}: train corr={tr:.3f}  held-out corr={ho_mean:.3f}  "
              f"margin over control={margin:+.3f}  "
              f"({'GENERALISES' if margin > 0.05 else 'no'})  [{time.time()-t0:.0f}s]")

    best = max(rows, key=lambda r: r[3])
    print(f"\nBest K_train={best[0]}: held-out {best[2]:.3f} vs control {ctrl:.3f} "
          f"(margin {best[3]:+.3f})")
    if best[3] > 0.05:
        print(f"=> POSITIVE amortized generativity: a frozen generic body + {D}-param code "
              f"reaches held-out targets.")
        print(f"   marginal Delta = DL - |code| = {DL_held:.0f} - {code_bits} "
              f"= {DL_held - code_bits:+.0f} bits, measure-consistent (body is a proven-"
              f"agnostic shared codebook).")
    else:
        print(f"=> NEGATIVE even at K_train={best[0]}: no generic prior emerges under ES; "
              f"positive marginal gap NOT realised. Bounds the generativity (see notes).")

    trend = "RISING" if rows[-1][2] > rows[0][2] + 0.03 else ("flat/none")
    print(f"held-out trend across K_train: {[round(r[2],3) for r in rows]}  ({trend})")
    print(f"\n[done] runtime={time.time()-t0:.0f}s")
    print("JSON_V6_BEGIN")
    print(json.dumps(dict(control=ctrl, DL_held=float(DL_held), code_bits=code_bits,
                          rows=rows, best_K=best[0], best_margin=best[3])))
    print("JSON_V6_END")


if __name__ == "__main__":
    main()
