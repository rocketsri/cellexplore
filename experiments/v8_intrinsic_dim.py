"""
v8_intrinsic_dim.py
===================

Tests the efficiency hypothesis: v6 concluded that shaping the shared codebook
("toolkit") with a low-I evolution strategy FAILED -- but that used vanilla ES on
the full ~1800-dim dense body, the least sample-efficient possible setup. The
hypothesis (following Li et al. 2018, "Measuring the Intrinsic Dimension of
Objective Landscapes"): the toolkit-shaping objective may have a much smaller
INTRINSIC dimension, so a low-I, non-differentiable search over a COMPRESSED
coordinate g (body = theta0 + P g, P a fixed random projection) can build a
generic codebook where full-dimensional ES could not.

This is the measurable core of the (iii) "developmental / indirect encoding with
measurable generativity" route, and it directly answers the user's point that
efficient search / compressive encodings should make the toolkit cheap to shape.

Design: FiLM-modulated NCA (HID=32 body, the capacity where v6 backprop found a
generic prior). Body is reconstructed from a d-dim vector g via a FIXED random
orthonormal projection P (d << |theta|). A single low-I ES jointly searches
[g ; K_train codes] on train targets (target-minibatched, scalar corr per
rollout -- non-differentiable). Then the body is FROZEN and held-out targets are
selected by scalar ES on their code. Held-out realizability vs d, against a
random-body control, is the curve that decides whether compression closes the gap.

Pure numpy. Deterministic. ~8-12 min.
"""

import numpy as np
import time
import json

H = W = 8
C = 8
HID = 32
D = 12
STEPS = 20

SX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float) / 8.0
FILT = [np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], float), SX, SX.T]
PERC = C * 3
_SIZES = [("W1", PERC * HID), ("b1", HID), ("W2", HID * C),
          ("b2", C), ("Wg", HID * D), ("Wb", HID * D)]
BODY = sum(s for _, s in _SIZES)


def unpack_body(b):
    P = b.shape[0]; i = 0; o = {}
    for name, s in _SIZES:
        o[name] = b[:, i:i + s]; i += s
    return (o["W1"].reshape(P, PERC, HID), o["b1"], o["W2"].reshape(P, HID, C),
            o["b2"], o["Wg"].reshape(P, HID, D), o["Wb"].reshape(P, HID, D))


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
    gamma = 1.0 + np.einsum('phd,pd->ph', Wg, codes)
    beta = np.einsum('phd,pd->ph', Wb, codes)
    p = np.concatenate([dw(s, f) for f in FILT], -1)
    h = np.tanh(np.einsum('phwq,pqk->phwk', p, W1) + b1[:, None, None, :])
    h = gamma[:, None, None, :] * h + beta[:, None, None, :]
    du = np.einsum('phwk,pkc->phwc', h, W2) + b2[:, None, None, :]
    return np.clip(s + du, -3, 3)


def grow(body, codes, steps=STEPS):
    P = body.shape[0]
    s = np.zeros((P, H, W, C)); s[:, H // 2, W // 2, :] = 1.0
    for _ in range(steps):
        s = step(s, body, codes)
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


def random_projection(d, seed):
    """Fixed random orthonormal BODY x d projection (columns orthonormal)."""
    rng = np.random.default_rng(seed)
    if d >= BODY:
        return np.eye(BODY)
    Aq, _ = np.linalg.qr(rng.normal(0, 1, (BODY, d)))
    return Aq                                                 # (BODY, d), orthonormal cols


def train_toolkit(train_targets, d, generations=280, pop=44, M=5, seed=0):
    """Low-I ES over [g (d-dim body coords) ; K codes]. body = theta0 + P g."""
    K = len(train_targets)
    theta0 = np.random.default_rng(100 + seed).normal(0, 0.1, BODY)
    P = random_projection(d, 7 + seed)
    dim = d + K * D
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 0.1, dim)
    half = pop // 2
    for g in range(generations):
        sigma = 0.15 * 0.4 ** (g / generations); lr = 0.12 * 0.4 ** (g / generations)
        eps = rng.normal(0, 1, (half, dim)); eps = np.concatenate([eps, -eps], 0)
        cand = z[None, :] + sigma * eps
        gcoords = cand[:, :d]
        codes_all = cand[:, d:].reshape(pop, K, D)
        bodies = theta0[None, :] + gcoords @ P.T               # (pop, BODY)
        idx = rng.choice(K, size=min(M, K), replace=False)
        fit = np.zeros(pop)
        for k in idx:
            fit += corr_batch(grow(bodies, codes_all[:, k, :]), train_targets[k])
        fit /= len(idx)
        z = z + lr * (eps * rankn(fit)[:, None]).mean(0) / sigma - 1e-4 * z
    body = (theta0 + z[:d] @ P.T)[None, :]
    codes = z[d:].reshape(K, D)
    tr = float(np.mean([corr_batch(grow(body, codes[k][None]), train_targets[k])[0]
                        for k in range(K)]))
    return body, tr


def es_select(body, tgt, pop=32, gens=120, seed=1):
    """Scalar-ES per-target code selection on a FROZEN body (low-I selection)."""
    rng = np.random.default_rng(seed); c = np.zeros(D); half = pop // 2
    bb = np.broadcast_to(body[0], (pop, BODY))
    for g in range(gens):
        sigma = 0.3 * 0.3 ** (g / gens)
        eps = rng.normal(0, 1, (half, D)); eps = np.concatenate([eps, -eps], 0)
        fit = corr_batch(grow(bb, c[None, :] + sigma * eps), tgt)
        c = c + 0.2 * (eps * rankn(fit)[:, None]).mean(0) / sigma
    return float(corr_batch(grow(body, c[None]), tgt)[0])


def main():
    t0 = time.time()
    print("=" * 74)
    print("v8: INTRINSIC-DIMENSION test -- can low-I ES build a codebook in COMPRESSED coords?")
    print("=" * 74)
    print(f"grid {H}x{W}, body |theta|={BODY} (HID={HID}), code dim d={D}, STEPS={STEPS}")
    pool = [target(s) for s in range(24)]
    held = [target(200 + k) for k in range(4)]

    # control: random frozen body, scalar-ES held-out selection
    rng = np.random.default_rng(5)
    ctrl = float(np.mean([es_select(rng.normal(0, 0.3, BODY)[None], h, seed=40 + i)
                          for i, h in enumerate(held)]))
    print(f"\ncontrol (random frozen body + scalar-ES code): held-out corr = {ctrl:.3f}\n")

    K_train = 12
    rows = []
    for d in (32, 128, 512, BODY):
        body, tr = train_toolkit(pool[:K_train], d=d, seed=0)
        ho = float(np.mean([es_select(body, h, seed=60 + i) for i, h in enumerate(held)]))
        rows.append((d, tr, ho, ho - ctrl))
        tag = "FULL" if d >= BODY else f"{d}/{BODY}"
        print(f"  search dim d={tag:>10s}: train corr={tr:.3f}  held-out corr={ho:.3f}  "
              f"margin over control={ho-ctrl:+.3f}  ({'GENERALISES' if ho-ctrl>0.05 else 'no'})"
              f"  [{time.time()-t0:.0f}s]")

    comp = [r for r in rows if r[0] < BODY]
    full = [r for r in rows if r[0] >= BODY][0]
    best_comp = max(comp, key=lambda r: r[2])
    print(f"\n  best COMPRESSED d={best_comp[0]}: held-out {best_comp[2]:.3f} (margin {best_comp[3]:+.3f})")
    print(f"  FULL-dim ES (d={BODY}):            held-out {full[2]:.3f} (margin {full[3]:+.3f})")
    if best_comp[3] > 0.05 and best_comp[2] > full[2] + 0.03:
        print(f"  => COMPRESSION HELPS: low-I ES builds a generalising codebook in {best_comp[0]} dims")
        print(f"     that full-dim ES ({BODY} dims) does not. The toolkit-shaping objective has low")
        print(f"     intrinsic dimension; 'expensive/deep-time' was an artifact of bad coordinates.")
    elif best_comp[3] > 0.05:
        print(f"  => Compressed search generalises (margin {best_comp[3]:+.3f}), but not clearly better")
        print(f"     than full-dim here; low-I codebook-building is feasible in compressed coords.")
    else:
        print(f"  => NEGATIVE: even compressed low-I ES does not build a generalising codebook here;")
        print(f"     the gap is not merely coordinates at this scale.")
    print(f"\n[done] runtime={time.time()-t0:.0f}s")
    print("JSON_V8_BEGIN")
    print(json.dumps(dict(control=ctrl, rows=rows, BODY=BODY)))
    print("JSON_V8_END")


if __name__ == "__main__":
    main()
