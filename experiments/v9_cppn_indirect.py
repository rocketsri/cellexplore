"""
v9_cppn_indirect.py
===================

DEVELOPMENTAL / INDIRECT ENCODING with measurable generativity (the CPPN/HyperNEAT
family, outside the diffusion/flow/LV/variational/RG registry).

Question this settles (the CRUX of the task): v6 showed low-I ES on the full
~1832-dim DENSE body fails to build a target-agnostic codebook (held-out margin
NEGATIVE). v8 asks whether a RANDOM LINEAR PROJECTION (= the generic Li-et-al.
intrinsic-dimension probe) rescues it. This script adds the STRUCTURED alternative:
a HyperNEAT-style CPPN that DEVELOPS the whole 1832-param FiLM-NCA body from a tiny
genotype g (|g| ~ 100-200 << |theta|), and runs it HEAD-TO-HEAD against a random
linear projection at the SAME search dimension. That comparison separates the two
hypotheses:
   - "the win is just lower dimension"  (random projection already gets it), vs
   - "the win is developmental STRUCTURE" (CPPN's geometric regularity /
      sin/gauss/abs primitives beat a random subspace of equal size).

Everything else -- body, targets, ES, held-out protocol, random-body control -- is
IDENTICAL to v8, so the ONLY independent variable is the decoder g -> theta.

Measure-consistent generativity (B1): the codebook is charged ONCE at |g| bits (not
|theta|), because under one fixed decoder U (= CPPN-develop + run-NCA) the codebook's
description length is K_U(theta) <= |g| + O(1). The recurring per-target cost is the
|code| = D*precision bits. Marginal generativity Delta = D_bar - |code|, provably
bounded above by |g| + O(1) (Thm in v9.md). Held-out realizability on the FROZEN
developed body is the acid test.

Pure numpy. Deterministic. ~8-12 min.
"""

import numpy as np
import time
import json

# ---- body / substrate: IDENTICAL to v8 ------------------------------------
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


# ---- CPPN developmental decoder (HyperNEAT-style) -------------------------
# The genotype g is a small fixed-topology CPPN with MIXED activation primitives
# (tanh / sin / gaussian / abs) -- the "compositional pattern producing" bias that
# yields geometric regularity (symmetry, repetition, gradients). It maps
# (source_coord, target_coord, tensor_id) -> weight, painting all 1832 body params.
NF = 10           # coordinate feature count


def cppn_feats(a, b):
    """(S,) source coord a, (S,) target coord b  ->  (S, NF) coordinate features.
    These bake in the CPPN primitives: linear (gradients), product (interaction),
    sin (repetition), gaussian of |a-b| (locality/symmetry), squares (curvature)."""
    return np.stack([np.ones_like(a), a, b, a * b, np.abs(a - b),
                     np.sin(np.pi * a), np.sin(np.pi * b),
                     np.exp(-4.0 * (a - b) ** 2), a * a, b * b], axis=1)


# tensor table: (name, source-coord array, target-coord array, head index)
def _coords():
    perc = np.linspace(-1, 1, PERC)
    hid = np.linspace(-1, 1, HID)
    chan = np.linspace(-1, 1, C)
    cod = np.linspace(-1, 1, D)
    zero = np.array([0.0])
    T = [
        ("W1", perc, hid, 0),   # (PERC, HID)
        ("b1", zero, hid, 1),   # (HID,)
        ("W2", hid, chan, 2),   # (HID, C)
        ("b2", zero, chan, 3),  # (C,)
        ("Wg", hid, cod, 4),    # (HID, D)
        ("Wb", hid, cod, 5),    # (HID, D)
    ]
    grids = {}
    for name, sc, tc, head in T:
        A = np.repeat(sc, len(tc)); B = np.tile(tc, len(sc))
        grids[name] = (cppn_feats(A, B), len(sc), len(tc), head)
    return T, grids


_TTAB, _GRIDS = _coords()
NHEAD = 6


def cppn_dg(Hc):
    return NF * Hc + Hc * NHEAD + NHEAD + NHEAD   # Wc1 + Wc2 + bc + gain


def cppn_init(Hc, seed):
    rng = np.random.default_rng(seed)
    Wc1 = rng.normal(0, 0.6, (NF, Hc))
    Wc2 = rng.normal(0, 0.6, (Hc, NHEAD))
    bc = np.zeros(NHEAD)
    gain = np.full(NHEAD, 0.3)
    return np.concatenate([Wc1.ravel(), Wc2.ravel(), bc, gain])


def _mixed_act(pre):
    """pre (pop,S,Hc) -> mixed activations by hidden-unit index mod 4."""
    Hc = pre.shape[-1]
    out = np.empty_like(pre)
    idx = np.arange(Hc) % 4
    m0 = idx == 0; m1 = idx == 1; m2 = idx == 2; m3 = idx == 3
    out[..., m0] = np.tanh(pre[..., m0])
    out[..., m1] = np.sin(pre[..., m1])
    out[..., m2] = np.exp(-pre[..., m2] ** 2)
    out[..., m3] = np.abs(pre[..., m3]) - 0.5   # centered abs
    return out


def decode_cppn(G, Hc):
    """G (pop, d_g) -> bodies (pop, BODY). Develops the full body from the genotype."""
    pop = G.shape[0]
    i = 0
    Wc1 = G[:, i:i + NF * Hc].reshape(pop, NF, Hc); i += NF * Hc
    Wc2 = G[:, i:i + Hc * NHEAD].reshape(pop, Hc, NHEAD); i += Hc * NHEAD
    bc = G[:, i:i + NHEAD]; i += NHEAD
    gain = G[:, i:i + NHEAD]
    parts = []
    for name, s in _SIZES:
        F, m, n, head = _GRIDS[name]
        pre = np.einsum('sf,pfh->psh', F, Wc1)          # (pop, S, Hc)
        Hh = _mixed_act(pre)
        out = np.einsum('psh,pht->pst', Hh, Wc2) + bc[:, None, :]  # (pop, S, NHEAD)
        w = out[:, :, head] * gain[:, head][:, None]    # (pop, S)
        parts.append(w)                                  # S == m*n (row-major)
    return np.concatenate(parts, axis=1)                 # (pop, BODY)


def random_projection(d, seed):
    rng = np.random.default_rng(seed)
    if d >= BODY:
        return np.eye(BODY)
    Aq, _ = np.linalg.qr(rng.normal(0, 1, (BODY, d)))
    return Aq


# ---- ES codebook builders (identical loop; only the decoder differs) ------
def train_cppn(train_targets, Hc, generations=240, pop=44, M=5, seed=0):
    K = len(train_targets)
    g0 = cppn_init(Hc, 100 + seed)
    dg = len(g0)
    dim = dg + K * D
    rng = np.random.default_rng(seed)
    z = np.concatenate([g0, rng.normal(0, 0.1, K * D)])
    half = pop // 2
    for gg in range(generations):
        sigma = 0.15 * 0.4 ** (gg / generations); lr = 0.12 * 0.4 ** (gg / generations)
        eps = rng.normal(0, 1, (half, dim)); eps = np.concatenate([eps, -eps], 0)
        cand = z[None, :] + sigma * eps
        bodies = decode_cppn(cand[:, :dg], Hc)
        codes_all = cand[:, dg:].reshape(pop, K, D)
        idx = rng.choice(K, size=min(M, K), replace=False)
        fit = np.zeros(pop)
        for k in idx:
            fit += corr_batch(grow(bodies, codes_all[:, k, :]), train_targets[k])
        fit /= len(idx)
        z = z + lr * (eps * rankn(fit)[:, None]).mean(0) / sigma - 1e-4 * z
    body = decode_cppn(z[None, :dg], Hc)
    codes = z[dg:].reshape(K, D)
    tr = float(np.mean([corr_batch(grow(body, codes[k][None]), train_targets[k])[0]
                        for k in range(K)]))
    return body, tr, dg


def train_rp(train_targets, d, generations=240, pop=44, M=5, seed=0):
    K = len(train_targets)
    theta0 = np.random.default_rng(100 + seed).normal(0, 0.1, BODY)
    P = random_projection(d, 7 + seed)
    dim = d + K * D
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 0.1, dim)
    half = pop // 2
    for gg in range(generations):
        sigma = 0.15 * 0.4 ** (gg / generations); lr = 0.12 * 0.4 ** (gg / generations)
        eps = rng.normal(0, 1, (half, dim)); eps = np.concatenate([eps, -eps], 0)
        cand = z[None, :] + sigma * eps
        bodies = theta0[None, :] + cand[:, :d] @ P.T
        codes_all = cand[:, d:].reshape(pop, K, D)
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
    """Scalar-ES per-target code selection on a FROZEN body (the low-I acid test)."""
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
    print("=" * 78)
    print("v9: CPPN DEVELOPMENTAL encoding vs RANDOM PROJECTION at EQUAL search dim")
    print("=" * 78)
    print(f"grid {H}x{W}, body |theta|={BODY} (HID={HID}), code dim d={D}, STEPS={STEPS}")
    pool = [target(s) for s in range(24)]
    held = [target(200 + k) for k in range(4)]

    rng = np.random.default_rng(5)
    ctrl = float(np.mean([es_select(rng.normal(0, 0.3, BODY)[None], h, seed=40 + i)
                          for i, h in enumerate(held)]))
    print(f"\ncontrol (random frozen body + scalar-ES code): held-out corr = {ctrl:.3f}")
    print(f"[v6 baseline: DENSE ES on full {BODY}-dim body -> held-out margin NEGATIVE]\n")

    K_train = 12
    rows = []
    for Hc in (6, 12):
        body_c, tr_c, dg = train_cppn(pool[:K_train], Hc=Hc, seed=0)
        ho_c = float(np.mean([es_select(body_c, h, seed=60 + i) for i, h in enumerate(held)]))
        body_r, tr_r = train_rp(pool[:K_train], d=dg, seed=0)
        ho_r = float(np.mean([es_select(body_r, h, seed=60 + i) for i, h in enumerate(held)]))
        rows.append(dict(Hc=Hc, dg=dg, cppn_tr=tr_c, cppn_ho=ho_c,
                         rp_tr=tr_r, rp_ho=ho_r, cppn_margin=ho_c - ctrl,
                         rp_margin=ho_r - ctrl, struct_edge=ho_c - ho_r))
        print(f"  d_g={dg:4d} (Hc={Hc:2d}) | CPPN: train={tr_c:.3f} held-out={ho_c:.3f} "
              f"(margin {ho_c-ctrl:+.3f}) || RandProj: train={tr_r:.3f} held-out={ho_r:.3f} "
              f"(margin {ho_r-ctrl:+.3f}) || STRUCT EDGE (CPPN-RP)={ho_c-ho_r:+.3f}  "
              f"[{time.time()-t0:.0f}s]")

    best = max(rows, key=lambda r: r["cppn_ho"])
    print(f"\n  best CPPN d_g={best['dg']}: held-out {best['cppn_ho']:.3f} "
          f"(margin {best['cppn_margin']:+.3f}); matched RandProj {best['rp_ho']:.3f}")
    print(f"  Delta_marginal ceiling = |g| = {best['dg']} params "
          f"(the Thm bound on amortized generativity)")
    if best["cppn_margin"] > 0.05 and best["struct_edge"] > 0.03:
        print("  => STRUCTURE WINS: CPPN develops a generalising codebook that (a) DENSE ES")
        print("     could not build and (b) beats a random projection of EQUAL dimension.")
        print("     Low-I codebook-building is closed by DEVELOPMENTAL structure, not dim alone.")
    elif best["cppn_margin"] > 0.05 and abs(best["struct_edge"]) <= 0.03:
        print("  => DIMENSION, NOT STRUCTURE: CPPN generalises but ties the random projection;")
        print("     the win is intrinsic-dimension reduction, which any compressor delivers.")
    elif best["rp_margin"] > 0.05:
        print("  => RANDOM PROJECTION generalises but CPPN does not: developmental bias HURTS here.")
    else:
        print("  => NEGATIVE: neither structured nor random compression builds a generalising")
        print("     codebook at this scale; the gap is not merely coordinates.")
    print(f"\n[done] runtime={time.time()-t0:.0f}s")
    print("JSON_V9_BEGIN")
    print(json.dumps(dict(control=ctrl, BODY=BODY, K_train=K_train, rows=rows)))
    print("JSON_V9_END")


if __name__ == "__main__":
    main()
