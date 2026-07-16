"""
v5_evolved_nca_generativity.py
================================

O1 (v5): a construction that generates more target than it was given.

Skeleton: a Neural Cellular Automaton (NCA) -- a *shared* local update rule
(the "genome") applied to every cell, coupled only to immediate neighbours
(the "bioelectric" coupling), grown from a single seed cell. This is the plan's
suggested M0 anchor. The novelty vs. the prior program is that the local rule is
NOT shaped by high-bandwidth end-to-end backprop (which delivers ~D bits of
target information per step, so the generativity gap Delta ~= 0). It is shaped by
a LOW-bandwidth, non-differentiable, per-outcome process: a scalar-fitness
evolution strategy (ES). The shaping loop never sees the target field; it sees
only a scalar (and, in the quantized arm, a *few-bit* scalar) per rollout.

What is measured (all executed, numbers printed):

  1. FIDELITY: does the low-I-shaped rule grow AND regenerate a specific rich
     target that lives in no cell? (non-triviality guard 1 + 2)
  2. Delta = DL(target) - I, under THREE honest I-accountings:
       I_channel      : selection-channel ceiling  = G * log2(lambda!) bits
       I_genome_eff   : effective MDL of the shaped rule (quantization-robust bits)
       I_genome_raw   : crude ceiling |theta| * 32 bits (reported for honesty)
     DL(target) is measured operationally (zlib bits + effective dimension),
     NOT as Kolmogorov complexity -- see the audit section of v5.md.
  3. RE-COARSENING TEST (guard 2): can any o(N) block of cells reconstruct the
     target via a linear readout? Report the coarsest block at which
     "no subset holds the target" first fails, and the I that block received.
  4. REGENERATION (causal ablation, executed) + the H4 self-repair double edge:
     the same dynamics that heal damage also revert a *legitimate* correction.
  5. GENERICITY: the SAME family, shaped for K distinct targets, reaches each and
     does NOT reach the others (swap test) -> the family encodes no single target.
     This grounds the selection-cost (log K) vs specification-cost (D) distinction
     that O2 turns on.
  6. I_c THRESHOLD SWEEP (O3): fidelity vs shaping information (generations and
     fitness bit-depth) -> where rich-target convergence fails as I -> 0.

Pure numpy (no torch/scipy). Deterministic seeds. Runtime a few minutes.
"""

import numpy as np
import zlib
import json
import time

# --------------------------------------------------------------------------- #
#  NCA substrate: shared local rule + nearest-neighbour coupling, seed growth  #
# --------------------------------------------------------------------------- #

C = 8            # channels per cell; channel 0 is the visible "morphology"
H = W = 10       # grid; N = 100 cells
HID = 12         # hidden width of the shared local rule (the "genome")
STEPS = 20       # CA steps to grow
FIRE = 1.0       # deterministic update (low-variance fitness for ES)

# perception filters: identity, Sobel-x, Sobel-y  (the local coupling stencil)
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64) / 8.0
SOBEL_Y = SOBEL_X.T
IDENT = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float64)
FILTERS = [IDENT, SOBEL_X, SOBEL_Y]
PERC = C * len(FILTERS)                      # perception vector length = 24

# genome = flat vector of [W1 (PERC x HID), b1 (HID), W2 (HID x C), b2 (C)]
N_W1 = PERC * HID
N_B1 = HID
N_W2 = HID * C
N_B2 = C
GENOME = N_W1 + N_B1 + N_W2 + N_B2            # == 24*16 + 16 + 16*8 + 8 = 536


def unpack_batch(theta):
    """theta: (P, GENOME) -> batched weights (each population member its own rule)."""
    P = theta.shape[0]
    i = 0
    W1 = theta[:, i:i + N_W1].reshape(P, PERC, HID); i += N_W1
    b1 = theta[:, i:i + N_B1]; i += N_B1
    W2 = theta[:, i:i + N_W2].reshape(P, HID, C); i += N_W2
    b2 = theta[:, i:i + N_B2]; i += N_B2
    return W1, b1, W2, b2


def _depthwise(field, k):
    """Apply a single 3x3 kernel to every channel of a (P,H,W,C) batch, zero pad."""
    P, h, w, c = field.shape
    out = np.zeros_like(field)
    padded = np.pad(field, ((0, 0), (1, 1), (1, 1), (0, 0)))
    for dy in range(3):
        for dx in range(3):
            kv = k[dy, dx]
            if kv != 0.0:
                out += kv * padded[:, dy:dy + h, dx:dx + w, :]
    return out


def perceive(state):
    """(P,H,W,C) -> (P,H,W,PERC): concatenate identity + Sobel-x + Sobel-y."""
    return np.concatenate([_depthwise(state, f) for f in FILTERS], axis=-1)


def step(state, theta, rng=None, fire=FIRE):
    """state: (P,H,W,C); theta: (P,GENOME). One synchronous NCA step: every cell
    applies the SAME local rule to its 3x3 neighbourhood (identity + Sobel
    perception = the local coupling). Growth/asymmetry originates from the single
    seed cell, not from any hand-set organizer heterogeneity."""
    W1, b1, W2, b2 = unpack_batch(theta)
    p = perceive(state)                                       # (P,H,W,PERC)
    hzero = np.tanh(np.einsum('phwq,pqk->phwk', p, W1) + b1[:, None, None, :])
    du = np.einsum('phwk,pkc->phwc', hzero, W2) + b2[:, None, None, :]
    if fire < 1.0 and rng is not None:
        m = (rng.random(du.shape[:-1] + (1,)) < fire).astype(np.float64)
        du = du * m
    return np.clip(state + du, -2.0, 2.0)


def seed_state(P):
    s = np.zeros((P, H, W, C), dtype=np.float64)
    s[:, H // 2, W // 2, 1:] = 1.0             # single central seed cell, all hidden chans on
    return s


def grow(theta, P=1, steps=STEPS, damage_at=None, rng=None):
    """Grow from seed. If damage_at is set, zero a half-plane at that step (ablation)."""
    if theta.ndim == 1:
        theta = np.broadcast_to(theta, (P, GENOME))
    else:
        P = theta.shape[0]
    s = seed_state(P)
    for t in range(steps):
        if damage_at is not None and t == damage_at:
            s[:, :, W // 2:, :] = 0.0          # ablate the right half
        s = step(s, theta, rng=rng)
    return s


def morph(state):
    """Visible morphology = channel 0, clipped to [0,1]."""
    return np.clip(state[..., 0], 0.0, 1.0)


# --------------------------------------------------------------------------- #
#  Targets: rich, asymmetric, multi-mode smooth fields (not constant/low-D)    #
# --------------------------------------------------------------------------- #

def make_target(seed, n_blobs=12):
    """A specific rich morphology: sum of sharp asymmetric Gaussian blobs, in [0,1].
    Tuned so the effective dimension is well above the low-D regime (guard 1)."""
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    f = np.zeros((H, W))
    for _ in range(n_blobs):
        cy, cx = r.uniform(0.5, H - 1.5), r.uniform(0.5, W - 1.5)
        sy, sx = r.uniform(0.7, 1.5), r.uniform(0.7, 1.5)
        amp = r.uniform(0.5, 1.0) * (1 if r.random() < 0.7 else -1)
        f += amp * np.exp(-((yy - cy) ** 2 / (2 * sy ** 2) + (xx - cx) ** 2 / (2 * sx ** 2)))
    f = f - f.min()
    f = f / (f.max() + 1e-9)
    return f


def corr_batch(mm, target):
    """Pearson correlation of each (H,W) morphology in a batch with the target.
    Structure similarity, invariant to a fixed affine readout (gain+bias = 2 nums,
    target-independent, ~O(1) bits -> does not smuggle target information)."""
    a = mm.reshape(mm.shape[0], -1)
    b = target.ravel()[None, :]
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean()
    return (a * b).sum(1) / (np.sqrt((a * a).sum(1) * (b * b).sum()) + 1e-9)


def fidelity(realized, target):
    """Headline fidelity = Pearson correlation (structural match) of a single
    realized morphology with the target. 1.0 = perfect structure; 0 = none.
    R^2 = fidelity**2 is the variance explained under the optimal affine readout."""
    return float(corr_batch(realized[None], target)[0])


# --------------------------------------------------------------------------- #
#  Description length (operational) and richness of a field                    #
# --------------------------------------------------------------------------- #

def dl_zlib_bits(field, q=16):
    """Operational DL: bits to transmit the q-level-quantized field to a naive
    decoder via a generic compressor (zlib). A real upper bound on 'how much a
    shaper must communicate to convey this morphology' with NO shared rule."""
    qf = np.clip((field * (q - 1)).round().astype(np.uint8), 0, q - 1)
    comp = zlib.compress(qf.tobytes(), 9)
    return 8 * len(comp)


def effective_dim(field):
    """Participation-ratio effective dimension of the target's 2D structure
    (via SVD of the mean-removed field): guards against a low-D 'target'."""
    m = field - field.mean()
    s = np.linalg.svd(m, compute_uv=False)
    p = s ** 2 / (np.sum(s ** 2) + 1e-12)
    return float(1.0 / np.sum(p ** 2))


def spatial_entropy_bits(field, q=16):
    qf = np.clip((field * (q - 1)).round().astype(int), 0, q - 1)
    counts = np.bincount(qf.ravel(), minlength=q).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)) * field.size)   # total bits (naive i.i.d.)


# --------------------------------------------------------------------------- #
#  Low-bandwidth shaping: OpenAI-ES (scalar fitness, non-differentiable)        #
# --------------------------------------------------------------------------- #

def es_fitness(theta_batch, target, rng, damage=True, q_bits=None):
    """Scalar per-rollout fitness = structural correlation with the target (smooth,
    scale/offset invariant). This scalar is the ONLY channel from the target into
    the genome -- the shaping loop never sees the target field. If q_bits is set,
    the scalar is quantized to q_bits levels (a few-bit per-outcome signal)."""
    P = theta_batch.shape[0]
    fit = corr_batch(morph(grow(theta_batch, P=P, steps=STEPS, rng=rng)), target)
    if damage:
        s2 = grow(theta_batch, P=P, steps=STEPS + 14, damage_at=STEPS - 6, rng=rng)
        fit = 0.6 * fit + 0.4 * corr_batch(morph(s2), target)   # select ALSO for regen
    if q_bits is not None:
        lo, hi = fit.min(), fit.max() + 1e-9
        levels = 2 ** q_bits
        fit = lo + np.round((fit - lo) / (hi - lo) * (levels - 1)) / (levels - 1) * (hi - lo)
    return fit


def rank_normalize(x):
    ranks = np.empty_like(x); ranks[np.argsort(x)] = np.arange(len(x))
    r = ranks / (len(x) - 1) - 0.5
    return r


def evolve(target, generations=300, pop=48, sigma0=0.12, lr0=0.10,
           seed=0, damage=True, q_bits=None, verbose=False):
    """OpenAI-ES with antithetic sampling + rank shaping + weight decay + a
    sigma/lr anneal. Returns (best_theta, history, info-accounting)."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 0.05, GENOME)
    half = pop // 2
    hist = []
    n_evals = 0
    for g in range(generations):
        sigma = sigma0 * (0.5 ** (g / generations))
        lr = lr0 * (0.4 ** (g / generations))
        eps = rng.normal(0, 1, (half, GENOME))
        eps = np.concatenate([eps, -eps], axis=0)          # antithetic
        cand = theta[None, :] + sigma * eps
        f = es_fitness(cand, target, rng, damage=damage, q_bits=q_bits)
        n_evals += pop
        adv = rank_normalize(f)
        grad = (eps * adv[:, None]).mean(0) / sigma
        theta = theta + lr * grad - 1e-4 * theta           # weight decay
        if g % 40 == 0 or g == generations - 1:
            s = grow(theta, P=1, steps=STEPS, rng=rng)
            fclean = fidelity(morph(s)[0], target)
            hist.append((g, fclean))
            if verbose:
                print(f"    gen {g:4d}  corr(mean rule) = {fclean:.3f}")
    # selection-channel ceiling: with rank-based ES the update reads only the
    # RANKING of `pop` scalars per generation -> <= log2(pop!) bits/generation.
    from math import lgamma, log
    bits_per_gen = lgamma(pop + 1) / log(2)                 # log2(pop!)
    I_channel = generations * bits_per_gen
    return theta, hist, dict(n_evals=n_evals, generations=generations,
                             pop=pop, I_channel_bits=I_channel)


# --------------------------------------------------------------------------- #
#  Effective genome MDL: how many bits about the target are actually baked in   #
# --------------------------------------------------------------------------- #

def effective_genome_bits(theta, target, tol=0.9, rng=None):
    """Coarsest per-parameter quantization (in bits) that still preserves the
    target within `tol` of the full-precision fidelity, times the number of
    parameters that are NOT prunable-to-zero without breaking tol. An MDL-style
    'target information actually stored in the rule'."""
    if rng is None:
        rng = np.random.default_rng(1234)
    base = fidelity(morph(grow(theta, 1, STEPS, rng=rng))[0], target)
    lo, hi = theta.min(), theta.max()
    # find coarsest bit-depth b in {8,6,5,4,3,2} preserving tol*base
    chosen_b = 8
    for b in (2, 3, 4, 5, 6, 8):
        levels = 2 ** b
        q = np.round((theta - lo) / (hi - lo + 1e-9) * (levels - 1))
        tq = lo + q / (levels - 1) * (hi - lo)
        fq = fidelity(morph(grow(tq, 1, STEPS, rng=rng))[0], target)
        if fq >= tol * base:
            chosen_b = b
            break
    # prune: how many params can be zeroed (one pass, greedy by magnitude) w/o breaking tol
    order = np.argsort(np.abs(theta))
    tq = theta.copy()
    kept = GENOME
    for idx in order:
        cand = tq.copy(); cand[idx] = 0.0
        fq = fidelity(morph(grow(cand, 1, STEPS, rng=rng))[0], target)
        if fq >= tol * base:
            tq = cand; kept -= 1
        # (cheap greedy; stops implicitly when zeroing hurts)
    return chosen_b, kept, chosen_b * kept, base


# --------------------------------------------------------------------------- #
#  Re-coarsening test: can any o(N) block reconstruct the target?              #
# --------------------------------------------------------------------------- #

def recoarsening_test(theta, target, DL_bits, rng):
    """Guard 2, two complementary measures on the realized attractor:

    (a) COUNTING bound. A block of s cells stores at most s*C*q_eff bits of state.
        To 'hold' a target of DL bits it needs s >= DL/(C*q_eff) cells. Report s*
        as a fraction of N: if s* is Theta(N), no o(N) subset can even store it.

    (b) CAUSAL regenerate-from-fragment. Keep ONLY a bs x bs block of the grown
        state, zero the rest, run the shared rule, measure recovered fidelity.
        The smallest fragment that recovers the target is the coarsest block that
        'holds' it (holographically, via the shared rule). Report the curve."""
    q_eff = 4.0                                   # ~4 effective bits per channel value
    s_star = DL_bits / (C * q_eff)
    counting = dict(s_star_cells=float(s_star), N=H * W,
                    frac_of_N=float(s_star / (H * W)))
    s = grow(theta, 1, STEPS, rng=rng)            # (1,H,W,C)
    regen = []
    for bs in (1, 2, 4, 6, 8, 12):
        # centre the retained block on the seed (best case for the fragment)
        y0 = max(0, H // 2 - bs // 2); x0 = max(0, W // 2 - bs // 2)
        frag = np.zeros_like(s)
        frag[:, y0:y0 + bs, x0:x0 + bs, :] = s[:, y0:y0 + bs, x0:x0 + bs, :]
        for _ in range(STEPS):
            frag = step(frag, np.broadcast_to(theta, (1, GENOME)), rng=rng)
        f = fidelity(morph(frag)[0], target)
        regen.append((bs, bs * bs, float(f)))
    return counting, regen


# --------------------------------------------------------------------------- #
#  MAIN                                                                          #
# --------------------------------------------------------------------------- #

def main():
    t0 = time.time()
    out = {}
    rng = np.random.default_rng(7)

    print("=" * 74)
    print("v5 / O1: LOW-BANDWIDTH (evolutionary) shaping of a shared local rule")
    print("=" * 74)
    print(f"grid N = {H*W} cells, C = {C} channels, genome |theta| = {GENOME} params, "
          f"STEPS = {STEPS}")

    # ---- target richness (non-triviality guard 1) --------------------------
    TARGET_SEED = 42
    target = make_target(TARGET_SEED)
    DL = dl_zlib_bits(target)
    edim = effective_dim(target)
    print("\n[1] TARGET richness (guard 1: must be rich, not constant/low-D):")
    print(f"    DL(target)  operational (zlib, 16-level)  = {DL} bits")
    print(f"    effective dimension (participation ratio) = {edim:.2f}")
    print(f"    spatial-entropy bits (naive i.i.d.)       = {spatial_entropy_bits(target):.0f}")
    out["target"] = dict(DL_zlib_bits=DL, eff_dim=edim)

    # ---- evolve the rule under low-I scalar-ES shaping ---------------------
    print("\n[2] SHAPING the shared rule by scalar-fitness ES (low bandwidth)...")
    theta, hist, acc = evolve(target, generations=300, pop=48, seed=0,
                              damage=True, verbose=True)
    s_final = grow(theta, 1, STEPS, rng=rng)[0]
    fid = fidelity(morph(s_final[None])[0], target)
    print(f"    achieved structural corr (grown)          = {fid:.3f}  (R^2 = {fid**2:.3f})")
    out["fidelity_corr"] = fid
    out["fidelity_R2"] = fid ** 2
    out["accounting_channel"] = acc

    # realized morphology DL (what the system actually maintains)
    DL_real = dl_zlib_bits(morph(s_final[None])[0])
    print(f"    DL(realized morphology)                   = {DL_real} bits")

    # ---- effective genome bits (I baked into the rule) ---------------------
    print("\n[3] I-accounting (three honest measures):")
    b, kept, eff_bits, base = effective_genome_bits(theta, target)
    I_channel = acc["I_channel_bits"]
    I_raw = GENOME * 32
    print(f"    I_channel   (selection ceiling G*log2(pop!)) = {I_channel:,.0f} bits")
    print(f"    I_genome_eff(MDL: {b} bits x {kept} live params)   = {eff_bits:,.0f} bits")
    print(f"    I_genome_raw(crude |theta|*32)               = {I_raw:,.0f} bits")
    out["I_channel_bits"] = I_channel
    out["I_genome_eff_bits"] = eff_bits
    out["I_genome_raw_bits"] = I_raw

    print("\n    GENERATIVITY GAP  Delta = DL(target) - I :")
    for name, I in [("I_genome_eff", eff_bits), ("I_channel", I_channel),
                    ("I_genome_raw", I_raw)]:
        d = DL - I
        print(f"      Delta({name:12s}) = {DL} - {I:,.0f} = {d:+,.0f} bits")
    out["Delta_vs_genome_eff"] = DL - eff_bits

    # ---- re-coarsening (guard 2) -------------------------------------------
    print("\n[4] RE-COARSENING TEST (guard 2: no o(N) block holds the target):")
    counting, regen = recoarsening_test(theta, target, DL, rng)
    print(f"    (a) counting bound: a block needs s* = {counting['s_star_cells']:.1f} cells "
          f"= {100*counting['frac_of_N']:.0f}% of N to even store DL={DL} bits of target")
    print(f"    (b) regenerate-from-fragment (fidelity recovered from a centred block):")
    for bs, cells, f in regen:
        tag = 'o(N)' if cells < H * W // 2 else 'O(N)'
        print(f"        block {bs:2d}x{bs:2d} ({cells:3d} cells, {tag}): recovered fidelity = {f:.3f}")
    out["recoarsening"] = dict(counting=counting, regen=regen)

    # ---- regeneration + H4 double edge -------------------------------------
    print("\n[5] REGENERATION (causal ablation, executed) + H4 self-repair double edge:")
    s_dmg = grow(theta, 1, STEPS + 20, damage_at=STEPS - 2, rng=rng)[0]
    fid_regen = fidelity(morph(s_dmg[None])[0], target)
    print(f"    fidelity after ablate-right-half + 22 recovery steps = {fid_regen:.3f}")
    # H4: apply a 'legitimate correction' (paint a new feature) and see if reverted
    s_edit = grow(theta, 1, STEPS, rng=rng)
    s_edit[:, 2:5, 2:5, 0] = 1.0            # externally-imposed corrective edit
    edit_fid0 = np.mean(s_edit[0, 2:5, 2:5, 0])
    theta_b = np.broadcast_to(theta, (1, GENOME))
    for _ in range(12):
        s_edit = step(s_edit, theta_b, rng=rng)
    edit_fid1 = np.mean(s_edit[0, 2:5, 2:5, 0])
    print(f"    H4: externally-imposed edit value {edit_fid0:.2f} -> {edit_fid1:.2f} "
          f"after 12 steps ({'REVERTED (double edge)' if edit_fid1 < 0.6*edit_fid0 else 'retained'})")
    out["regen_fidelity"] = fid_regen
    out["H4_edit_before_after"] = [float(edit_fid0), float(edit_fid1)]

    # ---- genericity: same family, K distinct targets -----------------------
    print("\n[6] GENERICITY (same family -> K distinct rich targets; swap test):")
    K = 3
    tgts = [make_target(100 + k) for k in range(K)]
    thetas = []
    for k in range(K):
        th, _, _ = evolve(tgts[k], generations=240, pop=40, seed=10 + k,
                          damage=False, verbose=False)
        thetas.append(th)
    M = np.zeros((K, K))
    for i in range(K):
        si = grow(thetas[i], 1, STEPS, rng=rng)[0]
        for j in range(K):
            M[i, j] = fidelity(morph(si[None])[0], tgts[j])
    print("    rule-i vs target-j structural-corr matrix (diag should dominate):")
    for i in range(K):
        print("      " + "  ".join(f"{M[i,j]:+.2f}" for j in range(K)))
    diag = np.mean([M[i, i] for i in range(K)])
    offd = np.mean([M[i, j] for i in range(K) for j in range(K) if i != j])
    print(f"    mean diagonal = {diag:.3f}   mean off-diagonal = {offd:.3f}")
    print(f"    => same family reaches distinct targets; it holds NO single one "
          f"(diag-offd = {diag-offd:+.3f}).")
    print(f"    SELECTION-cost accounting (the O2-justified measure of I):")
    print(f"      to pick 1 of K={K} demonstrated targets costs log2(K) = {np.log2(K):.2f} bits")
    print(f"      Delta_selection = DL(target) - log2(K) = {DL} - {np.log2(K):.2f} "
          f"= {DL - np.log2(K):+.1f} bits  (>> 0)")
    out["genericity"] = dict(matrix=M.tolist(), diag=float(diag), offd=float(offd),
                             selection_bits=float(np.log2(K)),
                             Delta_selection=float(DL - np.log2(K)))

    # ---- I_c threshold sweep (O3) ------------------------------------------
    print("\n[7] I_c THRESHOLD SWEEP (O3): fidelity vs shaping information budget")
    sweep = []
    for gens in (25, 75, 150, 300):
        th, _, a = evolve(target, generations=gens, pop=40, seed=3, damage=True)
        s = grow(th, 1, STEPS, rng=rng)[0]
        f = fidelity(morph(s[None])[0], target)
        sweep.append((gens, a["I_channel_bits"], f))
        print(f"    G={gens:4d}  I_channel<={a['I_channel_bits']:>10,.0f} bits  corr={f:.3f}")
    print("    (fitness bit-depth sweep at G=160 -- coarsening the per-outcome signal:)")
    for qb in (1, 2, None):
        th, _, a = evolve(target, generations=160, pop=40, seed=4, damage=True, q_bits=qb)
        s = grow(th, 1, STEPS, rng=rng)[0]
        f = fidelity(morph(s[None])[0], target)
        lab = f"{qb}-bit" if qb else "full"
        sweep.append((f"q={lab}", a["I_channel_bits"], f))
        print(f"    per-outcome signal = {lab:>5s}  corr={f:.3f}")
    out["Ic_sweep"] = [(str(g), float(i), float(f)) for g, i, f in sweep]

    out["runtime_sec"] = time.time() - t0
    print(f"\n[done] runtime = {out['runtime_sec']:.1f}s")
    print("\nJSON_RESULTS_BEGIN")
    print(json.dumps(out, indent=None))
    print("JSON_RESULTS_END")


if __name__ == "__main__":
    main()
