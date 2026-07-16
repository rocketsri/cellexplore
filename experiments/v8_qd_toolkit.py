"""
v8_qd_toolkit.py
================

The DEEP-TIME question left open by v6/v7: can a LOW-I (non-differentiable,
per-outcome) process BUILD the target-agnostic toolkit (shared FiLM-NCA body)
that backprop builds but vanilla aggregate-ES could not?

v6 negative: OpenAI-ES on a SINGLE aggregate objective  f(theta) = mean_k corr(Phi(theta,c_k), t_k)
jointly over body + K codes -> held-out FELL below a random-body control and got
WORSE as K_train grew (train corr collapsed 0.51 -> 0.15). Diagnosis: (i) the
shared theta is pulled in K conflicting directions (the averaged ES gradient's
useful *consensus* component shrinks while variance stays -> SNR ~ 1/sqrt(K));
(ii) the reachable-from-random-init basin is a "mean-blob" body that ignores the
codes (mode collapse) -- a deceptive local optimum.

The QD hypothesis (tested AND stress-tested here): Quality-Diversity keeps an
ARCHIVE of behaviorally diverse elites, selects by PER-OUTCOME behavior
descriptors, never averages across targets, and cannot collapse onto the mean
blob (novelty pushes off it). So it may reach the shared-codebook basin that
aggregate ES cannot.

CRITICAL DESIGN (see the analysis doc, parts 1-2): a *vanilla* MAP-Elites
archive stores one INDEPENDENT full genome per cell -- that is literally B1's
LOOKUP-TABLE counterexample (K*|theta| bits, marginal cost |theta|, Delta<=0),
NOT a codebook. To make the archive a genuine B1 codebook we must TIE the body:
one shared theta, a small per-cell CODE. So the QD process here shapes the SHARED
theta (low-I, scalar QD-score per candidate) so that its CODE space, illuminated
by MAP-Elites, covers a diverse high-fitness behavior repertoire. Then we FREEZE
theta and run the exact v6 held-out B1 test (fit a d-code to held-out targets vs a
random-body control). Tying the body is honest but it REINTRODUCES the cross-cell
coupling that sank aggregate ES -- that tension is the whole experiment.

ARMS
  R0  reference: aggregate-ES toolkit (the known v6 negative) + random-body control
  A1  QD-shared-body, target-AGNOSTIC descriptor, sweep archive resolution (=K niches)
  A2  QD novelty-ONLY (drop the fitness term)  -> does diversity alone suffice?
  A3  QD with a target-REFERENCING descriptor (corr-to-train)  -> smuggling control
  D   lookup-vs-generative discriminator: does a FRESH code reach held-out targets
      whose descriptor cell was EMPTY in the train archive (true extrapolation),
      or only recall visited cells?

Honest accounting (analysis part 2): the behavior-descriptor CHOICE is a shaping
channel. It is admissible (O(1) bits, NOT charged to I) IFF it is computed from
the OUTCOME ALONE, references no target, and is fixed a priori. A1 uses such a
descriptor; A3 is the positive control that quantifies how much a target-
referencing descriptor smuggles. The archive resolution log2(K_cells) IS the
per-target code length ell = log2 K -- charged where O2/B1 already put it.

Pure numpy; reuses the v6 FiLM-NCA substrate. Budgets are modest/configurable;
default run ~ a few minutes. Deterministic seeds. Prints query counts and a verdict.
"""

import numpy as np
import time
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v6_amortization_curve as v6   # grow, morph, corr_batch, rankn, target,
                                     # dl_bits, fit_code, train_body, BODY, D, H, W

BODY, D, H, W = v6.BODY, v6.D, v6.H, v6.W
grow, morph, corr_batch, rankn = v6.grow, v6.morph, v6.corr_batch, v6.rankn


# --------------------------------------------------------------------------- #
#  Behavior descriptors: outcome -> cell id.                                   #
#  AGNOSTIC ones take NO target; the CHEAT one references training targets.    #
# --------------------------------------------------------------------------- #
def bd_com(mm, res, _t=None):
    """AGNOSTIC: 2D center-of-mass of the morphology, binned to res x res.
    Computed from the outcome alone; no target reference -> O(1) description,
    not chargeable to I (like the affine-invariant readout in corr_batch)."""
    B, h, w = mm.shape
    tot = mm.reshape(B, -1).sum(1) + 1e-9
    ys = (mm.sum(2) * np.arange(h)[None, :]).sum(1) / tot / (h - 1)
    xs = (mm.sum(1) * np.arange(w)[None, :]).sum(1) / tot / (w - 1)
    iy = np.clip((ys * res).astype(int), 0, res - 1)
    ix = np.clip((xs * res).astype(int), 0, res - 1)
    return iy * res + ix


def bd_mass_energy(mm, res, _t=None):
    """AGNOSTIC #2 (for the garden-of-forking-paths / BD-robustness check):
    (total mass, edge energy). Fixed scaling, no target reference."""
    B, h, w = mm.shape
    mass = mm.reshape(B, -1).mean(1)                          # in [0,1]
    en = (np.abs(np.diff(mm, axis=2)).reshape(B, -1).mean(1)
          + np.abs(np.diff(mm, axis=1)).reshape(B, -1).mean(1))
    im = np.clip((mass * res).astype(int), 0, res - 1)
    ie = np.clip((en / 0.5 * res).astype(int), 0, res - 1)    # fixed scale ~0.5 max
    return im * res + ie


def make_bd_cheat(train_targets):
    """TARGET-REFERENCING descriptor: (corr-to-train0, corr-to-train1). This
    SMUGGLES target specifications into the archive geometry -- the positive
    control for the part-2 accounting. Its description embeds >= 2 target images."""
    t0, t1 = train_targets[0], train_targets[1]

    def bd(mm, res, _t=None):
        c0 = np.clip((corr_batch(mm, t0) + 1) / 2, 0, 1)
        c1 = np.clip((corr_batch(mm, t1) + 1) / 2, 0, 1)
        i0 = np.clip((c0 * res).astype(int), 0, res - 1)
        i1 = np.clip((c1 * res).astype(int), 0, res - 1)
        return i0 * res + i1
    return bd


# --------------------------------------------------------------------------- #
#  MAP-Elites in CODE space on a FIXED body: illuminate the code repertoire.   #
# --------------------------------------------------------------------------- #
def illuminate(body, train_targets, bd_fn, res, n_batches, batch, sigma, rng,
               use_fitness=True):
    """On a fixed shared body, run MAP-Elites over codes c in R^D. Fitness is a
    PER-OUTCOME scalar (max corr over training targets -- the only target signal,
    one number per rollout, like ES). Returns (archive, coverage, mean_elite_fit,
    qd_score). Novelty-only mode (use_fitness=False) keeps first occupant per cell."""
    archive = {}                                             # cell -> (fitness, code)
    bodyb = np.broadcast_to(body, (batch, BODY))
    for bi in range(n_batches):
        if archive and bi > 0:                              # perturb existing elites
            elites = [v[1] for v in archive.values()]
            base = np.stack([elites[rng.integers(len(elites))] for _ in range(batch)])
            codes = base + sigma * rng.normal(0, 1, (batch, D))
        else:                                               # seed with random codes
            codes = rng.normal(0, 0.6, (batch, D))
        mm = morph(grow(bodyb, codes))                       # (batch,H,W)
        fits = np.stack([corr_batch(mm, t) for t in train_targets], 0).max(0)
        cells = bd_fn(mm, res)
        for j in range(batch):
            key = int(cells[j]); f = float(fits[j])
            if use_fitness:
                if key not in archive or f > archive[key][0]:
                    archive[key] = (f, codes[j].copy())
            elif key not in archive:
                archive[key] = (f, codes[j].copy())
    cov = len(archive)
    meanf = float(np.mean([v[0] for v in archive.values()])) if archive else 0.0
    qd = cov * max(meanf, 0.0)                               # QD-score (coverage*quality)
    return archive, cov, meanf, qd


# --------------------------------------------------------------------------- #
#  QD as the LOW-I toolkit shaper: outer ES on theta maximizes the QD-score of  #
#  the code archive it induces. Only a scalar QD-score per candidate (rank-     #
#  based) enters theta -> low-I in kind, just like OpenAI-ES.                    #
# --------------------------------------------------------------------------- #
def qd_build_toolkit(train_targets, bd_fn, res, outer_gens, outer_pop,
                     inner_batches, inner_batch, seed, use_fitness=True):
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 0.05, BODY)
    half = outer_pop // 2
    n_eval = 0
    hist = []
    for g in range(outer_gens):
        sigma = 0.10 * 0.5 ** (g / outer_gens)
        lr = 0.08 * 0.5 ** (g / outer_gens)
        eps = rng.normal(0, 1, (half, BODY)); eps = np.concatenate([eps, -eps], 0)
        cand = theta[None, :] + sigma * eps
        qd = np.zeros(outer_pop)
        for m in range(outer_pop):
            _, _, _, qdscore = illuminate(cand[m], train_targets, bd_fn, res,
                                          inner_batches, inner_batch, 0.25, rng,
                                          use_fitness=use_fitness)
            qd[m] = qdscore
            n_eval += inner_batches * inner_batch
        adv = rankn(qd)
        theta = theta + lr * (eps * adv[:, None]).mean(0) / sigma - 1e-4 * theta
        if g % 5 == 0 or g == outer_gens - 1:
            _, cov, meanf, qds = illuminate(theta, train_targets, bd_fn, res,
                                            inner_batches, inner_batch, 0.25, rng,
                                            use_fitness=use_fitness)
            hist.append((g, cov, round(meanf, 3), round(qds, 2)))
    return theta, n_eval, hist


# --------------------------------------------------------------------------- #
def heldout_corr(body, held, tag):
    """The exact v6 B1 test: freeze body, fit only a D-dim code per held-out
    target by scalar ES, return mean held-out corr."""
    return float(np.mean([v6.fit_code(body, h, seed=1000 + i) for i, h in enumerate(held)]))


def main(quick=True):
    t0 = time.time()
    # modest, configurable budgets (quick default keeps runtime to a few minutes)
    OG, OP = (12, 8) if quick else (24, 12)          # outer gens, pop
    IB, IBZ = (4, 40) if quick else (6, 56)          # inner batches, batch size
    RES_SWEEP = (4, 6, 8)                            # archive res -> K_cells = res^2

    print("=" * 74)
    print("v8: QUALITY-DIVERSITY as the low-I builder of the target-agnostic toolkit")
    print("=" * 74)
    print(f"grid {H}x{W}, shared body |theta|={BODY}, code dim D={D}")

    pool = [v6.target(s) for s in range(30)]
    held = [v6.target(200 + k) for k in range(4)]    # never trained, the B1 held-out set
    DL_held = float(np.mean([v6.dl_bits(t) for t in held]))
    code_bits = D * 6
    train = pool[:24]
    out = {}

    # ---- R0 reference points -------------------------------------------------
    print("\n[R0] reference points (same held-out benchmark)")
    rngb = np.random.default_rng(99)
    ctrl = float(np.mean([v6.fit_code(rngb.normal(0, 0.5, BODY), h, seed=500 + i)
                          for i, h in enumerate(held)]))
    agg_body, agg_tr = v6.train_body(train, generations=(160 if quick else 260), seed=0)
    agg_ho = heldout_corr(agg_body, held, "aggES")
    print(f"     random-body control            held-out corr = {ctrl:+.3f}")
    print(f"     aggregate-ES toolkit (v6 neg.) held-out corr = {agg_ho:+.3f}"
          f"  (train {agg_tr:.3f})  margin {agg_ho-ctrl:+.3f}")
    print(f"     DL(held-out) ~ {DL_held:.0f} bits, marginal |code| ~ {code_bits} bits")
    out["control"] = ctrl; out["agg_es"] = dict(held=agg_ho, train=agg_tr)

    # ---- A1 QD, agnostic descriptor, resolution (=K niches) sweep -----------
    print("\n[A1] QD-shared-body toolkit, AGNOSTIC descriptor (center-of-mass), res sweep")
    a1 = []
    best = None
    for res in RES_SWEEP:
        th, nev, hist = qd_build_toolkit(train, bd_com, res, OG, OP, IB, IBZ, seed=res)
        ho = heldout_corr(th, held, f"qd-res{res}")
        a1.append((res, res * res, ho, ho - ctrl, nev))
        print(f"     res={res} (K_cells={res*res:2d}, ell~log2K={np.log2(res*res):.1f} bits): "
              f"held-out {ho:+.3f}  margin {ho-ctrl:+.3f}  [{nev} inner-grows, {time.time()-t0:.0f}s]")
        if best is None or ho > best[2]:
            best = (res, th, ho)
    out["A1"] = [(r, k, ho, m, nev) for r, k, ho, m, nev in a1]

    # ---- A2 novelty-only ablation -------------------------------------------
    print("\n[A2] QD novelty-ONLY (no fitness term): does diversity alone build a codebook?")
    th_nov, nev, _ = qd_build_toolkit(train, bd_com, best[0], OG, OP, IB, IBZ,
                                      seed=77, use_fitness=False)
    ho_nov = heldout_corr(th_nov, held, "novelty")
    print(f"     novelty-only held-out {ho_nov:+.3f}  margin {ho_nov-ctrl:+.3f}")
    out["A2_novelty"] = ho_nov

    # ---- A3 smuggling control: target-referencing descriptor ----------------
    print("\n[A3] QD with TARGET-REFERENCING descriptor (corr-to-train) -- smuggling control")
    th_ch, nev, _ = qd_build_toolkit(train, make_bd_cheat(train), best[0], OG, OP,
                                     IB, IBZ, seed=88)
    ho_ch = heldout_corr(th_ch, held, "cheat")
    print(f"     cheat-BD held-out {ho_ch:+.3f}  margin {ho_ch-ctrl:+.3f}")
    print(f"     smuggled-generativity attributable to the descriptor >= "
          f"{ho_ch-best[2]:+.3f} corr over the agnostic BD")
    out["A3_cheat"] = ho_ch

    # ---- D lookup-vs-generative discriminator -------------------------------
    print("\n[D]  lookup-table vs generative codebook discriminator (best agnostic toolkit)")
    th = best[1]; rng = np.random.default_rng(7)
    arch, cov, meanf, _ = illuminate(th, train, bd_com, best[0], IB * 3, IBZ, 0.25, rng)
    # which held-out targets land in a cell the TRAIN archive never filled?
    held_cells = bd_com(np.stack([h.numpy() if hasattr(h, "numpy") else h for h in held]),
                        best[0])
    empty = [i for i, c in enumerate(held_cells) if int(c) not in arch]
    ho_all = [v6.fit_code(th, h, seed=1000 + i) for i, h in enumerate(held)]
    ho_empty = float(np.mean([ho_all[i] for i in empty])) if empty else float("nan")
    print(f"     train-archive coverage {cov}/{best[0]**2} cells, mean elite fit {meanf:.3f}")
    print(f"     held-out targets whose descriptor cell was EMPTY in train archive: "
          f"{len(empty)}/{len(held)}")
    print(f"     fresh-code held-out corr on those EMPTY-cell targets = {ho_empty:+.3f} "
          f"(vs control {ctrl:+.3f})")
    print(f"     -> reaching empty-cell targets above control = GENERATIVE (extrapolation);")
    print(f"        at/below control = LOOKUP TABLE (only recalls visited descriptors).")
    out["D"] = dict(coverage=cov, n_empty=len(empty), ho_empty=ho_empty)

    # ---- verdict -------------------------------------------------------------
    print("\n" + "=" * 74)
    a1_trend = [r[2] for r in a1]
    rising = a1_trend[-1] > a1_trend[0] + 0.03
    closes = best[2] - ctrl > 0.10 and (not np.isnan(ho_empty) and ho_empty - ctrl > 0.05)
    print("VERDICT")
    print(f"  best agnostic-QD held-out margin over control : {best[2]-ctrl:+.3f}")
    print(f"  vs aggregate-ES margin (the v6 negative)       : {agg_ho-ctrl:+.3f}")
    print(f"  held-out trend across K_niches (B1 signature)  : {[round(x,3) for x in a1_trend]} "
          f"({'RISING' if rising else 'flat/none'})")
    print(f"  extrapolation to empty descriptor cells        : {ho_empty:+.3f} vs ctrl {ctrl:+.3f}")
    if closes and rising:
        print("  => QD CLOSES the deep-time gap: low-I diversity search builds a generative,")
        print("     target-agnostic codebook (held-out rises with K, extrapolates > control).")
    elif best[2] - ctrl > 0.05 and best[2] > agg_ho + 0.05:
        print("  => QD NARROWS but does not close: beats aggregate ES (escapes the mean-blob")
        print("     trap) yet the frozen body is more lookup-table than generative prior.")
    else:
        print("  => QD does NOT close the gap here: the archive is a lookup table (B1 necessity")
        print("     clause) -- diversity in behavior space != target-agnosticism of a shared theta.")
    print(f"\n[done] runtime={time.time()-t0:.0f}s")
    print("JSON_V8_BEGIN"); print(json.dumps(out)); print("JSON_V8_END")


if __name__ == "__main__":
    main(quick="--full" not in sys.argv)
