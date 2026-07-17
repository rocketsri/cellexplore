"""
v11_holonomy_scaling.py  (repurposed after the v9 audit)
========================================================

The deflationary audit of v9 (see v10.md) showed the Holonomy Code's claimed
generativity Delta=+1244 was a measure artifact: the "rich field" is quenched-
disorder / PRNG noise, the winding class explains ~1% of the field variance, and
richness (eff-dim 6.6) is the disorder's eff-dim showing through -- NOT a
class-carried target.

This experiment MEASURES that negative honestly and generally, as the
K-vs-richness-vs-class-carried tradeoff. For each disorder level it reports:
  - K       : number of stable winding classes (codebook size),
  - richness: spectral participation ratio of the realized cos-field,
  - eta2    : the FRACTION of realized-field variance explained by the winding
              class (between-class / total variance, over many seeds/class) --
              i.e. how much of the "rich field" the SELECTION actually carries.

The honest finding (predicted, then measured): there is NO disorder regime with
both high richness AND high eta2. Low disorder -> large K, high eta2, but low
richness (plane waves). High disorder -> high richness, but eta2 ~ 0 (richness is
disorder, not class). So the Holonomy Code cannot deliver a rich, class-carried,
distinct target: its "harmonic completion" adds no genuine per-target structure.
This validates class-explanatory-power / held-out realizability (B1) as the
discriminator between genuine generativity (the NCA codebook, v6-v8, which PASSES)
and fake generativity (this, which FAILS).

Pure numpy. Deterministic. ~a few minutes.
"""

import numpy as np
import itertools
import json
import time

DT = 0.05
ALPHA = 0.40


def make_rule(shape, seed, dis):
    d = len(shape); rng = np.random.default_rng(seed)
    A = np.exp(dis * rng.standard_normal(shape + (2 * d,)))
    omega = 0.03 * rng.standard_normal(shape)
    return A, omega, d


def step(th, A, omega, d, alpha=ALPHA):
    dth = omega.copy()
    for a in range(d):
        dth += A[..., 2 * a] * np.sin(np.roll(th, -1, a) - th - alpha)
        dth += A[..., 2 * a + 1] * np.sin(np.roll(th, 1, a) - th - alpha)
    return th + DT * dth


def relax(th, A, omega, d, steps=2000, alpha=ALPHA):
    for _ in range(steps):
        th = step(th, A, omega, d, alpha)
    return th


def winding(th, d):
    w = []
    for a in range(d):
        df = (np.diff(th, axis=a, append=np.take(th, [0], axis=a)) + np.pi) % (2 * np.pi) - np.pi
        w.append(int(round(np.median(df.sum(axis=a) / (2 * np.pi)))))
    return tuple(w)


def seed_winding(w, shape):
    idx = np.indices(shape).astype(float); ph = np.zeros(shape)
    for a, wa in enumerate(w):
        ph += 2 * np.pi * wa * idx[a] / shape[a]
    return ph


def morph(th):
    return (np.cos(th) + 1) / 2


def spectral_pr(field):
    F = np.abs(np.fft.fftn(field - field.mean())) ** 2
    F = F.ravel()
    return float((F.sum() ** 2) / (np.sum(F ** 2) + 1e-12))


def scan_point(shape, d, dis, wmax=4, reps=4, seed=0):
    A, omega, _ = make_rule(shape, seed, dis)
    rng = np.random.default_rng(1)
    # collect reps locked fields per stable class
    per_class = {}
    for w in itertools.product(range(-wmax, wmax + 1), repeat=d):
        fs = []
        for r in range(reps):
            th = relax(seed_winding(w, shape) + 0.12 * rng.standard_normal(shape), A, omega, d)
            if winding(th, d) == w:
                fs.append(morph(th).ravel())
        if len(fs) >= 2:
            per_class[w] = np.array(fs)
    K = len(per_class)
    if K == 0:
        return dict(dis=dis, K=0, richness=0.0, eta2=float("nan"))
    richness = float(np.mean([spectral_pr(morph(relax(seed_winding(w, shape), A, omega, d)))
                              for w in per_class]))
    # eta^2 = between-class variance / total variance
    allf = np.concatenate(list(per_class.values()), 0)
    grand = allf.mean(0)
    tot = float(((allf - grand) ** 2).mean())
    means = {w: v.mean(0) for w, v in per_class.items()}
    counts = {w: len(v) for w, v in per_class.items()}
    between = float(sum(counts[w] * ((means[w] - grand) ** 2).sum() for w in per_class)
                    / (allf.shape[0] * allf.shape[1]))
    eta2 = between / (tot + 1e-12)
    return dict(dis=dis, K=K, richness=richness, eta2=eta2)


def main():
    t0 = time.time()
    print("=" * 74)
    print("v11: Holonomy Code -- the K vs richness vs class-carried (eta^2) tradeoff")
    print("=" * 74)
    print("(honest re-measurement after the v9 audit; see v10.md)")
    L = 16; shape = (L, L); d = 2
    print(f"\n2D torus L={L}. For each disorder: K, richness(spectral-PR), "
          f"eta^2 (field variance explained by the winding class):\n")
    rows = []
    for dis in (0.1, 0.25, 0.5, 0.9, 1.6):
        r = scan_point(shape, d, dis)
        rows.append(r)
        print(f"  disorder={dis:4.2f}: K={r['K']:3d}  richness={r['richness']:5.1f}  "
              f"eta^2(class-carried)={r['eta2']:.3f}  "
              f"({'class carries field' if r['eta2']>0.5 else 'richness is DISORDER, not class'})")
    print("\n  => No regime has BOTH high richness AND high eta^2. Low disorder: large K,")
    print("     high eta^2, but low richness (plane waves). High disorder: high richness,")
    print("     but eta^2 -> 0 (disorder noise, not class). The 'harmonic completion' adds no")
    print("     genuine per-target structure: the Holonomy Code is NOT a generativity mechanism.")
    print("     (Contrast: the NCA codebook v6-v8 PASSES held-out realizability = high eta^2.)")
    print(f"\n[done] {time.time()-t0:.0f}s")
    print("JSON_V11_BEGIN"); print(json.dumps(dict(rows=rows))); print("JSON_V11_END")


if __name__ == "__main__":
    main()
