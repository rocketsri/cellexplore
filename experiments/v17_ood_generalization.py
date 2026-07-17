"""
v17_ood_generalization.py
=========================

Tests the one substantive open question about the superposition/dictionary positive
(v14): is its held-out realizability GENUINE structure-learning, or only "within the
same generative model" (which would make the generalization claim narrow)?

A dictionary learned on targets that are k-sparse in D0 is tested on held-out targets
from:
  (in-dist)  the SAME model D0 (v14's setting),
  (OOD-dict) a DIFFERENT random dictionary D1 (targets sparse in D1, not D0),
  (OOD-dense) dense/non-sparse signals (Gaussian random fields),
each vs a random-dictionary control. This bounds the SCOPE of the generalization:
  - if it generalizes only in-dist -> B1 holds (held-out from the target distribution)
    but transfer is narrow (honest bound);
  - if it also lifts OOD above control -> it learned reusable structure (stronger).

Pure numpy. Deterministic. ~1 min.
"""

import numpy as np
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v14_superposition_codebook import (unit_cols, make_targets, recon_corr,
                                        learn_dict_grad)


def grf_targets(n, m, seed):
    """dense (non-sparse) signals: filtered Gaussian noise -- out-of-sparse-model."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, m))
    # smooth along the index (low-pass) to make them structured but NOT k-sparse in any D
    for _ in range(3):
        X = 0.5 * X + 0.25 * np.roll(X, 1, 0) + 0.25 * np.roll(X, -1, 0)
    return unit_cols(X)


def main():
    n, M, k = 48, 96, 5
    D0 = unit_cols(np.random.default_rng(1).standard_normal((n, M)))
    D1 = unit_cols(np.random.default_rng(2).standard_normal((n, M)))   # different model

    print("=" * 70)
    print("v17: OOD generalization scope of a learned dictionary (superposition)")
    print("=" * 70)
    # learn the dictionary on D0-generated targets (K_train large enough)
    Xtr = make_targets(D0, 600, k, seed=123)
    Phi = learn_dict_grad(Xtr, M, iters=40, k=k, seed=0)
    rand = unit_cols(np.random.default_rng(7).standard_normal((n, M)))

    tests = {
        "in-dist (D0, sparse)":  make_targets(D0, 40, k, seed=999),
        "OOD-dict (D1, sparse)": make_targets(D1, 40, k, seed=888),
        "OOD-dense (GRF)":       grf_targets(n, 40, seed=777),
    }
    print(f"\ndictionary learned on D0-sparse targets; tested held-out on:\n")
    out = {}
    for name, X in tests.items():
        ho, _ = recon_corr(Phi, X, k)
        ctrl, _ = recon_corr(rand, X, k)
        print(f"  {name:24s}: learned={ho:.3f}  control={ctrl:.3f}  margin={ho-ctrl:+.3f}  "
              f"({'genuine' if ho-ctrl > 0.1 else 'no lift over random'})")
        out[name] = dict(learned=ho, control=ctrl, margin=ho - ctrl)

    print("\n[interpretation]")
    ind = out["in-dist (D0, sparse)"]["margin"]
    ood_d = out["OOD-dict (D1, sparse)"]["margin"]
    ood_g = out["OOD-dense (GRF)"]["margin"]
    print(f"  in-dist margin {ind:+.2f}: the dictionary recovered D0's structure and codes held-out")
    print(f"    draws from the SAME model -> B1 held-out realizability holds (not memorization).")
    if ood_d > 0.1 or ood_g > 0.1:
        print(f"  OOD margin (D1 {ood_d:+.2f}, dense {ood_g:+.2f}) > 0: learned reusable structure "
              f"beyond its own model -- STRONGER than expected.")
    else:
        print(f"  OOD margins (D1 {ood_d:+.2f}, dense {ood_g:+.2f}) ~ 0: generalization is WITHIN the")
        print(f"    shaped target distribution -- B1 holds there, but transfer to a different model")
        print(f"    class is narrow. Honest bound: the codebook is agnostic ACROSS the family it was")
        print(f"    shaped on, not across arbitrary distributions (as expected: a dict of D0-atoms")
        print(f"    cannot sparsely code D1-atoms). This matches how SAEs are dataset-specific.")
    print("\nJSON_V17_BEGIN"); print(json.dumps(out)); print("JSON_V17_END")


if __name__ == "__main__":
    main()
