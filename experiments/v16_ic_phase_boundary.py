"""
v16_ic_phase_boundary.py
========================

Empirical O3: the ACHIEVABILITY THRESHOLD I_c for building a GENERALIZING codebook,
measured on the fast superposition/dictionary substrate (numpy; GPU would not help).

O2 gives the lower bound (selection cost log2 K), B1 the condition (target-agnostic
codebook = held-out realizability). O3 asks: how much SHAPING (here: how many
training targets K_train, at what sparsity k) is needed before held-out
generalization actually appears -- and where does it fail? This maps to the known
sample-complexity / phase transitions of dictionary learning and sparse recovery.

Sweeps the (K_train, k) plane and measures held-out realizability + atom recovery,
locating the phase boundary:
  - K_train axis: dictionary IDENTIFIABILITY needs K_train >> M (sample complexity).
  - k axis: sparse RECOVERY needs k below the OMP/RIP threshold ~ n/(2 log(M/n)).
The held-out-realizability surface is the empirical I_c(K_train, k).

Pure numpy. Deterministic. ~1-2 min.
"""

import numpy as np
import json
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v14_superposition_codebook import (unit_cols, make_targets, sparse_code,
                                        recon_corr, learn_dict_grad, atom_recovery)


def main():
    t0 = time.time()
    n, M = 48, 96
    print("=" * 72)
    print("v16: empirical I_c PHASE BOUNDARY for a generalizing codebook (superposition)")
    print("=" * 72)
    print(f"n={n} units, M={M} atoms.  RIP/OMP sparse-recovery guide: k_max ~ n/(2 ln(M/n)) "
          f"= {n/(2*np.log(M/n)):.1f}")
    print(f"dictionary identifiability needs K_train >> M={M}.\n")
    D0 = unit_cols(np.random.default_rng(1).standard_normal((n, M)))

    Ks = [24, 48, 96, 192, 384, 768]
    ks = [3, 5, 8, 12]
    rand = unit_cols(np.random.default_rng(7).standard_normal((n, M)))

    print("held-out realizability (learned dict), rows=k, cols=K_train:")
    header = "   k \\ K_train " + "".join(f"{K:>7d}" for K in Ks) + "   |  ctrl"
    print(header)
    surface = {}
    for k in ks:
        held = make_targets(D0, 40, k, seed=999)
        ctrl, _ = recon_corr(rand, held, k)
        row = []
        for K in Ks:
            Xtr = make_targets(D0, K, k, seed=K * 10 + k)
            Phi = learn_dict_grad(Xtr, M, iters=40, k=k, seed=0)
            ho, _ = recon_corr(Phi, held, k)
            row.append(ho)
        surface[k] = dict(K=Ks, held=row, control=ctrl)
        print(f"   k={k:<2d}         " + "".join(f"{v:>7.2f}" for v in row) + f"   |  {ctrl:.2f}")

    # locate the threshold K_train* where held-out first clears control+0.1, per k
    print("\nphase boundary K_train*(k) = smallest K_train with held-out > control + 0.10:")
    for k in ks:
        ctrl = surface[k]["control"]; Kstar = None
        for K, ho in zip(Ks, surface[k]["held"]):
            if ho > ctrl + 0.10:
                Kstar = K; break
        print(f"   k={k:<2d}: K_train* = {Kstar if Kstar else '> ' + str(Ks[-1])}  "
              f"(ratio K*/M = {Kstar/M:.1f})" if Kstar else
              f"   k={k:<2d}: K_train* > {Ks[-1]}  (no generalization in range -- above recovery threshold)")

    print("\n[interpretation] Two thresholds bound achievability (the empirical I_c):")
    print("  (i)  SAMPLE complexity: held-out generalization appears only once K_train exceeds")
    print("       ~a few x M (dictionary identifiability) -- a clean sample-complexity threshold.")
    print("  (ii) SPARSE-RECOVERY: for k above the OMP/RIP limit, no amount of K_train recovers")
    print("       the codebook (the selection is not decodable). This is O3's 'fails' regime.")
    print("  => achievable generativity requires BOTH enough shaping data (K_train >> M) AND a")
    print("     selection sparse enough to be decodable (k below the recovery threshold).")
    print(f"\n[done] {time.time()-t0:.0f}s")
    print("JSON_V16_BEGIN"); print(json.dumps({str(k): surface[k] for k in ks})); print("JSON_V16_END")


if __name__ == "__main__":
    main()
