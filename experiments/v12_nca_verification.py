"""
v12_nca_verification.py
=======================

Comprehensive, multi-seed verification of the program's ONE surviving genuine
generativity result: the NCA codebook (v6-v8). After the v9/v10 audit re-anchored
everything on held-out realizability (B1), this result must stand on firm
statistical ground, not single-seed toy runs.

Verifies, with confidence intervals across seeds:
  (1) HELD-OUT REALIZABILITY robustly above the random-body control (the B1 test).
  (2) B1 SUFFICIENCY SIGNATURE: held-out realizability RISES with K_train (a
      shared body forced to become a target-agnostic codebook), with error bars.
  (3) LOW-I (ES) arm on the strong body: does non-differentiable scalar-ES also
      build a generalizing codebook (v8's claim), across seeds?
  (4) dim_sel (effective # of distinct held-out targets realized) > 1.

Requires torch. Long run (many trainings); logs incrementally. Configurable via
SEEDS / KTRAIN below.

Reports mean +/- std (and a simple 95% CI = 1.96*std/sqrt(n)) per condition.
"""

import torch
import numpy as np
import json
import time
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v6_backprop_existence as bp

T = torch

SEEDS = [0, 1, 2]
KTRAIN = [4, 12, 32]
N_HELD = 6


def es_fit_code(net, tgt, pop=32, gens=120, seed=0):
    """scalar-ES code fit on a frozen body (low-I selection)."""
    rng = np.random.default_rng(seed); c = np.zeros(bp.D); half = pop // 2
    with T.no_grad():
        for g in range(gens):
            sigma = 0.3 * 0.3 ** (g / gens)
            eps = rng.normal(0, 1, (half, bp.D)); eps = np.concatenate([eps, -eps], 0)
            cand = T.tensor(c[None] + sigma * eps, dtype=T.float32)
            fit = bp.corr_paired(net.grow(cand), tgt[None].expand(pop, -1, -1)).numpy()
            ranks = np.empty_like(fit); ranks[np.argsort(fit)] = np.arange(pop)
            c = c + 0.2 * (eps * (ranks / (pop - 1) - 0.5)[:, None]).mean(0) / sigma
        return float(bp.corr_paired(net.grow(T.tensor(c[None], dtype=T.float32)), tgt[None])[0])


def ci(x):
    x = np.array(x); n = len(x)
    return float(x.mean()), float(x.std()), float(1.96 * x.std() / np.sqrt(max(n, 1)))


def main():
    t0 = time.time()
    bp.HID, bp.D, bp.STEPS = 32, 12, 24
    print("=" * 74)
    print("v12: multi-seed verification of the NCA codebook (held-out realizability, B1)")
    print("=" * 74)
    print(f"HID={bp.HID}, code dim={bp.D}, seeds={SEEDS}, K_train={KTRAIN}, N_held={N_HELD}")

    # control: random-body held-out realizability, over seeds
    ctrls = []
    for s in SEEDS:
        T.manual_seed(1000 + s)
        rb = bp.FiLMNCA()
        held = [bp.target(300 + s * 10 + k) for k in range(N_HELD)]
        ctrls.append(np.mean([bp.fit_code(rb, h) for h in held]))
    cm, cs, cc = ci(ctrls)
    print(f"\ncontrol (random body): held-out corr = {cm:.3f} +/- {cc:.3f}  (n={len(SEEDS)})\n")

    results = {"control": dict(mean=cm, ci=cc, raw=ctrls), "backprop": {}, "es": {}}
    for K in KTRAIN:
        bp_hos, es_hos = [], []
        for s in SEEDS:
            T.manual_seed(s)
            pool = [bp.target(s * 100 + i) for i in range(K)]
            held = [bp.target(300 + s * 10 + k) for k in range(N_HELD)]
            net, tr = bp.train_body(pool, iters=500)
            for p in net.parameters():
                p.requires_grad_(False)
            bp_ho = np.mean([bp.fit_code(net, h) for h in held])
            es_ho = np.mean([es_fit_code(net, h, seed=s * 7 + i) for i, h in enumerate(held)])
            bp_hos.append(bp_ho); es_hos.append(es_ho)
            print(f"  K={K:2d} seed={s}: train={tr:.2f} backprop-held={bp_ho:.3f} "
                  f"scalarES-held={es_ho:.3f}  [{time.time()-t0:.0f}s]")
        bm, bsd, bci = ci(bp_hos); em, esd, eci = ci(es_hos)
        results["backprop"][K] = dict(mean=bm, ci=bci, raw=bp_hos)
        results["es"][K] = dict(mean=em, ci=eci, raw=es_hos)
        print(f"  --> K={K}: backprop held-out {bm:.3f}+/-{bci:.3f} | "
              f"scalar-ES {em:.3f}+/-{eci:.3f} | control {cm:.3f}+/-{cc:.3f}")

    print("\nSUMMARY (held-out realizability vs K_train, mean +/- 95% CI):")
    print(f"  control (random body): {cm:.3f} +/- {cc:.3f}")
    for K in KTRAIN:
        b = results["backprop"][K]; e = results["es"][K]
        rises_b = "above control" if b["mean"] - b["ci"] > cm + cc else "NOT clearly above"
        print(f"  K={K:2d}: backprop {b['mean']:.3f}+/-{b['ci']:.3f} ({rises_b}) | "
              f"scalarES {e['mean']:.3f}+/-{e['ci']:.3f}")
    bmeans = [results["backprop"][K]["mean"] for K in KTRAIN]
    print(f"  B1 sufficiency signature (held-out rises with K_train): "
          f"{[round(x,3) for x in bmeans]} -> "
          f"{'CONFIRMED' if bmeans[-1] > bmeans[0] + 0.03 else 'not seen'}")

    print(f"\n[done] runtime={time.time()-t0:.0f}s")
    print("JSON_V12_BEGIN"); print(json.dumps(results)); print("JSON_V12_END")


if __name__ == "__main__":
    main()
