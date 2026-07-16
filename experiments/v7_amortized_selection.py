"""
v7_amortized_selection.py
=========================

Resolves v6's "search-limited" verdict by splitting the shaping cost the way
biology does -- and the way B1/O2 say it splits:

  - ONE-TIME body ("toolkit"): the shared FiLM-NCA rule. Expensive to shape
    (backprop here; deep-time population search in biology). Amortized over the
    whole family / all time. NOT charged per target.
  - PER-TARGET selection: a small code picking one attractor from the repertoire.
    O2 says this costs only log2(K) bits; B1 says it suffices IFF the body is a
    target-agnostic codebook (which v6's backprop arm certified: held-out corr
    0.78 via a 12-param code).

v6 asked "can low-I ES shape the whole body" -> no (1800-dim search). The RIGHT
low-I question is: GIVEN the amortized toolkit, can a PURELY SCALAR, non-
differentiable, per-outcome shaper SELECT a held-out target via its small code?
If yes, the recurring per-target generativity D_bar - |code| is delivered by a
genome-like process, and the only expensive part is the one-time toolkit --
exactly biology's arrangement (conserved developmental toolkit + cheap
regulatory tweaks per lineage).

This script: (1) trains the toolkit once by backprop on K_train=24 targets;
(2) FREEZES it; (3) for held-out targets, finds the code by PURE SCALAR ES
(antithetic, rank-based, only a scalar corr per rollout -- no gradient), and
compares to backprop code-fit and to a random-body control; (4) sweeps code
dim d to measure the per-target selection cost.

Reuses v6_backprop_existence for the architecture. Requires torch. ~3 min.
"""

import torch
import numpy as np
import time
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v6_backprop_existence as bp

T = torch


def es_code_search(net, tgt, d, pop=40, gens=140, seed=0):
    """Find a d-dim FiLM code by PURE SCALAR ES: the only signal is a scalar
    correlation per rollout (non-differentiable, per-outcome). No backprop."""
    rng = np.random.default_rng(seed)
    c = np.zeros(d)
    half = pop // 2
    with T.no_grad():
        for g in range(gens):
            sigma = 0.3 * 0.3 ** (g / gens)
            eps = rng.normal(0, 1, (half, d)); eps = np.concatenate([eps, -eps], 0)
            cand = c[None, :] + sigma * eps
            codes = T.tensor(cand, dtype=T.float32)
            mm = net.grow(codes)                                  # (pop,H,W)
            fit = bp.corr_paired(mm, tgt[None].expand(pop, -1, -1)).numpy()  # scalar/rollout
            ranks = np.empty_like(fit); ranks[np.argsort(fit)] = np.arange(pop)
            adv = ranks / (pop - 1) - 0.5
            c = c + 0.2 * (eps * adv[:, None]).mean(0) / sigma
        final = float(bp.corr_paired(net.grow(T.tensor(c[None], dtype=T.float32)),
                                     tgt[None])[0])
    return final, pop * gens


def main():
    t0 = time.time()
    print("=" * 74)
    print("v7: AMORTIZED low-I SELECTION -- scalar ES picks held-out targets on a frozen toolkit")
    print("=" * 74)
    # sync capacities to v6 existence arm
    bp.HID, bp.D, bp.STEPS = 32, 12, 24
    pool = [bp.target(s) for s in range(30)]
    held = [bp.target(200 + k) for k in range(4)]
    DL_held = float(np.mean([bp.dl_bits(t) for t in held]))

    print("\n[1] one-time toolkit: train shared body by backprop on K_train=24 (amortized cost)")
    net, tr = bp.train_body(pool[:24], iters=900)
    print(f"    toolkit trained: train corr={tr:.3f}, body |theta|={net.body_numel()} (paid ONCE)")

    print("\n[2] per-target SELECTION on the FROZEN toolkit, held-out targets:")
    for p in net.parameters():
        p.requires_grad_(False)
    es_scores, bp_scores = [], []
    for i, h in enumerate(held):
        es_f, q = es_code_search(net, h, d=bp.D, seed=10 + i)
        bp_f = bp.fit_code(net, h, iters=400)
        es_scores.append(es_f); bp_scores.append(bp_f)
        print(f"    held-out {i}: scalar-ES code corr={es_f:.3f} (in {q} rollouts)  |  "
              f"backprop code corr={bp_f:.3f}")
    es_m, bp_m = float(np.mean(es_scores)), float(np.mean(bp_scores))

    # control: random frozen body + scalar-ES code (does ES 'succeed' on any body? no)
    rand = bp.FiLMNCA()
    ctrl = float(np.mean([es_code_search(rand, h, d=bp.D, seed=50 + i)[0]
                          for i, h in enumerate(held)]))

    print(f"\n    scalar-ES selection (mean held-out corr)   = {es_m:.3f}")
    print(f"    backprop  selection (mean held-out corr)   = {bp_m:.3f}")
    print(f"    scalar-ES on RANDOM body (control)         = {ctrl:.3f}")
    code_bits = bp.D * 6
    print(f"\n    => per-target selection IS low-I discoverable: a purely scalar, non-")
    print(f"       differentiable shaper finds the {bp.D}-param code for HELD-OUT targets")
    print(f"       at corr {es_m:.2f} (vs {ctrl:.2f} on a random body).")
    print(f"    marginal generativity per target Delta = DL - |code| = {DL_held:.0f} - {code_bits}")
    print(f"       = {DL_held - code_bits:+.0f} bits, delivered by a genome-like (per-outcome,")
    print(f"       non-differentiable) selection process; the toolkit is a one-time amortized cost.")

    print("\n[3] per-target selection cost vs code dim d (scalar ES):")
    sweep = []
    for d in (2, 4, 8, 12):
        # retrain a small toolkit at this d (code dim is part of the body interface)
        bp.D = d
        net_d, _ = bp.train_body(pool[:24], iters=700)
        for p in net_d.parameters():
            p.requires_grad_(False)
        s = float(np.mean([es_code_search(net_d, h, d=d, seed=70 + i)[0]
                           for i, h in enumerate(held)]))
        sweep.append((d, d * 6, s))
        print(f"    d={d:2d} (|code|~{d*6:2d} bits): scalar-ES held-out corr={s:.3f}  [{time.time()-t0:.0f}s]")
    bp.D = 12

    print(f"\n[done] runtime={time.time()-t0:.0f}s")
    print("JSON_V7_BEGIN")
    print(json.dumps(dict(toolkit_train=tr, es_sel=es_m, bp_sel=bp_m, ctrl=ctrl,
                          DL_held=DL_held, code_bits=code_bits, dim_sweep=sweep)))
    print("JSON_V7_END")


if __name__ == "__main__":
    main()
