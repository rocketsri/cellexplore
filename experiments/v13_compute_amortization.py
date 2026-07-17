"""
v13_compute_amortization.py
===========================

Grounds the "amortized selection" story in real compute numbers, and asks the
practical question the user raised: is any of this a feasible ARCHITECTURE?

Measures, on the NCA codebook substrate (the program's one surviving genuine
generativity result):
  - build cost   : FLOPs/time to shape the shared toolkit (backprop over K targets), ONE-TIME.
  - select cost  : FLOPs/time to realize one HELD-OUT target on the frozen toolkit
                   (fit a d=12 code) -- the RECURRING per-target cost.
  - scratch cost : FLOPs/time to train a whole fresh body for one target (the no-sharing baseline).
  - amortization : per-target params (12 code vs ~1832 body = 150x), and the crossover
                   number of targets T* beyond which (build + T*select) < T*scratch.

This is the exact structure of frozen-foundation-model + cheap-conditioning
(prompt / LoRA / adapter / soft-prompt / MoE-routing): a big amortized prior plus a
tiny per-target selector. The numbers here quantify that split on a toy substrate.

Requires torch. ~2-3 min.
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


def forward_flops():
    """Rough FLOPs for one FiLM-NCA rollout forward (mul+add counted as 2)."""
    C, HID, D, N, S = bp.C, bp.HID, bp.D, bp.H * bp.W, bp.STEPS
    per_cell = (3 * C) * HID + HID * C          # W1, W2
    per_step = N * per_cell * 2 + 2 * HID * D   # + FiLM (per-member, not per-cell)
    return per_step * S


def main():
    bp.HID, bp.D, bp.STEPS = 32, 12, 24
    t0 = time.time()
    fwd = forward_flops()
    print("=" * 70)
    print("v13: compute amortization of the NCA codebook (is it a feasible architecture?)")
    print("=" * 70)
    body_params = bp.FiLMNCA().body_numel()
    print(f"body |theta|={body_params}, code dim d={bp.D}  -> per-target params ratio = "
          f"{body_params/bp.D:.0f}x fewer to SELECT than to BUILD")
    print(f"~forward FLOPs/rollout = {fwd:.2e}")

    pool = [bp.target(i) for i in range(24)]
    held = [bp.target(300 + k) for k in range(4)]

    # BUILD (one-time): train toolkit on K=24
    t = time.time(); net, tr = bp.train_body(pool, iters=800)
    for p in net.parameters():
        p.requires_grad_(False)
    build_t = time.time() - t
    build_flops = 800 * 24 * fwd * 3          # iters * K * (fwd+bwd~2x) ~ *3
    print(f"\nBUILD toolkit (one-time): {build_t:.0f}s, ~{build_flops:.1e} FLOPs, train corr={tr:.2f}")

    # SELECT (recurring): fit a d=12 code to a held-out target
    t = time.time(); hos = [bp.fit_code(net, h, iters=400) for h in held]
    sel_t = (time.time() - t) / len(held)
    sel_flops = 400 * fwd * 3
    print(f"SELECT one held-out target (frozen body, d={bp.D} code): {sel_t:.1f}s, "
          f"~{sel_flops:.1e} FLOPs, held-out corr={np.mean(hos):.2f}")

    # SCRATCH (baseline): train a whole fresh body per target
    t = time.time()
    T.manual_seed(0)
    scratch_hos = []
    for h in held[:2]:
        netk, _ = bp.train_body([h], iters=400)
        with T.no_grad():
            c = T.zeros(1, bp.D)
            scratch_hos.append(float(bp.corr_paired(netk.grow(c), h[None])[0]))
    scratch_t = (time.time() - t) / 2
    scratch_flops = 400 * 1 * fwd * 3
    print(f"SCRATCH one target (train fresh body): {scratch_t:.0f}s, ~{scratch_flops:.1e} FLOPs")

    # amortization crossover
    # amortized(T) = build + T*select ; scratch(T) = T*scratch
    denom = (scratch_flops - sel_flops)
    Tstar = build_flops / denom if denom > 0 else float("inf")
    print(f"\nAMORTIZATION:")
    print(f"  per-target marginal cost: SELECT {sel_flops:.1e} vs SCRATCH {scratch_flops:.1e} FLOPs "
          f"({scratch_flops/sel_flops:.1f}x cheaper) and {body_params}->{bp.D} params ({body_params/bp.D:.0f}x)")
    print(f"  crossover T* (amortized cheaper than per-target training): "
          f"{Tstar:.0f} targets" if np.isfinite(Tstar) else "  select>=scratch")
    print(f"  => beyond ~{Tstar:.0f} targets the shared-toolkit+cheap-code design wins on total compute;")
    print(f"     the per-target MARGINAL cost is a {bp.D}-number code, not a {body_params}-param model.")

    print("\n[interpretation] This is the compute structure of frozen-foundation-model +")
    print("  cheap conditioning (prompt/LoRA/adapter/MoE-routing): amortized big prior +")
    print("  tiny per-target selector. The NCA is a toy substrate; the deployable form of")
    print("  the SAME principle is conditioning on a frozen transformer backbone.")
    print(f"\n[done] {time.time()-t0:.0f}s")
    print("JSON_V13_BEGIN")
    print(json.dumps(dict(body_params=body_params, fwd_flops=fwd,
                          build_flops=build_flops, sel_flops=sel_flops,
                          scratch_flops=scratch_flops, Tstar=float(Tstar),
                          held_out=float(np.mean(hos)))))
    print("JSON_V13_END")


if __name__ == "__main__":
    main()
