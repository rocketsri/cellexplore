"""
v1.md's immune-network route found: "the same mechanism that resists
adversarial perturbation mechanically resists legitimate correction, with
no term in the equations distinguishing the two."

PROPOSAL: route legitimate correction through a unit's own local
equilibrium-defining parameter (capacity K_i), not as an external force
competing against the network's homeostatic pull (which is what a hostile
perturbation and a naive "just push x_i down" correction both look like
to the dynamics -- structurally identical, hence identically resisted).

This test checks ONE narrow, precise, mechanistic claim: does a
capacity-modulation correction produce a LASTING re-equilibration
(sticks without continued enforcement) where a raw additive-force
correction does NOT (network homeostasis pulls the unit back once
forcing stops)? It does NOT claim to solve the general "can a
sufficiently adaptive hostile defector eventually learn to resist this
too" problem -- that is a separate, harder, explicitly unresolved
question, flagged at the end.
"""
import numpy as np

rng = np.random.default_rng(0)
N = 12

# Farmer-Packard-Perelson-style clonal network dynamics (v1.md's immune-network route)
M = rng.normal(0, 0.15, size=(N, N))
np.fill_diagonal(M, 0)
M = (M + M.T) / 2  # symmetric idiotypic interaction for stability, as v1 discusses
K = np.full(N, 5.0)   # capacity
d = np.full(N, 0.1)   # death rate
k_supp = 1.0           # suppression scale
s = np.full(N, 0.3)    # baseline local antigen-drive

def dynamics(x, K_, d_, s_):
    stim = M @ x
    supp = k_supp * (M.T @ x)
    return x * (stim - supp - x / K_ - d_) + s_

def simulate(x0, K_, d_, s_, steps, dt=0.02):
    x = x0.copy()
    traj = [x.copy()]
    for _ in range(steps):
        x = np.maximum(x + dt * dynamics(x, K_, d_, s_), 0)
        traj.append(x.copy())
    return np.array(traj)

# 1) Reach baseline equilibrium
x0 = np.full(N, 1.0)
traj0 = simulate(x0, K, d, s, 3000)
x_eq = traj0[-1]
print(f"Baseline equilibrium reached. unit-3 baseline x* = {x_eq[3]:.4f}")

TARGET = 3  # the unit we'll try to correct -- an ordinary, non-defecting unit

# 2a) "Correction A": raw additive negative force on x_3, applied for a window, then removed
def dynamics_forced(x, K_, d_, s_, force_idx, force_mag):
    base = dynamics(x, K_, d_, s_)
    base[force_idx] -= force_mag * x[force_idx]
    return base

x = x_eq.copy()
traj_a = [x.copy()]
force_window = 800
for t in range(2500):
    force_mag = 1.5 if t < force_window else 0.0
    x = np.maximum(x + 0.02 * dynamics_forced(x, K, d, s, TARGET, force_mag), 0)
    traj_a.append(x.copy())
traj_a = np.array(traj_a)
print(f"\nCorrection A (raw additive force, removed after step {force_window}):")
print(f"  x_3 during forcing (step {force_window-1}): {traj_a[force_window-1, TARGET]:.4f}")
print(f"  x_3 long after forcing removed (final):      {traj_a[-1, TARGET]:.4f}")
print(f"  original equilibrium: {x_eq[TARGET]:.4f}")
rebound_frac_A = (traj_a[-1, TARGET] - traj_a[force_window-1, TARGET]) / (x_eq[TARGET] - traj_a[force_window-1, TARGET] + 1e-9)
print(f"  rebound fraction (0=stayed corrected, 1=fully rebounded to original): {rebound_frac_A:.3f}")

# 2b) "Correction B": permanently modulate K_3 (the unit's own capacity), no ongoing force
K_corrected = K.copy()
K_corrected[TARGET] = 1.2  # structural change to the unit's own equilibrium-defining parameter
x = x_eq.copy()
traj_b = [x.copy()]
for t in range(2500):
    K_use = K_corrected if t >= 0 else K  # applied from the start, then left in place permanently
    x = np.maximum(x + 0.02 * dynamics(x, K_use, d, s), 0)
    traj_b.append(x.copy())
traj_b = np.array(traj_b)
print(f"\nCorrection B (permanent capacity modulation K_3: {K[TARGET]} -> {K_corrected[TARGET]}):")
print(f"  x_3 shortly after (step {force_window-1}): {traj_b[force_window-1, TARGET]:.4f}")
print(f"  x_3 long after (final, no forcing ever removed since it's structural): {traj_b[-1, TARGET]:.4f}")
rebound_frac_B = abs(traj_b[-1, TARGET] - traj_b[force_window-1, TARGET]) / (x_eq[TARGET] - traj_b[force_window-1, TARGET] + 1e-9)
print(f"  drift-back fraction after initial settling: {rebound_frac_B:.3f}")

# 3) Now test against a HOSTILE DEFECTOR: does the same asymmetry hold, or does a
#    sufficiently motivated defector resist both equally (the honest limit)?
DEFECTOR = 7
s_defect = s.copy()
s_defect[DEFECTOR] = 2.5  # sharp local-drive spike, simulating reward-hacking pressure

print(f"\n=== Defector case (unit {DEFECTOR}, elevated local drive from {s[DEFECTOR]} to {s_defect[DEFECTOR]}) ===")
traj_def_baseline = simulate(x_eq, K, d, s_defect, 2500)
print(f"Unmitigated defector equilibrium x_{DEFECTOR} = {traj_def_baseline[-1, DEFECTOR]:.4f} "
      f"(vs non-defecting baseline {x_eq[DEFECTOR]:.4f}) -- network suppression contains it: "
      f"{'YES' if traj_def_baseline[-1, DEFECTOR] < 3*x_eq[DEFECTOR] else 'NO, runaway'}")

# Try correction A (force) against the defector
x = x_eq.copy()
for t in range(2500):
    force_mag = 1.5 if t < force_window else 0.0
    x = np.maximum(x + 0.02 * dynamics_forced(x, K, d, s_defect, DEFECTOR, force_mag), 0)
final_A_defector = x[DEFECTOR]

# Try correction B (capacity) against the defector
K_corr_def = K.copy(); K_corr_def[DEFECTOR] = 1.2
traj_def_B = simulate(x_eq, K_corr_def, d, s_defect, 2500)
final_B_defector = traj_def_B[-1, DEFECTOR]

print(f"Defector under Correction A (force) at steady state: x_{DEFECTOR} = {final_A_defector:.4f}")
print(f"Defector under Correction B (capacity) at steady state: x_{DEFECTOR} = {final_B_defector:.4f}")
print(f"(For reference, unmitigated defector = {traj_def_baseline[-1, DEFECTOR]:.4f}, honest-unit target level = {x_eq[DEFECTOR]:.4f})")
