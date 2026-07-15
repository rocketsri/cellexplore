"""
Does the completion/denoising theory (derived for an analytically hand-set eta)
survive when eta is instead reached by REAL, stochastic equilibrium-propagation
training from random initialization?

Procedure per training step (literal EqProp, as specified in v2.md Phase 1-2):
  free phase:   x0 = argmin_x E(x; eta)                      [all nodes free]
  nudged phase: x_b = argmin_x E(x; eta) + (beta/2)||x-T||^2 [weak pull toward T]
  eqprop grad estimate: d eta_i = (1/beta) * mu * (x0_i - x_b_i)   [see derivation]
  eta <- eta - lr * d_eta   (gradient DESCENT on the eqprop-estimated gradient
                             of the true objective ||x0(eta) - T||^2)

Note: dE/d(eta_i) = -mu*(x_i - eta_i), so the EqProp estimator for d/deta of the
loss is (1/beta)[dE/deta(x_b) - dE/deta(x0)] = (1/beta)*mu*[(x0_i-eta_i)-(x_b_i-eta_i)]
                                              = (mu/beta)*(x0_i - x_b_i)
"""
import numpy as np
rng = np.random.default_rng(0)

def grid_laplacian(rows, cols):
    n = rows*cols
    W = np.zeros((n,n))
    for r in range(rows):
        for c in range(cols):
            i = r*cols+c
            for dr,dc in [(0,1),(1,0)]:
                rr,cc = r+dr, c+dc
                if rr<rows and cc<cols:
                    j = rr*cols+cc
                    W[i,j]=W[j,i]=1.0
    return np.diag(W.sum(1))-W, W

rows, cols = 6, 6
Nfull = rows*cols
L, W = grid_laplacian(rows, cols)
D = 4
T = rng.normal(0, 1, size=(Nfull, D))

MU, KAPPA = 1.0, 3.0
A_full = MU*np.eye(Nfull) + KAPPA*L   # full-graph settling operator

def settle_full(eta, beta=0.0, target=None):
    if beta == 0.0:
        return np.linalg.solve(A_full, MU*eta)
    A = A_full + beta*np.eye(Nfull)
    return np.linalg.solve(A, MU*eta + beta*target)

eta_star_closed_form = T + (KAPPA/MU)*(L @ T)

# --- Real, stochastic EqProp training from random init ---
eta = rng.normal(0, 0.5, size=(Nfull, D))   # random init, NOT the closed form
beta = 0.05
lr = 0.3
n_steps = 4000
losses = []
for step in range(n_steps):
    x0 = settle_full(eta, beta=0.0)
    xb = settle_full(eta, beta=beta, target=T)
    grad_eta = (MU/beta) * (x0 - xb)          # EqProp gradient estimate
    eta = eta - lr*grad_eta
    if step % 200 == 0:
        losses.append(np.linalg.norm(x0-T)/np.linalg.norm(T))

x0_final = settle_full(eta, beta=0.0)
print(f"Loss (||x0-T||/||T||) trajectory (every 200 steps): {[f'{l:.4f}' for l in losses]}")
print(f"\nFinal EqProp-trained ||x0_final - T|| / ||T|| = {np.linalg.norm(x0_final-T)/np.linalg.norm(T):.5f}")
print(f"Distance from trained eta to analytic eta*: ||eta_trained - eta*||/||eta*|| = "
      f"{np.linalg.norm(eta-eta_star_closed_form)/np.linalg.norm(eta_star_closed_form):.5f}")

# Now: does the SAME completion/denoising theory hold for this REALISTICALLY TRAINED eta?
def complete(mu, kappa, O_idx, F_idx, c_O, eta, L, W):
    LFF = L[np.ix_(F_idx, F_idx)]
    WFO = W[np.ix_(F_idx, O_idx)]
    A = mu*np.eye(len(F_idx)) + kappa*LFF
    b = mu*eta[F_idx] + kappa*(WFO @ c_O)
    return np.linalg.solve(A, b)

def run_trial(eta_use, frac_observed, noise_std, seed):
    r = np.random.default_rng(seed)
    n_obs = int(frac_observed*Nfull)
    perm = r.permutation(Nfull)
    O_idx, F_idx = perm[:n_obs], perm[n_obs:]
    delta = r.normal(0, noise_std, size=(n_obs, D))
    c_O = T[O_idx] + delta
    x_F = complete(MU, KAPPA, O_idx, F_idx, c_O, eta_use, L, W)
    return np.linalg.norm(x_F - T[F_idx]) / np.linalg.norm(T[F_idx])

print("\n=== Completion robustness: EqProp-TRAINED eta vs analytically hand-set eta* ===")
print(f"{'noise_std':<12}{'trained eta':<18}{'closed-form eta*':<20}{'relative difference'}")
for noise_std in [0.1, 0.2, 0.4]:
    errs_trained = [run_trial(eta, 0.5, noise_std, s) for s in range(30)]
    errs_closed  = [run_trial(eta_star_closed_form, 0.5, noise_std, s) for s in range(30)]
    mt, mc = np.mean(errs_trained), np.mean(errs_closed)
    print(f"{noise_std:<12.2f}{mt:<18.4f}{mc:<20.4f}{abs(mt-mc)/mc:+.2%}")

print("\n=== Does the connectivity-buffering (frac_observed) effect ALSO replicate with trained eta? ===")
for frac_obs in [0.15, 0.5, 0.9]:
    errs_trained = [run_trial(eta, frac_obs, 0.4, s) for s in range(30)]
    print(f"frac_observed={frac_obs:.2f}  trained-eta completion error = {np.mean(errs_trained):.4f} +/- {np.std(errs_trained):.4f}")
