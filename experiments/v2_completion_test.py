import numpy as np
rng = np.random.default_rng(0)

N = 30    # units, arranged on a 2D grid for a genuine "inpainting" flavor
D = 4     # embedding dim per unit
side = 6  # 6x6 grid ~ 36, trim to 30

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
    D_ = np.diag(W.sum(1))
    return D_-W, W

rows, cols = 6, 6
Nfull = rows*cols
L, W = grid_laplacian(rows, cols)

T = rng.normal(0, 1, size=(Nfull, D))  # a fixed "target pattern"

def eta_from_T(mu, kappa, T, L):
    # eta_i = T_i + (kappa/mu) * (L T)_i    [since sum_j w_ij(T_i-T_j) = (L T)_i]
    return T + (kappa/mu) * (L @ T)

def complete(mu, kappa, O_idx, F_idx, c_O, eta, L, W):
    LFF = L[np.ix_(F_idx, F_idx)]
    WFO = W[np.ix_(F_idx, O_idx)]
    A = mu*np.eye(len(F_idx)) + kappa*LFF
    b = mu*eta[F_idx] + kappa*(WFO @ c_O)
    return np.linalg.solve(A, b)

def run_trial(mu, kappa, frac_observed, noise_std, seed):
    r = np.random.default_rng(seed)
    eta = eta_from_T(mu, kappa, T, L)
    n_obs = int(frac_observed*Nfull)
    perm = r.permutation(Nfull)
    O_idx, F_idx = perm[:n_obs], perm[n_obs:]
    delta = r.normal(0, noise_std, size=(n_obs, D))
    c_O = T[O_idx] + delta
    x_F = complete(mu, kappa, O_idx, F_idx, c_O, eta, L, W)
    err = np.linalg.norm(x_F - T[F_idx]) / np.linalg.norm(T[F_idx])
    return err

print("=== Prediction 1: completion error scales linearly with corruption magnitude ===")
mu, kappa, frac_obs = 1.0, 3.0, 0.5
for noise_std in [0.0, 0.1, 0.2, 0.4, 0.8]:
    errs = [run_trial(mu, kappa, frac_obs, noise_std, s) for s in range(20)]
    print(f"  noise_std={noise_std:.2f}  relative completion error = {np.mean(errs):.4f} +/- {np.std(errs):.4f}")

print("\n=== Prediction 2: error shrinks as local prior strength mu grows (resists bad evidence) ===")
kappa, frac_obs, noise_std = 3.0, 0.5, 0.4
for mu in [0.1, 0.5, 1.0, 3.0, 8.0]:
    errs = [run_trial(mu, kappa, frac_obs, noise_std, s) for s in range(20)]
    print(f"  mu={mu:<6.2f} relative completion error = {np.mean(errs):.4f} +/- {np.std(errs):.4f}")

print("\n=== Prediction 3: error shrinks as fraction of clean/observed evidence grows ===")
mu, kappa, noise_std = 1.0, 3.0, 0.4
for frac_obs in [0.15, 0.3, 0.5, 0.7, 0.9]:
    errs = [run_trial(mu, kappa, frac_obs, noise_std, s) for s in range(20)]
    print(f"  frac_observed={frac_obs:.2f}  relative completion error = {np.mean(errs):.4f} +/- {np.std(errs):.4f}")

print("\n=== Confirm the exact linear gain relationship (measured slope vs ||G|| operator norm) ===")
mu, kappa, frac_obs = 1.0, 3.0, 0.5
r = np.random.default_rng(0)
eta = eta_from_T(mu, kappa, T, L)
perm = r.permutation(Nfull)
n_obs = int(frac_obs*Nfull)
O_idx, F_idx = perm[:n_obs], perm[n_obs:]
LFF = L[np.ix_(F_idx, F_idx)]; WFO = W[np.ix_(F_idx, O_idx)]
A = mu*np.eye(len(F_idx)) + kappa*LFF
G = np.linalg.solve(A, kappa*WFO)
predicted_gain = np.linalg.norm(G, 2)  # spectral norm
measured_slopes = []
for noise_std in [0.1, 0.2, 0.4]:
    errs_abs = []
    for s in range(30):
        rr = np.random.default_rng(1000+s)
        delta = rr.normal(0, noise_std, size=(n_obs, D))
        c_O = T[O_idx] + delta
        x_F = complete(mu, kappa, O_idx, F_idx, c_O, eta, L, W)
        errs_abs.append(np.linalg.norm(x_F - T[F_idx]) / np.linalg.norm(delta))
    measured_slopes.append(np.mean(errs_abs))
print(f"  theoretical ||G||_2 (operator norm bound) = {predicted_gain:.4f}")
print(f"  measured ||x_F*-T_F|| / ||delta|| at noise_std=0.1,0.2,0.4: {[f'{m:.4f}' for m in measured_slopes]}")
