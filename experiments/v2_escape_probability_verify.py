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
eta_star = T + (KAPPA/MU)*(L @ T)

def gain_matrix(O_idx, F_idx):
    LFF = L[np.ix_(F_idx, F_idx)]
    WFO = W[np.ix_(F_idx, O_idx)]
    A = MU*np.eye(len(F_idx)) + KAPPA*LFF
    G = np.linalg.solve(A, KAPPA*WFO)
    return G, A

print("=== Exact identity check: s_i + q_i = 1 ===")
r = np.random.default_rng(1)
perm = r.permutation(Nfull)
n_obs = 18
O_idx, F_idx = perm[:n_obs], perm[n_obs:]
G, A = gain_matrix(O_idx, F_idx)
s = G.sum(axis=1)                              # escape probability, from gain matrix row sums
q = MU * np.linalg.solve(A, np.ones(len(F_idx)))  # kill probability, from the resolvent formula
print(f"max |s_i + q_i - 1| over all free nodes = {np.max(np.abs(s+q-1)):.2e}  (should be ~0, exact identity)")

print("\n=== Monotonicity check: does escape probability for a FIXED node increase as the free region shrinks around it? ===")
target_node = 14  # a fixed node, track its escape probability as coverage grows around it
r2 = np.random.default_rng(7)
base_perm = r2.permutation([i for i in range(Nfull) if i != target_node])
for n_obs in [5, 10, 15, 20, 25, 30]:
    O_idx = np.array(base_perm[:n_obs])
    F_idx = np.array([target_node] + [i for i in base_perm[n_obs:]])
    G, A = gain_matrix(O_idx, F_idx)
    s_target = G[0].sum()   # target_node is F_idx[0]
    print(f"  n_observed={n_obs:<4} (free region size={len(F_idx)})  escape probability s_target = {s_target:.4f}")
