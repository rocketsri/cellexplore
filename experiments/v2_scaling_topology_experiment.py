import numpy as np
rng = np.random.default_rng(0)
MU, KAPPA = 1.0, 3.0

def laplacian_from_edges(n, edges):
    W = np.zeros((n,n))
    for i,j in edges:
        W[i,j]=W[j,i]=1.0
    return np.diag(W.sum(1))-W, W

def grid_graph(rows, cols):
    n = rows*cols; edges=[]
    for r in range(rows):
        for c in range(cols):
            i=r*cols+c
            for dr,dc in [(0,1),(1,0)]:
                rr,cc=r+dr,c+dc
                if rr<rows and cc<cols: edges.append((i, rr*cols+cc))
    return laplacian_from_edges(n, edges)

def chain_graph(n):
    return laplacian_from_edges(n, [(i,i+1) for i in range(n-1)])

def random_regular_graph(n, degree, rng, max_tries=200):
    # simple random regular-ish graph via repeated random pairing + rewiring to avoid multi-edges/self-loops
    for _ in range(max_tries):
        stubs = np.repeat(np.arange(n), degree)
        rng.shuffle(stubs)
        edges = set()
        ok = True
        for k in range(0, len(stubs)-1, 2):
            a,b = stubs[k], stubs[k+1]
            if a==b or (a,b) in edges or (b,a) in edges:
                ok = False; break
            edges.add((a,b))
        if ok: return laplacian_from_edges(n, list(edges))
    return laplacian_from_edges(n, list(edges))  # fall back to whatever we got

def avg_escape_prob(L, W, n, frac_obs, trials, rng):
    n_obs = int(frac_obs*n)
    vals = []
    for _ in range(trials):
        perm = rng.permutation(n)
        O_idx, F_idx = perm[:n_obs], perm[n_obs:]
        LFF = L[np.ix_(F_idx,F_idx)]; WFO = W[np.ix_(F_idx,O_idx)]
        A = MU*np.eye(len(F_idx)) + KAPPA*LFF
        G = np.linalg.solve(A, KAPPA*WFO)
        vals.append(G.sum(axis=1).mean())
    return np.mean(vals), np.std(vals)

print("=== Topology comparison at matched N=64, degree~4, fixed coverage=0.5 ===")
n = 64
L_grid, W_grid = grid_graph(8, 8)                         # 2D grid, degree 2-4, large diameter (~14)
L_reg, W_reg = random_regular_graph(n, 4, np.random.default_rng(3))  # random 4-regular, small diameter (~log n)
L_chain, W_chain = chain_graph(n)                          # 1D chain, degree ~2, huge diameter (~63)

for name, (L,W) in [("2D grid (degree~4, diam~14)", (L_grid,W_grid)),
                     ("random 4-regular (degree=4, diam~log n small)", (L_reg,W_reg)),
                     ("1D chain (degree~2, diam~63)", (L_chain,W_chain))]:
    m, s = avg_escape_prob(L, W, n, 0.5, 25, np.random.default_rng(11))
    print(f"  {name:<48} avg escape prob (completion fragility) = {m:.4f} +/- {s:.4f}")

print("\n=== N-scaling for 2D grid at FIXED coverage fraction=0.3 (does larger diameter -> more robust?) ===")
for side in [6, 10, 14, 20]:
    n = side*side
    L,W = grid_graph(side, side)
    m, s = avg_escape_prob(L, W, n, 0.3, 25, np.random.default_rng(21))
    print(f"  side={side:<4} N={n:<5} avg escape prob = {m:.4f} +/- {s:.4f}")

print("\n=== N-scaling for random regular graph (degree=4) at FIXED coverage fraction=0.3 (small diameter regardless of N -> should NOT improve much) ===")
for n in [64, 144, 256, 400]:
    L,W = random_regular_graph(n, 4, np.random.default_rng(5))
    m, s = avg_escape_prob(L, W, n, 0.3, 25, np.random.default_rng(21))
    print(f"  N={n:<5} avg escape prob = {m:.4f} +/- {s:.4f}")
