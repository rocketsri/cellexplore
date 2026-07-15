import numpy as np
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
    return laplacian_from_edges(n, edges), (rows,cols)

def random_regular_graph(n, degree, rng, max_tries=200):
    for _ in range(max_tries):
        stubs = np.repeat(np.arange(n), degree); rng.shuffle(stubs)
        edges=set(); ok=True
        for k in range(0, len(stubs)-1, 2):
            a,b = stubs[k], stubs[k+1]
            if a==b or (a,b) in edges or (b,a) in edges: ok=False; break
            edges.add((a,b))
        if ok: return laplacian_from_edges(n, list(edges))
    return laplacian_from_edges(n, list(edges))

def escape_prob(L, W, O_idx, F_idx):
    LFF = L[np.ix_(F_idx,F_idx)]; WFO = W[np.ix_(F_idx,O_idx)]
    A = MU*np.eye(len(F_idx)) + KAPPA*LFF
    return np.linalg.solve(A, KAPPA*WFO).sum(axis=1)

# ============ TEST 1: clustered vs scattered corruption, N-scaling ============
print("=== TEST 1: does scale help when corruption is CLUSTERED (a compact patch) rather than scattered? ===")
print(f"{'side':<6}{'N':<7}{'scattered (uniform random)':<30}{'clustered (compact square patch)'}")
for side in [6, 10, 14, 20]:
    (L,W), (rows,cols) = grid_graph(side, side)
    n = side*side
    frac = 0.3
    n_obs = int(frac*n)

    # scattered baseline
    rng = np.random.default_rng(21)
    scat_vals=[]
    for _ in range(25):
        perm = rng.permutation(n)
        O_idx, F_idx = perm[:n_obs], perm[n_obs:]
        scat_vals.append(escape_prob(L,W,O_idx,F_idx).mean())

    # clustered: a compact square patch of ~n_obs nodes placed at a random corner/location
    patch_side = int(np.sqrt(n_obs))
    clus_vals=[]
    rng2 = np.random.default_rng(22)
    for _ in range(25):
        r0 = rng2.integers(0, max(1,rows-patch_side+1))
        c0 = rng2.integers(0, max(1,cols-patch_side+1))
        O_idx = [ (r0+dr)*cols+(c0+dc) for dr in range(patch_side) for dc in range(patch_side)
                  if r0+dr<rows and c0+dc<cols ]
        O_idx = np.array(O_idx[:n_obs] if len(O_idx)>=n_obs else O_idx)
        F_idx = np.array([i for i in range(n) if i not in set(O_idx.tolist())])
        clus_vals.append(escape_prob(L,W,O_idx,F_idx).mean())

    print(f"{side:<6}{n:<7}{np.mean(scat_vals):.4f} +/- {np.std(scat_vals):.4f}{'':<10}{np.mean(clus_vals):.4f} +/- {np.std(clus_vals):.4f}")

# ============ TEST 2: modular/compartmentalized topology -- does it protect clean modules from corrupted ones? ============
print("\n=== TEST 2: modular (compartmentalized) topology -- does it CONTAIN module-level corruption? ===")
def modular_graph(n_modules, module_size, internal_degree, n_bridges_per_pair, rng):
    n = n_modules*module_size
    edges=[]
    module_of = {}
    for m in range(n_modules):
        base = m*module_size
        for i in range(module_size): module_of[base+i]=m
        # dense-ish internal connectivity via random regular within module
        stubs = np.repeat(np.arange(base, base+module_size), internal_degree)
        rng.shuffle(stubs)
        seen=set()
        for k in range(0, len(stubs)-1, 2):
            a,b = stubs[k], stubs[k+1]
            if a!=b and (a,b) not in seen and (b,a) not in seen:
                edges.append((a,b)); seen.add((a,b))
    # sparse bridges: connect module m to module m+1 (ring of modules)
    for m in range(n_modules):
        m2 = (m+1) % n_modules
        for _ in range(n_bridges_per_pair):
            a = rng.integers(m*module_size, (m+1)*module_size)
            b = rng.integers(m2*module_size, (m2+1)*module_size)
            edges.append((a,b))
    return laplacian_from_edges(n, edges), module_of

n_modules, module_size = 8, 8
n = n_modules*module_size  # 64, matched to earlier topology comparison
rng = np.random.default_rng(9)
(L_mod, W_mod), module_of = modular_graph(n_modules, module_size, internal_degree=4, n_bridges_per_pair=1, rng=rng)
(L_reg, W_reg) = random_regular_graph(n, 4, np.random.default_rng(3))

def module_corruption_test(L, W, module_of, n_modules, module_size, n_corrupted_modules, trials, rng):
    # corrupt ALL nodes in n_corrupted_modules randomly chosen modules; measure escape
    # probability for FREE nodes specifically in the CLEAN (uncorrupted) modules
    vals=[]
    n = n_modules*module_size
    for _ in range(trials):
        corrupted_modules = set(rng.choice(n_modules, size=n_corrupted_modules, replace=False).tolist())
        O_idx = np.array([i for i in range(n) if module_of[i] in corrupted_modules])
        F_idx = np.array([i for i in range(n) if module_of[i] not in corrupted_modules])
        clean_module_free_escape = escape_prob(L, W, O_idx, F_idx)
        vals.append(clean_module_free_escape.mean())
    return np.mean(vals), np.std(vals)

print(f"{'n_corrupted_modules':<22}{'modular topology (clean-module escape prob)':<48}{'random-regular (same overall size/density)'}")
for n_corrupted in [1, 2, 3, 4]:
    m_mean, m_std = module_corruption_test(L_mod, W_mod, module_of, n_modules, module_size, n_corrupted, 30, np.random.default_rng(50))
    # for random-regular, define "modules" as arbitrary same-size groupings for a fair frac-matched comparison
    module_of_reg = {i: i//module_size for i in range(n)}
    r_mean, r_std = module_corruption_test(L_reg, W_reg, module_of_reg, n_modules, module_size, n_corrupted, 30, np.random.default_rng(50))
    print(f"{n_corrupted:<22}{m_mean:.4f} +/- {m_std:.4f}{'':<26}{r_mean:.4f} +/- {r_std:.4f}")

# Also check ablation-recovery speed (lambda2) within a module vs across the whole graph, to confirm
# the modular design doesn't SACRIFICE repair speed to get this containment benefit
def algebraic_connectivity(L):
    eigs = np.linalg.eigvalsh(L)
    return eigs[1]  # second-smallest (first is ~0)

lam2_mod_whole = algebraic_connectivity(L_mod)
lam2_mod_module = algebraic_connectivity(L_mod[0:module_size,0:module_size][np.ix_(range(module_size),range(module_size))])
# proper within-module connectivity: extract actual module subgraph laplacian from W_mod
W_sub = W_mod[0:module_size,0:module_size]
L_sub = np.diag(W_sub.sum(1)) - W_sub
lam2_mod_module_correct = algebraic_connectivity(L_sub)
lam2_reg = algebraic_connectivity(L_reg)

print(f"\nAblation-recovery-relevant connectivity check:")
print(f"  modular graph, WHOLE-graph lambda2 (cross-module) = {lam2_mod_whole:.4f}")
print(f"  modular graph, WITHIN-module lambda2 (local repair speed) = {lam2_mod_module_correct:.4f}")
print(f"  random-regular graph, whole-graph lambda2 = {lam2_reg:.4f}")
