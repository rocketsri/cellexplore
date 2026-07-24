"""Generate a valid .ipynb via json.dump (guarantees well-formed JSON)."""
import json

def _lines(src):
    """nbformat requires each source line to retain its trailing newline
    (all but the last); otherwise the cell concatenates into a single line."""
    body = src.strip("\n").split("\n")
    return [ln + "\n" for ln in body[:-1]] + [body[-1]]

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(src)}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": _lines(src)}

cells = []

cells.append(md(r"""
# GPU scaling experiments — graph-coupled alignment architecture

Scales the CPU experiments from `paper.md` (which topped out at $N=400$) to $N \sim 10^4$–$10^5$ on GPU,
to test three questions currently listed as **open** in `paper.md` §5:

1. **§5.2 / §5.6 — scaling.** Does the compartmentalization advantage (measured 8x at $N=64$) grow,
   shrink, or stay constant with $N$? Is there an optimal module size $m^*$?
2. **§5.5 — bridge vulnerability.** The sparse inter-module bridges are a concentrated attack surface.
   How much cheaper is a targeted bridge attack than a random attack of equal budget?
3. **§3.4 vs §3.5 — threat-model dependence.** Does the scattered-vs-clustered corruption asymmetry
   hold at scale?

**Core quantity** (Theorem 5, `paper.md`): completion fragility equals the escape probability of a
killed random walk,
$$s_F \;=\; (\mu I_F + \kappa L_{FF})^{-1}\,\kappa\, W_{FO}\,\mathbf{1}_O ,$$
where $O$ = corrupted/clamped nodes, $F$ = free nodes, $\mu$ = local prior strength, $\kappa$ = coupling.
High $s$ = fragile (corruption propagates in), low $s$ = robust.

**Runtime:** set Runtime → Change runtime type → GPU (T4 is sufficient).
"""))

cells.append(md("## 0. Setup"))

cells.append(code(r"""
import numpy as np, torch, time, math
import matplotlib.pyplot as plt

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# float64 keeps CG well-conditioned. On T4, fp64 runs at 1/32 fp32 rate but problem sizes here
# are solver-bound, not flop-bound. Drop to float32 only if you hit memory limits at N > 1e5.
DTYPE = torch.float64

print("torch:", torch.__version__)
print("device:", DEV)
if DEV.type == "cuda":
    print("gpu:", torch.cuda.get_device_name(0))
else:
    print("WARNING: no GPU detected - this will still run but slowly at large N.")

MU, KAPPA = 1.0, 3.0   # matches paper.md throughout
"""))

cells.append(md("""
## 1. Graph builders

Three topologies, all returned as a deduped, symmetrized edge list `(lo, hi, n)`:

- **grid** — 2D lattice, large diameter $O(\\sqrt{N})$
- **random** — Erdős–Rényi-ish with mean degree $\\approx d$, small diameter $O(\\log N)$ (expander-like)
- **modular** — $M$ modules of $m$ nodes, dense within, `b` bridge edges per adjacent module pair in a ring

Note: the `random` builder gives *mean* degree $\\approx d$, not exact $d$-regularity (exact regular-graph
sampling is slow at $N\\sim10^5$). Actual mean degree is reported for every graph so comparisons stay honest.
"""))

cells.append(code(r"""
def _sym_dedupe(src, dst, n):
    src = np.asarray(src); dst = np.asarray(dst)
    keep = src != dst                                  # drop self-loops
    src, dst = src[keep], dst[keep]
    lo = np.minimum(src, dst); hi = np.maximum(src, dst)
    key = lo.astype(np.int64) * n + hi.astype(np.int64)
    _, idx = np.unique(key, return_index=True)         # drop duplicate edges
    return lo[idx], hi[idx]

def grid_graph(side):
    n = side * side
    idx = np.arange(n).reshape(side, side)
    src = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()])
    dst = np.concatenate([idx[:, 1:].ravel(),  idx[1:, :].ravel()])
    lo, hi = _sym_dedupe(src, dst, n)
    return lo, hi, n

def random_graph(n, deg, rng):
    '''Configuration-model pairing: n*deg/2 edges, so MEAN DEGREE ~= deg (not 2*deg).
    Matching degree honestly matters - an unmatched control would inflate any comparison.'''
    stubs = np.repeat(np.arange(n), deg).copy()
    rng.shuffle(stubs)
    k = (len(stubs) // 2) * 2
    lo, hi = _sym_dedupe(stubs[0:k:2], stubs[1:k:2], n)
    return lo, hi, n

def modular_graph(M, m, deg_in, b, rng):
    '''M modules of m nodes; internal mean degree ~= deg_in; b bridges per adjacent pair (ring).'''
    n = M * m
    src_l, dst_l = [], []
    for k in range(M):                       # stub-pair WITHIN each module
        stubs = np.repeat(np.arange(k * m, (k + 1) * m), deg_in).copy()
        rng.shuffle(stubs)
        kk = (len(stubs) // 2) * 2
        src_l.append(stubs[0:kk:2]); dst_l.append(stubs[1:kk:2])
    src = np.concatenate(src_l); dst = np.concatenate(dst_l)
    bs, bd = [], []
    for k in range(M):
        k2 = (k + 1) % M
        if k2 == k:
            continue
        bs.extend(rng.integers(k * m, (k + 1) * m, size=b).tolist())
        bd.extend(rng.integers(k2 * m, (k2 + 1) * m, size=b).tolist())
    src = np.concatenate([src, np.array(bs, dtype=np.int64)])
    dst = np.concatenate([dst, np.array(bd, dtype=np.int64)])
    lo, hi = _sym_dedupe(src, dst, n)
    return lo, hi, n

def module_id(n, m):
    return np.arange(n) // m

def bridge_mask(lo, hi, m):
    '''True for edges whose endpoints lie in different modules.'''
    return (lo // m) != (hi // m)

def mean_degree(lo, hi, n):
    return 2.0 * len(lo) / n
"""))

cells.append(md("""
## 2. Sparse Laplacian + GPU conjugate-gradient solver

We never form $L_{FF}$ explicitly. Keeping iterates zero on $O$, for $i \\in F$:
$(L v)_i = d_i v_i - \\sum_j w_{ij} v_j = (L_{FF} v_F)_i$ automatically, because $v$ vanishes on $O$.
So one full-graph sparse matvec plus a mask gives the restricted operator. The operator
$\\mu I + \\kappa L_{FF}$ is symmetric positive definite for $\\mu>0$ (Theorem 1), so CG is guaranteed to converge.
"""))

cells.append(code(r"""
def build_sparse(lo, hi, n, dev=DEV, dtype=DTYPE):
    i = np.concatenate([lo, hi]); j = np.concatenate([hi, lo])
    idx = torch.tensor(np.stack([i, j]), dtype=torch.long, device=dev)
    val = torch.ones(idx.shape[1], dtype=dtype, device=dev)
    W = torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()
    deg = torch.sparse.sum(W, dim=1).to_dense()
    return W, deg

def _spmv(W, v):
    return torch.sparse.mm(W, v.unsqueeze(1)).squeeze(1)

def escape_probability(W, deg, O_mask, mu=MU, kappa=KAPPA, tol=1e-9, maxit=20000):
    '''Returns (s, iters): s[i] = escape probability for free node i (0 on O).'''
    Ffloat = (~O_mask).to(deg.dtype)
    b = kappa * _spmv(W, O_mask.to(deg.dtype)) * Ffloat

    def A(v):
        return (mu * v + kappa * (deg * v - _spmv(W, v))) * Ffloat

    x = torch.zeros_like(b); r = b.clone(); p = r.clone()
    rs = torch.dot(r, r)
    if torch.sqrt(rs) < tol:
        return x, 0
    for it in range(1, maxit + 1):
        Ap = A(p)
        alpha = rs / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)
        if torch.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x, it

def mean_escape(W, deg, O_mask, **kw):
    s, _ = escape_probability(W, deg, O_mask, **kw)
    F = ~O_mask
    return (s[F].mean().item() if F.any() else float('nan'))

def algebraic_connectivity(lo, hi, n, dense_limit=2500):
    '''lambda_2 of the graph Laplacian. Dense below dense_limit, LOBPCG above.'''
    if n <= dense_limit:
        Wd = np.zeros((n, n))
        Wd[lo, hi] = 1.0; Wd[hi, lo] = 1.0
        L = np.diag(Wd.sum(1)) - Wd
        return float(np.linalg.eigvalsh(L)[1])
    W, deg = build_sparse(lo, hi, n)
    i = torch.arange(n, device=DEV)
    Ld = torch.sparse_coo_tensor(torch.stack([i, i]), deg, (n, n))
    L = (Ld - W).coalesce()
    try:
        vals, _ = torch.lobpcg(L, k=2, largest=False, niter=400, tol=1e-7)
        return float(vals.sort().values[1])
    except Exception as e:
        print("  lobpcg failed:", e); return float('nan')
"""))

cells.append(md("""
## 3. Validation — verify the GPU solver before trusting any new number

Two independent checks:
1. **CG vs. dense direct solve** on a small graph — pure linear-algebra correctness.
2. **Theorem 5's identity** $s_i + q_i = 1$, where $q_i = \\mu[(\\mu I_F + \\kappa L_{FF})^{-1}\\mathbf{1}_F]_i$
   is the kill probability. Verified to machine precision on CPU in `paper.md`; must hold here too.
"""))

cells.append(code(r"""
rng = np.random.default_rng(0)
lo, hi, n = grid_graph(8)
W, deg = build_sparse(lo, hi, n)
O_mask = torch.zeros(n, dtype=torch.bool, device=DEV)
O_mask[torch.tensor(rng.choice(n, 20, replace=False), device=DEV)] = True

# --- check 1: CG vs dense ---
s_cg, iters = escape_probability(W, deg, O_mask)
Wd = np.zeros((n, n)); Wd[lo, hi] = 1.0; Wd[hi, lo] = 1.0
Ld = np.diag(Wd.sum(1)) - Wd
Om = O_mask.cpu().numpy(); Fm = ~Om
A_dense = MU * np.eye(Fm.sum()) + KAPPA * Ld[np.ix_(Fm, Fm)]
b_dense = KAPPA * Wd[np.ix_(Fm, Om)] @ np.ones(Om.sum())
s_dense = np.linalg.solve(A_dense, b_dense)
err = np.abs(s_cg.cpu().numpy()[Fm] - s_dense).max()
print(f"check 1  CG vs dense: max abs diff = {err:.3e}  ({iters} CG iters)   {'PASS' if err < 1e-8 else 'FAIL'}")

# --- check 2: Theorem 5 identity s + q = 1 ---
Ffloat = (~O_mask).to(DTYPE)
def A_op(v):
    return (MU * v + KAPPA * (deg * v - _spmv(W, v))) * Ffloat
rhs = MU * Ffloat
x = torch.zeros_like(rhs); r = rhs.clone(); p = r.clone(); rs = torch.dot(r, r)
for _ in range(20000):
    Ap = A_op(p); alpha = rs / torch.dot(p, Ap)
    x = x + alpha * p; r = r - alpha * Ap
    rs_new = torch.dot(r, r)
    if torch.sqrt(rs_new) < 1e-12: break
    p = r + (rs_new / rs) * p; rs = rs_new
ident = (s_cg + x)[~O_mask]
dev_max = (ident - 1.0).abs().max().item()
print(f"check 2  max |s_i + q_i - 1| = {dev_max:.3e}   {'PASS' if dev_max < 1e-8 else 'FAIL'}")
"""))

cells.append(md("""
## 4. Experiment A — does the modular advantage hold at scale?

`paper.md` §3.6 measured ~8x better containment for a modular graph at $N=64$ (escape prob 0.066 vs 0.537
when one module is fully corrupted). **Open question:** does that ratio grow, shrink, or stay constant as
$N$ grows? Three regimes are swept separately, because they plausibly differ:

- **(a) $M$ fixed, $m$ growing** — bigger modules, same count
- **(b) $m$ fixed, $M$ growing** — more modules, same size
- **(c) $m = M = \\sqrt{N}$** — both grow

In each case corruption is one whole module, and the non-modular control is a random graph of the same $N$
and matched mean degree, with the same *number* of corrupted nodes (chosen at random).
"""))

cells.append(code(r"""
def modular_vs_random(M, m, deg_in=4, b=1, seed=0, n_trials=5):
    rng = np.random.default_rng(seed)
    n = M * m
    lo, hi, _ = modular_graph(M, m, deg_in, b, rng)
    Wm, dm = build_sparse(lo, hi, n)
    md_mod = mean_degree(lo, hi, n)

    # corrupt one whole module; measure escape prob for free nodes in clean modules
    mod_vals = []
    for t in range(n_trials):
        k = rng.integers(0, M)
        O = torch.zeros(n, dtype=torch.bool, device=DEV)
        O[k * m:(k + 1) * m] = True
        mod_vals.append(mean_escape(Wm, dm, O))

    # control: random graph, matched N and mean degree, same COUNT of corrupted nodes.
    # Degree matching is checked in the output columns - an unmatched control would
    # inflate the modular advantage, since more connectivity => higher escape probability.
    deg_r = max(1, int(round(md_mod)))
    lo2, hi2, _ = random_graph(n, deg_r, rng)
    Wr, dr = build_sparse(lo2, hi2, n)
    md_rnd = mean_degree(lo2, hi2, n)
    rnd_vals = []
    for t in range(n_trials):
        O = torch.zeros(n, dtype=torch.bool, device=DEV)
        O[torch.tensor(rng.choice(n, m, replace=False), device=DEV)] = True
        rnd_vals.append(mean_escape(Wr, dr, O))

    return dict(N=n, M=M, m=m, mod=np.mean(mod_vals), rnd=np.mean(rnd_vals),
                ratio=np.mean(rnd_vals) / max(np.mean(mod_vals), 1e-12),
                deg_mod=md_mod, deg_rnd=md_rnd)

print("REGIME (a): M = 8 fixed, m growing")
print(f"{'N':>8} {'m':>6} {'modular':>10} {'random':>10} {'ratio':>8}  {'deg_mod':>8} {'deg_rnd':>8}")
rows_a = []
for m in [8, 16, 32, 64, 128, 256]:
    r = modular_vs_random(M=8, m=m); rows_a.append(r)
    print(f"{r['N']:>8} {r['m']:>6} {r['mod']:>10.4f} {r['rnd']:>10.4f} {r['ratio']:>8.2f}  {r['deg_mod']:>8.2f} {r['deg_rnd']:>8.2f}")

print("\nREGIME (b): m = 8 fixed, M growing")
print(f"{'N':>8} {'M':>6} {'modular':>10} {'random':>10} {'ratio':>8}")
rows_b = []
for M in [8, 16, 32, 64, 128, 256]:
    r = modular_vs_random(M=M, m=8); rows_b.append(r)
    print(f"{r['N']:>8} {r['M']:>6} {r['mod']:>10.4f} {r['rnd']:>10.4f} {r['ratio']:>8.2f}")

print("\nREGIME (c): m = M = sqrt(N)")
print(f"{'N':>8} {'m=M':>6} {'modular':>10} {'random':>10} {'ratio':>8}")
rows_c = []
for s in [8, 12, 16, 24, 32, 48]:
    r = modular_vs_random(M=s, m=s); rows_c.append(r)
    print(f"{r['N']:>8} {r['m']:>6} {r['mod']:>10.4f} {r['rnd']:>10.4f} {r['ratio']:>8.2f}")
"""))

cells.append(code(r"""
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for rows, lab in [(rows_a, "(a) M=8 fixed, m grows"), (rows_b, "(b) m=8 fixed, M grows"), (rows_c, "(c) m=M=sqrt(N)")]:
    ax[0].plot([r['N'] for r in rows], [r['mod'] for r in rows], 'o-', label=lab)
    ax[1].plot([r['N'] for r in rows], [r['ratio'] for r in rows], 'o-', label=lab)
ax[0].set_xscale('log'); ax[0].set_xlabel('N'); ax[0].set_ylabel('modular escape prob'); ax[0].set_title('Modular fragility vs N'); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_xlabel('N'); ax[1].set_ylabel('advantage ratio (random / modular)')
ax[1].set_title('Compartmentalization advantage vs N'); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()
"""))

cells.append(md("""
## 5. Experiment B — optimal module size $m^*$

`paper.md` §5.6 flags this as open. At **fixed** $N$, sweep module size $m$. Two competing costs:

- small $m$ → many modules, many bridges relative to volume → worse containment
- large $m$ → better containment, but one compromised module takes out a larger share of the system

We report both terms plus a combined damage objective
$\\text{damage}(m) = s(m) \\cdot (1 - m/N) + m/N$,
read as: a corrupted module always costs its own $m/N$ share, and additionally leaks into the remaining
$(1-m/N)$ share with gain $s(m)$. This objective is a modelling choice, stated explicitly rather than
assumed — the two raw components are reported separately so a different objective can be applied.
"""))

cells.append(code(r"""
def module_size_sweep(N=4096, deg_in=4, b=1, seed=0, n_trials=5):
    out = []
    for m in [8, 16, 32, 64, 128, 256, 512]:
        if N % m: continue
        M = N // m
        if M < 3: continue
        r = modular_vs_random(M=M, m=m, deg_in=deg_in, b=b, seed=seed, n_trials=n_trials)
        blast = m / N
        damage = r['mod'] * (1 - blast) + blast
        out.append(dict(m=m, M=M, escape=r['mod'], blast=blast, damage=damage))
    return out

sweep = module_size_sweep()
print(f"{'m':>6} {'M':>6} {'escape s(m)':>13} {'blast m/N':>11} {'damage':>10}")
for r in sweep:
    print(f"{r['m']:>6} {r['M']:>6} {r['escape']:>13.4f} {r['blast']:>11.4f} {r['damage']:>10.4f}")
best = min(sweep, key=lambda r: r['damage'])
print(f"\noptimal module size m* = {best['m']}  (M={best['M']}, damage={best['damage']:.4f})")

plt.figure(figsize=(6, 4))
plt.plot([r['m'] for r in sweep], [r['escape'] for r in sweep], 'o-', label='escape prob s(m)')
plt.plot([r['m'] for r in sweep], [r['blast'] for r in sweep], 's-', label='blast radius m/N')
plt.plot([r['m'] for r in sweep], [r['damage'] for r in sweep], '^-', lw=2, label='combined damage')
plt.axvline(best['m'], ls='--', c='k', alpha=.5)
plt.xscale('log', base=2); plt.xlabel('module size m'); plt.ylabel('value'); plt.title('Optimal module size (N=4096)')
plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.show()
"""))

cells.append(md("""
## 6. Experiment C — bridge vulnerability (`paper.md` §5.5)

The open question: modularity buys containment by making bridges sparse, but that makes the bridges a
concentrated attack surface. We compare, at **matched edge-removal budget** $B$:

- **targeted** — remove $B$ bridge edges (attacker knows the topology)
- **random** — remove $B$ edges uniformly at random

and measure two things: whole-graph $\\lambda_2$ (global coordination / consensus speed) and escape
probability (containment). Note the possible self-defeating tension flagged in `paper.md`: cutting bridges
should *help* containment while *hurting* coordination, so a bridge attack may not be unambiguously good
for an attacker.
"""))

cells.append(code(r"""
def bridge_attack(M=16, m=32, deg_in=4, b=2, seed=0, budgets=(0, 2, 4, 8, 16, 32)):
    rng = np.random.default_rng(seed)
    n = M * m
    lo, hi, _ = modular_graph(M, m, deg_in, b, rng)
    isb = bridge_mask(lo, hi, m)
    b_idx = np.where(isb)[0]; nb_idx = np.where(~isb)[0]
    print(f"graph: N={n}, {len(lo)} edges, {isb.sum()} bridges, mean degree {mean_degree(lo,hi,n):.2f}\n")
    print(f"{'budget':>7} | {'TARGETED (bridges)':>28} | {'RANDOM (any edge)':>28}")
    print(f"{'':>7} | {'lambda2':>12} {'escape':>13} | {'lambda2':>12} {'escape':>13}")
    rows = []
    for B in budgets:
        # targeted
        drop = rng.choice(b_idx, size=min(B, len(b_idx)), replace=False) if B else np.array([], int)
        keep = np.setdiff1d(np.arange(len(lo)), drop)
        l2_t = algebraic_connectivity(lo[keep], hi[keep], n)
        Wt, dt = build_sparse(lo[keep], hi[keep], n)
        O = torch.zeros(n, dtype=torch.bool, device=DEV); O[0:m] = True
        es_t = mean_escape(Wt, dt, O)
        # random
        drop_r = rng.choice(len(lo), size=B, replace=False) if B else np.array([], int)
        keep_r = np.setdiff1d(np.arange(len(lo)), drop_r)
        l2_r = algebraic_connectivity(lo[keep_r], hi[keep_r], n)
        Wr_, dr_ = build_sparse(lo[keep_r], hi[keep_r], n)
        es_r = mean_escape(Wr_, dr_, O)
        rows.append(dict(B=B, l2_t=l2_t, es_t=es_t, l2_r=l2_r, es_r=es_r))
        print(f"{B:>7} | {l2_t:>12.5f} {es_t:>13.4f} | {l2_r:>12.5f} {es_r:>13.4f}")
    return rows

rows_atk = bridge_attack()

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
B = [r['B'] for r in rows_atk]
ax[0].plot(B, [r['l2_t'] for r in rows_atk], 'o-', label='targeted (bridges)')
ax[0].plot(B, [r['l2_r'] for r in rows_atk], 's-', label='random edges')
ax[0].set_xlabel('edges removed'); ax[0].set_ylabel(r'$\lambda_2$'); ax[0].set_title('Global coordination capacity'); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(B, [r['es_t'] for r in rows_atk], 'o-', label='targeted (bridges)')
ax[1].plot(B, [r['es_r'] for r in rows_atk], 's-', label='random edges')
ax[1].set_xlabel('edges removed'); ax[1].set_ylabel('escape probability'); ax[1].set_title('Containment (lower = more robust)'); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()
"""))

cells.append(md("""
## 7. Experiment D — clustered vs. scattered corruption at scale

`paper.md` §3.4 found scale does *not* help against uniformly scattered corruption, while §3.5 found it
helps a lot against clustered corruption (escape prob more than halved as $N$ grew $36\\to400$).
Both were measured only up to $N=400$. This extends the comparison by two orders of magnitude.
"""))

cells.append(code(r"""
def clustered_vs_scattered(sides=(8, 16, 32, 64, 128, 200), frac=0.3, seed=0, n_trials=3):
    rng = np.random.default_rng(seed)
    print(f"{'side':>6} {'N':>8} {'scattered':>12} {'clustered':>12}")
    rows = []
    for side in sides:
        n = side * side
        lo, hi, _ = grid_graph(side)
        W, deg = build_sparse(lo, hi, n)
        n_obs = int(frac * n)
        sc, cl = [], []
        for t in range(n_trials):
            O = torch.zeros(n, dtype=torch.bool, device=DEV)
            O[torch.tensor(rng.choice(n, n_obs, replace=False), device=DEV)] = True
            sc.append(mean_escape(W, deg, O))
            ps = int(math.sqrt(n_obs))                      # compact square patch
            r0 = rng.integers(0, max(1, side - ps + 1)); c0 = rng.integers(0, max(1, side - ps + 1))
            idx = np.arange(n).reshape(side, side)[r0:r0+ps, c0:c0+ps].ravel()
            O = torch.zeros(n, dtype=torch.bool, device=DEV)
            O[torch.tensor(idx, device=DEV)] = True
            cl.append(mean_escape(W, deg, O))
        rows.append(dict(N=n, scattered=np.mean(sc), clustered=np.mean(cl)))
        print(f"{side:>6} {n:>8} {np.mean(sc):>12.4f} {np.mean(cl):>12.4f}")
    return rows

rows_cs = clustered_vs_scattered()

plt.figure(figsize=(6, 4))
plt.plot([r['N'] for r in rows_cs], [r['scattered'] for r in rows_cs], 'o-', label='scattered corruption')
plt.plot([r['N'] for r in rows_cs], [r['clustered'] for r in rows_cs], 's-', label='clustered corruption')
plt.xscale('log'); plt.xlabel('N'); plt.ylabel('escape probability (lower = robust)')
plt.title('Threat-model dependence of scaling'); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.show()
"""))

cells.append(md("""
## 8. Summary

Fill this in from the runs above. The specific things to record, and what each would mean:

| Question | `paper.md` status | What this notebook shows |
|---|---|---|
| Modular advantage vs $N$, three regimes | open (§5.2, §5.6) | grows / shrinks / constant, and functional form |
| Optimal module size $m^*$ | open (§5.6) | interior optimum, or monotone (no optimum) |
| Bridge attack vs random attack | open, flagged as most important (§5.5) | budget ratio for equal $\\lambda_2$ damage |
| Is a bridge attack self-defeating? | conjectured in §3.6 | does containment *improve* while coordination degrades |
| Clustered vs scattered scaling | measured to $N=400$ only | whether the asymmetry survives 2 more orders of magnitude |

**Reporting discipline** (matching the rest of this project): predictions that come out wrong get recorded
as wrong, not quietly dropped. Two prior predictions in `paper.md` were falsified this way — "more cue
coverage helps" (false; connectivity/hitting-time governs it) and "bigger domain buffers better" (false
for scattered corruption, true for clustered) — and both corrections were more informative than the
original guesses.
"""))

nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = "/home/user/cellexplore/experiments/gpu_scaling_experiments.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", out)

# validate it parses back cleanly
with open(out, encoding="utf-8") as f:
    parsed = json.load(f)
print("valid JSON, cells:", len(parsed["cells"]))
