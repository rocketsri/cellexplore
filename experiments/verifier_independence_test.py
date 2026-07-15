"""
directions1.md found Direction 4 (multiple verifiers) fatal absent a
measured independence/correlation estimate -- "different team, different
data" is vague. PROPOSAL: operationalize independence as escape
probability (Theorem 5, paper.md) between two verifiers' computational
modules through however much they share, and test whether that predicts
correlated failure rate.

EXPLICIT SCOPE, stated up front: this tests whether escape probability
predicts correlated blind spots arising from SHARED TRAINING LINEAGE
(the failure mode directions1.md's cross-cutting synthesis flagged). It
does NOT test, and is not claimed to address, Obfuscated Gradients'
actual finding -- an adversary with query access to both verifiers
running a JOINT optimization against them simultaneously. Those are
different failure modes; conflating them would be exactly the kind of
overclaim this investigation has been trying not to make.
"""
import numpy as np
rng = np.random.default_rng(0)

MU, KAPPA = 1.0, 3.0

# Model: verifier A's "private" module (pA nodes), verifier B's "private" module
# (pB nodes), and a SHARED component (s nodes) both route through, with coupling
# strength to the shared component controlled by alpha (0=fully independent,
# 1=fully shared/coupled).
pA, pB, shared = 6, 6, 6

def build_graph(alpha):
    n = pA + pB + shared
    W = np.zeros((n, n))
    # dense internal coupling within each private module
    for block in [range(0, pA), range(pA, pA+pB), range(pA+pB, n)]:
        idx = list(block)
        for i in idx:
            for j in idx:
                if i < j:
                    W[i, j] = W[j, i] = 1.0
    # coupling from each private module to the shared component, strength = alpha
    shared_idx = range(pA+pB, n)
    for i in range(0, pA):
        for j in shared_idx:
            W[i, j] = W[j, i] = alpha
    for i in range(pA, pA+pB):
        for j in shared_idx:
            W[i, j] = W[j, i] = alpha
    L = np.diag(W.sum(1)) - W
    return L, W

def escape_prob_between_modules(alpha):
    # escape probability FROM verifier-A's private module INTO verifier-B's
    # private module (treated as the "boundary" O), routed only through the
    # shared component -- this is the exact Theorem 5 computation.
    L, W = build_graph(alpha)
    n = pA + pB + shared
    O_idx = np.array(range(pA, pA+pB))          # verifier B's private module = boundary
    F_idx = np.array([i for i in range(n) if i not in O_idx])  # A's module + shared
    LFF = L[np.ix_(F_idx, F_idx)]
    A_mat = MU*np.eye(len(F_idx)) + KAPPA*LFF
    G = np.linalg.solve(A_mat, KAPPA*W[np.ix_(F_idx, O_idx)])
    s = G.sum(axis=1)
    # average escape probability specifically for nodes in verifier A's private module
    A_local_idx = [F_idx.tolist().index(i) for i in range(pA)]
    return s[A_local_idx].mean()

print("=== Escape probability between verifier modules, as a function of shared coupling alpha ===")
alphas = [0.0, 0.1, 0.3, 0.6, 1.0]
escape_by_alpha = {}
for alpha in alphas:
    ep = escape_prob_between_modules(alpha)
    escape_by_alpha[alpha] = ep
    print(f"  alpha={alpha:.1f}  escape probability (A-module -> B-module) = {ep:.4f}")

# --- Now the empirical test: train two verifiers whose feature extraction shares
# a "confound" pathway to degree alpha, and measure their CORRELATED failure rate
# on an OOD test set where a spurious feature decouples from the true label. ---
print("\n=== Empirical test: does higher alpha (higher escape probability) predict more CORRELATED failures? ===")

N_TRAIN, N_TEST, D_PRIVATE, D_SHARED = 4000, 2000, 4, 4

def make_data(n, tamper_frac, rng):
    z = rng.integers(0, 2, size=n).astype(np.float64)
    shared_feat = np.stack([z]*D_SHARED, axis=1) + rng.normal(0, 0.4, size=(n, D_SHARED))
    privateA = np.stack([z]*D_PRIVATE, axis=1) + rng.normal(0, 0.5, size=(n, D_PRIVATE))
    privateB = np.stack([z]*D_PRIVATE, axis=1) + rng.normal(0, 0.5, size=(n, D_PRIVATE))
    tampered = rng.random(n) < tamper_frac
    # the SHARED feature is the one that gets spuriously decoupled from z (the confound)
    shared_feat[tampered] = 1.0 + rng.normal(0, 0.2, size=(tampered.sum(), D_SHARED))
    return shared_feat, privateA, privateB, z, tampered

def softmax_bce_train(X, y, epochs=250, lr=0.15, seed=0):
    r = np.random.default_rng(seed)
    D = X.shape[1]; H = 12
    W1 = r.normal(0, 1/np.sqrt(D), (D, H)); b1 = np.zeros(H)
    W2 = r.normal(0, 1/np.sqrt(H), (H,)); b2 = 0.0
    n = len(y)
    for ep in range(epochs):
        idx = r.permutation(n)
        for b0 in range(0, n, 256):
            bidx = idx[b0:b0+256]
            h = np.maximum(0, X[bidx]@W1+b1)
            logit = h@W2+b2
            p = 1/(1+np.exp(-logit))
            dlogit = (p - y[bidx]) / len(bidx)
            dW2 = h.T@dlogit; db2=dlogit.sum()
            dh = np.outer(dlogit, W2); dh[h<=0]=0
            dW1 = X[bidx].T@dh; db1=dh.sum(0)
            W1-=lr*dW1; b1-=lr*db1; W2-=lr*dW2; b2-=lr*db2
    return (W1,b1,W2,b2)

def predict(params, X):
    W1,b1,W2,b2 = params
    h = np.maximum(0, X@W1+b1); return 1/(1+np.exp(-(h@W2+b2)))

rng_data = np.random.default_rng(11)
shared_tr, pA_tr, pB_tr, z_tr, _ = make_data(N_TRAIN, 0.0, rng_data)
shared_te, pA_te, pB_te, z_te, tamp_te = make_data(N_TEST, 0.5, rng_data)

for alpha in alphas:
    # verifier features = private features + alpha * shared (confound) features concatenated,
    # scaled so alpha=0 means the verifier ignores the shared/confound pathway entirely,
    # alpha=1 means it has full access to (and, since it's spuriously predictive, will lean on) it
    XA_tr = np.concatenate([pA_tr, alpha*shared_tr], axis=1)
    XB_tr = np.concatenate([pB_tr, alpha*shared_tr], axis=1)
    XA_te = np.concatenate([pA_te, alpha*shared_te], axis=1)
    XB_te = np.concatenate([pB_te, alpha*shared_te], axis=1)

    pA_params = softmax_bce_train(XA_tr, z_tr, seed=1)
    pB_params = softmax_bce_train(XB_tr, z_tr, seed=2)

    predA = (predict(pA_params, XA_te) > 0.5).astype(float)
    predB = (predict(pB_params, XB_te) > 0.5).astype(float)

    wrongA = (predA[tamp_te] != z_te[tamp_te])
    wrongB = (predB[tamp_te] != z_te[tamp_te])
    # correlated failure rate: P(both wrong | at least one wrong) -- the quantity that
    # actually matters for "does a second verifier catch what the first missed"
    both_wrong = (wrongA & wrongB).sum()
    either_wrong = (wrongA | wrongB).sum()
    corr_failure_rate = both_wrong / max(either_wrong, 1)
    ep = escape_by_alpha[alpha]
    print(f"  alpha={alpha:.1f}  escape_prob={ep:.4f}  accA={1-wrongA.mean():.3f}  accB={1-wrongB.mean():.3f}  "
          f"P(both wrong | either wrong)={corr_failure_rate:.3f}")
