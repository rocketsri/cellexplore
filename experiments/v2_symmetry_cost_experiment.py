"""
Does v2's Onsager-symmetric coupling requirement (w_ij = w_ji, hence bidirectional
influence on every existing edge) impose a real capability cost relative to
standard causal/asymmetric attention, specifically for autoregressive generation?

Task: order-2 Markov chain over a small vocabulary -- token_t is a fixed
(near-deterministic) function of (token_{t-1}, token_{t-2}). Genuinely solvable
from PAST context alone; no oracle need to see the future.

Model A (causal/asymmetric, the standard-transformer analog):
  predicts token_t from [embed(token_{t-1}), embed(token_{t-2})] only, both at
  train and generation time -- never sees anything not yet generated.

Model B (v2-faithful, symmetric energy coupling over a chain graph):
  x_t has its own local drive U_t(x_t) = (mu/2)||x_t - embed(token_t)||^2 and is
  symmetrically coupled to x_{t-1} AND x_{t+1} (kappa term), exactly v2's energy.
  Closed-form settling: x* = (mu*I + kappa*L)^{-1} mu * target, L = graph Laplacian
  of whatever positions currently EXIST in the graph.
    - Train time (teacher forcing): the WHOLE sequence exists as nodes, so x_t's
      settled value is influenced by x_{t+1} = embed(token_{t+1}) -- which is
      LITERALLY the label being predicted. A readout r(x_t) -> predict token_{t+1}
      can shortcut through this direct coupling rather than learning genuine
      past-context inference.
    - Generation time: only positions 1..t exist (t+1 hasn't been generated yet),
      so x_t must settle using ONLY the causal-prefix-restricted graph -- the
      future-coupling shortcut is structurally unavailable.

Prediction: Model A shows consistent accuracy in both regimes (no shortcut ever
available). Model B shows artificially high teacher-forced accuracy (shortcut)
and a real drop in true autoregressive generation accuracy (shortcut gone).
"""
import numpy as np

rng = np.random.default_rng(0)
V = 8        # vocab size
D = 6        # embedding dim
L_SEQ = 14   # sequence length
N_SEQ = 1500 # number of training sequences
N_TEST = 400

# Fixed, mostly-deterministic order-2 transition table: f(a,b) -> next token
transition_table = rng.integers(0, V, size=(V, V))
noise_prob = 0.05  # small stochasticity so it's not trivially memorizable by a lookup alone

def gen_sequence(rng):
    seq = list(rng.integers(0, V, size=2))
    for _ in range(L_SEQ - 2):
        nxt = transition_table[seq[-2], seq[-1]]
        if rng.random() < noise_prob:
            nxt = rng.integers(0, V)
        seq.append(int(nxt))
    return np.array(seq)

train_seqs = np.array([gen_sequence(rng) for _ in range(N_SEQ)])
test_seqs = np.array([gen_sequence(rng) for _ in range(N_TEST)])

embed_table = rng.normal(0, 1/np.sqrt(D), size=(V, D))

def embed(tok_ids):
    return embed_table[tok_ids]

# ---------- Model A: causal, order-2 context MLP ----------
H = 24
def initA(rng):
    return {
        "W1": rng.normal(0, 1/np.sqrt(2*D), size=(2*D, H)),
        "b1": np.zeros(H),
        "W2": rng.normal(0, 1/np.sqrt(H), size=(H, V)),
        "b2": np.zeros(V),
    }

def forwardA(p, ctx):  # ctx: (n,2D)
    h = np.maximum(0, ctx @ p["W1"] + p["b1"])
    logits = h @ p["W2"] + p["b2"]
    return h, logits

def softmax(logits):
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def train_A(seqs, epochs=40, lr=0.1, seed=0):
    r = np.random.default_rng(seed)
    p = initA(r)
    ctxs, labels = [], []
    for s in seqs:
        for t in range(2, L_SEQ):
            ctxs.append(np.concatenate([embed_table[s[t-1]], embed_table[s[t-2]]]))
            labels.append(s[t])
    X = np.array(ctxs); Y = np.array(labels)
    n = len(Y)
    for ep in range(epochs):
        idx = r.permutation(n)
        for b0 in range(0, n, 256):
            bidx = idx[b0:b0+256]
            h, logits = forwardA(p, X[bidx])
            probs = softmax(logits)
            onehot = np.eye(V)[Y[bidx]]
            dlogits = (probs - onehot) / len(bidx)
            dW2 = h.T @ dlogits; db2 = dlogits.sum(0)
            dh = dlogits @ p["W2"].T; dh[h <= 0] = 0
            dW1 = X[bidx].T @ dh; db1 = dh.sum(0)
            for k, g in zip(["W1","b1","W2","b2"], [dW1,db1,dW2,db2]):
                p[k] -= lr * g
    return p

def eval_A_teacher_forced(p, seqs):
    correct = total = 0
    for s in seqs:
        for t in range(2, L_SEQ):
            ctx = np.concatenate([embed_table[s[t-1]], embed_table[s[t-2]]])[None,:]
            _, logits = forwardA(p, ctx)
            pred = logits.argmax(-1)[0]
            correct += (pred == s[t]); total += 1
    return correct/total

def eval_A_autoregressive(p, seqs):
    correct = total = 0
    for s in seqs:
        gen = list(s[:2])
        for t in range(2, L_SEQ):
            ctx = np.concatenate([embed_table[gen[-1]], embed_table[gen[-2]]])[None,:]
            _, logits = forwardA(p, ctx)
            pred = int(logits.argmax(-1)[0])
            correct += (pred == s[t]); total += 1
            gen.append(pred)  # feed own prediction forward, genuine generation
    return correct/total

# ---------- Model B: v2-faithful symmetric energy coupling on a chain ----------
MU = 1.0
KAPPA = 2.0

def chain_laplacian(n):
    L = np.zeros((n, n))
    for i in range(n-1):
        L[i, i] += 1; L[i+1, i+1] += 1
        L[i, i+1] -= 1; L[i+1, i] -= 1
    return L

def settle_x(n, targets_emb):  # targets_emb: (n, D); returns x*: (n, D)
    L = chain_laplacian(n)
    A = MU * np.eye(n) + KAPPA * L
    return np.linalg.solve(A, MU * targets_emb)

def initB(rng):
    return {
        "Wr": rng.normal(0, 1/np.sqrt(D), size=(D, V)),
        "br": np.zeros(V),
    }

def readout(p, x):
    return x @ p["Wr"] + p["br"]

def train_B(seqs, epochs=40, lr=0.1, seed=0):
    r = np.random.default_rng(seed)
    p = initB(r)
    # precompute, for every training sequence, the FULL bidirectional settle
    # (train-time regime: whole sequence exists, x_t is influenced by x_{t+1})
    all_x, all_y = [], []
    for s in seqs:
        targets = embed(s)                      # (L_SEQ, D)
        x_star = settle_x(L_SEQ, targets)        # full bidirectional settle
        for t in range(2, L_SEQ):
            all_x.append(x_star[t-1])            # "current" node's settled state
            all_y.append(s[t])                   # predict the NEXT token
    X = np.array(all_x); Y = np.array(all_y)
    n = len(Y)
    for ep in range(epochs):
        idx = r.permutation(n)
        for b0 in range(0, n, 256):
            bidx = idx[b0:b0+256]
            logits = readout(p, X[bidx])
            probs = softmax(logits)
            onehot = np.eye(V)[Y[bidx]]
            dlogits = (probs - onehot) / len(bidx)
            dWr = X[bidx].T @ dlogits; dbr = dlogits.sum(0)
            p["Wr"] -= lr * dWr; p["br"] -= lr * dbr
    return p

def eval_B_teacher_forced(p, seqs):
    # standard eval regime: full ground-truth sequence given, bidirectional settle allowed
    correct = total = 0
    for s in seqs:
        targets = embed(s)
        x_star = settle_x(L_SEQ, targets)
        for t in range(2, L_SEQ):
            logits = readout(p, x_star[t-1][None,:])
            pred = logits.argmax(-1)[0]
            correct += (pred == s[t]); total += 1
    return correct/total

def eval_B_autoregressive(p, seqs):
    # true generation: only the CAUSAL PREFIX exists at each step -- settle
    # restricted to positions that have actually been generated so far.
    correct = total = 0
    for s in seqs:
        gen = list(s[:2])
        for t in range(2, L_SEQ):
            prefix_len = len(gen)                       # positions 0..prefix_len-1 exist
            targets = embed(np.array(gen))
            x_star = settle_x(prefix_len, targets)       # coupling restricted to existing prefix
            logits = readout(p, x_star[-1][None,:])       # last (most recent) node
            pred = int(logits.argmax(-1)[0])
            correct += (pred == s[t]); total += 1
            gen.append(pred)
    return correct/total

print("Training Model A (causal/asymmetric, order-2 context)...")
pA = train_A(train_seqs)
tf_A = eval_A_teacher_forced(pA, test_seqs)
ar_A = eval_A_autoregressive(pA, test_seqs)

print("Training Model B (v2-faithful symmetric energy coupling)...")
pB = train_B(train_seqs)
tf_B = eval_B_teacher_forced(pB, test_seqs)
ar_B = eval_B_autoregressive(pB, test_seqs)

print(f"\n{'Model':<45}{'teacher-forced acc':<22}{'true autoregressive acc':<25}{'gap'}")
print(f"{'A: causal/asymmetric (GPT-style analog)':<45}{tf_A:.3f}{'':<17}{ar_A:.3f}{'':<20}{tf_A-ar_A:+.3f}")
print(f"{'B: symmetric energy coupling (v2-faithful)':<45}{tf_B:.3f}{'':<17}{ar_B:.3f}{'':<20}{tf_B-ar_B:+.3f}")
