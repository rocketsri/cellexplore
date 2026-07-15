import numpy as np
exec(open("v2_symmetry_cost_experiment.py").read().split("print(\"Training Model A")[0])

KAPPA = 2.0

# Model B', trained with CONSISTENT causal-restricted settling: at every
# position t, x_1..t is settled using ONLY the graph over {1,...,t} -- exactly
# matching what will be available at generation time. No future leakage.
def initB_mlp(rng):
    return {"W1": rng.normal(0, 1/np.sqrt(D), size=(D, H)),
            "b1": np.zeros(H),
            "W2": rng.normal(0, 1/np.sqrt(H), size=(H, V)),
            "b2": np.zeros(V)}

def readout_mlp(p, x):
    h = np.maximum(0, x @ p["W1"] + p["b1"])
    return h, h @ p["W2"] + p["b2"]

def train_B_causal_consistent(seqs, epochs=40, lr=0.1, seed=0):
    r = np.random.default_rng(seed)
    p = initB_mlp(r)
    all_x, all_y = [], []
    for s in seqs:
        for t in range(2, L_SEQ):
            prefix = s[:t]                          # only positions 0..t-1 "exist"
            targets = embed(prefix)
            x_star = settle_x(t, targets)            # causal-restricted settle, matches generation
            all_x.append(x_star[-1])                 # most recent node's settled state
            all_y.append(s[t])
    X = np.array(all_x); Y = np.array(all_y)
    n = len(Y)
    for ep in range(epochs):
        idx = r.permutation(n)
        for b0 in range(0, n, 256):
            bidx = idx[b0:b0+256]
            h, logits = readout_mlp(p, X[bidx])
            probs = softmax(logits)
            onehot = np.eye(V)[Y[bidx]]
            dlogits = (probs - onehot) / len(bidx)
            dW2 = h.T @ dlogits; db2 = dlogits.sum(0)
            dh = dlogits @ p["W2"].T; dh[h <= 0] = 0
            dW1 = X[bidx].T @ dh; db1 = dh.sum(0)
            for k, g in zip(["W1","b1","W2","b2"], [dW1,db1,dW2,db2]):
                p[k] -= lr * g
    return p

def eval_causal_consistent(p, seqs, use_own_predictions):
    correct = total = 0
    for s in seqs:
        gen = list(s[:2])
        for t in range(2, L_SEQ):
            src = gen if use_own_predictions else list(s[:t])  # teacher-forced ground truth prefix, or own generations
            targets = embed(np.array(src))
            x_star = settle_x(len(src), targets)
            _, logits = readout_mlp(p, x_star[-1][None,:])
            pred = int(logits.argmax(-1)[0])
            correct += (pred == s[t]); total += 1
            if use_own_predictions:
                gen.append(pred)
    return correct/total

print("Training Model B' with CONSISTENT causal-restricted settling (train == generation procedure)...")
pBc = train_B_causal_consistent(train_seqs)
tf_Bc = eval_causal_consistent(pBc, test_seqs, use_own_predictions=False)  # ground-truth prefix each step ("teacher-forced")
ar_Bc = eval_causal_consistent(pBc, test_seqs, use_own_predictions=True)   # true autoregressive generation

print(f"\n{'Model':<50}{'teacher-forced acc':<22}{'true autoregressive acc':<25}{'gap'}")
print(f"{'A: causal/asymmetric MLP (reference)':<50}0.936{'':<17}0.688{'':<20}+0.247")
print(f"{'B: symmetric, full-sequence training (mismatch)':<50}0.688{'':<17}0.222{'':<20}+0.466")
print(f"{'B prime: symmetric, causal-consistent training':<50}{tf_Bc:.3f}{'':<17}{ar_Bc:.3f}{'':<20}{tf_Bc-ar_Bc:+.3f}")
