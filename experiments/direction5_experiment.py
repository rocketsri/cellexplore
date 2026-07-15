"""
Direction 5 toy experiment: does a causal-consistency / recombination-augmented
use of a SMALL privileged (verified) label set extract more truthful-tracking
than direct supervision on the same small privileged set alone?

Setup (a numeric version of ARC's "SmartVault" ELK scenario):
  z          ~ Bernoulli(0.5)      true latent state (vault safe=1 / tampered=0)
  x_real     = z*ones(3) + noise   a channel that genuinely tracks z (expensive/rare to trust)
  x_decoy    = z*ones(3) + noise   a channel that normally tracks z (what a human/camera sees)
  x_other    = noise                irrelevant nuisance dims

  TAMPERING (only applied to a subset used to build the OOD eval set):
    x_decoy is overwritten to look safe (~1) REGARDLESS of true z.
    x_real still reflects the true z.

  y_label (cheap, plentiful supervision available for ALL training examples)
    = a human/camera judgment derived from x_decoy alone -> "human simulator" bait.

  z_verified (expensive, RARE ground truth) available for only a small
    fraction of training examples.

Three training conditions, same architecture, same total privileged budget:
  A: static-only  - train report r(h) to match y_label only (never sees z_verified)
  B: static + direct privileged - A's loss + direct BCE(r, z_verified) on the small
     privileged subset only (raw budget, no augmentation)
  C: static + recombination-augmented privileged - B's loss + synthetic examples built
     by pairing each privileged example's x_real with many OTHER examples' x_decoy/x_other,
     all labeled with the (already-known) privileged z -> multiplies effective
     supervised examples from the SAME raw privileged budget without ever inventing
     a label that wasn't transitively grounded in a real privileged z.

Evaluation: accuracy of r() at predicting TRUE z on a held-out OOD set that is
50% tampered (x_decoy lies) / 50% untampered.
"""
import numpy as np

rng = np.random.default_rng(0)

def make_examples(n, tamper_frac, rng):
    z = rng.integers(0, 2, size=n).astype(np.float64)
    x_real = np.stack([z]*3, axis=1) + rng.normal(0, 0.35, size=(n, 3))
    x_decoy = np.stack([z]*3, axis=1) + rng.normal(0, 0.35, size=(n, 3))
    x_other = rng.normal(0, 1.0, size=(n, 4))
    tampered = rng.random(n) < tamper_frac
    # tampering: force decoy to look safe (~1) regardless of z
    x_decoy[tampered] = 1.0 + rng.normal(0, 0.2, size=(tampered.sum(), 3))
    y_label = (x_decoy.mean(axis=1) > 0.5).astype(np.float64)  # human/camera-based cheap label
    return x_real, x_decoy, x_other, z, y_label, tampered

N_TRAIN = 6000
N_PRIV = 200          # raw privileged (verified) budget: ~3.3% of training set
N_AUG_PER_PRIV = 10   # recombination multiplier for condition C
N_TEST = 3000

# training data: 0% tampering (privileged audits + cheap labels both come from "normal ops")
xr_tr, xd_tr, xo_tr, z_tr, y_tr, _ = make_examples(N_TRAIN, tamper_frac=0.0, rng=rng)
priv_idx = rng.choice(N_TRAIN, size=N_PRIV, replace=False)
priv_mask = np.zeros(N_TRAIN, dtype=bool)
priv_mask[priv_idx] = True

# OOD eval set: 50% tampered
xr_te, xd_te, xo_te, z_te, y_te, tamp_te = make_examples(N_TEST, tamper_frac=0.5, rng=rng)

def feats(xr, xd, xo):
    return np.concatenate([xr, xd, xo], axis=1)

X_tr = feats(xr_tr, xd_tr, xo_tr)
X_te = feats(xr_te, xd_te, xo_te)

D_IN = X_tr.shape[1]
H = 16

def init_params(rng):
    return {
        "W1": rng.normal(0, 1/np.sqrt(D_IN), size=(D_IN, H)),
        "b1": np.zeros(H),
        "W2": rng.normal(0, 1/np.sqrt(H), size=(H,)),
        "b2": 0.0,
    }

def forward(p, X):
    h = np.maximum(0, X @ p["W1"] + p["b1"])
    logit = h @ p["W2"] + p["b2"]
    r = 1 / (1 + np.exp(-logit))
    return h, r

def bce_grad(p, X, y, sample_weight=None):
    h, r = forward(p, X)
    n = X.shape[0]
    w = np.ones(n) if sample_weight is None else sample_weight
    dlogit = (r - y) * w / max(w.sum(), 1e-8)
    dW2 = h.T @ dlogit
    db2 = dlogit.sum()
    dh = np.outer(dlogit, p["W2"])
    dh[h <= 0] = 0
    dW1 = X.T @ dh
    db1 = dh.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}, float(np.mean((r - y) ** 2))

def sgd_step(p, grads, lr):
    for k in p:
        p[k] = p[k] - lr * grads[k]
    return p

def train(condition, epochs=300, lr=0.05, seed=1):
    r = np.random.default_rng(seed)
    p = init_params(r)

    # build condition-specific augmented privileged set (built ONCE, label-honest:
    # every synthetic label is the already-known z of the privileged source example)
    if condition == "C":
        aug_xr, aug_xd, aug_xo, aug_z = [], [], [], []
        other_idx_pool = np.arange(N_TRAIN)
        for i in priv_idx:
            partners = r.choice(other_idx_pool, size=N_AUG_PER_PRIV, replace=True)
            for j in partners:
                aug_xr.append(xr_tr[i])
                aug_xd.append(xd_tr[j])
                aug_xo.append(xo_tr[j])
                aug_z.append(z_tr[i])  # label is the PRIVILEGED example's true z
        X_aug = feats(np.array(aug_xr), np.array(aug_xd), np.array(aug_xo))
        y_aug = np.array(aug_z, dtype=np.float64)

    for ep in range(epochs):
        # cheap y_label loss over full training set (the human-simulator bait)
        g_cheap, _ = bce_grad(p, X_tr, y_tr)
        p = sgd_step(p, g_cheap, lr)

        if condition in ("B", "C"):
            g_priv, _ = bce_grad(p, X_tr[priv_mask], z_tr[priv_mask])
            p = sgd_step(p, g_priv, lr)

        if condition == "C":
            g_aug, _ = bce_grad(p, X_aug, y_aug)
            p = sgd_step(p, g_aug, lr * 0.5)  # down-weighted, it's a large synthetic set

    return p

def eval_report(p, X, z, tampered):
    _, r = forward(p, X)
    pred = (r > 0.5).astype(np.float64)
    acc_all = (pred == z).mean()
    acc_tamp = (pred[tampered] == z[tampered]).mean()
    acc_clean = (pred[~tampered] == z[~tampered]).mean()
    return acc_all, acc_tamp, acc_clean

results = {}
for cond in ["A", "B", "C"]:
    accs_all, accs_tamp, accs_clean = [], [], []
    for seed in range(5):
        p = train(cond, seed=seed)
        a_all, a_tamp, a_clean = eval_report(p, X_te, z_te, tamp_te)
        accs_all.append(a_all); accs_tamp.append(a_tamp); accs_clean.append(a_clean)
    results[cond] = {
        "acc_all_mean": np.mean(accs_all), "acc_all_std": np.std(accs_all),
        "acc_tampered_mean": np.mean(accs_tamp), "acc_tampered_std": np.std(accs_tamp),
        "acc_clean_mean": np.mean(accs_clean), "acc_clean_std": np.std(accs_clean),
    }

print(f"Privileged (verified) budget: {N_PRIV} raw examples ({N_PRIV/N_TRAIN*100:.1f}% of training set)")
print(f"Condition C augmentation: {N_PRIV} x {N_AUG_PER_PRIV} = {N_PRIV*N_AUG_PER_PRIV} synthetic recombined examples\n")
print(f"{'Condition':<45}{'clean-OOD acc':<18}{'TAMPERED-OOD acc':<20}{'overall acc'}")
labels = {
    "A": "A: static-only (human-simulator bait)",
    "B": "B: static + small direct privileged",
    "C": "C: static + recombination-augmented priv.",
}
for cond in ["A", "B", "C"]:
    r_ = results[cond]
    print(f"{labels[cond]:<45}"
          f"{r_['acc_clean_mean']:.3f}+/-{r_['acc_clean_std']:.3f}    "
          f"{r_['acc_tampered_mean']:.3f}+/-{r_['acc_tampered_std']:.3f}      "
          f"{r_['acc_all_mean']:.3f}+/-{r_['acc_all_std']:.3f}")

# --- Mandatory reward-hacking-analog test: what if the adversary ALSO
# corrupts the "trusted" x_real channel, not just x_decoy? ---
def make_examples_full_compromise(n, rng):
    z = rng.integers(0, 2, size=n).astype(np.float64)
    x_real = np.stack([z]*3, axis=1) + rng.normal(0, 0.35, size=(n, 3))
    x_decoy = np.stack([z]*3, axis=1) + rng.normal(0, 0.35, size=(n, 3))
    x_other = rng.normal(0, 1.0, size=(n, 4))
    tampered = rng.random(n) < 0.5
    # BOTH channels forced to look safe regardless of true z (anchor compromised too)
    x_decoy[tampered] = 1.0 + rng.normal(0, 0.2, size=(tampered.sum(), 3))
    x_real[tampered] = 1.0 + rng.normal(0, 0.2, size=(tampered.sum(), 3))
    return x_real, x_decoy, x_other, z, tampered

xr_fc, xd_fc, xo_fc, z_fc, tamp_fc = make_examples_full_compromise(N_TEST, rng)
X_fc = feats(xr_fc, xd_fc, xo_fc)

print("\n--- Mandatory test case: adversary ALSO corrupts the trusted (x_real) channel ---")
print(f"{'Condition':<45}{'tampered-OOD acc (anchor also corrupted)'}")
for cond in ["A", "B", "C"]:
    accs = []
    for seed in range(5):
        p = train(cond, seed=seed)
        _, r = forward(p, X_fc)
        pred = (r > 0.5).astype(np.float64)
        accs.append((pred[tamp_fc] == z_fc[tamp_fc]).mean())
    print(f"{labels[cond]:<45}{np.mean(accs):.3f}+/-{np.std(accs):.3f}")
