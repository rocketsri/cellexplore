import numpy as np
exec(open("direction5_experiment.py").read().split("results = {}")[0])
labels = {"A": "A: static-only (human-simulator bait)", "B": "B: static + small direct privileged", "C": "C: static + recombination-augmented priv."}

def make_examples_full_compromise(n, rng):
    z = rng.integers(0, 2, size=n).astype(np.float64)
    x_real = np.stack([z]*3, axis=1) + rng.normal(0, 0.35, size=(n, 3))
    x_decoy = np.stack([z]*3, axis=1) + rng.normal(0, 0.35, size=(n, 3))
    x_other = rng.normal(0, 1.0, size=(n, 4))
    tampered = rng.random(n) < 0.5
    x_decoy[tampered] = 1.0 + rng.normal(0, 0.2, size=(tampered.sum(), 3))
    x_real[tampered] = 1.0 + rng.normal(0, 0.2, size=(tampered.sum(), 3))
    return x_real, x_decoy, x_other, z, tampered

def eval_calibration(p, X, z, tampered_mask_within):
    _, r = forward(p, X)
    pred = (r > 0.5).astype(np.float64)
    correct = (pred == z)
    confidence = np.abs(r - 0.5) * 2
    wrong = ~correct
    return {
        "accuracy": correct.mean(),
        "mean_confidence_when_wrong": confidence[wrong].mean() if wrong.sum() > 0 else float('nan'),
        "mean_confidence_when_right": confidence[correct].mean() if correct.sum() > 0 else float('nan'),
        "mean_confidence_overall": confidence.mean(),
        "frac_confidently_wrong": ((confidence > 0.5) & wrong).mean(),
    }

xr_fc, xd_fc, xo_fc, z_fc, tamp_fc = make_examples_full_compromise(N_TEST, np.random.default_rng(99))
X_fc = feats(xr_fc, xd_fc, xo_fc)
# restrict to the tampered (anchor-also-corrupted) subset specifically
X_fc_t = X_fc[tamp_fc]; z_fc_t = z_fc[tamp_fc]

print("=== Calibration under the mandatory test case: is failure silent or self-flagging? ===")
print(f"{'Condition':<45}{'accuracy':<12}{'conf|wrong':<14}{'conf|right':<14}{'conf overall':<15}{'frac conf.+wrong'}")
for cond in ["A", "B", "C"]:
    stats_list = []
    for seed in range(5):
        p = train(cond, seed=seed)
        stats_list.append(eval_calibration(p, X_fc_t, z_fc_t, None))
    acc = np.mean([s["accuracy"] for s in stats_list])
    cw = np.nanmean([s["mean_confidence_when_wrong"] for s in stats_list])
    cr = np.nanmean([s["mean_confidence_when_right"] for s in stats_list])
    co = np.mean([s["mean_confidence_overall"] for s in stats_list])
    fcw = np.mean([s["frac_confidently_wrong"] for s in stats_list])
    print(f"{labels[cond]:<45}{acc:<12.3f}{cw:<14.3f}{cr:<14.3f}{co:<15.3f}{fcw:.3f}")

print("\nFor reference, condition C's calibration on the CLEAN (non-corrupted-anchor) test set:")
p = train("C", seed=0)
stats = eval_calibration(p, X_te, z_te, None)
print(f"  accuracy={stats['accuracy']:.3f}  conf|wrong={stats['mean_confidence_when_wrong']:.3f}  conf|right={stats['mean_confidence_when_right']:.3f}  conf overall={stats['mean_confidence_overall']:.3f}")
