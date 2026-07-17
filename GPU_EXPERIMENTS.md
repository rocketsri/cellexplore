# GPU / Colab experiments to run (collected)

These are the runs that would be **materially sped up by a GPU** (large backprop training). CPU-efficient
work is done inline in the repo; these are queued for you. Each says what it tests and what a positive
result looks like. Paste the printed summary / JSON back and I'll fold the numbers into the write-ups.

## 1. Scale the amortized NCA codebook to real, rich images  ⟵ highest priority
`experiments/gpu_scale_codebook.py` (ready, CPU-smoke-tested). Tests whether the program's #1 positive
result (held-out realizability of a shared learned prior + cheap code) survives at realistic size on
**genuinely rich targets** — the honest limitation the audit flagged (everything so far is 8×8 blobs).

```bash
# Runtime → GPU (T4 ok)
git clone -b claude/plan-implementation-7rdcg7 https://github.com/rocketsri/cellexplore
cd cellexplore/experiments && pip -q install torchvision
python gpu_scale_codebook.py --targets mnist --grid 28 --hid 128 --channels 16 --ktrain 8,32,128 --iters 2500
python gpu_scale_codebook.py --targets cifar --grid 32 --hid 128 --channels 16 --ktrain 8,32,128
python gpu_scale_codebook.py --targets grf  --grid 32 --hid 128 --channels 16
```
**Look for:** held-out realizability ≫ random-body control on real images; whether it rises with
K_train; the scalar-ES (low-`I`) arm ≈ backprop; the amortized-inference **encoder zero-shot** > control
(the prompting-like regime). Saves `gpu_scale_results.json`.

## 2. Superposition/dictionary at SAE scale (optional, GPU helps with large M + data)
The CPU version (`experiments/v14_superposition_codebook.py`) already gives the clean result
(held-out 0.98 vs 0.69 control, 88% atom recovery). A GPU version would test it at **SAE scale** —
real image patches (or LLM activations), `M` in the thousands, `k`~10–30 — via a torch SAE
(tied ReLU encoder + dictionary, L1). *If you want this, tell me and I'll write the torch script;* the
CPU result already establishes the mechanism, so this is confirmation-at-scale, not load-bearing.

## 3. The most practically-relevant test: held-out realizability on a real architecture (design, not yet scripted)
The B1 claim predicts: a **frozen small transformer + a cheap per-task selector** (soft prompt / LoRA)
should realize **held-out tasks** via a small code, with held-out realizability ≫ a random-backbone
control. This is the program's principle on the substrate industry actually uses. It needs a GPU (train
a small transformer, then freeze + fit soft-prompts to held-out tasks). *If you want it, I'll write a
self-contained script* (e.g., a tiny character-transformer over a family of synthetic sequence tasks,
frozen backbone + soft-prompt per task, held-out task generalization vs control). Highest external
relevance; medium effort to script.

---
**Note on interpretation:** the audit-validated discriminator is **held-out realizability vs a
random-prior control** (equivalently `dim_sel` / class-explanatory-power). For every run, the load-bearing
comparison is *learned prior beats a random prior on held-out targets* — not raw fidelity, and not any
zlib-based Δ (which the audit voided).
