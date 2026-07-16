# Practicality & compute: are any of these feasible architectures?

A cross-cutting analysis prompted by a fair challenge: the program has been about *whether* B-bio
generativity is possible/measurable; this asks whether any of it is a **deployable architecture**, at
what **compute**, and — as transformers went from a clever idea to a feasible one through many
efficiency improvements (flash-attention, sparse/linear attention, MoE, quantization, distillation) —
what the **efficiency path** is for the parts that look expensive now.

## 1. The honest headline

**The program's one surviving positive result is already the dominant deployed paradigm.** B1's amortized
codebook — *a shared learned prior + a cheap per-target selector, valid iff the prior generalizes to
held-out targets* — is exactly **a frozen foundation model + cheap conditioning** (prompt / soft-prompt /
prefix / LoRA / adapter / MoE routing). The program did not invent this architecture; it supplies the
**theory of when and why it works**, and falsifiable **discriminators** for when cheap conditioning
genuinely generalizes versus fakes it. That theory is compute-practical precisely because its
instantiation is what industry already runs.

| program result | deployed architecture it describes | per-target compute |
|---|---|---|
| B1 amortized codebook (shared prior + code) | frozen backbone + prompt / LoRA / adapter / soft-prompt | tiny (tokens or rank-`r` params) |
| O2 selection cost `log₂K`, not specification cost `D` | *why* conditioning is cheap: you pay to **select** among the backbone's reachable behaviours, not to **specify** one | `log₂K` bits |
| held-out realizability (B1 necessary+sufficient) | in-context / zero-shot generalization to new tasks via cheap conditioning | — |
| class-explanatory-power / `dim_sel` discriminator (v10) | test of whether cheap conditioning is **real generalization** vs Claim-A memorization/retrieval | — |
| MoE selection = pick 1 of `K` experts | Mixture-of-Experts routing (Mixtral, etc.) | top-`k` of `E`, `log E` bits |

## 2. Measured compute amortization (NCA substrate)

Analytic FLOP estimate for the HID=32, `d=12`, `8×8`, 24-step FiLM-NCA (`experiments/v13_compute_amortization.py`
measures wall-clock to confirm): forward `≈ 3.2×10⁶` FLOPs/rollout; **build** the shared toolkit
(backprop, 800 iters × 24 targets) `≈ 1.8×10¹¹` FLOPs, **one-time**; **select** a held-out target
(fit a `d=12` code, ~400 iters on the frozen body) `≈ 3.8×10⁹` FLOPs, **recurring**.

**The honest accounting — where the win is and is not.** Per *iteration*, fitting a 12-D code and training
a fresh body cost ~the same FLOPs (both backprop through the identical 24-step rollout). So the amortized
design's advantage is **not** raw FLOPs-per-iter. It is three other things, all real:
1. **Params / memory per target: `1832 → 12` (`~150×`).** You store a 12-number code per target, not a
   model — the LoRA/soft-prompt economics.
2. **Iterations to converge:** a 12-D selector search converges in far fewer steps than training an
   1832-D body from scratch, so *effective* per-target compute is lower (the selector is low-D).
3. **No re-learning the prior + generalization:** the frozen backbone is reused and generalizes to
   held-out targets (B1), so most tasks need *no* training of the expensive part.

**The prompting-like limit needs one more step (amortized inference).** True "zero per-task compute"
(prompt in, output out, no optimization) requires an **encoder** that maps a target to its code in a
single forward pass, rather than optimizing the code per target. That is a concrete, standard efficiency
step (amortized/feedforward inference of the selector) — and it is exactly what turns "LoRA-like cheap
fine-tune" into "prompting-like zero-shot." The program's substrate demonstrates the LoRA-like regime; the
prompting-like regime is one amortized-encoder away.

## 3. Compute per substrate, and the efficiency path

**(a) Frozen backbone + cheap conditioning — already feasible, deployed.**
Backbone forward is `O(L·d² + L²·d)` (or `O(L·d)` with linear/sparse attention), amortized across all
tasks; per-task cost is prompt tokens (`O(K·d)`) or LoRA rank (`O(r·d·layers)`) — negligible beside the
backbone. This is the practical form of the entire program's positive result. **No efficiency work is
owed; the efficiency is inherited from the backbone.** The program's value here is diagnostic, not
architectural: O2/B1 say *how many bits* a task-selector needs (selection, not specification), and
held-out realizability says *whether* a given backbone is a genuine codebook for the task family.

**(b) NCA / iterative-refinement substrate — expensive now, known efficiency path.**
NCA is `O(steps · N · rule)` recurrent; the toy runs here are `8×8`, `24` steps. As a general architecture
this is costly (the step count is the tax), and it is *not* competitive with a feedforward transformer
for most tasks. What it uniquely gives is **self-repair / robustness as a dynamical property**. The
efficiency path is the same one diffusion models walked:
- **step distillation** (diffusion went `1000 → 1–4` steps; consistency models): distil the many-step
  growth into few steps;
- **deep-equilibrium / implicit models**: solve directly for the attractor fixed point instead of
  iterating (`O(1)` "steps" via a root-finder + implicit-function-theorem gradients);
- **channel/step sparsity, learned step size, early-stopping on converged cells.**
So "iterative refinement for robustness" is feasible at a step-cost that is reducible by the same tricks
that made diffusion deployable — the recurrence is not a fundamental barrier.

**(c) Holonomy / oscillator substrate — not a compute engine, but a cheap monitoring layer.**
Simulating analog oscillator dynamics digitally is `O(steps · E)` and buys nothing as a *generator* (the
audit showed it is Claim-A or noise). **But the surviving Type-B result is architecturally cheap and
deployable**: computing a **relational / global invariant** (a loop integral, a conserved quantity) over
`N` units is `O(N)`, and the theorem says such an invariant is **unfakeable by self-report up to the code
distance** (min-collusion = systole). As a *monitoring / oversight* layer on any backbone — read a
relational invariant of the population's states rather than trusting per-unit self-reports — this is
practical, cheap, and directly targets deceptive misreport (H3). The compute cost is a single `O(N)`
reduction per check.

## 4. The transformer-efficiency analogy, applied honestly

The user's analogy is apt but cuts a specific way:
- The **amortized-selection principle** needs no efficiency rescue — it is *already* the efficient thing
  (prompting/LoRA/MoE). The program explains it, it doesn't have to make it feasible.
- The **iterative-robustness substrate** (NCA/diffusion-like) has a *demonstrated* efficiency path (step
  distillation, equilibrium solving) — the diffusion community already walked exactly this from 1000 to
  a few steps.
- The **oscillator substrate** does not become an efficient generator with more engineering (the audit
  is a *content* problem, not a compute one) — but its *monitoring* use is already cheap.

So there is no case here of "a beautiful-but-infeasible architecture awaiting its flash-attention." The
practical architecture (frozen prior + cheap selector) is here and efficient; the robustness substrate
has a known acceleration path; the oscillator is a monitoring primitive, not an engine.

## 5. What a practitioner should actually take

1. **The theory is the deliverable, and it is architecture-agnostic.** "Adapt a big frozen model to a new
   task with a small selector, and certify it with **held-out realizability**" — O2/B1 quantify the bit
   cost (selection, `log₂K`) and give the necessary+sufficient generalization condition. This applies to
   prompts, LoRA, adapters, MoE routers, soft prompts — any frozen-prior + cheap-selector design.
2. **The generativity discriminator is a QA test for conditioning.** `dim_sel` / class-explanatory-power /
   held-out realizability separates *genuine* cheap-conditioning generalization from Claim-A retrieval and
   from fake (over-counted) richness. Use it to check that a cheap adapter is really generalizing, not
   memorizing — the same test that caught two of this program's own over-claims.
3. **For robustness, use iterative-refinement backbones and distil the steps.** Self-repair is a property
   of attractor dynamics; you get it from diffusion/equilibrium-style models, and you pay for it at a
   step-cost that distillation reduces.
4. **For deception-resistant oversight, read relational/global invariants, not self-reports.** Cheap
   (`O(N)`), and unfakeable up to the code distance — the one genuinely novel *architectural* suggestion
   the program produced, and it is compute-trivial.

## 6. Honest limits of this analysis

- All experiments are toy-scale (`≤ 256` units, CPU); the amortization *ratio* and *crossover* are
  measured, but absolute FLOPs are not a large-model claim. The scaling argument to real systems rests on
  the mapping to prompting/LoRA/MoE, which are independently known to scale — not on scaling the NCA
  itself.
- "Already deployed" is a double edge: much of the *practical* content restates known paradigms. The
  program's genuine additions are the **theoretical discriminators** (selection vs specification cost;
  held-out realizability as necessary+sufficient; class-explanatory-power) and the **relational-invariant
  monitoring** primitive — not a new high-performance architecture.
