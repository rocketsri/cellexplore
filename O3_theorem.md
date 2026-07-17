# O3 — the general achievability theorem for low-bandwidth codebook generativity

Attempt at the general form of O3 (the achievability threshold), completing O2 (lower bound `I ≥ log₂K`)
and B1 (achievable iff held-out realizability) into a full quantitative characterization. The empirical
versions are v16 (dictionary phase boundary) and v8 (intrinsic dimension); this generalizes them.

**The one-line result.** *Building a target-agnostic codebook by a low-bandwidth, per-outcome process
costs a number of shaped targets that scales with the **prior's** complexity `P`, not with the **target's**
complexity `D` — which is exactly why the generativity gap `Δ = D̄ − log₂K` can be made arbitrarily large
at fixed shaping budget.* It is a meta-learning (learning-to-learn) statement.

---

## 1. Setup and definitions

A **substrate** is a decoder `Φ : Θ × C → X`, where `θ ∈ Θ` is the shared prior ("toolkit" / rule /
dictionary / body), `c ∈ C` is a per-target code, and `X` is target space with a metric `d`. A **target
distribution** `μ` on `X`. A **shaping process** observes `m` i.i.d. targets `t₁,…,t_m ∼ μ` through a
low-bandwidth, per-outcome channel and outputs a prior `θ̂`.

- **Realizability loss** `ℓ(θ, t) := min_{c∈C} d(Φ(θ,c), t)` — the best a prior `θ` can do for target `t`
  via *some* code.
- **Held-out realizability** of `θ`: `L_μ(θ) := E_{t∼μ}[ℓ(θ,t)] ≤ ε`. (B1's necessary+sufficient condition,
  in expectation.)
- `θ̂ := argmin_{θ∈Θ} (1/m) Σᵢ ℓ(θ, tᵢ)` (empirical risk minimizer over shaped targets); `L* := inf_θ L_μ(θ)`.
- **Prior complexity** `P`: a capacity measure of the loss class `F := {t ↦ ℓ(θ,t) : θ∈Θ}` — e.g. its
  pseudo-dimension, or `Θ`'s parameter count when `Φ` is Lipschitz in `θ`, or `log N(Θ, ρ)` (metric entropy).
- **Target complexity** `D̄ := E_{t∼μ}[DL(t)]`; **repertoire** `K := |C|` (so `log₂K` = code length = O2
  selection cost).

---

## 2. The theorem

> **Theorem O3 (sample complexity of a generalizing codebook).**
> Let `F = {t ↦ ℓ(θ,t) : θ∈Θ}` have Rademacher complexity `R_m(F) ≤ O(√(P/m))` (equivalently pseudo-
> dimension `P`, or a `P`-parameter class with `L`-Lipschitz `Φ`). Then with probability `≥ 1−δ` over the
> `m` shaped targets, the empirical minimizer `θ̂` satisfies
> `L_μ(θ̂) ≤ L* + O(√(P/m) + √(log(1/δ)/m))`.
> Hence **`m = Θ̃(P/ε²)`** shaped targets are sufficient (and, for a genuinely `P`-dimensional family,
> necessary up to log factors, by a Fano/packing lower bound) for held-out realizability within `ε` of the
> best achievable. Crucially, **`P` is the complexity of the prior class `Θ`, and the bound is independent
> of the target complexity `D̄` and the repertoire size `K`.**

**Proof sketch.** Sufficiency is agnostic-PAC uniform convergence over the loss class `F`: standard
symmetrization gives `sup_θ |L_μ(θ) − L̂(θ)| ≤ 2R_m(F) + O(√(log(1/δ)/m))` (Bartlett–Mendelson), and
`R_m(F) = O(√(P/m))` for a pseudo-dimension-`P` (or `P`-parameter Lipschitz) class; applying it to `θ̂` and
`argmin_θ L_μ` gives the excess-risk bound. Necessity: identifying a member of a `P`-dimensional family to
accuracy `ε` in a non-degenerate metric requires `Ω(P/ε²)` samples by a standard Fano argument over an
`ε`-packing of `Θ`. `D̄` and `K` never enter: `D̄` is a property of one target (it affects `L*`'s absolute
scale, not the *estimation* rate), and `K = |C|` is optimized *inside* `ℓ` (the `min_c`), so it changes the
per-target code length but not the sample complexity of learning `θ`. ∎

This is exactly the **learning-to-learn / representation-learning** sample complexity (Baxter 2000; Maurer,
Pontil & Romera-Paredes 2016): the cost of learning a shared representation that transfers to new tasks
scales with the *representation's* complexity, amortized over tasks — here, tasks = targets.

---

## 3. The generativity corollary (why B-bio is achievable)

> **Corollary (unbounded generativity at fixed shaping).**
> Amortize the build cost over the targets the codebook serves. Per-target shaping tends to the O2 selection
> cost: `I_per-target → log₂K` as the codebook is reused (it generalizes to held-out targets, so its
> `Θ̃(P/ε²)` build cost is paid once). The achievable marginal generativity is
> `Δ = D̄ − log₂K`. Since **neither `P` nor the build cost `Θ̃(P/ε²)` scales with `D̄`**, `Δ` is **unbounded
> in `D̄` at fixed prior complexity `P`** — provided `P ≪ D̄`.

`P ≪ D̄` is the whole content of B-bio: a **low-complexity prior generating high-complexity targets**,
the decoder `Φ` doing the computational work (development / dynamics / sparse synthesis) that the shaping
loop does *not* pay for. This is the quantitative form of the biological "outsource to physics." Empirically
`P` is small: v8 finds the toolkit-shaping objective has intrinsic dimension `~10²` (a random `100`-dim
subspace suffices) while realized targets are far richer; v16 finds the dictionary needs `m ≳ M log M`
shaped targets (`P ↔ M log M`) regardless of signal dimension.

Contrast: in **Claim A / backprop**, the shaping channel delivers `~D̄` bits per target (per-pixel gradient),
so `P` effectively equals `D̄` (`P ≈ D̄`, no gap) — the corner the prior program measured and mistook for the
general case.

---

## 4. The two side conditions — O3's "fails" regime

The sample-complexity bound gives a good `θ̂` in *expectation*, but two further conditions are necessary for
the generativity to be **realizable and measurable**:

- **(N1) Decodability.** Held-out realizability requires that for a fresh `t`, a code `c` with
  `d(Φ(θ̂,c),t) ≤ ε` is **findable by a tractable search** (poly in `|c|`). Worst-case code inference can be
  intractable (sparse coding is NP-hard in general), so a codebook can *exist* yet be *unusable*. It is
  tractable under structural conditions — RIP/incoherence for dictionaries (Candès–Tao), smooth
  code→output Jacobian for the NCA. This is v16's **sparsity threshold** (`k` below the OMP/RIP limit).
- **(N2) Non-triviality.** The prior must do non-trivial work: a **random** `θ` must *not* already `ε`-cover
  `μ`. If it does (`μ` low-complexity relative to the substrate's default reach), `Δ_effective = 0` and
  there is no generativity to measure. This is v16's **densification** regime (dense selections → a random
  prior suffices → the learned-vs-random margin vanishes) and v14's soft-coding control-match.

Both are exactly the audit-validated **discriminator**: generativity is real only when a *learned* prior
beats a *random* one on *held-out* targets — (N2) is the "beats random," (N1) is the "on held-out via a
findable code."

---

## 5. Instantiation on the two verified substrates

| | prior complexity `P` | side conditions | matches |
|---|---|---|---|
| **Dictionary / superposition** | `P ↔ M log M` (dict identifiability) | (N1) `k` below RIP/OMP limit; (N2) hard `k`-sparse coding (random dict fails) | v16: held-out turns on at `K_train ≈ M`; vanishes when `k` dense |
| **FiLM-NCA** | `P ↔` intrinsic dim of the body (`~10²`, not the `~10³` params) | (N1) smooth code→attractor; (N2) random-body control `≈0` | v8 (intrinsic dim), v12 (held-out `~0.7` ≫ `0.006` control) |

Both realize `Δ = D̄ − log₂K > 0` with `P ≪ D̄`, `m` scaling with `P` not `D̄`, and both side conditions
met — i.e. both sit in the achievable regime the theorem describes.

---

## 6. Relation to O2 and B1, and honest scope

- **O2** (lower bound `I ≥ log₂K`) is the *per-target* selection cost; **O3** is the *one-time build* cost
  `Θ̃(P/ε²)` of the shared prior; **B1** (held-out realizability) is the property `θ̂` must have, which O3
  says is attained once `m ≳ P/ε²` and (N1),(N2) hold. Together: `I_total(for T targets) = Θ̃(P/ε²) + T·log₂K`,
  and per-target `→ log₂K`.
- **Honest scope.** This is a *framework* theorem: rigorous wherever the prior-class complexity `P` and a
  uniform-convergence bound are well-defined (both verified substrates qualify), with the decodability
  condition (N1) the subtlest part (it hides the worst-case hardness of code inference, tractable only under
  structural assumptions). The constants and exact rates are substrate-dependent; the *scaling* claims —
  `m = Θ̃(P/ε²)`, independent of `D̄` and `K`; `Δ` unbounded in `D̄` at fixed `P` — are the general content
  and are corroborated by v8 and v16. It does **not** claim `P` is always small (that is a property of the
  substrate/target-distribution pair); it says B-bio is achievable *exactly when* `P ≪ D̄` with (N1),(N2),
  and quantifies the cost when it is.

**Bottom line.** O3 completes the characterization: the low-bandwidth build cost of a generalizing codebook
is `Θ̃(P/ε²)` — the *prior's* complexity, not the *target's* — so the generativity `Δ = D̄ − log₂K` grows
without bound in `D̄` at fixed shaping whenever a low-complexity prior can generate high-complexity targets
through a decodable, non-trivial selection. That is a theorem-level statement of *why* B-bio is not just
possible but generic, and it is the achievability companion to O2's lower bound.

*Citations (honest):* Baxter, *A Model of Inductive Bias Learning* (2000); Maurer, Pontil, Romera-Paredes,
*The Benefit of Multitask Representation Learning* (2016); Bartlett & Mendelson, Rademacher complexity
(2002); standard Fano/packing lower bounds; Candès & Tao, RIP (2005); Li et al., *Intrinsic Dimension of
Objective Landscapes* (2018).
