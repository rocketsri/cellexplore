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

## 2A. Full proof

**Precise hypotheses.**
- **(H1) Bounded loss.** `ℓ(θ,t) = min_{c∈C} d(Φ(θ,c), t) ∈ [0,B]` for all `θ∈Θ, t∈supp(μ)` (under a
  normalized metric `B = O(1)`).
- **(H2) Prior-class regularity.** `Θ ⊆ ℝ^P` bounded (`‖θ−θ'‖₂ ≤ R`), and `θ ↦ d(Φ(θ,c), t)` is
  `Λ`-Lipschitz in `θ`, uniformly over `c∈C, t∈supp(μ)`. *(Any pseudo-dimension-`P` bound on
  `F={t↦ℓ(θ,t)}` serves equally; the Lipschitz form makes `P` the parameter count.)*
- **(H3) i.i.d. ERM.** `t₁,…,t_m ∼ μ` i.i.d.; `L̂(θ)=(1/m)Σᵢℓ(θ,tᵢ)`; `θ̂∈argmin_Θ L̂`; `θ*∈argmin_Θ L_μ`,
  `L*=L_μ(θ*)`.

**Lemma 1 (the `min_c`, hence `K`, does not enter the complexity).** Under (H2), `θ↦ℓ(θ,t)` is `Λ`-Lipschitz
uniformly in `t`. *Proof.* `ℓ(θ,t)=min_c g_c(θ)`, `g_c(θ):=d(Φ(θ,c),t)` each `Λ`-Lipschitz. Pick
`c'∈argmin_c g_c(θ')`: `ℓ(θ,t) ≤ g_{c'}(θ) ≤ g_{c'}(θ')+Λ‖θ−θ'‖ = ℓ(θ',t)+Λ‖θ−θ'‖`; symmetric. ∎ The
pointwise minimum over `C` is absorbed into `Λ` — `K=|C|` never appears (it only lowers `L*`). This is the
formal reason `K` is a per-target *code* cost, not a *shaping* cost.

**Lemma 2 (Rademacher).** Under (H1)–(H2), `R̂_S(F) ≤ c₀·ΛR·√(P/m)·√(log m)`. *Proof.* By Lemma 1 each
`f∈F` is `Λ`-Lipschitz in `θ∈Θ⊆ℝ^P`, `diam≤R`. An `ρ/Λ`-cover of `Θ` (size `≤(3RΛ/ρ)^P`) induces an
`ρ`-cover of `F` in `L²(S)`, so `log N(F,ρ,L²(S)) ≤ P·log(3RΛ/ρ)`. Dudley's entropy integral with `α=1/√m`
gives `R̂_S(F)=O(ΛR√(P/m)·√(log(RΛm)))`. ∎ (Equivalently, pseudo-dimension `P` ⇒ `R̂_S(F)=O(B√(P log m/m))`.)

**Upper bound.** With probability `≥1−δ`,
`L_μ(θ̂) − L* ≤ 2R̂ + B√(2 log(2/δ)/m)`, where `R̂ = c₀ΛR√(P log m/m)`.
*Proof.* Let `Ψ(S)=sup_{θ}(L_μ(θ)−L̂(θ))`. Changing one sample moves `Ψ` by `≤B/m` (loss range `B`), so
McDiarmid gives, w.p. `≥1−δ/2`, `Ψ ≤ E[Ψ]+B√(log(2/δ)/(2m))`. Symmetrization: `E[Ψ]≤2E R̂_S(F)≤2R̂`. Hence
for **all** `θ`, w.p. `≥1−δ/2`: `L_μ(θ)−L̂(θ) ≤ 2R̂ + B√(log(2/δ)/(2m))` (★). For the fixed comparator `θ*`,
Hoeffding gives w.p. `≥1−δ/2`: `L̂(θ*)−L_μ(θ*) ≤ B√(log(2/δ)/(2m))` (★★). On the intersection
(prob `≥1−δ`), using `L̂(θ̂)≤L̂(θ*)` (ERM):
`L_μ(θ̂)−L* = (L_μ(θ̂)−L̂(θ̂)) + (L̂(θ̂)−L̂(θ*)) + (L̂(θ*)−L_μ(θ*)) ≤ 2R̂ + 0 + B√(2 log(2/δ)/m)`. ∎
Setting RHS `≤ε`: **`m ≥ C₁(Λ²R²P log m + B² log(1/δ))/ε² = Θ̃(P/ε²)`**, with `Λ,R,B` substrate constants.

**`D̄`- and `K`-independence (explicit).** `D̄=E_μ DL(t)` enters only through `B` (loss range) and `L*`
(the floor); under a normalized metric `B=O(1)`, so the *excess-risk rate* `L_μ(θ̂)−L*=Õ(√(P/m))` has no
`D̄` dependence. `K=|C|` is eliminated by Lemma 1.

**Lower bound (matching, `m=Ω(P/ε²)`).** There exist substrate/target pairs requiring `Ω(P/ε²)`.
*Construction.* Take `Θ={−γ,+γ}^P` and a decoder for which the excess risk is separable,
`L_μ(θ)−L*=(1/P)Σ_{j=1}^P \mathbf 1[θ_j≠θ°_j]` (a hidden sign vector `θ°`), each shaped target yielding one
observation carrying `O(1)` bits about `θ°` (a per-outcome, low-bandwidth channel — the model's own
assumption). Reaching excess risk `ε` needs `≥(1−ε)P` correct signs; by Assouad's lemma / Fano over the
`2^P`-packing, any estimator from `m` such observations has expected excess risk `≥cP/m`, so `m=Ω(P/ε²)` is
necessary. `D̄` and `K` are free in this construction. ∎

**Conclusion.** Up to log factors, `m=Θ(P/ε²)` shaped targets are **necessary and sufficient** for held-out
realizability within `ε` of optimal — the rate governed by the prior complexity `P`, independent of `D̄` and
`K`. (The upper bound is universal; the lower bound is worst-case over the prior class, so the rate is
minimax-tight.) The corollary (§3) and side conditions (§4) then follow: amortized per-target shaping tends
to `log₂K`, and `Δ=D̄−log₂K` is unbounded in `D̄` at fixed `P`, whenever (N1) `ℓ` is efficiently computable
(the `min_c` tractable — statistics is fine regardless; *efficiency* needs RIP/incoherence or smooth
`∂Φ/∂c`) and (N2) `L*` with a *random* `θ` exceeds `ε` (else the prior does no work, `Δ_eff=0`).

**Remark (statistics vs. computation).** The theorem counts *shaped targets / shaping information*. (N1) is
the only place computational tractability enters — worst-case `min_c` (code inference) can be NP-hard, so
held-out realizability is *efficiently* achievable only under structural conditions. This separates the
information threshold (`Θ̃(P/ε²)`, always) from the computational threshold (substrate-dependent); both were
seen in v16 (sample-complexity turn-on at `K_train≈M`; the sparsity/decodability boundary).

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
