# v9 — Developmental (indirect) encoding: a CPPN toolkit-builder with measurable, audit-surviving generativity

**Task slot:** the "mechanism OUTSIDE the prior families" search, developmental/indirect-encoding
branch. Registry check — this is **not** diffusion/Laplacian-consensus, flow/allocation, Lotka–Volterra,
variational-reducing-to-diffusion, nor RG-as-diagnostic. It is a **compositional pattern-producing
network (CPPN) developing a neural-cellular-automaton codebook** (HyperNEAT genotype → dense body →
run-to-attractor). The generativity is scored information-theoretically under a single fixed decoder, so
it survives the Kolmogorov measure-mismatch audit.

Builds on (does not re-derive): **O2** (necessary shaping `I ≥ log₂K`), **B1** (marginal generativity
`D̄ − log₂K`, realizable iff the shared rule is a target-agnostic codebook, tested by held-out
realizability, charged by `|θ|` once and claiming only the amortized gap), and **v6/v7/v8**: a codebook
*exists* (backprop), per-target *selection* on a frozen codebook is low-`I` (scalar ES ≈ gradient ≫
control), dense-ES *building* of the codebook *fails* (v6: held-out margin −0.04…−0.20), and a **random
linear projection** of the body already rescues building at low dimension (v8: `d=32` → held-out
margin **+0.096**, so the toolkit-shaping objective has **low intrinsic dimension**).

The open gap this file attacks: **can a *developmental* (structured, indirect) encoding build the
one-time codebook with a low-`I`, non-differentiable, per-outcome process — and does its *structure* buy
anything a generic random projection does not?**

---

## 1. The encoding, precisely

**Substrate (unchanged from v5–v8).** `N = 8×8` cells, per-cell state `s ∈ ℝ^{C}` (`C=8`). A **shared
FiLM-modulated neural CA** with parameters `θ` (`|θ| = 1832`) is the local rule: each step perceives via
fixed Sobel/identity filters (`PERC = 3C = 24` channels), a hidden layer `HID=32`, and a per-target code
`z ∈ ℝ^{D}` (`D=12`) enters only as a **FiLM** modulation `h ← γ(z)·h + β(z)` with
`γ = 1 + W_g z`, `β = W_b z`. Growth = 20 steps from a seed; the **attractor** channel-0 field is the
phenotype/target `P`. `θ` partitions into 6 tensors `{W1(24×32), b1(32), W2(32×8), b2(8),
W_g(32×12), W_b(32×12)}`.

**Genotype (the indirect encoding).** A **fixed-topology CPPN** `C_g` — the HyperNEAT weight-painter
(Stanley, D'Ambrosio & Gauci 2009; CPPN: Stanley 2007). Each body parameter is a *weight between two
substrate units*; give every unit a geometric coordinate in `[−1,1]` (perception channel, hidden unit,
output channel, code dimension). `C_g` maps the pair (source-coord `a`, target-coord `b`) plus a tensor
tag to the weight value:

```
weight(a, b, tensor t)  =  gain_t · ( Σ_h Wc2[h,t] · φ_h( Σ_f Wc1[f,h] · feat_f(a,b) ) + bc_t )
feat(a,b) = [1, a, b, a·b, |a−b|, sin(πa), sin(πb), exp(−4(a−b)²), a², b²]          (NF = 10)
φ_h ∈ {tanh, sin, gaussian(x)=e^{−x²}, abs}   assigned by  h mod 4       (the CPPN primitives)
```

The genotype is `g = {Wc1(10×Hc), Wc2(Hc×6), bc(6), gain(6)}`, size `|g| = 16·Hc + 12` params
(`Hc=6 → 108`, `Hc=12 → 204`). **Decoding** `g → θ`: for each of the 6 tensors, evaluate `C_g` on that
tensor's coordinate grid (a single vectorized forward pass) and read its head → the full `1832`-param
body. `|g| = 108 ≪ |θ| = 1832` (≈17× compression). The mixed sin/gauss/abs primitives are exactly the
inductive bias that produces **geometric regularity** — symmetry, repetition, gradients — the structure a
reusable developmental toolkit is made of, and precisely what a random linear subspace lacks.

**Two levels of indirection, matching biology's arrangement.**
`g  ──develop (CPPN)──▶  θ (shared codebook)  ──run NCA with code z──▶  P (target attractor).`
The codebook `θ` is expensive to *describe* (1832 params) but cheap to *generate* (`|g|=108`). The
low-`I` search runs over `g`, not `θ`.

Runnable spec: `experiments/v9_cppn_indirect.py` (pure numpy, deterministic, ~10 min). It is the v8
harness with the *only* change being the decoder `g→θ`, and it runs the CPPN **head-to-head against a
random projection of equal search dimension**.

---

## 2. Measure-consistent generativity + bound (theorem)

Fix **one** universal decoder `U` that contains (i) the CPPN development map and (ii) the NCA run-to-
attractor operator `Φ`. A system realizes targets `{P_m}` via a **shared genotype** `g` (→ codebook
`θ = C_g`) and **per-target codes** `{z_m}`: `P_m = Φ(C_g, z_m)`, all computable under `U`. Let
`D_m = K_U(P_m)` (target's own description length under `U`, estimated conservatively by the strongest
available generic compressor — the *smallest* `D_m`, the audit-surviving choice). Define the **marginal
(amortized) generativity**
`Δ = (1/M) Σ_m [ D_m − |z_m| ]`, with the codebook charged once at `|g|`, not `|θ|`.

> **Theorem (indirect-encoding generativity).**
> **(i) Ceiling.** `Δ ≤ |g| + O(1)`.
> **(ii) Codebook is charged `|g|`, not `|θ|`.** Under `U`, `K_U(θ) ≤ |g| + O(1)` (θ is *generated* from
> g), so the one-time fixed cost is `|g|`. Total generativity over `M` targets is
> `G(M) = Σ_m D_m − ( |g| + Σ_m |z_m| ) = M·Δ − |g| + O(1)`, positive once `M > |g|/Δ`.
> **(iii) No free codebook generativity.** The claim "the codebook is worth `|θ| − |g|`" is **void**:
> since `K_U(θ) ≤ |g|+O(1)`, the CPPN-developed body carries no more information than `g`. Indirect
> encoding manufactures **no** generativity on the codebook side; its entire contribution is to lower the
> fixed cost from `K_U(θ_dense)` to `|g|` and to make building feasible (§3).
> **(iv) Attribution (B1 acid test).** `Δ` is attributable to the codebook only if it survives on
> **held-out** targets never used to shape `g`, realized with `|z_m| ≈ log₂K`. Otherwise the codes are
> smuggling the target and the gap is a measure artifact.

**Proof.** `P_m = U(decoder, g, z_m)` ⟹ `K_U(P_m) ≤ |g| + |z_m| + O(1)`, so `D_m − |z_m| ≤ |g| + O(1)`
for every `m`; averaging gives (i). (ii): `θ = U(develop, g)` ⟹ `K_U(θ) ≤ |g| + O(1)`; substitute into
the total-cost ledger. (iii) is the contrapositive of (ii). (iv) is B1 restated: without held-out
realizability the decomposition `(g, z_m)` is not identified — a lookup table achieves `Δ` spuriously. ∎

**Why this survives the Kolmogorov audit.** Both sides are coded under the same `U`. The target side is
`D_m` (strongest compressor). The shaping side is `|g|` (once) `+ |z_m|` (per target) — *not* the naive
`|θ|·(bits/param)`. The theorem's ceiling `Δ ≤ |g|` is the honest statement that **a compressive genotype
caps amortized generativity at its own size**; you cannot claim more shared structure than `g` contains.

**Numbers (this substrate).** From v6/v7: held-out `D̄ ≈ 400` bits, `|z| = D·(bits/param) ≈ 72` bits
(`D=12` at ~6 bits/param) → target marginal `Δ = 400 − 72 = +328` bits. Ceiling `|g| = 108` params ×
6 bits ≈ **648 bits ≥ 328** — the CPPN genotype is *large enough in bits* to supply the 328-bit marginal
gap, yet *small enough in params* (108) to be ES-feasible. Total generativity `G(M) = 328M − 648`, so
**positive for M ≥ 3 held-out targets**. A dense incompressible codebook would instead carry a fixed cost
`≈ |θ|·b` bits, needing far more targets to amortize — *and* it is infeasible to build (v6).

---

## 3. CRUX — intrinsic dimension & efficiency: does indirect encoding close the gap dense ES could not?

**The claim.** The toolkit-shaping objective `J(θ) = held-out realizability of the codebook θ` has an
**intrinsic dimension** `d_int ≪ |θ|` (Li, Farhi, Risteski & Yosinski 2018). v8 *measures* this directly:
a random affine subspace of the body of dimension `d` solves `J` — and it already does at **`d=32`**
(held-out margin **+0.096** vs the failed dense `1832`-dim ES). So `d_int ≲ 32` for this objective.

**Why dense ES fails but a `d`-dim search succeeds.** Zeroth-order / ES optimization is
**dimension-limited**. The antithetic ES gradient estimate has error `E‖ĝ − ∇J‖² ∝ (n / pop)·‖∇J‖²`
(and Nesterov–Spokoiny 2017 give `O(n/ε²)` oracle calls for `n`-dim smooth problems — **linear in `n`**).
At fixed population `pop=44`, the estimator's signal-to-noise degrades like `1/n`: at `n=1832` the useful
gradient is buried in `≈1800` non-functional directions and the search stalls (v6's negative). Restricting
to an `n=d`-dim coordinate collapses the noise by `|θ|/d` and puts the problem back in the feasible SNR
regime.

**Query-complexity estimate.**
- Dense ES: to reach a fixed SNR needs `pop ∝ |θ|`; total rollouts `∝ |θ|·G`. Empirically at `pop=44`
  it does not just run slow — it **fails** (below the SNR threshold). Matching the compressed SNR would
  need `≈ |θ|/d ≈ 1832/108 ≈ 17×` the population.
- CPPN / compressed ES: `N ≈ pop·G·M ≈ 44·240·5 ≈ 5·10⁴` rollouts at `d_g = 108`, which **succeeds**
  (§4). Cost scales with `d_g`, i.e. with the *desired generativity* (`d_g ≳ Δ/bits-per-param`), not with
  the ambient body size. **Ratio ≈ |θ|/d_g ≈ 17×**, and — decisively — the *absolute* dimension crosses
  from the ES-failure regime (`~1800`) into the ES-feasible regime (`~10²`).

**Structure vs. dimension (the part specific to *developmental* encoding) — and the MEASURED answer.**
A random projection already captures `d_int`, so "compression helps" is really "intrinsic dimension is
low." The stronger *developmental* claim is that the CPPN spans a **structured** `d`-manifold aligned to
the codebook's regularities (symmetry/repetition/gradients from its sin/gauss primitives), so its
**effective** intrinsic dimension is `≤` a random subspace's, and it should beat a random projection at
equal `d` (the HyperNEAT-vs-direct-encoding intuition; Clune, Stanley, Pennock & Ofria 2011). v9 tests
this by pitting CPPN against a random projection **at identical search dimension `d_g`**.

**Measured (`experiments/v9_cppn_results.log`, `v9_crux_results.log`).** The CPPN **does** build a
generalizing codebook by low-`I` scalar ES over the compact genotype: held-out margin over the
memorization control **+0.044 (`d_g=108`)** and **+0.123 (`d_g=204`)**, and a 4-seed mean **+0.078** at
`d_g=108` — feasible where dense-scale search struggled. **But developmental structure does *not* beat a
random subspace of equal size.** The 4-seed STRUCT EDGE (CPPN − RandProj at equal `d_g=108`) is
**−0.093 ± 0.183** (only 1/4 seeds positive); the random projection generalizes *more* often (3/4 vs 2/4)
and with a larger mean margin (**+0.171** vs +0.078). At `d_g=204` the random projection wins outright
(margin +0.326 vs the CPPN's +0.123). **Verdict: F3 — the feasibility win is intrinsic-dimension
reduction, which any low-dim encoding delivers; the CPPN's structural prior adds nothing here and, on
average, slightly hurts.** This is exactly the v8 result (a random subspace at `d=32` already generalizes,
margin +0.096) reproduced with a *developmental* decoder that fails to improve on it. Honest and
deflationary: on this substrate/objective, "developmental" is not doing independent work beyond
compression. The high variance (edge sd 0.18) says single-seed "structure wins" claims (like this run's
own +0.175 at seed 0) are noise.

---

## 4. Held-out realizability — the B1 acid test (protocol)

1. **Fix substrate & targets.** FiLM-NCA body (`|θ|=1832`), `K=16` code cells; `K_train=12` shaping
   targets, `4` **held-out** targets (seeds `200+k`) never seen while shaping `g`.
2. **Build the codebook, low-`I`.** Run **scalar, antithetic, rank-based ES** (non-differentiable,
   per-outcome; the *only* signal is a scalar correlation per rollout) jointly over `[g ; K_train codes]`.
   Decode `θ = C_g` each generation. No gradients through `Φ`.
3. **FREEZE `θ`.** Acid test: for each **held-out** target, search **only** the per-target code `z` by
   scalar ES on the frozen `θ` (never touch `θ`). Measure held-out correlation with `|z| ≈ log₂K`.
4. **Controls.** (a) random frozen body + scalar-ES code (memorization floor); (b) **random projection of
   equal `d_g`** built by the identical ES loop — the crux control that separates *structure* from
   *dimension*.
5. **Read-out.** Report held-out corr, margin over control (a), and **STRUCT EDGE = CPPN − RandProj** at
   equal `d`. Then `Δ = D̄ − |z|` with ceiling `|g|`.

**Confirms measurable generativity iff:** held-out targets are realized on the **frozen** developed body
with small codes (margin over control `> 0.05`), i.e. `θ = C_g` is a genuine **target-agnostic codebook**
built by a low-`I` process. Then `Δ = D̄ − |z| > 0` is real and audit-consistent (codebook charged `|g|`
once).

**Result (measured).** Control (memorization floor) = 0.218. CPPN develops a codebook that passes the
acid test — held-out margins **+0.044 (`d_g=108`)**, **+0.123 (`d_g=204`)**; 4-seed mean **+0.078** at
`d_g=108`. So `Δ = D̄ − |z| > 0` is real and audit-consistent. **But** the equal-`d` random-projection
control passes at least as well (mean margin +0.171; STRUCT EDGE −0.093 ± 0.183), so the generativity is
attributable to *compression/low intrinsic dimension*, **not** to developmental structure. Companion v8:
random projection at `d=32` already generalizes (margin +0.096), and even the full `d=1832` ES in v8's
harness generalizes (+0.241) — locating the earlier "dense fails" (v6) as a coordinate/SNR artifact, not
a fundamental barrier.

---

## 5. What would falsify the claim (honest)

- **F1 — no codebook (generativity void).** Held-out targets *not* realizable on the frozen developed
  body with small codes → `θ` memorized the train set; `Δ` is a lookup artifact. **Kills the claim.**
- **F2 — no efficiency win.** CPPN-ES needs comparable/greater query budget than dense ES to reach a
  held-out-passing codebook → the intrinsic-dimension argument fails for this objective; indirect encoding
  does not close the gap.
- **F3 — dimension, not development. ← THIS IS THE REALIZED OUTCOME.** CPPN ties-or-loses the equal-`d`
  random projection (measured STRUCT EDGE −0.093 ± 0.183, 1/4 seeds positive). The mechanism reduces to
  generic intrinsic-dimension reduction — feasibility is real (both compressed encodings generalize where
  dense-scale search struggled), but the *developmental/CPPN* content is empty on this substrate. Honest,
  deflationary, and the actual result.
- **F4 — ceiling violation / hidden channel.** Observed `Δ > |g| + O(1)`. By the theorem this is
  impossible under honest accounting, so it **flags a leak** — target information entering through
  topology, fitness shaping, or a target-specialized decoder — i.e. mis-counted shaping `I`.
- **F5 — measure mismatch resurfaces.** Small `|z|` achievable only by making the CPPN decoder or `U`
  target-specific → the codebook is not shared; the amortization ledger is broken.

**Bottom line.** Two things stand and one falls.
1. **Theorem (stands, encoding-independent):** under one fixed decoder, amortized generativity
   `Δ ≤ |g| + O(1)`, and an indirect encoding legitimately charges the codebook `|g|` (not `|θ|`) because
   `K_U(θ) ≤ |g| + O(1)` — lowering the fixed cost and making total generativity `G(M)=MΔ−|g|` positive at
   small `M`. It manufactures **no** codebook-side generativity (Thm iii): the CPPN body is compressible to
   `g` by construction.
2. **Feasibility (stands, empirically):** a low-`I`, non-differentiable, per-outcome scalar ES **can**
   build a held-out-realizing codebook when it searches a **compact** genotype — CPPN margins +0.04…+0.12,
   confirming the toolkit-shaping objective has low intrinsic dimension (v8: already at `d=32`).
3. **The developmental claim (falls, F3):** the CPPN's structural bias does **not** beat a random subspace
   of equal dimension (STRUCT EDGE −0.093 ± 0.183). The win is *compression*, not *development*. So the
   honest register-entry is: **"indirect/low-dimensional encoding closes the low-`I` codebook-building gap;
   the CPPN is one valid instance, but its geometric-regularity prior earns no measurable premium over a
   random projection on this substrate."** The equal-dimension random-projection control — not the CPPN —
   is what carries the positive result, and saying so is the point.
