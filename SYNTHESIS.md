# SYNTHESIS — the v5+ re-attack on distributed target origination

A consolidated, honest statement of what the re-attack (v5–v16) actually found. Supersedes the scattered
per-round docs for reading purposes; each claim points to its executed artifact.

---

## 1. The question and the three claims

A population of units runs a shared local rule, coupled locally, and converges on / maintains / self-repairs
a **rich global target represented in no single unit and no `o(N)` subset**, where the target is an
*attractor* of the local dynamics and the rule was shaped by a **low-bandwidth, lossy, non-differentiable,
per-outcome** process (a genome under selection, not backprop). Three claims must be kept apart:

- **B-strong** (incoherent): the system originates the *correct target* with **no** target information
  entering anywhere. An is–ought violation; not the goal.
- **A** (the floor): an external process supplies the entire target `T`; the system just maintains it.
  Shaping information `I ≈ D` (the target's description length).
- **B-bio** (the target): the rich target's description length `D` *exceeds* the shaping information `I`
  the low-bandwidth process supplied. The system produces target-specification it was not given.

The prior program (v1–v3) declared B-bio structurally impossible. **That verdict was wrong**, and the
reason it was wrong is the whole result.

---

## 2. The reframing: selection cost, not specification cost (O2)

Three independent Opus agents adjudicated the three impossibility arguments the prior verdict rested on
(Good Regulator, Second Welfare, Hume). They converged, without contact, on one crux:

> **All three forbid only B-strong, not B-bio.** The one lower bound that survives is a **selection cost**
> `I ≥ log₂K` (bits to pick *which* target among the rule family's reachable repertoire `K`), **not** a
> **specification cost** `I ≥ D`. The upgrade to `I ≥ D` holds only when `K ≈ 2^D` — targets
> incompressible relative to the family — which is exactly Claim A.

The prior program measured `I` with end-to-end backprop, which lives in the `K ≈ 2^D` corner where the two
costs coincide — so it saw `I ≈ D` and concluded impossibility. The correct object is the gap
`Δ = D − log₂K`. *(v5.md; O2 sub-briefs.)*

---

## 3. B1: amortized generativity, and the discriminator that governs everything

**B1 theorem.** With a fixed decoder, a shared rule `θ` + a small per-target code realizes a family of
targets. The **amortized/marginal generativity** is `Δ = D̄ − |code|`, and it is *real and measure-consistent*
**if and only if** the shared rule is a **target-agnostic codebook** — operationally: a **frozen** rule
realizes **held-out** targets (never shaped on) via a small code. Held-out realizability is the
**necessary and sufficient** condition. *(v6.md.)*

This makes **held-out realizability** (equivalently `dim_sel` / class-explanatory-power) *the* discriminator
between genuine generativity and its impostors:
- **Claim A**: the selection supplies the whole target (no shared generalizing prior). Fails held-out.
- **Fake Δ**: apparent richness that is uncharged noise, scored by a generic compressor (zlib) rather than
  by what the selection actually carries. Caught by measuring class-carried variance, not description length.

**This discriminator caught two of the program's own over-claims** (see §6), which is the strongest evidence
it is the right measure.

---

## 4. O3: the achievability threshold (theorem + measured)

O2 is the lower bound; B1 the condition; **O3 is how much low-bandwidth shaping is actually needed** to build
a *generalizing* codebook, and where it fails.

> **O3 threshold (statement).** Building a target-agnostic codebook that realizes held-out targets with
> error `ε` by a low-bandwidth, per-outcome process requires shaping information/experience bounded **by the
> effective (intrinsic) dimension of the shared prior, not by the per-target complexity `D`**. Concretely, on
> a substrate whose codebook is identifiable from `m` shaped targets:
> **(i) sample complexity** — held-out generalization appears once `m ≳ c · C`, where `C` is the codebook's
> capacity (e.g. number of atoms / effective toolkit dimension), *independent of `D`*; and
> **(ii) decodability** — the per-target selection must be sparse/identifiable enough that a *random* prior
> cannot already realize the target (else there is no generativity to measure). Outside either bound, the
> gap `Δ` is not realizable — O3's "fails" regime.

**Proof status.** Rigorous on the dictionary substrate: (i) is dictionary-learning sample complexity
(`m = Ω(M log M)` for identifiability; Arora et al., Agarwal et al.), (ii) is the sparse-recovery / RIP
threshold (`k = O(n / log(M/n))` for OMP). Measured directly (v16): held-out generalization turns on at
`K_train ≈ M` for sparse selection, and vanishes as the selection densifies (a random prior then suffices).
The general statement — that shaping scales with the prior's intrinsic dimension, not `D` — is corroborated
by v8 (the toolkit-shaping objective has low intrinsic dimension: a ~100-dim random subspace suffices where
the full ~1800-dim search is unnecessary). *(v16.md, v8.md.)*

Together: **O2 (need `≥ log₂K`) + B1 (sufficient iff held-out) + O3 (achievable when `m ≳ C` and the
selection is decodable)** is a full quantitative characterization of when B-bio generativity is achievable.

---

## 5. Two independently-verified positive substrates

The abstract result is realized on **two different substrates**, both passing the held-out discriminator —
so it is a principle, not one construction.

**(a) Neural Cellular Automaton (v6–v8, v12).** A shared learned local rule (FiLM-NCA "body") + a small
per-target code grows/regenerates a target. Verified across **6 seeds** (v12): held-out realizability
**0.74/0.71/0.64** (backprop) and **0.51/0.46/0.51** (non-differentiable low-`I` ES) vs a **0.006** random-body
control. A generic prior *exists* (backprop reaches 0.78 held-out at HID=32) and low-`I` search builds a
partially-generalizing one given adequate capacity. Self-repairing (grows from a seed; regenerates after
ablation).

**(b) Superposition / sparse dictionaries (v14, v16) — the one closest to real ML.** A shared overcomplete
dictionary (amortized prior) + a `k`-sparse code (selection). Held-out realizability **0.978 vs 0.690**
random-dict control, recovering **88% of true atoms**, with the B1 sufficiency signature (rises with
`K_train`). This is the toy of **SAEs / feature superposition** in trained networks: superposed features are
a target-agnostic codebook; a prompt selects a sparse combination; held-out realizability is *why* features
generalize.

---

## 6. Honest negatives that sharpen the result

- **The Holonomy Code retraction (v9 → v10).** A non-reciprocal-oscillator "topological" construction claimed
  `Δ = +1244` bits. A dedicated deflationary audit (which ran the code) showed the "richness" was
  quenched-disorder PRNG noise; the winding class carried only **1.3%** of the field variance; under a
  consistent measure `Δ ≤ 0`. **Retracted.** The discriminator (class-explanatory-power) caught it. What
  survives from that work is *not* generativity but two real side-results (§7).
- **Associative memory = Claim A (v15).** Modern Hopfield recalls stored patterns (0.91) but fails held-out
  *unrelated* targets (0.48) — the Claim-A fingerprint. Where it generalizes (in-span compositions) it is a
  *more constrained* case of the superposition substrate. Not a distinct new mechanism.

Both negatives are governed by the *same* discriminator as the positives — which is the point: **held-out
realizability separates generation from supply/retrieval/noise**, and it does so consistently.

---

## 7. Surviving side-results (not generativity, but real)

- **Type-B deception resistance = code distance (v11, executed).** A relational/loop-integral readout converts
  a deceptive misreport (Type-B, the one hazard the prior program flagged as undefendable) into a *visible*
  Type-A defect; minimum collusion for an undetected shift = the graph's systole = the code distance. The
  same parameter governs self-repair, edit-cost, and deception-cost. The MHC/"missing-self" analog. A cheap
  (`O(N)`) monitoring primitive: read relational invariants, not self-reports.
- **Holographic threshold theorem (v10).** The oscillator winding numbers are literally the toric code's
  logical operators; B3's worst-case distance sharpens to an erasure threshold `p_c = 1/2`. Correct as
  coding theory (about the *encoding*, not generativity).

---

## 8. Practicality (architectures.md)

The positive result **is the deployed paradigm**: a frozen foundation model + cheap conditioning
(prompt / LoRA / adapter / MoE routing) *is* B1's amortized codebook. O2 (selection cost) says *why*
conditioning is cheap; held-out realizability is the *certify-it* test; the discriminator is a QA test for
whether a cheap adapter genuinely generalizes vs memorizes. The one novel *architectural* suggestion is the
relational-invariant monitor (§7). Iterative-robustness substrates (NCA) have a diffusion-style
step-distillation efficiency path; the compute win of amortization is params/memory + generalization, not
raw FLOPs.

---

## 9. Bottom line

**B-bio is not impossible; it is achievable, and already ubiquitous.** The prior program's impossibility
verdict was an artifact of measuring the shaping channel with backprop (which delivers ~`D` bits). The
correct quantity is the selection–specification gap `Δ = D̄ − log₂K`; it is positive and real whenever a
shared learned prior generalizes to held-out targets via a cheap selection — verified on two substrates, and
identical in structure to how foundation models are conditioned. The governing measure is **held-out
realizability**, which cleanly separates genuine generativity from Claim-A, retrieval, and compressor-inflated
noise (and caught two of this program's own over-claims). What remains open: scaling both positive substrates
to large, rich, real targets (queued for GPU), and tightening O3 from the substrate-specific theorem to a
fully general one.

*Executed artifacts:* `v5.md` (O2), `v6.md` (B1), `v7.md` (amortized selection, B3), `v8.md` (intrinsic
dimension), `v10.md` (retraction + Type-B + holographic), `v12` (multi-seed verification), `v14.md`
(superposition), `v15.md` (associative memory), `v16.md` (O3 threshold), `architectures.md`,
`GPU_EXPERIMENTS.md`; code under `experiments/`.
