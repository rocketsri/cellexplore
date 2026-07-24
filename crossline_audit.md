# Cross-line audit — reconciling the v1–v3/paper line with the v5–v17 line

Two lines in this repo attacked the same question and reached apparently conflicting verdicts. This is an
adversarial adjudication of both, with claims checked against executed code and logs rather than prose.
Corrections have been applied to both sides' documents where verified.

## Verdict

**The v5–v17 line partially corrects one document of the v1–v3 line and leaves its central conclusion
standing.** It also contains one retraction-level bug of its own, independently confirmed here.

## 1. The correction to this line is real, but aimed at the wrong document

`SYNTHESIS.md` states: "The prior program (v1–v3) declared B-bio structurally impossible. That verdict was
wrong." This is a **misattribution** with respect to `v1.md`, which said the opposite:

> "Whether that disanalogy is closeable at all (e.g., via genuinely blind, non-differentiable,
> population-scale search over coupling topologies) is the honest open question this investigation leaves on
> the table, **not a claim that has been resolved either way**." — `v1.md` §8

`discussion1.md` §1 and §3 go further, already naming the bandwidth axis ("a matter of degree, not kind,
along four measurable dials… a testable threshold question rather than a binary property") and the
physics-does-the-work mechanism ("outsourcing pattern complexity to self-organizing physics/chemistry rather
than encoding it"). That is B-bio, anticipated.

**But the criticism lands on `discussion1.md` §7**, which overstated to "not achievable by any architecture"
and "blocked for structural, not engineering, reasons." That is inconsistent with `v1.md` §8, and
`directions1.md` and `v3.md` both cite it as settled. **Corrected in place**; `v1.md` §8 is now the canonical
statement.

Note what §7's Claim B actually required, though: that no *target-aware supplier* exist. B-bio still requires
$m = \tilde\Theta(P/\varepsilon^2)$ correct exemplar targets from exactly such a supplier — a point the other
line concedes in its own Hume adjudication ("target-designation is external in both"). So B-bio does not
satisfy §7's Claim B even after the correction; the other line re-cast Claim B into an information quantity
($I$ vs $D$) that §7 never used, then refuted the recast.

## 2. Good Regulator does bite on B-bio — as an achievability theorem

The other line's reading of Conant–Ashby (a fixed-point characterization, silent on build cost, not forcing
localization) is technically right, and its refutation of "the model must be localized" attacks a claim
`v1.md` §3.5 never made — v1 explicitly granted the model is real but distributed.

Where GR genuinely bites: **recurse it to the codebook builder.** The shaping process is itself a regulator
whose regulated system is the target distribution $\mu$; its output $\theta$ must model $\mu$, and requisite
variety floors $\theta$'s realized variety by $\mu$'s outcome-relevant variety. That floor is exactly the
prior complexity $P$, and its sample cost is exactly the other line's $m=\tilde\Theta(P/\varepsilon^2)$.
**`O3_theorem.md` is the Good Regulator theorem applied one level up.** Requisite variety was never evaded;
it was paid in $P$ — and it buys nothing outside $\mu$, which that line's own `v17_results.log` confirms
(off-distribution margin over a random prior: **+0.004**, and **−0.030** on dense targets).

## 3. Does B-bio bear on alignment? It *increases* difficulty

The other line concedes its positive result "**is** the deployed paradigm" — frozen foundation model plus
cheap conditioning — and is "already ubiquitous." That concession is self-undercutting as a rebuttal: every
prompted foundation model is a B-bio system, and alignment is not solved for them. Proving the deployed
paradigm is *possible* says nothing about whether it is *aligned*.

Four reasons difficulty rises, three supported by that line's own data:

1. **Error amplification is the same quantity as generativity.** $\Delta = \bar D - \log_2 K$ means the
   decoder amplifies the code by $2^\Delta$ in target-space volume — and amplification is not selective. A
   wrong bit in a 53-bit code is amplified into the 192-bit target exactly as a right one is. Under Claim A
   specification errors stay local; under B-bio they are amplified by construction.
2. **Verification does not compress.** Specifying costs $\log_2 K$; *checking* the realized target still costs
   $\sim\bar D$. B-bio therefore widens the specification/verification asymmetry — you can cheaply request
   what you cannot cheaply check. Neither line noted this, and it is the direction that is bad for alignment.
3. **Cheapness is conditional on $\mu$, and alignment targets live off $\mu$** (`v17`: in-distribution margin
   +0.289, off-distribution +0.004). Novel-situation alignment is exactly the regime where the cheap channel
   buys nothing and you are back to paying $\tilde\Theta(P/\varepsilon^2)$ — i.e. retraining.
4. **Cheap conditioning is uncorrectable conditioning**, if Theorem 7's duality holds on codebook substrates
   (see §5).

**Answer:** orthogonal to achievability, adverse on difficulty.

## 4. Confirmed defects in the other line's empirical claims

**(a) The low-$I$ dictionary arm is invalid — seed collision, independently verified here.**
`experiments/v14_superposition_codebook.py` line 152 builds the ground-truth dictionary as
`D0 = unit_cols(np.random.default_rng(1).standard_normal((n, M0)))`; line 182 calls
`learn_dict_es(Xtr, M, gens=150, seed=1)`, which at lines 120–122 does
`rng = np.random.default_rng(1); Phi0 = unit_cols(rng.standard_normal((n, M)))` — same seed, same shape.
Verified: **byte-identical, max abs diff 0.0.** The "non-differentiable low-$I$ shaper" *starts at* the true
dictionary (atom recovery 1.000) and degrades it. Re-run with clean seeds, held-out corr is 0.694–0.700
against a 0.690 control with **0% atom recovery** — it fails that line's own discriminator. `v14.md`'s low-$I$
claim should be retracted.

**(b) The headline margin is regime-dependent.** The 0.978-vs-0.690 figure is one cell ($k{=}5$, $K{=}600$)
of a table whose margin runs +0.37 at $k{=}3$ down to **+0.05 at $k{=}12$ — below that line's own +0.10 pass
threshold** (`v16_results.log`). Also, 0.690 for a *random* dictionary is legitimate but scale-compressing:
in $R^2$ terms the comparison is 0.956 vs 0.476.

**(c) The discriminator is necessary, not sufficient.** At $K{=}200$ held-out realizability is 0.853 (passing)
with **10.4% atom recovery** — it certifies a dictionary that recovered almost none of the true structure.
And `v12_results.log` prints its own contrary verdict: the sufficiency signature "held-out rises with
$K_{train}$" is "**not seen**" (0.741 → 0.714 → 0.642, falling).

**(d) The Type-B result does not defeat Type-B.** In the Kuramoto ring each unit's entire state *is* its
report, so the substrate cannot express a hidden causally-divergent state — the defining feature of Type-B.
Further: the claimed branch-cut detector **raised 0 flags**; detection came from a separate pairwise
`std(edge diffs)` monitor, not the loop integral. The adversary is non-adaptive (two hand-written candidate
ramps, no optimization against the monitor). The substrate is noiseless (honest edge diffs $\pm0.000$), so
the monitor tolerance — the security parameter — is free; with real heterogeneity it must exceed the honest
noise floor, and that tradeoff is unmeasured. Finally "min collusion = systole = code distance" was not
attained: the log gives min-$k$ = **12 of 24** at the operative tolerance and 23 at the tightest swept.

What survives is real but narrower: *a relational ($H^1$) readout has higher misreport-distance than a
self-reported scalar against a non-adaptive liar on a noiseless substrate.* That is worth keeping. It is not
Type-B resistance, and it does not test this line's `bridge_analysis.md` P14 (no bridge topology, no hidden
state) — P14 remains an untested conjecture.

## 5. Genuine cross-connections neither line had

**(a) $b$ is a physical realization of $\log_2 K$.** In the modular architecture, the code the rest of the
system can impose on a module is carried by its $b$ bridges, so module code length is $O(b)$ and module target
complexity is $O(m)$ — giving $\Delta_{\text{module}} \propto m/b$. **This line's containment ratio and that
line's generativity gap are the same ratio in this architecture.** That converts `paper.md` §3.6b's three
measured scaling regimes into predictions about *generativity* scaling that the other line has no way to
derive — notably that "add more modules" buys **zero** asymptotic gain.

**(b) $M$ and $m/b$ protect against different threats.** Measured here: the wrap systole of the modular ring
(= minimum colluders for an undetected topological class flip) grows roughly as $2.6M$ — 9, 21, 45, 167 for
$M = 4,8,16,64$ — while growing $m$ at fixed $M$ moves it only 21→32. So module count $M$, which `paper.md`
§3.6b showed buys **nothing** for containment, buys collusion resistance **linearly**. Two orthogonal knobs
for two orthogonal threats; this line had only measured one because Type-B resistance had no metric until the
other line's code-distance framing supplied one. *(Caveat: one data point failed to close a ring — the $b{=}1$
builder can drop a bridge in dedupe, exactly the fragility the other line's bridge analysis flagged.)*

**(c) Held-out realizability is the wrong certificate for Direction 5; atom recovery is the right one.**
`v3.md` Direction 5 was downgraded because its recombination trick presupposes separable channels and real
representations are superposed. A sparse dictionary *does* supply that separability ($x = \Phi s$ splices at
the atom level). But splicing requires the recovered atoms to be the **true causal** ones, and reconstruction
quality does not certify that — the other line's own data dissociates them ($K{=}200$: held-out 0.853,
atom recovery 0.104). So Direction 5 is revivable on the dictionary substrate, **gated on atom recovery, not
on held-out realizability** — which is simultaneously a real limitation on that line's discriminator.

**(d) Verifier independence needs both channels.** `reorientations.md` §2's escape-probability criterion is
computable from architecture alone and tracks correlated failure well, but can only see confounds routed
through *shared architectural components* — two verifiers with disjoint architecture trained on the same data
would score escape 0 and still fail together. The other line's `v17` OOD-margin protocol (is A's failure set
realizable by B's prior?) sees exactly that distributional channel. **Complementary; neither alone covers
both.**

## 6. Highest-value next direction

**Measure the generativity–correctability duality on the NCA codebook** — i.e. test whether Theorem 7's
$s_i + q_i = 1$ governs a substrate that is *simultaneously* a codebook and a relaxation.

Protocol: for each frozen FiLM-NCA body from `v12` (6 seeds × $K_{train}\in\{6,12,24\}$, spanning held-out
0.43–0.84, plus the 0.006 random-body floor), measure (i) $s$ = attractor displacement per bit of code change,
(ii) $q$ = displacement per equivalent-magnitude change to the body $\theta$. Prediction: normalized,
$s+q=1$, so as generativity rises, code authority per bit must fall. Falsifiable both ways.

Why this beats continuing either line alone:

- **Only the composite substrate can host the question.** This line has the exact identity but a pure Claim-A
  substrate with no codebook. That line has the codebook but no correctability metric — and its *dictionary*
  substrate provably cannot host the duality (measured flat, §5 of `paper.md`). The NCA is the unique point in
  either program that is both a target-agnostic codebook **and** a self-repairing attractor.
- **It is already half-corroborated from two independent formalisms** (potential theory here; code distance in
  `v7.md` §B3(c)) with no contact between them.
- **It converts §3 into a number.** If the duality holds, "the cheaper the conditioning, the lower that
  channel's correction authority" becomes a *conservation law*: prompt-steerability and weight-level
  correctability are exactly zero-sum in frozen-model deployment; a model's prompt-injection resistance is the
  same quantity as its resistance to legitimate prompt-level correction; and correcting a highly-generative
  prior must go through the $\tilde\Theta(P/\varepsilon^2)$ channel, not the $\log_2 K$ one. That is an
  alignment-load-bearing claim about the deployed paradigm, derivable from neither line alone.
- **It is cheap.** The v12 bodies and the Theorem 7 verification code both exist; the addition is two
  Jacobian-norm measurements.

## 7. Not verified

`v12_results.log` reflects a 2182-second run not re-executed here (its internal consistency — 0.006 control,
six seeds, and a printed self-critical "not seen" verdict — makes it credible). The GPU artifacts in both
lines are queued, not executed. Whether an adversary optimizing directly against v11's `std(edge diffs)`
monitor can flip the class with $k \ll 12$ is the obvious next test; it was established only that the other
line did not run it.
