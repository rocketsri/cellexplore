# Discussion 1 — from the v1 negative result to a concrete architecture proposal

This is a synthesis of the conversation that followed the v1 investigation (`v1.md`). v1 established a convergent negative/partial result: every one of nine independent formal routes for "cellular-style" alignment relocated the global target into an outer loop rather than eliminating it. This discussion pushes on that finding, tests it against real biology, and ends at a concrete architecture proposal (`v2.md`).

## 1. Is "relocation, not elimination" just an artifact of comparing against backprop?

Objection: individual neurons in a standard NN don't know the global loss either — each weight's update is a locally-stored quantity (∂L/∂w_i). Isn't this the same kind of locality the biological routes were going for?

Resolution: no — locality of *storage* is not locality of *computation*. Backprop's per-weight gradient is the output of a globally-orchestrated, exact, lossless, synchronized sweep (a centrally-evaluated loss, full forward-pass memory, exact chain-rule propagation). None of the nine routes' coupling channels (gap junctions, gossip, price signals) have any of those four properties. The target is fully and exactly present in the loss computation graph — just spread across edges rather than sitting in one node. This is *more* complete access than any biological analog, not comparable locality.

Reframed as a matter of degree, not kind, along four measurable dials: **bandwidth** of feedback, **exactness**, **centralization of the update computation**, and **granularity** (per-step vs. per-generation). Evolution and backprop sit at opposite extremes of the same dials — this makes "does the target get relocated" a testable threshold question rather than a binary property.

## 2. Can training and inference-time coupling be unified into one architecture?

Real precedent exists: **equilibrium propagation** (Scellier & Bengio), predictive coding, forward-forward — these use the *same* local settling dynamics for both inference and learning, with no separate backward pass. Structurally close to the bioelectric/mean-field/immune-network routes' math, just not previously proposed as the *learning* mechanism, only the *inference-time* one.

Separately, a genuinely blind outer loop (evolving the local rule / topology via population-based search with a coarse, lossy fitness signal, not gradient descent on hyperparameters) would close the access-asymmetry gap at the design level — expensive, unproven at scale, but a real, different research direction from anything in v1.

## 3. Does biological evolution refute the negative result? (No — it confirms it, with real mechanism behind both halves.)

**How local rules produced specialization without global access:** (a) an almost incomprehensible trial budget (billions of organisms × billions of generations); (b) reuse of a deep, conserved genetic toolkit (Hox genes, ~600My) rather than re-deriving specialization from scratch; (c) outsourcing pattern complexity to self-organizing physics/chemistry (reaction-diffusion, mechanical folding) rather than encoding it; (d) denser effective selection than "one bit per generation," since failed development is continuously filtered before birth.

**But evolution did not solve alignment with its own implicit objective (inclusive fitness).** Modern humans show real goal misgeneralization specifically where a technology severs a proximate reward from the ultimate objective faster than generational selection can respond (contraception decoupling sex from reproduction; hyperpalatable food decoupling taste from caloric scarcity). And within a single organism, cell-level coordination with the organism-level target is not free — it requires continuous, costly policing (immune surveillance, apoptosis), and still fails at meaningful, nonzero rates (cancer — the literal biological instance of the "loses contact inhibition" test case from v1). The same mechanism that resists adversarial defection (cancer) also mechanically resists legitimate correction, with nothing in the biology distinguishing the two cases.

**Correction from user:** not all local-goal pursuit is proxy-capture. Adaptive plasticity (e.g., r/K life-history shifts under different mortality regimes) is an organism robustly re-solving for the *same* true objective via a different strategy — a genuinely positive instance of goal-tracking under environmental perturbation, cleaner than anything the nine formal routes achieved. This sharpens the finding into a precise condition: **alignment under a local-coupling mechanism holds as long as the rate at which the environment can decouple a local proxy from the true objective stays slower than the outer loop's correction rate, and breaks exactly when that ordering reverses.** This is the same timescale-separation condition several routes (VSM, active inference) already needed as an explicit stability assumption.

## 4. Level-of-analysis correction

The original premise (v1) is specifically about **Level 1**: cells within one organism, one developmental lifetime, coupled by bioelectric/biochemical signaling. Evolutionary-psychology arguments are **Level 2** (organisms across generations, differential reproduction). "Division of labor once baseline needs are met" (civilizational specialization) is **Level 3** (individuals within a society, coordinated by institutions/culture/markets) — which turns out to be a re-instantiation of v1's Route 7 (market price discovery) at a different scale, inheriting the same relocation finding: division of labor works because the coordinating infrastructure was itself built by a process with target access. Using evidence from one level to argue about another without an explicit bridging argument is exactly the "scale mismatch" failure mode v1's skeptic agent was built to catch — flagged and corrected mid-conversation.

Also flagged as an open tension, not resolved: whether human specialization away from base survival/reproductive goals is actually evidence of "success," given sub-replacement fertility in developed societies is arguably itself an instance of proxy decoupling whose long-run multi-level-selection consequences haven't played out yet.

## 5. Semblance in the original premise

The synthesis that emerged (credit-aware local learning + deliberate redundancy + independent verification) was not stated in the original premise, but its two problem-halves were: Requirement 1 explicitly demanded a local rule with "its own local drive" that "responds to the coupling signal" — exactly the tension that turned out to require something like equilibrium propagation to resolve cleanly (several v1 routes hit this same circularity independently: bioelectricity's local proxy, consensus's Type A/B split, active inference's precision dynamics). Requirement 4's mandatory "loses contact inhibition" test case is exactly the scenario whose precise treatment, across routes, kept requiring an independent verification/audit channel. The premise licensed departing from pure biology explicitly ("agents should follow [a more productive substrate], even if it departs entirely from biology"), which sanctions this direction.

## 6. Is a verification layer alone sufficient?

No. It needs its own reference, supplied from outside — this doesn't eliminate the relocation problem, it adds a second, ideally independently-sourced instance of it (a real redundancy benefit, not an elimination). It risks being a centralized controller in decentralized costume if not itself genuinely distributed. And it is fundamentally reactive/defensive — it can catch known-shape defection but is structurally weakest against genuinely novel failure modes it was never built to check for (the "coverage gap," same shape as antigenic escape in the immune-network route).

## 7. Claim A vs. Claim B — the load-bearing distinction

**Claim A (achievable):** given an externally supplied reference T, an architecture provably converges to and maintains behavior within ε of T, robust to a stated class of perturbation and defection.

**Claim B (not achievable by any architecture):** the system converges on the *correct* target using only internal dynamics, without T ever having been supplied by a process that already had access to what "aligned" means.

Claim B is blocked for structural, not engineering, reasons — independently reached by three separate formalisms in v1 (VSM's Good Regulator theorem, the market route's Second Welfare Theorem, and evolution's own externally-imposed fitness pressure). It is a version of Hume's is-ought gap: no dynamical system derives values from nothing. The honest, buildable target is a tight, quantified, falsifiable version of Claim A — which is what `v2.md` proposes.
