# Discussion 2 — how biology addresses inner alignment, deceptive alignment, and scalable oversight, and efficient AI-native analogs

Continuing from `discussion1.md` and `v2.md`. `v2.md` closed by naming four named AI-safety sub-problems it does not solve: outer alignment, inner alignment, deceptive alignment, scalable oversight. This asks how biology handles each, and how to get the useful parts without brute-forcing evolution's trial count.

## How biology addresses each problem

**Outer alignment (is $T$ correct?) — not solved, dissolved.** Evolution has no external standard to answer to. Fitness is defined circularly as whatever caused more copies to persist; there is no sense in which it can be "wrong," because nothing outside the process checks it against anything. This is the sharpest disanalogy in the whole investigation: AI's outer alignment problem exists specifically because we want the system to track an independent, pre-existing standard (human values/intent) — evolution never had to do that. There is no biological mechanism to import here.

**Inner alignment (does the system actually pursue $T$, not merely correlate with it during training?) — real, partial mechanisms.** Development is filtered continuously and cheaply, not evaluated once at the end: most failed local cellular coordination is purged early (embryonic loss), before a full lifetime is invested. Developmental systems are also selected for *canalization* (Waddington) — robustness to internal noise built into the basin of attraction by selection pressure, not free. And checking continues within a single lifetime: p53-mediated damage response and apoptosis are a real-time, continuously-running inner-alignment check on individual cells.

**Deceptive alignment (a component that passes scrutiny while diverging) — not solved; biology's clearest negative case.** Tumors undergo real somatic evolution *under immune selection pressure*, evolving specifically to evade the markers (MHC presentation, stress ligands) immune surveillance uses to detect them — a live arms race with a nonzero, unresolved failure rate (~40% lifetime cancer incidence). Biology's mitigation is not one perfect check but *layered, structurally different* checks (immune surveillance, apoptosis, the Hayflick limit/replicative senescence, mechanical/contact-inhibition signaling), so evading all simultaneously is combinatorially harder than evading one — and it accepts residual risk rather than promising zero.

**Scalable oversight (checking something more capable than the checker) — a real, elegant trick.** The immune system never tries to understand a cell's full computational state. It checks a narrow, low-dimensional, mechanistically-privileged signal — MHC presentation, stress ligands — that is hard to fake without also breaking function. The checker doesn't need capability parity with the checked system; it needs a signal structurally coupled to the property of interest.

## Proposed efficient AI-native analogs (candidates for further scrutiny — see `directions1.md`)

1. Early, cheap, continuous filtering instead of one expensive final eval (process-level reward models, successive halving, killing bad trajectories before full investment).
2. Explicit canalization/robustness training (adversarial internal perturbation, noise injection as a deliberate objective, not a hoped-for side effect).
3. Real-time, component-level runtime monitoring with actual intervention authority, not just output-level filters.
4. Multiple, structurally different verifiers running together (behavioral evals + mechanistic interpretability + causal tracing + red-teaming), rather than one homogeneous verifier population.
5. Checking internal representations rather than only external behavior (the MHC/stress-ligand trick ported directly — close to existing ELK / weak-to-strong-generalization directions).
6. Treating verification as a continuously updated arms race (ongoing red-teaming, scheduled verifier retraining) rather than a fixed, one-time-trained checker.
7. For the piece that genuinely needs blind/evolutionary search (evolving the local rule/topology from `discussion1.md` §2): restrict the search to a small, high-leverage space on top of an already gradient-trained capability base, rather than searching the whole system from scratch — shrinking the needed trial budget by orders of magnitude, mirroring how biology reuses a deep conserved toolkit instead of searching from a blank slate each time.

These seven are candidates, not established findings — `directions1.md` is the adversarial assessment of each, ahead of committing further research effort to any of them.
