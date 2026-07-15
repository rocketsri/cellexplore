# Reorientations — testing proposed fixes for two previously-negative findings

This document tests two proposals that emerged from pushing back on negative findings elsewhere in the investigation, rather than accepting them as final. Both are extensions of results already established: the immune-network route's Lotka–Volterra dynamics (`v1.md`) and the escape-probability/killed-random-walk framework (`paper.md`, Theorem 5). Both proposals are tested honestly, including the parts that didn't fully resolve.

## 1. Does routing corrections through a unit's own equilibrium parameters make them stick?

**The finding being addressed** (`v1.md`, immune-network route): "the identical mechanism that resists adversarial perturbation mechanically resists legitimate correction, with no term in the equations distinguishing the two."

**Mechanism tested.** Using the same Farmer–Packard–Perelson clonal-network dynamics as `v1.md`'s route ($\dot x_i = x_i[\sum_j m_{ij}x_j - k\sum_j m_{ji}x_j - x_i/K_i - d_i] + s_i$), compare two ways of correcting a unit's behavior: **Correction A**, a raw additive external force competing directly against the network's homeostatic pull (structurally identical, from the dynamics' point of view, to a hostile perturbation); **Correction B**, a permanent modulation of the unit's own capacity parameter $K_i$ — changing what the network's homeostasis is *itself* converging toward, rather than fighting it.

**Results, honest unit (no defection):**
- Correction A: **100% rebound** — once the external force is removed, the unit returns exactly to its pre-correction equilibrium. This is the literal mechanism behind `v1.md`'s finding: resistance to legitimate correction is real and total when correction takes this form.
- Correction B: **0% drift-back** — the unit settles at the new equilibrium and stays there permanently, with no ongoing enforcement.

**Results, hostile defector (elevated local drive, simulating reward-hacking pressure):** unmitigated, the defector runs away (3.29 vs. 1.0 baseline — suppression alone does not contain it, consistent with `v1.md`'s original honest limitation). Correction A has **no measurable effect at all** on the defector (identical 3.2944 with or without it — the elevated local drive fully absorbs and outlasts the temporary force). Correction B produces a real but **partial** correction (3.29→1.67, still above the honest-unit baseline of 1.0).

**What this establishes, precisely:** for well-behaved units, routing corrections through equilibrium-defining parameters rather than competing external force is not just different but strictly better — a genuine, validated, actionable design principle. Against an actual defector, it's still better than raw force (which does nothing), but only partially effective, not a full fix.

**What this does not establish, stated plainly:** the tested defector had a fixed, non-adaptive elevated drive. Whether a defector that can *adaptively* increase its own drive in response to a capacity cut — a genuine optimization process pushing back against the correction, not a static parameter — would eventually defeat this too is a separate, harder, untested question. This result narrows the original finding (correction-resistance is not an unconditional property of the mechanism, it depends on which channel correction uses) without claiming to solve the adaptive-adversary case.

## 2. Does escape probability between verifier modules predict correlated blind spots?

**The finding being addressed** (`directions1.md`, Direction 4): "the actual security gain from an ensemble scales with the verifiers' *failure independence*, not their count or surface methodological variety — and there is no evidence offered... that real verifier suites have low failure-correlation rather than assumed low correlation."

**Explicit scope, stated before any result:** this tests whether escape probability predicts correlated blind spots arising from *shared training lineage* — the specific failure mode `directions1.md`'s cross-cutting synthesis named. It does **not** test, and is not claimed to address, Obfuscated Gradients' actual finding (Athalye, Carlini & Wagner, ICML 2018): an adversary with query access to multiple verifiers running a *joint, adaptive optimization* against them simultaneously. Those are different failure modes. Conflating them would repeat exactly the kind of overclaim that got Direction 5 downgraded earlier in this investigation.

**Mechanism tested.** Two verifier populations, each with a "private" computational module plus shared access (controlled by a mixing parameter $\alpha\in[0,1]$) to a common upstream component that carries a spurious confound. Escape probability from verifier A's module to verifier B's module, routed through the shared component, is computed exactly via Theorem 5's formula as a function of $\alpha$. Separately, both verifiers are actually trained (on data where the confound is predictive in-distribution but decoupled from the true label out-of-distribution) and their **correlated failure rate** — $P(\text{both wrong}\mid\text{either wrong})$, the quantity that determines whether a second verifier adds real protective value — is measured empirically.

**Result: a clean, monotonic match.**

| $\alpha$ (shared coupling) | escape probability (Theorem 5) | $P(\text{both wrong}\mid\text{either wrong})$ |
|---|---|---|
| 0.0 | 0.000 | 0.000 |
| 0.1 | 0.336 | 0.057 |
| 0.3 | 0.629 | 0.241 |
| 0.6 | 0.777 | 0.448 |
| 1.0 | 0.855 | 0.609 |

Escape probability — a quantity computable from architecture alone, before any training or observed failures — tracks the empirically measured correlated-failure rate closely and monotonically across the full range. This gives Direction 4 the concrete, testable design criterion `directions1.md` said was missing: verifier independence can be estimated *in advance*, from how much two verifiers' computational modules are coupled through shared components, rather than assumed from organizational separation ("different team, different data") with no way to check it.

**What this does not establish:** nothing here bears on adaptive joint attacks. A low-escape-probability pair of verifiers could still, in principle, both be defeated by a single adversarial input specifically optimized against both simultaneously with query access — that is a different threat model this result says nothing about, and the paper does not claim otherwise.

## Summary

Both proposals survive honest testing, and both survive with real, stated limits rather than as unconditional fixes. The immune-network result is a validated, useful design principle for the honest-unit case, with an explicitly flagged, untested open question for the adaptive-defector case. The verifier-independence result is a validated, useful, computable design criterion for one specific failure mode (correlated blind spots from shared lineage), with an explicitly flagged, different failure mode (adaptive joint attack) it does not address. Neither is a rescue of the original negative finding in the strong sense; both are genuine narrowings that produce something a designer could actually use.

Experiment code: `experiments/immune_correction_channel_test.py`, `experiments/verifier_independence_test.py`.
