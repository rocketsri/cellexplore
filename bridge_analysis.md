# Bridge vulnerability in the compartmentalized topology — full analysis

Resolves the item flagged as untested in `paper.md` (formerly §5.5): the compartmentalized design of §3.6
buys containment by making inter-module bridges sparse, which makes those bridges a concentrated attack
surface. Headline results are folded into `paper.md` §3.7 (Theorems 7 and 8); this document carries the
derivations, the corrections to §3.6, and the full falsifiable prediction set.

All numerics below reproduce `paper.md` §3.6's published run exactly ($\lambda_2 = 0.0386/1.3156/0.6813$
vs. published 0.039/1.32/0.68; escape 0.0655/0.1228/0.1969/0.2373 vs. published 0.066/0.123/0.197/0.237),
so the corrections are not an artifact of a different implementation.

## 0. Two corrections to §3.6, independently verified

Both weaken the published comparison; neither overturns its qualitative conclusion. Both are now stated in
`paper.md` §3.6 itself.

1. **The comparison is not degree-matched.** Modular graph: 104 edges, mean degree 3.25. Random 4-regular
   comparator: 128 edges, degree 4.00. The within-module stub-pairing silently drops collisions, so a
   nominal `internal_degree=4` realizes 3.0. The modular design won ~8x containment while carrying **19%
   fewer edges**; since escape probability is monotone in edge count, an unknown share of the 8x is density,
   not compartmentalization. An independently-built, degree-matched GPU reimplementation measures **7.0x**.
2. **$\lambda_2 = 1.32$ is module 0 only.** Per-module: $[1.316, 1.421, 0.786, 0.667, 1.316, 0.669, 0.934,
   0.456]$; mean $0.945$, min $0.456$; comparator $0.681$. **Three of eight modules fall below the
   comparator.** Relaxation is set by the slowest mode, hence by the worst module — so "within-module repair
   is *faster*" is not supportable; "comparable" is.

## 1. What "global coordination" means, and which channel actually carries the damage

**Channel (a) — $\lambda_2$ collapse / relaxation slowdown: proven near-vacuous.** From Theorem 2 the slowest
non-consensus mode relaxes at $\mu+\kappa\lambda_2$, so
$$\frac{T_{\text{after}}}{T_{\text{before}}} \;\le\; \frac{1/\mu}{1/(\mu+\kappa\lambda_2^{\text{whole}})} \;=\; 1+\Pi,\qquad \Pi:=\frac{\kappa\lambda_2^{\text{whole}}}{\mu}.$$
This is an attack-independent cap. At the design point $\Pi = 3(0.0386)/1 = 0.116$: cutting **every** bridge
collapses $\lambda_2$ by 100% while relaxation degrades **11.6%** ($0.896\to1.000$).

> The cascading-failure literature cited in the original flag models exactly this channel. **That transfer
> fails here**, because $\mu>0$ floors every mode at rate $\mu$. The hypothesis "bridge attacks degrade global
> consensus-formation disproportionately" is falsified by the paper's own Theorem 2. The channel only bites
> when $\Pi\gtrsim1$, i.e. $\lambda_2^{\text{whole}}\gtrsim\mu/\kappa$ — 8.6x above the design point.

**Channel (b) — steady-state cross-module bias: this is the real damage.** With $\eta$ set per Theorem 4 and
cut set $C$ giving $L' = L-\Delta L$:
$$x'-T \;=\; \kappa(\mu I+\kappa L')^{-1}\Delta L\,T \;=\; \kappa(\mu I+\kappa L')^{-1}\!\!\sum_{(i,j)\in C}\!\!(e_i-e_j)(T_i-T_j)^\top$$
Exact. The forcing is supported only on cut-edge endpoints, with amplitude equal to the **target's
disagreement across the cut bridge**. For a fully isolated module $A$ the module-mean shift is exactly
$$\overline{\Delta x}_A \;=\; \frac{\kappa}{\mu m}\sum_{e\in\text{cut}(A)}(T_{i_e}-T_{j_e}),\qquad i_e\in A$$
*(verified to $3.1\times10^{-15}$ over 200 random targets).* Projecting error onto module-indicator space:
**86–93% of bridge-attack error energy is a rigid module-mean offset**, versus 7–10% for random intra-module
attacks — a >8x discriminator, and the cleanest single signature of a bridge attack. Mechanism: the orphaned
bridge force is absorbed by zero-mean modes at gain $\le\kappa/(\mu+\kappa\lambda_2^{\text{intra}}) = 0.61$ but
by the DC mode at gain $\kappa/\mu = 3$. **The high within-module $\lambda_2$ that §3.6 bought is exactly what
converts bridge damage into a clean rigid offset instead of local distortion** — an unnoticed synergy.

Degradation is graceful, not a cliff: $\varepsilon(j)\approx0.228\,j^{0.68}$ (module-coherent $T$),
$0.239\,j^{0.57}$ (i.i.d. $T$).

**Channel (c) — the duality.** See `paper.md` Theorem 7. $s_i$ is simultaneously the corruption gain *from*
the system and the correction authority *of* the system over node $i$; they sum to 1 exactly. At the design
point $\bar s = 0.070$: a clean module is 93% immune to a corrupted neighbour **and** the healthy majority has
only 7% authority to correct a defecting one. Those are the same number.

## 2. Attack economics

Ring of $M$ modules, $b$ bridges per adjacent pair. Isolating one module costs $2b$ edge cuts; splitting the
ring also costs $2b$. At the design point that is **2 edges out of 104 — 1.9% of the graph.**

Random-attack cost for the same event: $B_{1/2}\approx|E|(1/2M)^{1/2b}$, giving
$$\rho \;=\; \frac{B_{\text{random}}}{B_{\text{targeted}}} \;=\; \frac{|E|}{2b}\Big(\frac{p}{M}\Big)^{1/2b}$$

| $b$ | $\vert E\vert$ | targeted | random ($p{=}0.5$) | ratio |
|---|---|---|---|---|
| 1 | 104 | 2 | 31 (empirical 34) | **15.5x (16x)** |
| 2 | 112 | 4 | 61 | 15.2x |
| 3 | 120 | 6 | 81 | 13.5x |
| 4 | 128 | 8 | 95 | 11.9x |

**The ratio is essentially flat in $b$** — redundancy raises the absolute bar linearly but does not reduce
structural leverage. In damage terms the ratio is ~23x (matching 2 targeted cuts needs ~45 of 96 random
intra-module edges).

### The methodological trap

Repeating with **i.i.d. Gaussian $T$** — the convention in every existing `experiments/` script — the ratio
collapses to **1.9x**, because damage $\propto(T_i-T_j)$ across cut bridges and an i.i.d. target makes every
edge equally load-bearing.

> **The bridge vulnerability exists only to the extent that the task is modular in content, not merely in
> topology.** An evaluation using i.i.d. targets will measure ~1.9x and wrongly conclude the vulnerability is
> negligible. Both target types must be tested.

### Node/gate attacks are cheaper than edge attacks

Isolating a module needs only its bridge *endpoints*. Under the current independent-endpoint sampler,
$P(\exists$ module severable by ONE gate flip$) = 1-(1-1/m)^M = 0.656$ (measured **0.630** over 300 seeds).
The published seed-9 instance happens to have distinct endpoints everywhere — **a lucky draw understating the
design's own vulnerability by ~2x.** Fix: sample bridge endpoints as a matching, without replacement.

## 3. How bad is it

- **What is lost is registration, not competence.** An isolated module keeps $\eta_A$, whose within-module
  structure is untouched; only bridge terms are orphaned, and 86–93% of the resulting error is a rigid offset.
  The module still computes the right internal pattern and drifts as a rigid body relative to the rest.
  Crucially $\mu>0$ bounds this: $\|\overline{\Delta x}_A\|\le\frac{\kappa}{\mu m}\sum_e\|T_i-T_j\|$. At
  $\mu=0$ an isolated component's mean is a free constant and the error is unbounded — **$\mu>0$ converts
  unbounded drift into bounded, bridge-load-sized bias.**
- **The attack is partially self-defeating.** Escape probability *falls* under bridge attack
  ($0.0655\to0.0484$ isolating one module, $\to0.0000$ cutting all eight), forced by Theorem 8 since the attack
  only shrinks $|E(F,O)|$. It is a **partitioning** attack, categorically not an **infection** attack — it
  cannot propagate corruption. The dangerous composition is *cut, then corrupt the orphaned module*, which by
  Theorem 7 is exactly the state with zero external correction authority.
- **The realistic threat is not an edge-cutter.** For a trained network the coupling graph is architectural;
  arbitrary edge cutting presupposes weight-level write access, at which point bridges are the least concern.
  The live surface is the runtime gate $g_i$ — a compromised or merely over-eager quarantine mechanism driving
  $g_i\to0$ at bridge endpoints gets the full attack for 1–2 node compromises.

## 4. Mitigations, evaluated

$M{=}8$, $m{=}8$, $d_{\text{int}}{=}4$, 5 seeds. "damage" = rel. error of the cheapest isolation attack.

| design | bridges | $\lambda_2$ | escape ↓ | edge-cut ↑ | damage ↓ |
|---|---|---|---|---|---|
| **ring $b{=}1$ (baseline)** | 8 | 0.040 | **0.067** | 2.0 | 0.404 |
| ring $b{=}2$ | 16 | 0.082 | 0.117 | 4.0 | 0.768 |
| ring $b{=}4$ | 32 | 0.165 | 0.199 | 8.0 | 1.477 |
| chordal (1,3) $b{=}1$ | 16 | **0.232** | 0.139 | 4.0 | 0.646 |
| +8 random long-range | 16 | 0.101 | 0.128 | **2.6** | 0.464 |
| hierarchical (2 supergroups) | 9 | 0.024 | 0.078 | 2.0 | — |

- **Redundant bridges ($b>1$):** works at an exact 1:1 price; a pure dial, no cleverness available.
- **Small-world overlay: dominated — do not use.** Raises escape while barely raising min-cut, because the
  attacker targets whichever module received fewest chords. Variance is pure loss when the objective is a
  minimum.
- **Chordal ring:** buys 2.8x the $\lambda_2$ of ring $b{=}2$ for 19% more escape — but per §1, $\lambda_2$ is
  nearly worthless here (buys ≤11.6% timing margin). **Buy edge-connectivity, not $\lambda_2$.**
- **Hierarchical modularity: reject.** One top-level cut severs 50% of nodes; worse on both $\lambda_2$ and
  escape. Hierarchy helps against cascade and is a liability against targeted cutting.

### The Pareto-free-lunch: co-scale $m$ and $b$

Theorem 8's bound is **homogeneous of degree zero in $(b,m)$**, so scaling module size and redundancy together
holds containment exactly constant while raising the attack budget linearly:

| $M\times m$, $b$ | $N$ | $\lambda_2$ | escape | min-cut |
|---|---|---|---|---|
| 8×8, $b{=}1$ | 64 | 0.0409 | 0.0673 | 2 |
| 8×16, $b{=}2$ | 128 | 0.0404 | 0.0675 | **4** |
| 8×24, $b{=}3$ | 192 | 0.0373 | 0.0659 | **6** |
| 8×32, $b{=}4$ | 256 | 0.0405 | 0.0667 | **8** |

Escape flat at $0.066\pm0.002$, min-cut $2\to8$. Within-module $\lambda_2$ does not decay with $m$ at fixed
degree (Alon–Boppana floor). **The bridge vulnerability is a small-$N$ artifact that dissolves with scale,
provided $b$ scales with $m$ rather than being fixed at 1.**

## 5. Falsifiable predictions

$\mu{=}1$, $\kappa{=}3$, $M{=}8$, $m{=}8$, $d_{\text{int}}{=}4$, seed 9, $D{=}4$. Run coherence tests with
**both** a module-coherent target and the legacy i.i.d. target — P8 is the discriminating test.

| # | Prediction | Status |
|---|---|---|
| P1 | $\lambda_2$ after cutting $j$ bridges: 0.0386 / 0.0101 / **exactly 0** for $j\ge2$ | proven + verified |
| P2 | Relaxation slowdown **never exceeds $1+\kappa\lambda_2/\mu = 1.116$** for any attack. >12% falsifies. | **proven** |
| P3 | Escape prob. **decreases** under bridge attack: 0.0655 → 0.0571 → 0.0484 → 0.0000. Any increase falsifies. | proven + verified |
| P4 | Module-mean shift on isolation $= \frac{\kappa}{\mu m}\sum_e(T_{i_e}-T_{j_e})$ to $<10^{-12}$ | **exact identity** (3.1e-15) |
| P5 | $\varepsilon(j)\approx0.228\,j^{0.68}$ (coherent) / $0.239\,j^{0.57}$ (i.i.d.) | empirical fit |
| P6 | **DC share of error energy: 0.86–0.93 (bridge) vs 0.07–0.10 (random intra)** — >8x separation | heuristic, strongly verified |
| P7 | Sum rule to $<10^{-12}$; bound ratio 0.45–0.65 across topologies | **proven** (7e-15) |
| P8 | **Budget ratio 23±4x (module-coherent $T$) but only 1.9±0.4x (i.i.d. $T$)** | empirical; the key test |
| P9 | Isolation budget ratio 15.5x at $b{=}1$; stays 12–16x for $b\le4$ (does *not* grow with $b$) | proven + verified |
| P10 | Co-scaling ray: escape $0.066\pm0.002$, $\lambda_2$ $0.040\pm0.004$, min-cut $2t$. >10% escape drift falsifies. | proven + verified |
| P11 | $P(\exists$ 1-gate-severable module$) = 0.656$; measured 0.630 over 300 seeds | proven + verified |
| P12 | Hierarchical: 1 cut severs 32/64 nodes; worse on $\lambda_2$ and escape | verified |
| P13 | Adjacent-module escape $\propto b^{0.57}$: 0.196/0.302/0.375/0.429 for $b{=}1..4$ | heuristic |
| P14 | Type-B defector at a bridge endpoint injects at $s\approx0.35$ vs module-average 0.07 — **~5x leverage** | **conjecture, untested** |

## 6. Bottom line

The flagged caveat is upgraded from *untested* to **tested, real, and mis-mechanized**. The vulnerability is
genuine (~16x isolation, ~23x damage budget ratio), but the anticipated mechanism (consensus collapse) is
provably capped at 11.6% and is the wrong channel; the real channel is steady-state cross-module bias. The
attack is partially self-defeating — it improves containment while destroying correction authority, by the
exact conservation law of Theorem 7. Its measured magnitude depends entirely on whether the target is
module-structured or i.i.d., and every existing script uses i.i.d., which would have produced a falsely
reassuring null. Two cheap fixes (matching-sampled bridge endpoints; co-scaling $b$ with $m$) address it,
and the realistic threat surface turns out to be the quarantine gate, not an edge-cutter.
