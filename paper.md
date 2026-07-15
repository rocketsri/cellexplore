# Local Priors, Graph Coupling, and Provable Completion Robustness in an Equilibrium-Propagation-Trained Alignment Architecture

## Abstract

We give a closed-form theoretical treatment of the architecture proposed in `v2.md`: a population of units coupled by a symmetric, gated graph-Laplacian energy, locally driven by their own priors, and trained via equilibrium propagation. We prove the composed energy is globally strongly convex whenever local priors are (Theorem 1), which yields a unique fixed point and closes a gap `v2.md` left open — that equilibrium propagation's convergence guarantee actually survives composition with the graph coupling term. We derive a corrected relaxation-time formula that fixes a latent gap in `v1.md`'s regeneration-time result (Theorem 2). We give an exact closed-form solution for cued completion under partial/corrupted observation (Theorem 3) and a provable operator-norm robustness bound (Theorem 4). We validate all of this not only analytically but with real, stochastic equilibrium-propagation training from random initialization, confirming the trained system matches the idealized theory to within a few percent. We are explicit about what is and is not novel here: the underlying graph-regularization mathematics is well-established (Zhou et al. 2004; Smola & Kondor 2003), and the central empirical finding — that *connectivity*, not raw evidence quantity, governs robustness — is a specific instance of the oversmoothing/oversquashing phenomenon already studied in the GNN literature (Oono & Suzuki 2020; Topping et al. 2021). What is new is the closed form for this specific gated, EqProp-trainable composed energy, and its connection to the alignment-maintenance framing developed across `v1.md`–`v3.md`.

## 1. Setup

Recall from `v2.md`: $N$ units with coupling states $x_i$, gains $g_i\in[0,1]$, local energies $U_i(x_i;\theta_i)$, symmetric graph weights $w_{ij}=w_{ji}$, and composed energy
$$
E(x,\theta) = \sum_i U_i(x_i;\theta_i) + \frac{\kappa}{2}\sum_{i,j} g_ig_jw_{ij}\lVert x_i-x_j\rVert^2.
$$
We work in the homogeneous quadratic case $U_i(x_i)=\frac{\mu}{2}\lVert x_i-\eta_i\rVert^2$ ($\mu>0$) throughout, both because it admits exact closed forms and because it is the case in which $\eta_i$ is directly learnable via equilibrium propagation.

## 2. Theoretical results

**Theorem 1 (Global strong convexity).** For every $\kappa\ge0$, $g\in[0,1]^N$, and graph $G$, if each $U_i$ is $\mu$-strongly convex in $x_i$,
$$
\frac{\partial^2 E}{\partial x^2} = \mu I + \kappa(L_g\otimes I_d) \succeq \mu I \succ 0,
$$
where $L_g=D_g-W_g$ is the gated graph Laplacian, $(W_g)_{ij}=g_ig_jw_{ij}$. *Proof:* $L_g$ is a graph Laplacian, hence positive semi-definite for any nonnegative weights; adding it to $\mu I$ can only increase the minimum eigenvalue. $\blacksquare$

**Corollary (EqProp applicability, closing `v2.md`'s open gap).** $E$ is therefore globally $\mu$-strongly convex in $x$ for fixed $\theta$; it has a unique minimizer $x^*(\theta)$ depending smoothly on $\theta$ and on the nudging strength $\beta$ (via the implicit function theorem, since the Hessian is uniformly bounded away from singular). Scellier & Bengio's (2017) equilibrium-propagation convergence theorem therefore applies to the full composed system exactly as stated, with no additional unverified assumptions — resolving the "composed, plausible but not independently re-derived" status `v2.md` §3 left open.

**Theorem 2 (Corrected relaxation rate).** In the homogeneous quadratic case, the Hessian diagonalizes against $L_g$'s eigenbasis: mode $k$ relaxes at rate $\mu+\kappa\lambda_k(L_g)$. The common/consensus mode ($\lambda_1=0$) relaxes at rate exactly $\mu$, independent of $\kappa$ and graph structure. This corrects `v1.md`'s regeneration-time formula, which implicitly assumed $T_{\text{relax}}\sim1/(\kappa\lambda_2(G\setminus S))$ and therefore predicted no recovery guarantee for a fully-disconnected unit ($\lambda_2\to0$); the corrected formula $T_{\text{relax}}\sim1/(\mu+\kappa\lambda_2(G\setminus S))$ shows such a unit still recovers, at rate $\mu$, to its own local optimum.

**Theorem 3 (Closed-form Dirichlet completion).** Partition units into observed $O$ (clamped to cue values $c_O$) and free $F$. The energy-minimizing completion is
$$
x_F^* = (\mu I_F+\kappa L_{FF})^{-1}(\mu\eta_F+\kappa W_{FO}c_O),
$$
where $L_{FF}$, $W_{FO}$ are the Laplacian's free-free and coupling matrices' free-observed blocks. This is a discrete Dirichlet/Poisson problem, formalizing (with an explicit closed form for this specific energy) what `v1.md`'s bioelectricity route only described qualitatively as "harmonic inpainting."

**Theorem 4 (Operator-norm robustness bound).** Set $\eta$ so the full free-settling fixed point equals a target pattern $T$: $\eta_i=T_i+\frac{\kappa}{\mu}\sum_jw_{ij}(T_i-T_j)$. For a corrupted cue $c_O=T_O+\delta$,
$$
x_F^*-T_F = G\delta, \qquad G:=(\mu I_F+\kappa L_{FF})^{-1}\kappa W_{FO}, \qquad \lVert x_F^*-T_F\rVert\le\lVert G\rVert_2\lVert\delta\rVert.
$$
Since $L_{FF}\succeq0$ and is strictly positive-definite with minimum eigenvalue $\gamma:=\lambda_{\min}(L_{FF})>0$ whenever every connected component of the free-induced subgraph has at least one edge to $O$ (a classical grounded-Laplacian fact), $\mu I_F+\kappa L_{FF}\succeq(\mu+\kappa\gamma)I_F$, giving the a priori bound
$$
\lVert G\rVert_2 \;\le\; \frac{\kappa\lVert W_{FO}\rVert_2}{\mu+\kappa\gamma}.
$$

**Honest limitation on Theorem 4.** This bound is valid (confirmed numerically: predicted $\lVert G\rVert_2=0.957$ against a measured realized gain of $0.443$ for random Gaussian corruption — correctly an upper bound, not a tight equality, since random noise need not align with the top singular direction). It does **not**, by itself, explain the *direction* of the empirical connectivity-buffering effect in Section 3 below, because both $\lVert W_{FO}\rVert_2$ (numerator) and $\gamma$ (denominator) increase together as the free set shrinks and becomes more diagonally dominant — a joint, Cheeger/conductance-style bound relating the $(F,O)$ cut directly (in the spirit of Topping et al. 2021's curvature-bottleneck framework) would be needed to derive the empirical direction analytically rather than observe it. This is flagged as the clearest remaining theoretical gap, not papered over.

## 3. Experimental validation

**3.1 Analytically-idealized $\eta$ (6×6 grid, $D=4$, $N=36$, 20 seeds per condition).**
- Linear scaling with corruption: confirmed exactly (error doubles when noise doubles: 0.048→0.096→0.192→0.385).
- Error decreases with $\mu$: confirmed monotonically (0.227 at $\mu=0.1$ → 0.098 at $\mu=8.0$).
- Error vs. cue coverage: the naive prediction ("more evidence helps") is **false** — error *increases* with coverage (0.138 at 15% coverage → 0.220 at 90%). Mechanism: as the free set shrinks, remaining free nodes lose free-free neighbors to average against and become directly exposed to more noisy cue sources with less buffering. This is the connectivity-governs-robustness finding, now precisely diagnosed rather than merely observed.
- Bound check: measured realized gain (0.443) ≤ theoretical $\lVert G\rVert_2$ (0.957), the correct relationship for an operator-norm upper bound under random (non-adversarial) corruption.

**3.2 Real, stochastic equilibrium-propagation training (random initialization, $\beta=0.05$, 4000 steps).** Loss ($\lVert x_0-T\rVert/\lVert T\rVert$) falls from 1.04 to 0.022 over training; the trained $\eta$ converges to within 4.1% (relative) of the closed-form $\eta^*$ from Theorem 4. Completion robustness under the trained $\eta$ matches the idealized-$\eta$ predictions closely (2–22% relative difference depending on corruption level, converging as corruption grows and dominates residual training imprecision), and the connectivity-buffering effect **replicates** under the trained system (0.145→0.198→0.241 across the same coverage sweep) — the theory is not an artifact of hand-constructing $\eta$; it survives contact with real local learning.

## 4. Related work — honest positioning

The graph-Laplacian-regularized completion machinery (Theorem 3, the linear-system form) is not new: it is the same equation family as Zhou, Bousquet, Lal, Weston & Schölkopf's "Learning with Local and Global Consistency" (NeurIPS 2004) and Smola & Kondor's "Kernels and Regularization on Graphs" (2003), both foundational to graph-based semi-supervised learning. The empirical connectivity-governs-degradation finding (Section 3.1) is a specific instance of oversmoothing/oversquashing, studied extensively in graph neural network theory via Dirichlet-energy bounds tied to algebraic connectivity (Oono & Suzuki, 2020) and via curvature/bottleneck arguments (Topping et al., "Understanding Oversquashing and Bottlenecks on Graphs via Curvature," 2021). What we add: (i) the closed form for this specific composed energy, including the gated $g_i$ quarantine variable (not present in the standard graph-regularization or GNN-oversmoothing literature, and specifically motivated by the defection/verification framing developed in `v1.md`–`v3.md`); (ii) confirmation that the local priors driving this system can be reached by a provably-local, EqProp-based learning rule rather than a hand-designed kernel or globally-backpropagated loss; (iii) the connection to this investigation's Claim A / Claim B distinction (`discussion1.md` §7) — this entire apparatus is a Claim-A tool (converging to and robustly maintaining an externally-supplied target), not a route to originating one.

## 5. What's still needed for a fully rigorous submission

1. **The Cheeger/conductance-style tightening of Theorem 4**, to analytically (not just empirically) explain the direction of the coverage-vs-robustness effect.
2. **Scale beyond the 36-node toy grid** — larger $N$, non-grid topologies, and a check of whether the connectivity-buffering effect's magnitude follows a predictable scaling law in $N$ and average degree.
3. **A genuine multi-pattern extension** — the current construction (Theorem 1's strict convexity) provably supports only a single learned target per unit; extending to multiple recallable patterns requires relaxing to non-convex, multi-well local energies, reopening the basin-of-attraction analysis flagged in the earlier discussion, and is left as the natural next step rather than addressed here.
4. **A task beyond synthetic Gaussian patterns** — testing completion/denoising on a real structured dataset would be necessary to claim practical, not just mathematical, relevance.

All experiment code referenced here (`v2_completion_test.py`, `v2_eqprop_trained_validation.py`, `v2_symmetry_cost_experiment.py`, `v2_causal_consistent_training.py`) is available in this session's scratchpad and can be moved into the repository on request.
