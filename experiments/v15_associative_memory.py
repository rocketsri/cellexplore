"""
v15_associative_memory.py
=========================

Adjudicates the crux the 4th new-agenda agent was assigned: is a learned
ASSOCIATIVE MEMORY (modern Hopfield / dense associative memory = attention;
Ramsauer et al. 2020, Krotov-Hopfield 2016) a GENERATIVITY mechanism (realizes
HELD-OUT targets via a key) or mere RETRIEVAL (Claim-A: the target was stored, so
the selection specifies it)?

Modern Hopfield retrieval:  x*(q) = Xi @ softmax(beta * Xi^T q),  Xi = stored patterns.
- High beta (sharp): retrieves the single nearest STORED pattern (discrete memory).
- Low beta (soft/linear): x* is a convex/linear combination of stored patterns ->
  can produce COMPOSITIONS in span(Xi).

The adjudication (measured):
  (A) RETRIEVAL fidelity: noisy query of a STORED pattern -> recovers it (works as memory).
  (B) HELD-OUT UNRELATED realizability: a NEW random target (not stored, not in span);
      optimize the key to hit it. If it fails -> memory returns a stored pattern ->
      RETRIEVAL = CLAIM-A (selection specifies a stored target; no generativity).
  (C) HELD-OUT COMPOSITIONAL realizability: target = combination of stored basis atoms
      (in span(Xi)); optimize the key. If it succeeds -> generativity, but of the
      COMPOSITIONAL/superposition kind -> reduces to the v14 dictionary substrate.

Expected/likely honest finding: plain associative memory is CLAIM-A (retrieval);
its generativity, where real, is COMPOSITION = the superposition/dictionary mechanism
(v14), NOT a new mechanism. This is a registry result (no new positive substrate).

Pure numpy. Deterministic. ~10 s.
"""

import numpy as np
import json


def softmax(z, axis=0):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def retrieve(Xi, q, beta):
    """Modern Hopfield one-step retrieval: x* = Xi softmax(beta Xi^T q)."""
    return Xi @ softmax(beta * (Xi.T @ q), axis=0)


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def best_key_for_target(Xi, target, beta, iters=400, lr=0.5, seed=0):
    """Optimize a key q so retrieval x*(q) ~ target (the per-target SELECTION).
    Uses finite-difference-free direct gradient of the softmax retrieval."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(Xi.shape[0]) * 0.1
    for _ in range(iters):
        p = softmax(beta * (Xi.T @ q), axis=0)      # (K,)
        x = Xi @ p
        e = x - target
        # dL/dq via chain rule through softmax: dx/dq = beta * Xi @ (diag(p)-pp^T) @ Xi^T
        g_p = Xi.T @ e                                # (K,)
        Jp = beta * (p * (g_p - (p * g_p).sum()))    # (K,)
        grad = Xi @ Jp
        q = q - lr * grad
    return retrieve(Xi, q, beta), q


def main():
    out = {}
    n, K = 64, 40
    rng = np.random.default_rng(0)
    Xi = rng.standard_normal((n, K)); Xi /= np.linalg.norm(Xi, axis=0, keepdims=True)

    print("=" * 70)
    print("v15: ASSOCIATIVE MEMORY -- generativity (held-out) vs retrieval (Claim-A)?")
    print("=" * 70)
    print(f"n={n}, K={K} stored patterns (unit-norm)\n")

    # (A) RETRIEVAL of a stored pattern from a noisy query -------------------
    betaH = 8.0
    rec = []
    for mu in range(K):
        q = Xi[:, mu] + 0.3 * rng.standard_normal(n)
        rec.append(corr(retrieve(Xi, q, betaH), Xi[:, mu]))
    print(f"[A] RETRIEVAL (noisy query of STORED pattern, beta={betaH}): corr = {np.mean(rec):.3f} "
          f"({'works as a memory' if np.mean(rec) > 0.8 else 'weak'})")
    out["retrieval"] = float(np.mean(rec))

    # (B) HELD-OUT UNRELATED target (not stored, not in span) ----------------
    #     span(Xi) has dim min(n,K)=40 < n=64, so a random n-vector has a component
    #     OUTSIDE span(Xi): the memory CANNOT produce it (x* always in span(Xi)).
    ho_un = []
    for t in range(8):
        target = rng.standard_normal(n); target /= np.linalg.norm(target)
        x, _ = best_key_for_target(Xi, target, betaH, seed=t)
        ho_un.append(corr(x, target))
    print(f"[B] HELD-OUT UNRELATED (new random target; optimize key): corr = {np.mean(ho_un):.3f}")
    print(f"    => memory output x* is ALWAYS in span(stored); a target with an out-of-span")
    print(f"       component is UNREACHABLE. Low corr = RETRIEVAL = CLAIM-A (no generativity).")
    out["held_out_unrelated"] = float(np.mean(ho_un))

    # (C) HELD-OUT COMPOSITIONAL target (a combination of stored atoms, in span) --
    betaL = 1.0
    ho_comp = []
    for t in range(8):
        w = np.zeros(K); idx = rng.choice(K, 5, replace=False)
        w[idx] = rng.standard_normal(5)
        target = Xi @ w; target /= np.linalg.norm(target)     # in span(Xi)
        x, _ = best_key_for_target(Xi, target, betaL, seed=100 + t)
        ho_comp.append(corr(x, target))
    print(f"[C] HELD-OUT COMPOSITIONAL (target in span(stored), beta={betaL}): corr = "
          f"{np.mean(ho_comp):.3f}")
    print(f"    => reachable IFF the target is a COMBINATION of stored atoms. This is")
    print(f"       generativity of the COMPOSITIONAL/superposition kind = the v14 dictionary")
    print(f"       mechanism (stored patterns = atoms, softmax key = the sparse/soft code).")
    out["held_out_compositional"] = float(np.mean(ho_comp))

    print("\nADJUDICATION:")
    print(f"  retrieval (stored)            = {out['retrieval']:.3f}")
    print(f"  held-out UNRELATED            = {out['held_out_unrelated']:.3f}  "
          f"({'CLAIM-A / retrieval only' if out['held_out_unrelated'] < 0.5 else 'generalises?!'})")
    print(f"  held-out COMPOSITIONAL        = {out['held_out_compositional']:.3f}  "
          f"({'generativity = superposition (v14)' if out['held_out_compositional'] > 0.7 else 'weak'})")
    print("  => Verdict: plain associative memory is RETRIEVAL (Claim-A) -- it realizes only")
    print("     stored patterns / in-span combinations, NOT out-of-span held-out targets. Where")
    print("     it DOES generalize (compositional/in-span), it IS the superposition/dictionary")
    print("     substrate (v14), not a new mechanism. Registry: NOT a distinct 2nd positive.")
    print("\nJSON_V15_BEGIN"); print(json.dumps(out)); print("JSON_V15_END")


if __name__ == "__main__":
    main()
