"""
v9_cppn_crux_seeds.py
=====================
Robustness of the CRUX comparison: CPPN developmental encoding vs random linear
projection at EQUAL search dimension (d_g = 108), averaged over seeds. Single-seed
ES is noisy (v8 RP margin swung +0.20 @d128 vs v9 RP -0.13 @d108); this settles
whether the STRUCT EDGE (CPPN - RP at equal dim) is consistently positive.

Reuses the v9 harness verbatim. ~12-18 min.
"""
import numpy as np, time, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v9_cppn_indirect as v9

SEEDS = [0, 1, 2, 3]
Hc = 6
GEN = 220


def run():
    t0 = time.time()
    pool = [v9.target(s) for s in range(24)]
    held = [v9.target(200 + k) for k in range(4)]
    rng = np.random.default_rng(5)
    ctrl = float(np.mean([v9.es_select(rng.normal(0, 0.3, v9.BODY)[None], h, seed=40 + i)
                          for i, h in enumerate(held)]))
    dg = v9.cppn_dg(Hc)
    print(f"CRUX seeds test | control={ctrl:.3f} | d_g={dg} | seeds={SEEDS}")
    rows = []
    for sd in SEEDS:
        bc, trc, _ = v9.train_cppn(pool[:12], Hc=Hc, generations=GEN, seed=sd)
        hoc = float(np.mean([v9.es_select(bc, h, seed=60 + i) for i, h in enumerate(held)]))
        br, trr = v9.train_rp(pool[:12], d=dg, generations=GEN, seed=sd)
        hor = float(np.mean([v9.es_select(br, h, seed=60 + i) for i, h in enumerate(held)]))
        rows.append(dict(seed=sd, cppn_ho=hoc, rp_ho=hor, edge=hoc - hor,
                         cppn_margin=hoc - ctrl, rp_margin=hor - ctrl))
        print(f"  seed {sd}: CPPN ho={hoc:.3f} (m {hoc-ctrl:+.3f}) | RP ho={hor:.3f} "
              f"(m {hor-ctrl:+.3f}) | EDGE {hoc-hor:+.3f}  [{time.time()-t0:.0f}s]")
    ce = np.array([r["edge"] for r in rows])
    cm = np.array([r["cppn_margin"] for r in rows]); rm = np.array([r["rp_margin"] for r in rows])
    print(f"\n  STRUCT EDGE  mean {ce.mean():+.3f}  sd {ce.std():.3f}  "
          f"(min {ce.min():+.3f}, {int((ce>0).sum())}/{len(ce)} positive)")
    print(f"  CPPN margin  mean {cm.mean():+.3f}  ({int((cm>0.05).sum())}/{len(cm)} generalise)")
    print(f"  RP   margin  mean {rm.mean():+.3f}  ({int((rm>0.05).sum())}/{len(rm)} generalise)")
    verdict = ("STRUCTURE WINS (robust)" if ce.mean() > 0.03 and (ce > 0).mean() >= 0.75
               else "DIMENSION not structure" if abs(ce.mean()) <= 0.03
               else "mixed/RP-favoured")
    print(f"  => {verdict}")
    print("JSON_CRUX_BEGIN"); print(json.dumps(dict(control=ctrl, dg=dg, rows=rows,
          edge_mean=float(ce.mean()), edge_sd=float(ce.std())))); print("JSON_CRUX_END")


if __name__ == "__main__":
    run()
