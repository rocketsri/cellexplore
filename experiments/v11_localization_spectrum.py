"""
localization_spectrum.py
========================
A DL-free, causal/interventional replacement for the generativity gap Delta.

Measures a 4-channel SENSITIVITY PROFILE (the "localization spectrum") + a
healing coefficient + a causal generativity ratio G = dim_rule / dim_sel, and
shows the profile is a FINGERPRINT separating:
  - B-bio    (target in the shared dynamics; an attractor)  -- real Kuramoto Holonomy Code
  - Claim A  (target injected as a stored vector)           -- StoredVector control
  - Trivial  (constant attractor independent of everything) -- Trivial control

No zlib / description length anywhere. Pure numpy. ~1-2 min.
"""
import numpy as np

# ----------------------------------------------------------------------------
# generic distance on realized attractor morphologies (reparam-robust)
# ----------------------------------------------------------------------------
def d_field(m1, m2):
    a = m1.ravel() - m1.mean(); b = m2.ravel() - m2.mean()
    va=(a*a).sum(); vb=(b*b).sum()
    if va<1e-12 and vb<1e-12: return float(abs(m1.mean()-m2.mean())>1e-9)
    if va<1e-12 or vb<1e-12: return 1.0
    return float(1.0 - (a*b).sum()/np.sqrt(va*vb))   # 0 = identical, up to 2

def eff_dim_of_set(vectors):
    """participation ratio (effective dimension) of a set of response vectors."""
    M = np.stack([v.ravel() - v.ravel().mean() for v in vectors], 0)
    if M.shape[0] < 2: return 1.0
    s = np.linalg.svd(M - M.mean(0, keepdims=True), compute_uv=False)
    lam = s**2
    if lam.sum() < 1e-18: return 0.0
    return float((lam.sum())**2 / (lam**2).sum())

def richness(field):
    f = field.reshape(field.shape[0], -1) if field.ndim>2 else field
    m = f - f.mean()
    if (m**2).sum() < 1e-12: return 1.0            # constant field = trivial
    s = np.linalg.svd(m, compute_uv=False); p = s**2/(s**2).sum()
    return float(1.0/(p**2).sum())

def rel_perturb(x, rho, rng):
    """add Gaussian noise of std = rho * RMS(x)  -> a 'rho-relative' perturbation."""
    x = np.asarray(x, float)
    rms = np.sqrt((x**2).mean()) + 1e-12
    return x + rho*rms*rng.standard_normal(x.shape)

# ============================================================================
# SUBSTRATE INTERFACE
#   channels(): dict of the four intervention targets + the 'selection' channel
#   relax(ch, state0): run dynamics to the attractor, return the settled state
#   morph(state): map settled state -> morphology field in [0,1]
#   klass(state): discrete class label (or None)
#   seed(rng): a fresh runtime initial condition (state0)
#   select_states(): list of the K settled attractors (one per selectable target)
# ============================================================================

# ----------------------------- B-bio: Kuramoto Holonomy Code ----------------
class Kuramoto:
    name = "B-bio (Kuramoto Holonomy Code)"
    def __init__(self, L=16, alpha=0.40, dis=2.0, dt=0.05, ts=2000, seed=0):
        self.L=L; self.N=L*L; self.dt=dt; self.ts=ts
        rng = np.random.default_rng(seed)
        A = np.exp(dis*rng.standard_normal((L,L,4)))
        omega = 0.05*rng.standard_normal((L,L))
        self.ch = dict(rule=np.array([alpha]), coupling=A, drive=omega)
        self._classes = None
    def _step(self, th, ch):
        L=self.L; a=float(ch['rule'][0]); A=ch['coupling']; om=ch['drive']
        nb=[np.roll(th,-1,1),np.roll(th,1,1),np.roll(th,-1,0),np.roll(th,1,0)]
        d=om.copy()
        for k in range(4): d += A[...,k]*np.sin(nb[k]-th-a)
        return th + self.dt*d
    def relax(self, ch, state0, steps=None):
        th=state0.copy()
        for _ in range(steps or self.ts): th=self._step(th,ch)
        return th
    def morph(self, th):
        # DRIFT-INVARIANT texture: local phase-gradient field (global phase drift cancels).
        # This is where the disorder-pinned pattern AND the winding live.
        dx=(np.diff(th,axis=1,append=th[:,:1])+np.pi)%(2*np.pi)-np.pi
        dy=(np.diff(th,axis=0,append=th[:1,:])+np.pi)%(2*np.pi)-np.pi
        return np.stack([dx,dy],-1)
    def klass(self, th):
        L=self.L
        dx=(np.diff(th,axis=1,append=th[:,:1])+np.pi)%(2*np.pi)-np.pi
        dy=(np.diff(th,axis=0,append=th[:1,:])+np.pi)%(2*np.pi)-np.pi
        return (int(round(np.median(dx.sum(1)/(2*np.pi)))),
                int(round(np.median(dy.sum(0)/(2*np.pi)))))
    def seed(self, rng):
        # the system is realizing a locked target class (1,0) as its attractor
        return self.plane_seed(1,0,rng)
    def plane_seed(self, wx, wy, rng):
        ii,jj=np.mgrid[0:self.L,0:self.L]
        return 2*np.pi*(wx*jj+wy*ii)/self.L + 0.2*rng.standard_normal((self.L,self.L))
    def selection_states(self, rng):
        """the K selectable targets = the settled winding-class attractors (basin selection)."""
        seen={}
        for wx,wy in [(0,0),(1,0),(0,-1),(1,-1),(0,1),(-1,0),(1,1),(-1,-1)]:
            th=self.relax(self.ch, self.plane_seed(wx,wy,rng))
            seen.setdefault(self.klass(th), th)
        return list(seen.values())

# ----------------------------- Claim A: stored vector -----------------------
class StoredVector:
    """target IS the injected runtime state; dynamics are identity (nothing heals)."""
    name = "Claim A (stored injected vector)"
    def __init__(self, L=16, seed=0):
        self.L=L; self.N=L*L
        rng=np.random.default_rng(seed)
        # a rich 'rule' and 'coupling' that the output IGNORES (pass-through dynamics)
        self.ch=dict(rule=rng.standard_normal((8,8)),
                     coupling=rng.standard_normal((L,L,4)),
                     drive=rng.standard_normal((L,L)))
        self._targets=[self._richfield(rng) for _ in range(4)]
    def _richfield(self, rng):
        ii,jj=np.mgrid[0:self.L,0:self.L]
        f=np.zeros((self.L,self.L))
        for _ in range(6):
            f+=rng.standard_normal()*np.sin(rng.uniform(0.3,1.5)*ii+rng.uniform(0,6)) \
                *np.cos(rng.uniform(0.3,1.5)*jj+rng.uniform(0,6))
        return f
    def relax(self, ch, state0, steps=None): return state0.copy()   # identity: stores state
    def morph(self, th): return (th-th.min())/(np.ptp(th)+1e-9)
    def klass(self, th): return None
    def seed(self, rng): return self._targets[rng.integers(len(self._targets))].copy()
    def selection_states(self, rng): return [t.copy() for t in self._targets]

# ----------------------------- Trivial control ------------------------------
class Trivial:
    """constant attractor, independent of every channel and every seed."""
    name = "Trivial (constant attractor)"
    def __init__(self, L=16, seed=0):
        self.L=L; self.N=L*L
        rng=np.random.default_rng(seed)
        self.ch=dict(rule=rng.standard_normal((8,8)),
                     coupling=rng.standard_normal((L,L,4)),
                     drive=rng.standard_normal((L,L)))
        self._const=0.5*np.ones((L,L))
    def relax(self, ch, state0, steps=None): return self._const.copy()
    def morph(self, th): return th
    def klass(self, th): return 0
    def seed(self, rng): return rng.standard_normal((self.L,self.L))
    def selection_states(self, rng): return [self._const.copy()]*4

# ============================================================================
# THE MEASURE
# ============================================================================
def base_attractor(sub, rng):
    return sub.relax(sub.ch, sub.seed(rng))

def sensitivity(sub, channel, rho, base_morph, seed_fixed, rng, n=10):
    """post-relaxation attractor displacement per unit relative perturbation of `channel`.
       For state channels we perturb the initial condition; else perturb a rule/coupling/drive tensor."""
    ds=[]
    for _ in range(n):
        if channel=='state_init':
            s0 = rel_perturb(seed_fixed, rho, rng)
            f = sub.morph(sub.relax(sub.ch, s0))
        else:
            ch=dict(sub.ch); ch[channel]=rel_perturb(sub.ch[channel], rho, rng)
            f = sub.morph(sub.relax(ch, seed_fixed.copy()))
        ds.append(d_field(f, base_morph))
    return float(np.mean(ds))/rho

def healing(sub, rho, base_state, rng, n=8):
    """perturb the SETTLED (runtime) state, re-relax, ask if the TARGET is restored.
       H≈1 -> attractor heals the kick (target in dynamics); H≈0 -> kick persists (stored).
       For a topological/traveling attractor the target is the CLASS (drift-invariant);
       otherwise we drift-cancel by comparing re-relaxed-kicked to re-relaxed-unkicked."""
    base_m=sub.morph(base_state); c0=sub.klass(base_state)
    if c0 is not None:                                   # class-based (drift-invariant)
        restored=0
        for _ in range(n):
            kicked=rel_perturb(base_state, rho, rng)
            if sub.klass(sub.relax(sub.ch, kicked))==c0: restored+=1
        return float(restored/n), None
    imm=[]; rel=[]                                       # field-based, drift-cancelled
    ref=sub.relax(sub.ch, base_state.copy())            # unkicked reference, same evolution
    ref_m=sub.morph(ref)
    for _ in range(n):
        kicked=rel_perturb(base_state, rho, rng)
        imm.append(d_field(sub.morph(kicked), base_m))
        rel.append(d_field(sub.morph(sub.relax(sub.ch, kicked)), ref_m))
    imm=np.mean(imm)+1e-9; rel=np.mean(rel)
    return float(np.clip(1.0 - rel/imm,0,1)), float(rel)

def response_dim(sub, channel, rho, seed_fixed, base_morph, rng, n=24):
    """effective dimension of the attractor-change manifold under interventions on `channel`."""
    resp=[]
    for _ in range(n):
        if channel=='selection':
            pass
        ch=dict(sub.ch); ch[channel]=rel_perturb(sub.ch[channel], rho, rng)
        f=sub.morph(sub.relax(ch, seed_fixed.copy()))
        resp.append(f - base_morph)
    return eff_dim_of_set(resp)

def generativity_G(sub, rho, seed_fixed, base_morph, rng, n=24):
    """DL-FREE generativity:
        dim_rule = eff-dim of target structure the SHARED RULE+COUPLING commands (causal)
        dim_sel  = eff-dim of target structure the PER-TARGET SELECTION commands (causal)
       G = dim_rule / dim_sel  =  realized structure generated per unit of selection."""
    dim_rule = 0.5*(response_dim(sub,'rule',rho,seed_fixed,base_morph,rng,n)
                    + response_dim(sub,'coupling',rho,seed_fixed,base_morph,rng,n))
    sel = sub.selection_states(rng)
    sel_m = [sub.morph(s) for s in sel]
    dim_sel = eff_dim_of_set(sel_m)               # dims the selection channel spans
    return dim_rule, dim_sel, dim_rule/max(dim_sel,1e-6)

def levin_coupling_test(sub, rng):
    """Levin two-headed planarian formalized as a COUPLING-edit panel, GENOME (alpha) FIXED:
       a transient bioelectric-style edit to A_ij that, after restoration, leaves a PERSISTENT
       NEW winding class = a new attractor of the ORIGINAL rule. Weak edits heal back
       (topological protection = H4 double edge); strong-enough ones trap a new attractor.
       'coupling leverage' = fraction of edits depositing a persistent new attractor."""
    if not isinstance(sub, Kuramoto): return None
    base=sub.relax(sub.ch, sub.plane_seed(0,0,rng)); c0=sub.klass(base)
    persist=0; trials=0; new_classes=set()
    for g in (2,4,8,16,32):
        for ts in (600,1000,1600):
            A=sub.ch['coupling'].copy(); A[...,0]*=g; A[...,1]/=g   # transient directional gap-junction edit
            trans=sub.relax(dict(sub.ch, coupling=A), base.copy(), steps=ts)
            healed=sub.relax(sub.ch, trans.copy())                 # RESTORE coupling; genome untouched throughout
            c1=sub.klass(healed); trials+=1
            if c1!=c0: persist+=1; new_classes.add(c1)
    # matched-magnitude RULE edit (perturb alpha), genome-as-target control:
    rule_flip=0
    for g in (2,4,8):
        ch2=dict(sub.ch); ch2['rule']=sub.ch['rule']*g
        if sub.klass(sub.relax(sub.ch, sub.relax(ch2, base.copy(), steps=1000)))!=c0: rule_flip+=1
    return dict(genome_fixed=True, start_class=c0, coupling_leverage=persist/trials,
                persistent_new_classes=sorted(new_classes), n_trials=trials)

# ============================================================================
def run(sub, rho=0.05):
    rng=np.random.default_rng(7)
    seed_fixed=sub.seed(rng)
    base_state=sub.relax(sub.ch, seed_fixed)
    base_m=sub.morph(base_state)
    R=richness(base_m)
    s_rule = sensitivity(sub,'rule',rho,base_m,seed_fixed,rng)
    s_coup = sensitivity(sub,'coupling',rho,base_m,seed_fixed,rng)
    s_drive= sensitivity(sub,'drive',rho,base_m,seed_fixed,rng)
    s_state= sensitivity(sub,'state_init',rho,base_m,seed_fixed,rng)
    H, resid = healing(sub, rho, base_state, rng)
    dim_rule, dim_sel, G = generativity_G(sub, rho, seed_fixed, base_m, rng)
    lev = levin_coupling_test(sub, rng)
    print("="*74); print(sub.name); print("="*74)
    print(f"  realized richness (eff-dim of morphology)  = {R:5.2f}   {'[>trivial]' if R>2 else '[TRIVIAL]'}")
    print(f"  --- localization spectrum (attractor displacement per unit rho-perturbation) ---")
    print(f"    s_rule     (a) rule/genome     = {s_rule:6.3f}")
    print(f"    s_coupling (b) coupling/topo   = {s_coup:6.3f}")
    print(f"    s_state    (c) initial cond.   = {s_state:6.3f}")
    print(f"    s_drive    (d) drive/env       = {s_drive:6.3f}")
    print(f"    healing H  (c) runtime kick    = {H:6.3f}   (1=heals/attractor, 0=persists/stored)")
    ss=np.array([s_rule,s_coup,s_state,s_drive]); tot=ss.sum()+1e-9; L=ss/tot
    print(f"    normalized localization profile [rule,coup,state,drive] = "
          f"[{L[0]:.2f},{L[1]:.2f},{L[2]:.2f},{L[3]:.2f}]")
    print(f"  --- DL-free generativity ---")
    print(f"    dim_rule (structure rule commands) = {dim_rule:5.2f}")
    print(f"    dim_sel  (structure selection cmds)= {dim_sel:5.2f}")
    print(f"    G = dim_rule/dim_sel               = {G:6.2f}")
    if lev is not None:
        print(f"  --- Levin coupling test (GENOME/alpha FIXED throughout) ---")
        print(f"    start class {lev['start_class']}; transient coupling edits over {lev['n_trials']} settings")
        print(f"    coupling leverage (frac depositing a PERSISTENT new attractor) = {lev['coupling_leverage']:.2f}")
        print(f"    persistent new classes reached by coupling alone = {lev['persistent_new_classes']}")
    return dict(richness=R, s_rule=s_rule, s_coup=s_coup, s_state=s_state,
                s_drive=s_drive, healing=H, dim_rule=dim_rule, dim_sel=dim_sel, G=G, levin=lev)

if __name__=="__main__":
    import time; t=time.time()
    print("\nLOCALIZATION SPECTRUM  --  DL-free causal fingerprint\n")
    outs={}
    outs['Kuramoto']  = run(Kuramoto());     print()
    outs['StoredVec'] = run(StoredVector()); print()
    outs['Trivial']   = run(Trivial());      print()
    print("="*74); print("FINGERPRINT SUMMARY"); print("="*74)
    hdr=f"{'substrate':<34}{'rich':>6}{'s_rule':>8}{'s_state':>8}{'heal':>7}{'G':>8}"
    print(hdr); print("-"*len(hdr))
    for k,o in outs.items():
        print(f"{k:<34}{o['richness']:6.2f}{o['s_rule']:8.3f}{o['s_state']:8.3f}"
              f"{o['healing']:7.2f}{o['G']:8.2f}")
    print(f"\nruntime {time.time()-t:.0f}s")
