import numpy as np
from numpy.linalg import norm, eig
exec(open('/home/claude/krein2.py').read().split('# ======')[0])

L, N = 20.0, 240
x, D2, dx = make_grid(L, N)
sigma = -1.0

def profile(A):
    g  = A*x*np.exp(-x**2/2)
    gp = A*(1-x**2)*np.exp(-x**2/2)
    return -g**2, gp

def model(A):
    V, G = profile(A)
    return Model(V, G, sigma, x, D2, dx)

def adaptive(M_of, ph, p0, p1, minstep=1e-4):
    """continue from parameter p0 to p1 with step halving; M_of(p) -> Model"""
    p, step = p0, (p1-p0)/12
    while abs(p - p1) > 1e-12:
        if abs(step) > abs(p1-p): step = p1-p
        pn = p + step
        M = M_of(pn)
        p2 = M.newton(ph, pn[1] if isinstance(pn, tuple) else OM, 1.0)
        if p2 is None:
            step /= 2
            if abs(step) < minstep: return None, None, p
            continue
        ph, p = p2, pn
        step *= 1.3
    return M_of(p1), ph, p1

# --- stage 1: build the branch at omega0 by A-continuation ---
OM = -0.5
A_target = 1.0
seed = np.sqrt(2*abs(OM))/np.cosh(np.sqrt(abs(OM))*x)+0j
ph, Acur, step = seed, 0.0, 0.05
while Acur < A_target - 1e-12:
    step = min(step, A_target-Acur)
    M = model(Acur+step)
    p2 = M.newton(ph, OM, 1.0)
    if p2 is None:
        step /= 2
        if step < 1e-5: break
        continue
    ph, Acur = p2, Acur+step
    step *= 1.3
print(f"A-continuation at omega={OM} reached A={Acur:.4f}")
A = Acur
M = model(A)

def loc_eigs(Lm, N, xg, maxabs=60.0, locfrac=0.85):
    ev, Vc = eig(Lm)
    keep = []
    core = np.abs(xg) < 6.0
    for j in range(len(ev)):
        if abs(ev[j]) > maxabs: continue
        w = np.abs(Vc[:, j])**2
        w = w[:N] + w[N:]
        if w.sum() <= 0: continue
        if w[core].sum()/w.sum() < locfrac: continue
        keep.append(j)
    return ev[keep], Vc[:, keep]

def observe(M, ph, om):
    Phi = np.concatenate([ph, np.conj(ph)])
    Lm = M.Lop(ph, om, 1.0)
    ch, s1, s2 = adj_kernel(Lm, dx)
    ch = gaugefix(ch, Phi, dx)
    ch, c = real_ray(ch, N, dx)
    ch = ch*np.sqrt(np.sum(np.abs(Phi)**2)*dx)/np.sqrt(np.sum(np.abs(ch)**2)*dx)
    Q = 0.5*ip(ch, Phi, dx)
    if Q.real < 0: Q = -Q
    ev, _ = loc_eigs(Lm, N, x)
    if len(ev):
        j = np.argmax(ev.imag); lam = ev[j]
    else:
        lam = 0j
    P = np.sum(np.abs(ph)**2)*dx
    return P, Q, lam, np.angle(c), s1, s2

# --- stage 2: omega-continuation in both directions at fixed A ---
res = {}
for direction, lo, hi in [("down", OM, -2.6), ("up", OM, -0.15)]:
    p = ph.copy(); om = OM
    stp = -0.02 if direction == "down" else 0.02
    while (om > lo if direction=="down" else om < hi):
        om_new = om + stp
        p2 = M.newton(p, om_new, 1.0)
        if p2 is None:
            stp /= 2
            if abs(stp) < 1e-4: break
            continue
        p, om = p2, om_new
        stp *= 1.2
        stp = np.sign(stp)*min(abs(stp), 0.02)
        if abs(round(om,3)*100 % 5) < 1e-6 or True:
            res[round(om, 4)] = observe(M, p, om)
    print(f"  {direction}: stopped at omega = {om:.4f}")

oms = sorted(res)
print(f"\nA = {A:.3f};  {len(oms)} omega points from {oms[0]:.3f} to {oms[-1]:.3f}")
print(f"{'omega':>8} {'P':>9} {'Q':>9} {'ImQ/|Q|':>9} {'maxIm lam':>10} {'Re lam':>9} {'gap':>9}")
sel = oms[::max(1, len(oms)//24)]
for o in sel:
    P, Q, lam, argc, s1, s2 = res[o]
    print(f"{o:8.3f} {P:9.5f} {Q.real:9.5f} {Q.imag/abs(Q):9.1e} "
          f"{lam.imag:10.5f} {lam.real:9.4f} {s2:9.2e}")

# find sign changes of Q' and P'
Qs = np.array([res[o][1].real for o in oms]); Ps = np.array([res[o][0] for o in oms])
Ims = np.array([res[o][2].imag for o in oms]); O = np.array(oms)
dQ = np.gradient(Qs, O); dP = np.gradient(Ps, O)
print("\nsign changes:")
for nm, d in [("Q'", dQ), ("P'", dP)]:
    s = np.sign(d); idx = np.where(np.diff(s) != 0)[0]
    for i in idx:
        print(f"   {nm} changes sign near omega = {0.5*(O[i]+O[i+1]):.4f}"
              f"   (maxIm lambda there = {Ims[i]:+.5f})")
print(f"\nmax Im(lambda) over the whole branch: {Ims.max():+.5f}; min: {Ims.min():+.5f}")
np.save('/home/claude/sweepdata.npy',
        np.array([O, Qs, Ps, Ims, [res[o][2].real for o in oms]]))
