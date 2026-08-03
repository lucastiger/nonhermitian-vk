import numpy as np
from numpy.linalg import norm, eig
exec(open('/home/claude/krein2.py').read().split('# ======')[0])
sigma = -1.0

def build(Lx, Ng):
    x, D2, dx = make_grid(Lx, Ng)
    return x, D2, dx

def model(A, x, D2, dx):
    g  = A*x*np.exp(-x**2/2)
    gp = A*(1-x**2)*np.exp(-x**2/2)
    return Model(-g**2, gp, sigma, x, D2, dx)

def branch(A, om_target, Lx, Ng, om_start=-0.5):
    x, D2, dx = build(Lx, Ng)
    seed = np.sqrt(2*abs(om_start))/np.cosh(np.sqrt(abs(om_start))*x)+0j
    ph, Ac, st = seed, 0.0, 0.05
    while Ac < A-1e-12:
        st = min(st, A-Ac)
        M = model(Ac+st, x, D2, dx)
        p2 = M.newton(ph, om_start, 1.0)
        if p2 is None:
            st /= 2
            if st < 1e-6: return None
            continue
        ph, Ac = p2, Ac+st; st *= 1.3
    M = model(A, x, D2, dx)
    om, st = om_start, np.sign(om_target-om_start)*0.01
    while abs(om-om_target) > 1e-9:
        if abs(st) > abs(om_target-om): st = om_target-om
        p2 = M.newton(ph, om+st, 1.0)
        if p2 is None:
            st /= 2
            if abs(st) < 1e-7: return None
            continue
        ph, om = p2, om+st
        st *= 1.2; st = np.sign(st)*min(abs(st), 0.01)
    return M, ph, x, dx, Ng

def analyse(M, ph, x, dx, Ng, om):
    Phi = np.concatenate([ph, np.conj(ph)])
    Lm  = M.Lop(ph, om, 1.0)
    ch, s1, s2 = adj_kernel(Lm, dx)
    ch = gaugefix(ch, Phi, dx); ch, c = real_ray(ch, Ng, dx)
    ch = ch*np.sqrt(np.sum(np.abs(Phi)**2)*dx)/np.sqrt(np.sum(np.abs(ch)**2)*dx)
    Q = 0.5*ip(ch, Phi, dx)
    if Q.real < 0: ch, Q = -ch, -Q
    P = np.sum(np.abs(ph)**2)*dx
    ev, Vc = eig(Lm)
    core = np.abs(x) < 6.0
    best, lam = None, 0j
    for j in range(len(ev)):
        if abs(ev[j]) > 40: continue
        w = np.abs(Vc[:, j])**2; w = w[:Ng]+w[Ng:]
        if w.sum() <= 0 or w[core].sum()/w.sum() < 0.85: continue
        if ev[j].imag > lam.imag: lam, best = ev[j], j
    Km = np.nan
    if best is not None:
        evL, VL = eig(Lm.conj().T)
        jj = np.argmin(np.abs(np.conj(evL)-lam))
        vR, vL = Vc[:, best], VL[:, jj]
        Km = (np.sum(np.abs(vR)**2)*np.sum(np.abs(vL)**2)
              / np.abs(np.sum(np.conj(vL)*vR))**2)
    # Corollary A check on the localized part of the spectrum
    small = ev[np.abs(ev) < 40]
    d1 = max(np.min(np.abs(small - m)) for m in -np.conj(small)) if len(small) else np.nan
    return P, Q.real, lam, Km, s2, d1

A = 1.0
print("Convergence check: replacement counterexample, g = A x exp(-x^2/2), A = 1")
print(f"{'omega':>7} {'Lx':>5} {'N':>5} {'P':>9} {'Q':>9} {'Im lam':>10} "
      f"{'Re lam':>9} {'K_mode':>8} {'CorA':>9}")
store = {}
for om in [-0.30, -0.26, -0.24, -0.22, -0.20, -0.18]:
    for (Lx, Ng) in [(18.0, 220), (22.0, 260), (22.0, 320)]:
        out = branch(A, om, Lx, Ng)
        if out is None:
            print(f"{om:7.2f} {Lx:5.1f} {Ng:5d}    branch failed"); continue
        M, ph, x, dx, _ = out
        P, Q, lam, Km, s2, d1 = analyse(M, ph, x, dx, Ng, om)
        store[(om, Lx, Ng)] = (P, Q, lam)
        print(f"{om:7.2f} {Lx:5.1f} {Ng:5d} {P:9.5f} {Q:9.5f} {lam.imag:10.5f} "
              f"{lam.real:9.4f} {Km:8.3f} {d1:9.1e}")
np.save('/home/claude/validate.npy', np.array(
    [[k[0], k[1], k[2], v[0], v[1], v[2].real, v[2].imag] for k, v in store.items()]))
