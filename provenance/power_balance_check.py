"""
Admissibility / power-balance checks (Lemma P and hypothesis H5a).

Reconstruction of the checks that were originally run inline. Verifies:

 (1) Lemma P on a candidate profile: any decaying stationary state must satisfy
     int G |phi|^2 dx = 0, so a sign-definite G admits no stationary state.
     Demonstrated on G = A sech^2(x), for which the power-balance defect
     evaluated on the free soliton is O(1), not small.

 (2) The Wadati factorisation used in H5a, checked symbolically:
        -d_xx - g^2 + i g' = (d_x + i g)(-d_x + i g).

 (3) That the sign convention matters: with an admissible (odd, non-monotone)
     g the family exists for V = -g^2 and fails to exist for V = +g^2.

 (4) Lemma P along the converged Theorem A branch: int G |phi|^2 is satisfied
     automatically to ~1e-12 without ever being imposed.

Depends on the shared machinery at the top of krein2.py.
"""
import numpy as np
from numpy.linalg import norm
exec(open('/home/claude/krein2.py').read().split('# ======')[0])

L, N = 22.0, 300
x, D2, dx = make_grid(L, N)
om, sigma = -0.5, -1.0
seed = np.sqrt(2*abs(om))/np.cosh(np.sqrt(abs(om))*x) + 0j

print("=" * 74)
print("(1) Lemma P: a sign-definite G cannot support a stationary state")
print("=" * 74)
print("Power-balance defect int G|phi|^2 dx on the free soliton, G = A sech^2:")
for A in [0.05, 0.2, 0.8]:
    G = A / np.cosh(x) ** 2
    print(f"    A = {A:<5}  int G|phi|^2 = {np.sum(G*np.abs(seed)**2)*dx:.6f}"
          f"   (must be 0)")

print()
print("=" * 74)
print("(2) Symbolic check of the Wadati factorisation")
print("=" * 74)
import sympy as sp
xx = sp.Symbol('x')
f = sp.Function('f')
gg = sp.Function('g')
lhs = -sp.diff(f(xx), xx, 2) - gg(xx)**2*f(xx) + sp.I*sp.diff(gg(xx), xx)*f(xx)
inner = -sp.diff(f(xx), xx) + sp.I*gg(xx)*f(xx)
rhs = sp.diff(inner, xx) + sp.I*gg(xx)*inner
print("    (-d_xx - g^2 + i g') f  -  (d_x + i g)(-d_x + i g) f  simplifies to:",
      sp.simplify(sp.expand(lhs - rhs)))

print()
print("=" * 74)
print("(3) Sign convention: g odd and NON-monotone, so that g' changes sign")
print("=" * 74)
g_ = lambda A: A*x*np.exp(-x**2/2)
gp_ = lambda A: A*(1-x**2)*np.exp(-x**2/2)
for sV, label in [(-1, "V = -g^2  (Wadati factorisable form)"),
                  (+1, "V = +g^2  (non-factorisable)")]:
    ph, ok = seed, True
    for Av in np.linspace(0, 0.6, 25)[1:]:
        M = Model(sV*g_(Av)**2, gp_(Av), sigma, x, D2, dx)
        p2 = M.newton(ph, om, 1.0)
        if p2 is None:
            ok = False
            break
        ph = p2
    if ok:
        r = norm(M.res(ph.real, ph.imag, om, 1.0))*np.sqrt(dx)
        print(f"    {label}: family EXISTS, residual = {r:.1e}")
    else:
        print(f"    {label}: FAILS at A = {Av:.3f}")

print()
print("=" * 74)
print("(4) Lemma P along the Theorem A branch (never imposed, only checked)")
print("=" * 74)
A_target = 1.0
ph = seed
for Av in np.linspace(0, A_target, 41)[1:]:
    M = Model(-g_(Av)**2, gp_(Av), sigma, x, D2, dx)
    p2 = M.newton(ph, om, 1.0)
    assert p2 is not None, f"continuation failed at A = {Av}"
    ph = p2
for omv in [-0.30, -0.26, -0.24, -0.22, -0.20, -0.18]:
    p = ph
    st = np.sign(omv - om)*0.01
    o = om
    while abs(o - omv) > 1e-9:
        if abs(st) > abs(omv - o):
            st = omv - o
        p2 = M.newton(p, o + st, 1.0)
        if p2 is None:
            st /= 2
            continue
        p, o = p2, o + st
    pb = np.sum(gp_(A_target)*np.abs(p)**2)*dx
    print(f"    omega = {omv:6.2f}   int G|phi|^2 = {pb:+.3e}")
