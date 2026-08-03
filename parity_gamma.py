"""
Parity obstruction (Proposition 2, Section 3.7).

Computes the exceptional-point splitting coefficient

    gamma = (-i / kappa(omega)) * < chi_0 , i h(x) w_0 >                (Eq. 13-14)

for even, odd and generic (neither-parity) perturbation directions h, on the
PT-symmetric base

    D  = -d_xx + 0.3 sech^2(x)      (V even)
    G0 = -A sech(x) tanh(x)         (odd  ->  PT-symmetric),  A = 0.3
    N(s) = sigma s,  sigma = -1     (focusing cubic)

at omega in {-0.8, -1.5, -2.5}.

Proposition 2 asserts gamma == 0 identically for every EVEN h.

Note on well-posedness of the reported quantity.  Both the bracket
B = <chi_0, i h w_0> and kappa = <chi_0, d_omega Phi_0> are linear in chi_0, so
the ratio |gamma| = |B| / |kappa| is independent of the normalisation of chi_0.
It is likewise invariant under the residual U(1) phase of phi_0, because chi_0
and w_0 both carry the same Lambda = diag(e^{i th}, e^{-i th}) factor
(Lemma 2.4).  |gamma| is therefore a gauge-invariant number and needs no gauge
fixing; the gauges are nevertheless imposed below so that the parity structure
of Im(conj(eta_0) phi_0) can be checked explicitly.

Normalisation of h: each test direction is scaled to sup-norm 1, so that the
three magnitudes are directly comparable.  This choice is a convention and is
stated in the paper alongside the numbers.

Depends on the shared machinery at the top of krein2.py.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from numpy.linalg import norm

exec(open('/home/claude/krein2.py').read().split('# ======')[0])

# ------------------------------------------------------------------ setup
Lx, Ng = 20.0, 240
x, D2, dx = make_grid(Lx, Ng)
sigma = -1.0
V = 0.3 / np.cosh(x) ** 2                     # even
g = -np.tanh(x) / np.cosh(x)                  # odd  -> PT-symmetric
A = 0.3
hh = 2e-3                                     # finite-difference step in omega

M = Model(V, g, sigma, x, D2, dx)


def reflect(f):
    """x -> -x on the periodic Fourier grid x_j = -Lx + 2 Lx j / N.

    The reflection maps index j to (N - j) mod N, i.e. np.roll(f[::-1], 1);
    the naive f[::-1] is off by one grid spacing.
    """
    return np.roll(f[::-1], 1)


def pt_gauge(phi):
    """Rotate phi onto the PT-covariant ray, phi(-x) = conj(phi(x))."""
    z = np.sum(np.conj(phi) * np.conj(reflect(phi))) * dx
    return phi * np.exp(-0.5j * np.angle(z))


def solve_base(om, A_target, nsteps=30):
    """Free-soliton seed, then continuation in the gain-loss amplitude."""
    seed = np.sqrt(2 * abs(om)) / np.cosh(np.sqrt(abs(om)) * x) + 0j
    ph = M.newton(seed, om, 0.0)
    if ph is None:
        return None
    for e in np.linspace(0, A_target, nsteps + 1)[1:]:
        p2 = M.newton(ph, om, e)
        if p2 is None:
            return None
        ph = p2
    return pt_gauge(ph)


# test perturbation directions, each scaled to sup-norm 1
def test_directions(xg):
    h_even = 1.0 / np.cosh(xg) ** 2
    h_odd = np.tanh(xg) / np.cosh(xg)
    h_gen = 1.0 / np.cosh(xg - 1.0) ** 2          # neither even nor odd
    out = {}
    for name, h in [("even   h=sech^2(x)", h_even),
                    ("odd    h=sech(x)tanh(x)", h_odd),
                    ("generic h=sech^2(x-1)", h_gen)]:
        out[name] = h / np.max(np.abs(h))
    return out


print("=" * 78)
print("Proposition 2 -- parity obstruction:  |gamma| = |<chi_0, i h w_0>| / |kappa|")
print(f"base: V = 0.3 sech^2(x),  G0 = -A sech(x)tanh(x),  A = {A},  sigma = {sigma}")
print(f"grid: Lx = {Lx}, N = {Ng}")
print("=" * 78)
print(f"{'omega':>7} {'kernel sv':>11} {'gap':>9} {'|kappa|':>10} "
      f"{'parity resid':>13} {'direction':>26} {'|gamma|':>12}")

rows = []
for om in [-0.8, -1.5, -2.5]:
    ph = solve_base(om, A)
    assert ph is not None, f"base continuation failed at omega={om}"

    Phi = np.concatenate([ph, np.conj(ph)])
    w0 = np.concatenate([1j * ph, -1j * np.conj(ph)])
    Lm = M.Lop(ph, om, A)

    chi0, sv_kernel, sv_gap = adj_kernel(Lm, dx)
    chi0 = gaugefix(chi0, Phi, dx)
    chi0, _ = real_ray(chi0, Ng, dx)           # H18 local reality gauge

    # kappa = <chi_0, d_omega Phi>, unrescaled pairing (Eq. 12)
    php = M.newton(ph, om + hh, A)
    phm = M.newton(ph, om - hh, A)
    php = pt_gauge(php)
    phm = pt_gauge(phm)
    dphi = (php - phm) / (2 * hh)
    dPhi = np.concatenate([dphi, np.conj(dphi)])
    kappa = ip(chi0, dPhi, dx)

    # explicit check of the parity structure used in the proof:
    # Im(conj(eta_0) phi_0) must be an odd function of x
    eta0 = chi0[:Ng]
    prod_im = np.imag(np.conj(eta0) * ph)
    parity_resid = (np.max(np.abs(prod_im + reflect(prod_im)))
                    / np.max(np.abs(prod_im)))

    for name, h in test_directions(x).items():
        hw0 = np.concatenate([1j * h * w0[:Ng], 1j * h * w0[Ng:]])
        B = ip(chi0, hw0, dx)
        gam = -1j * B / kappa
        rows.append((om, name, abs(gam)))
        tag = (f"{om:7.2f} {sv_kernel:11.2e} {sv_gap:9.2e} {abs(kappa):10.5f} "
               f"{parity_resid:13.2e}") if name.startswith("even") else " " * 53
        print(f"{tag} {name:>26} {abs(gam):12.4e}")

print()
print("Summary  (Proposition 2 predicts |gamma| = 0 exactly for even h):")
print(f"{'direction':>26} " + "".join(f"{f'omega={o}':>16}" for o in [-0.8, -1.5, -2.5]))
for name in test_directions(x):
    vals = [v for (o, n, v) in rows if n == name]
    print(f"{name:>26} " + "".join(f"{v:16.4e}" for v in vals))
