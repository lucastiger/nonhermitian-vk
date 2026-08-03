"""
Gauge monodromy / amplitude sweep (Section 5.3).

Tracks the biorthogonal quasi-power

    Q(omega) = <<chi_0, Phi>> = (1/2) <chi_0, Phi>                       (Eq. 7)

as the gain-loss amplitude A is increased from 0 to 0.5 at fixed omega, with
chi_0 obtained by NAIVE CONTINUITY TRACKING of the adjoint-kernel line from the
exact Hermitian anchor -- that is, with no re-application of the real-ray gauge
of Lemma 2.6 at any point along the path.  The question is whether Q stays real.

Base (as in Section 3.7):

    D  = -d_xx + 0.3 sech^2(x)      (V even)
    G0 = -A sech(x) tanh(x)         (odd  ->  PT-symmetric)
    N(s) = sigma s,  sigma = -1     (focusing cubic),   omega = -1

Conventions.  phi is fixed at every A on the PT-covariant ray
phi(-x) = conj(phi(x)), which is canonical for a PT-symmetric base and removes
the residual U(1) freedom of the stationary problem.  chi_0 is then transported
by maximising the overlap with its value at the previous step, and rescaled to
the canonical norm ||chi_0|| = ||Phi|| of H13(ii).  Any drift of arg Q is
therefore attributable to chi_0 alone, which is the point of the test.

Also reported is the S-eigenvalue c, defined by S chi_0 = c chi_0 (Lemma 2.6),
together with its cumulative unwrapped phase, since a monodromy obstruction is
precisely a nonzero winding of arg c around the path.

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
g = -np.tanh(x) / np.cosh(x)                  # odd -> PT-symmetric
om = -1.0
A_max = 0.5
dA = 0.005                                    # fine steps: phase tracking must
                                              # be unambiguous between steps
M = Model(V, g, sigma, x, D2, dx)


def reflect(f):
    """x -> -x on the periodic Fourier grid (index j -> (N-j) mod N)."""
    return np.roll(f[::-1], 1)


def pt_gauge(phi):
    """Rotate phi onto the PT-covariant ray, phi(-x) = conj(phi(x))."""
    z = np.sum(np.conj(phi) * np.conj(reflect(phi))) * dx
    return phi * np.exp(-0.5j * np.angle(z))


def S_eigenvalue(chi):
    """c with S chi = c chi; |c| = 1 (Lemma 2.6)."""
    Sc = S_map(chi, Ng)
    return np.sum(np.conj(chi) * Sc) * dx / (np.sum(np.abs(chi) ** 2) * dx)


# ------------------------------------------------ Hermitian anchor, A = 0
seed = np.sqrt(2 * abs(om)) / np.cosh(np.sqrt(abs(om)) * x) + 0j
ph = M.newton(seed, om, 0.0)
assert ph is not None, "Hermitian anchor failed"
ph = ph * np.exp(-1j * np.angle(ph[Ng // 2]))          # make phi real
Phi = np.concatenate([ph, np.conj(ph)])
chi_prev = Phi.copy()                                  # H12: chi_0(A=0) = Phi_0
P0 = np.sum(np.abs(ph) ** 2) * dx

print("=" * 78)
print("Section 5.3 -- drift of Q under naive continuity tracking of chi_0")
print(f"base: V = 0.3 sech^2(x),  G0 = -A sech(x)tanh(x),  omega = {om},"
      f"  sigma = {sigma}")
print(f"grid: Lx = {Lx}, N = {Ng};  step dA = {dA}")
print(f"Hermitian anchor: P(omega) = {P0:.6f}  (= Q at A = 0, Lemma 2.5)")
print("=" * 78)
print(f"{'A':>6} {'P':>10} {'Re Q':>10} {'Im Q':>11} {'Im Q/|Q|':>10} "
      f"{'K=P^2/Q^2':>10} {'arg c':>9} {'S-resid':>10}")

report_at = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
argc_unwrapped = 0.0
argc_prev = 0.0
rows = []

A_grid = np.round(np.arange(0.0, A_max + 1e-12, dA), 10)
for A in A_grid:
    if A > 0:
        p2 = M.newton(ph, om, A)
        assert p2 is not None, f"continuation failed at A = {A}"
        ph = pt_gauge(p2)
        Phi = np.concatenate([ph, np.conj(ph)])
        Lm = M.Lop(ph, om, A)
        v, sv, gap = adj_kernel(Lm, dx)
        # ---- naive continuity transport: maximise overlap with previous chi_0
        z = np.sum(np.conj(chi_prev) * v) * dx
        v = v * np.exp(-1j * np.angle(z))
        # ---- canonical norm ||chi_0|| = ||Phi||  (H13(ii))
        v = v * np.sqrt(np.sum(np.abs(Phi) ** 2) * dx) / np.sqrt(
            np.sum(np.abs(v) ** 2) * dx)
        chi_prev = v
    else:
        sv, gap = 0.0, np.nan

    Q = 0.5 * ip(chi_prev, Phi, dx)
    P = np.sum(np.abs(ph) ** 2) * dx
    c = S_eigenvalue(chi_prev)
    a = np.angle(c)
    if A > 0:
        d = a - argc_prev
        d = (d + np.pi) % (2 * np.pi) - np.pi          # shortest step
        argc_unwrapped += d
    argc_prev = a

    if any(abs(A - r) < 1e-9 for r in report_at):
        K = P ** 2 / Q.real ** 2
        Sres = (np.linalg.norm(S_map(chi_prev, Ng) - chi_prev)
                / np.linalg.norm(chi_prev))
        rows.append((A, P, Q.real, Q.imag, Q.imag / abs(Q), K, a, Sres))
        print(f"{A:6.2f} {P:10.6f} {Q.real:10.6f} {Q.imag:11.2e} "
              f"{Q.imag/abs(Q):10.2e} {K:10.6f} {a:9.2e} {Sres:10.2e}")

print()
print(f"Total unwrapped drift of arg c over 0 <= A <= {A_max}: "
      f"{argc_unwrapped:+.3e} rad")
print()
print("Interpretation.  Phi = (phi, conj phi) always satisfies S Phi = Phi, and")
print("if S a = a and S b = b then <a,b> = int 2 Re(conj(a1) b1) dx is REAL.")
print("Hence the maximal-overlap transport phase is 0 or pi, the transported")
print("chi_0 stays on the real S-ray, and Q stays real along the whole path.")
print("The residual global freedom is therefore a SIGN (Z_2), not a phase (U(1)),")
print("and K = ||Phi||^2 ||chi_0||^2 / Q^2 is blind to it because it involves Q^2.")
