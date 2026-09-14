"""Machinery for the kappa-VK results: master identity, local bifurcation,
parity index, quartet symmetry and the K_2 closure.

This module is **additive**.  It imports the existing primitives from
:mod:`nhvk.core` rather than restating them, defines nothing that scripts 01-06
import, and changes no numerical constant, tolerance or algorithm used by them.
What it adds is the machinery the later sections of the paper need and
:mod:`nhvk.core` does not provide:

* :class:`QuinticModel` -- :class:`nhvk.core.Model` is cubic only,
  ``N(s) = sigma s``; Remark ``rem:M3real`` and Configuration IV of
  Table ``tab:K2`` need ``N(s) = sigma s + beta s^2``, for which ``a`` and ``b``
  in the doubled operator pick up the extra terms;
* :func:`dphi_domega` -- ``d phi / d omega`` from an exact bordered linear
  solve rather than a finite difference, which is what makes ``kappa`` accurate
  enough for the ``1e-14`` agreement of Theorem ``thm:targetB``;
* :func:`newton_robust` -- the symmetric bordering of
  :meth:`nhvk.core.Model.newton` satisfies ``<ker J^dagger, w> = 2 i Q(omega)``
  and is therefore *exactly* singular at the zeros of ``Q``; retrying with fixed
  pseudo-random column borders lets a continuation pass such a point.  The
  iteration is otherwise identical to the core solver, including its ``1e-12``
  early-exit and ``1e-8`` acceptance thresholds;
* :func:`contour_block` -- a Beyn projection onto the spectral subspace near
  ``lambda = 0``.  At the ``J_4`` point of Theorem ``thm:partI`` the individual
  eigenvalues are accurate only to ``eps^{1/4}``, while
  ``nu = (1/2) tr(L_Gamma^2)`` from the compressed block is accurate to
  ``~1e-10``;
* the two mode sets used by the later sections, which are **not**
  interchangeable -- see :func:`gap_modes` and :func:`enclosed_modes`.

Only NumPy is required.  The two routines that need pseudo-random data
(:func:`contour_block` and the least-squares system of script 10) take an
explicit integer seed, which the calling scripts record in their JSON sidecars,
so every result is deterministic.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import eig, lstsq, norm, solve, svd

from .core import Model, adj_kernel, ip, make_grid, real_ray

__all__ = [
    "QuinticModel",
    "first_derivative_matrix",
    "newton_robust",
    "dphi_domega",
    "Lplus",
    "Lminus",
    "chi0_at",
    "kappa_at",
    "jordan_chain",
    "gap_modes",
    "enclosed_modes",
    "sym_defect",
    "classify_gap",
    "n_unstable",
    "contour_block",
    "nu_from_block",
    "secant",
    "free_seed",
    "build_wadati",
    "move_omega",
    "Branch",
]


# ------------------------------------------------------------------ operators
class QuinticModel(Model):
    """:class:`nhvk.core.Model` with ``N(s) = sigma s + beta s^2``.

    ``beta = 0`` reproduces the base class exactly, term by term, so a
    ``QuinticModel(..., beta=0.0)`` and a ``Model(...)`` built on the same data
    return bit-identical residuals, Jacobians and doubled operators.
    """

    def __init__(self, V, g, sigma, x, D2, dx, beta: float = 0.0) -> None:
        super().__init__(V, g, sigma, x, D2, dx)
        self.beta = beta

    def _N(self, s):
        return self.sigma*s + self.beta*s**2

    def _Np(self, s):
        return self.sigma + 2*self.beta*s

    def res(self, u, v, om, eps):
        s = u**2 + v**2
        Nn = self._N(s)
        r1 = self.D@u + Nn*u - eps*self.g*v - om*u
        r2 = self.D@v + Nn*v + eps*self.g*u - om*v
        return np.concatenate([r1, r2])

    def jac(self, u, v, om, eps):
        s = u**2 + v**2
        Nn, Npv = self._N(s), self._Np(s)
        J11 = self.D + np.diag(Nn + 2*Npv*u**2 - om)
        J12 = np.diag(2*Npv*u*v - eps*self.g)
        J21 = np.diag(2*Npv*u*v + eps*self.g)
        J22 = self.D + np.diag(Nn + 2*Npv*v**2 - om)
        return np.block([[J11, J12], [J21, J22]])

    def Lop(self, phi, om, eps):
        s = np.abs(phi)**2
        a = self._N(s) + self._Np(s)*s
        b = self._Np(s)*phi**2
        A11 = self.D + np.diag(1j*eps*self.g - om + a)
        A12 = np.diag(b)
        A21 = np.diag(-np.conj(b))
        A22 = -(self.D + np.diag(-1j*eps*self.g - om + a))
        return np.block([[A11, A12], [A21, A22]])


def first_derivative_matrix(L: float, N: int) -> np.ndarray:
    """Dense spectral ``d/dx`` on the grid of :func:`nhvk.core.make_grid`."""
    k = np.fft.fftfreq(N, d=2*L/N)*2*np.pi
    F = np.fft.fft(np.eye(N), axis=0)
    return np.real(np.fft.ifft(1j*k[:, None]*F, axis=0))


def Lplus(M, phi0: np.ndarray, om: float) -> np.ndarray:
    """Hermitian-limit amplitude operator ``L_+`` (``phi_0`` real, ``G = 0``)."""
    s = phi0**2
    Nn = M._N(s) if isinstance(M, QuinticModel) else M.sigma*s
    Npv = M._Np(s) if isinstance(M, QuinticModel) else M.sigma*np.ones_like(s)
    return -M.D2 + np.diag(M.V + Nn + 2*Npv*s - om)


def Lminus(M, phi0: np.ndarray, om: float) -> np.ndarray:
    """Hermitian-limit phase operator ``L_-`` (``phi_0`` real, ``G = 0``)."""
    s = phi0**2
    Nn = M._N(s) if isinstance(M, QuinticModel) else M.sigma*s
    return -M.D2 + np.diag(M.V + Nn - om)


# ------------------------------------------------------------------ solvers
def newton_robust(M, phi0, om, eps, tol=1e-12, itmax=60, bcol=None,
                  robust=True, seed=11):
    """:meth:`nhvk.core.Model.newton` with a replaceable column border.

    The bordered matrix is ``[[J, c], [w^T, 0]]`` with ``w`` the unit U(1) orbit
    tangent.  With the symmetric choice ``c = w`` one has
    ``<ker J^dagger, w> = 2 i Q(omega)``, so the solve is exactly singular where
    ``Q`` vanishes.  ``robust=True`` retries with four fixed pseudo-random
    column borders, for which that obstruction is generically absent.  All
    tolerances, the iteration cap and the acceptance threshold are those of the
    core solver.
    """
    def attempt(col):
        u, v = phi0.real.copy(), phi0.imag.copy()
        n = M.n
        for _ in range(itmax):
            r = M.res(u, v, om, eps)
            if norm(r)*np.sqrt(M.dx) < tol:
                break
            J = M.jac(u, v, om, eps)
            w = np.concatenate([-v, u])
            nw = norm(w)
            if nw < 1e-14:
                return None
            w = w/nw
            B = np.zeros((2*n+1, 2*n+1))
            B[:2*n, :2*n] = J
            B[:2*n, 2*n] = w if col is None else col
            B[2*n, :2*n] = w
            try:
                d = np.linalg.solve(B, np.concatenate([-r, [0.0]]))
            except np.linalg.LinAlgError:
                return None
            if not np.all(np.isfinite(d)):
                return None
            u = u + d[:n]
            v = v + d[n:2*n]
        if norm(M.res(u, v, om, eps))*np.sqrt(M.dx) > 1e-8:
            return None
        return u + 1j*v

    p = attempt(bcol)
    if p is not None or not robust:
        return p
    rng = np.random.default_rng(seed)
    for _ in range(4):
        c = rng.standard_normal(2*M.n)
        c /= norm(c)
        p = attempt(c)
        if p is not None:
            return p
    return None


def dphi_domega(M, phi, om, eps):
    """``d phi / d omega`` from an exact bordered solve.

    Differentiating the stationary equation in ``omega`` gives
    ``J y = (u, v)``; the U(1) direction is removed by the same border as in
    the Newton solve, which is legitimate because ``kappa`` is insensitive to
    adding a multiple of ``w_0`` (Lemma ``lem:persistent``).
    """
    n = M.n
    u, v = phi.real, phi.imag
    J = M.jac(u, v, om, eps)
    w = np.concatenate([-v, u])
    w = w/norm(w)
    B = np.zeros((2*n+1, 2*n+1))
    B[:2*n, :2*n] = J
    B[:2*n, 2*n] = w
    B[2*n, :2*n] = w
    y = solve(B, np.concatenate([np.concatenate([u, v]), [0.0]]))
    return y[:n] + 1j*y[n:2*n]


# ------------------------------------------------------------------ chi_0, kappa
def chi0_at(M, phi, om, dx, eps=1.0, chi_prev=None):
    """``chi_0`` in the H18 real-ray gauge with the H13(ii) normalisation.

    The residual Z2 sign is fixed by **continuity transport** when ``chi_prev``
    is given (``Re <chi_prev, chi> > 0``), and otherwise by the rule ``Q > 0``.
    The two agree wherever ``Q`` stays away from zero; where they differ the
    transport rule is the correct one, because the ``Q > 0`` rule flips
    spuriously at a zero of ``Q``.

    Returns ``(chi, Q, s_last, s_prev, |arg c|, L)``.
    """
    n = M.n
    Phi = np.concatenate([phi, np.conj(phi)])
    Lm = M.Lop(phi, om, eps)
    chi, s1, s2 = adj_kernel(Lm, dx)
    chi, cS = real_ray(chi, n, dx)
    chi = chi*np.sqrt(np.sum(np.abs(Phi)**2)*dx)/np.sqrt(np.sum(np.abs(chi)**2)*dx)
    Q = 0.5*ip(chi, Phi, dx)
    if chi_prev is None:
        if Q.real < 0:
            chi, Q = -chi, -Q
    elif np.real(ip(chi_prev, chi, dx)) < 0:
        chi, Q = -chi, -Q
    return chi, Q, s1, s2, abs(np.angle(cS)), Lm


def kappa_at(M, phi, om, dx, chi, eps=1.0):
    """``kappa = <chi_0, d_omega Phi>`` and ``P'(omega)`` alongside."""
    dphi = dphi_domega(M, phi, om, eps)
    dPhi = np.concatenate([dphi, np.conj(dphi)])
    return ip(chi, dPhi, dx), 2*np.real(np.sum(np.conj(phi)*dphi)*dx), dphi


def jordan_chain(M, phi, om, dx, chi, eps=1.0):
    """The chain ``L e_1 = w_0``, ``L e_2 = e_1``, ``L e_3 = e_2`` and its scalars.

    Returns ``(kappa, kappa_2, kappa_3, residuals)`` with
    ``kappa_j = <chi_0, e_j>``.  ``e_2`` and ``e_3`` are obtained by least
    squares, which returns the minimum-norm solution; ``kappa_3`` is unaffected
    by the chain freedom because ``kappa = kappa_2 = 0`` at a zero of ``kappa``.
    """
    w0 = np.concatenate([1j*phi, -1j*np.conj(phi)])
    Lm = M.Lop(phi, om, eps)
    dphi = dphi_domega(M, phi, om, eps)
    dPhi = np.concatenate([dphi, np.conj(dphi)])
    e1 = 1j*dPhi
    r1 = norm(Lm@e1 - w0)/norm(w0)
    e2 = lstsq(Lm, e1, rcond=None)[0]
    r2 = norm(Lm@e2 - e1)/norm(e1)
    e3 = lstsq(Lm, e2, rcond=None)[0]
    r3 = norm(Lm@e3 - e2)/norm(e2)
    return ip(chi, dPhi, dx), ip(chi, e2, dx), ip(chi, e3, dx), (r1, r2, r3)


# ------------------------------------------------------------------ mode sets
def gap_modes(Lm, om, tol_zero=1e-5):
    """Eigenvalues inside the **gap disk** ``|lambda| < 0.999 |omega|``.

    By Lemma E the essential spectrum is real and outside this disk, so it
    contains only genuine isolated point spectrum.  The disk is invariant under
    both ``lambda -> -conj(lambda)`` and ``lambda -> conj(lambda)``, so it can
    neither manufacture nor destroy either symmetry.  **This is the only mode
    set on which Corollary 1 / Corollary 2 defects may be measured**; a
    mass-fraction filter is not conjugation-symmetric, because the two members
    of a quartet decay at different rates.

    Returns ``(all_in_disk, nonzero_in_disk)``.
    """
    ev = np.linalg.eigvals(Lm)
    inb = ev[np.abs(ev) < abs(om)*0.999]
    return inb, inb[np.abs(inb) > tol_zero]


def enclosed_modes(Lm, om, x, Lx, core=6.0, locfrac=0.75, maxabs=3.0):
    """The discrete spectrum enclosed by the contour ``Gamma``.

    The gap disk, plus the strongly localised modes outside it whose imaginary
    part exceeds the finite-domain artefact floor ``3/(2 L_x)`` (the spurious
    imaginary part of the discretised essential spectrum decays like
    ``0.8/(2 L_x)``, measured).  **This is the set on which ``n_unstable`` is
    counted**, because as ``omega -> 0^-`` the gap disk shrinks and no longer
    encloses the whole discrete spectrum -- exactly the situation Theorem
    ``thm:parity`` excludes by hypothesis, and which must be detected rather
    than silently mis-counted.  The localisation filter is not
    conjugation-symmetric and must never be used for a symmetry test.

    Returns ``(enclosed, in_disk)``.
    """
    n = len(x)
    ev, Vc = eig(Lm)
    in_disk = np.abs(ev) < abs(om)*0.999
    floor = 3.0/(2.0*Lx)
    m = np.abs(x) < core
    extra = []
    for j in np.where(~in_disk)[0]:
        if abs(ev[j]) > maxabs or abs(ev[j].imag) < floor:
            continue
        w = np.abs(Vc[:, j])**2
        w = w[:n] + w[n:]
        if w.sum() > 0 and w[m].sum()/w.sum() > locfrac:
            extra.append(j)
    idx = np.concatenate([np.where(in_disk)[0], np.array(extra, dtype=int)])
    return ev[idx], ev[in_disk]


def sym_defect(ev, kind: str) -> float:
    """``max_lambda dist(lambda, image)``; ``kind='A'``: ``-conj``, ``'B'``: ``conj``."""
    if len(ev) == 0:
        return float("nan")
    tgt = -np.conj(ev) if kind == "A" else np.conj(ev)
    return float(max(np.min(np.abs(ev - t)) for t in tgt))


def n_unstable(ev, thresh=1e-4) -> int:
    """Number of eigenvalues with ``Im lambda > thresh``, with multiplicity."""
    return int(np.sum(ev.imag > thresh))


def classify_gap(nz, tol_im=1e-6):
    """``(real pairs, imaginary pairs, quartets)`` of a quartet-symmetric set."""
    real = nz[np.abs(nz.imag) < tol_im]
    imag = nz[(np.abs(nz.real) < tol_im) & (np.abs(nz.imag) >= tol_im)]
    cplx = nz[(np.abs(nz.real) >= tol_im) & (np.abs(nz.imag) >= tol_im)]
    return len(real)//2, len(imag)//2, len(cplx)//4


# ------------------------------------------------------------------ contour block
def contour_block(Lm, r=0.10, nq=48, m=8, tol=1e-9, seed=0):
    """Beyn projection onto the spectral subspace inside ``|lambda| < r``.

    Returns ``(eigenvalues, compressed matrix, probe singular values)``.
    Deterministic for a fixed ``seed``.
    """
    n = Lm.shape[0]
    rng = np.random.default_rng(seed)
    Y = rng.standard_normal((n, m)) + 1j*rng.standard_normal((n, m))
    S0 = np.zeros((n, m), dtype=complex)
    th = 2*np.pi*(np.arange(nq)+0.5)/nq
    I = np.eye(n)
    for t in th:
        z = r*np.exp(1j*t)
        S0 += (r*np.exp(1j*t)/nq)*solve(z*I - Lm, Y)
    U, S, _ = svd(S0, full_matrices=False)
    k = int(np.sum(S > tol*S[0]))
    if k == 0:
        return np.array([]), None, S
    U = U[:, :k]
    return np.linalg.eigvals(U.conj().T @ Lm @ U), U.conj().T @ Lm @ U, S


def nu_from_block(Lc) -> float:
    """``nu = (1/2) tr(Lc^2)``, i.e. ``lambda^2`` of the migrating pair.

    Real by Corollary 1 and analytic through the collision, where individual
    eigenvalues are accurate only to ``eps^{1/4}``.
    """
    if Lc is None:
        return float("nan")
    return float(np.real(np.trace(Lc @ Lc))/2.0)


# ------------------------------------------------------------------ continuation
def secant(f, a, b, tol=1e-12, itmax=40):
    """Plain secant iteration; returns ``(root, f(root))``."""
    fa, fb = f(a), f(b)
    for _ in range(itmax):
        if abs(fb - fa) < 1e-300:
            break
        c = b - fb*(b - a)/(fb - fa)
        if not np.isfinite(c):
            break
        fc = f(c)
        a, fa, b, fb = b, fb, c, fc
        if abs(b - a) < tol:
            break
    return b, fb


def free_seed(om: float, x: np.ndarray) -> np.ndarray:
    """The free focusing soliton at ``omega``, used as the continuation seed."""
    return np.sqrt(2*abs(om))/np.cosh(np.sqrt(abs(om))*x) + 0j


def build_wadati(profile, A, om0, x, D2, dx, sigma=-1.0, beta=0.0,
                 dA=0.005, dbeta=0.005):
    """Build a Wadati branch: ramp ``A`` from the free soliton, then ``beta``.

    The order matters.  At ``A = 0`` the free soliton is translation invariant,
    so the phase-bordered Jacobian has a two-dimensional kernel; ramping
    ``beta`` first fails for that reason, while ramping ``A`` first does not.
    ``profile`` follows the :mod:`nhvk.profiles` signature ``profile(x, A)``.
    """
    ph = free_seed(om0, x)
    Ac, st = 0.0, dA
    M = None
    while Ac < A - 1e-13:
        st = min(st, A - Ac)
        V, G = profile(x, Ac + st)
        M = QuinticModel(V, G, sigma, x, D2, dx, 0.0)
        p = newton_robust(M, ph, om0, 1.0)
        if p is None or np.sum(np.abs(p)**2)*dx < 1e-8:
            st /= 2
            if st < 1e-7:
                raise RuntimeError(f"A-continuation stalled at A = {Ac:.6f}")
            continue
        ph, Ac = p, Ac + st
        st = min(st*1.3, 0.02)
    b, stb = 0.0, dbeta
    while b < beta - 1e-13:
        stb = min(stb, beta - b)
        V, G = profile(x, A)
        M = QuinticModel(V, G, sigma, x, D2, dx, b + stb)
        p = newton_robust(M, ph, om0, 1.0)
        if p is None or np.sum(np.abs(p)**2)*dx < 1e-8:
            stb /= 2
            if stb < 1e-7:
                raise RuntimeError(f"beta-continuation stalled at beta = {b:.6f}")
            continue
        ph, b = p, b + stb
        stb *= 1.3
    V, G = profile(x, A)
    M = QuinticModel(V, G, sigma, x, D2, dx, beta)
    ph = newton_robust(M, ph, om0, 1.0)
    if ph is None:
        raise RuntimeError("final Newton polish failed")
    return M, ph


def move_omega(M, ph, om0, om1, dom=0.01, eps=1.0):
    """``omega``-continuation with a secant predictor; returns ``(phi, omega)``.

    Returns ``(None, omega_reached)`` if the step size has to be halved below
    ``1e-8`` without success, so that a caller can report where it stopped
    rather than silently returning a profile at the wrong frequency.
    """
    om, prev = om0, None
    while abs(om - om1) > 1e-12:
        st = np.sign(om1 - om)*min(dom, abs(om1 - om))
        p = None
        if prev is not None:
            p = newton_robust(M, ph + (ph - prev), om + st, eps)
        if p is None:
            p = newton_robust(M, ph, om + st, eps)
        if p is None:
            dom /= 2
            if dom < 1e-8:
                return None, om
            continue
        prev, ph, om = ph, p, om + st
    return ph, om


class Branch:
    """A profile that can be moved in ``omega``, for root finding."""

    def __init__(self, M, phi, om, dx, dom=0.01):
        self.M, self.phi, self.om, self.dx, self.dom = M, phi, om, dx, dom

    def at(self, om):
        if abs(om - self.om) > 1e-14:
            p, o = move_omega(self.M, self.phi, self.om, om, self.dom)
            if p is None:
                raise RuntimeError(f"continuation failed to omega = {om}")
            self.phi, self.om = p, o
        return self.phi

    def kappa(self, om):
        ph = self.at(om)
        chi, Q, _, _, _, _ = chi0_at(self.M, ph, om, self.dx)
        kap, Pp, _ = kappa_at(self.M, ph, om, self.dx, chi)
        return float(kap.real), float(Q.real), float(Pp)

    def nu(self, om, r=0.10, seed=0):
        ph = self.at(om)
        _, Lc, _ = contour_block(self.M.Lop(ph, om, 1.0), r=r, seed=seed)
        if Lc is None or Lc.shape[0] != 4:
            return float("nan")
        return nu_from_block(Lc)
