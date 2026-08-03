"""Shared machinery for the non-Hermitian Vakhitov--Kolokolov calculations.

Paper convention:  i psi_t = D psi + N(|psi|^2) psi + i G(x) psi,
                   D = -d_xx + V,  N(s) = sigma s,  G = eps * g.
CP convention:     i psi_t = -psi_xx + (V + i gamma W) psi - g_cp |psi|^2 psi
                   => sigma = -g_cp,  G = gamma W,  mu = omega.

This module is a verbatim-behaviour extraction of the machinery defined at the
top of ``provenance/krein2.py``, together with the two helpers (``reflect``,
``pt_gauge``) that were duplicated in ``provenance/parity_gamma.py`` and
``provenance/gauge_monodromy.py``.  No numerical constant, tolerance, iteration
cap or algorithm has been changed.

Only NumPy is required.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm

__all__ = [
    "make_grid",
    "Model",
    "ip",
    "ip_cp",
    "adj_kernel",
    "S_map",
    "real_ray",
    "gaugefix",
    "reflect",
    "pt_gauge",
    "S_eigenvalue",
    "transport",
]


# ---------- grid ----------
def make_grid(L: float, N: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Uniform Fourier grid on the periodic interval ``[-L, L)``.

    The grid points are ``x_j = -L + 2 L j / N``, ``j = 0, ..., N-1``.  The
    second-derivative operator is built spectrally and returned as a dense
    ``(N, N)`` matrix, so that the doubled linearisation ``L(omega)`` of
    equation (4) can be assembled and diagonalised directly.

    Parameters
    ----------
    L : float
        Half-width of the computational domain.
    N : int
        Number of grid points.

    Returns
    -------
    x : ndarray, shape (N,)
        Grid points.
    D2 : ndarray, shape (N, N)
        Dense spectral matrix representing ``d^2/dx^2``.
    dx : float
        Grid spacing ``2 L / N``.
    """
    x = -L + 2*L*np.arange(N)/N
    k = np.fft.fftfreq(N, d=2*L/N)*2*np.pi
    F = np.fft.fft(np.eye(N), axis=0)
    D2 = np.real(np.fft.ifft(-(k**2)[:, None]*F, axis=0))
    return x, D2, x[1]-x[0]


class Model:
    """Stationary problem and doubled linearisation for one ``(V, g, sigma)``.

    The stationary equation solved by :meth:`newton` is

    ``D phi + sigma |phi|^2 phi + i eps g phi - omega phi = 0``,

    with ``D = -d_xx + V``.  The gain-loss profile enters as ``G = eps * g``, so
    ``eps`` is the continuation parameter used throughout the paper: ``eps = 0``
    is the Hermitian anchor and ``eps = 1`` restores the full profile when ``g``
    already carries its physical amplitude.

    Parameters
    ----------
    V : ndarray, shape (N,)
        External potential on the grid.
    g : ndarray, shape (N,)
        Gain-loss profile, multiplied by ``eps`` wherever it is used.
    sigma : float
        Sign/strength of the cubic nonlinearity, ``N(s) = sigma s``.
    x, D2, dx
        Grid data as returned by :func:`make_grid`.
    """

    def __init__(self, V: np.ndarray, g: np.ndarray, sigma: float,
                 x: np.ndarray, D2: np.ndarray, dx: float) -> None:
        self.V, self.g, self.sigma, self.x, self.D2, self.dx = V, g, sigma, x, D2, dx
        self.n = len(x)
        self.D = -D2 + np.diag(V)

    # ---- stationary residual in real coords ----
    def res(self, u: np.ndarray, v: np.ndarray, om: float, eps: float) -> np.ndarray:
        """Residual of the stationary equation in real coordinates.

        With ``phi = u + i v`` the returned vector is the concatenation of the
        real and imaginary parts of
        ``D phi + sigma|phi|^2 phi + i eps g phi - omega phi``.
        """
        s = self.sigma*(u**2+v**2)
        r1 = self.D@u + s*u - eps*self.g*v - om*u
        r2 = self.D@v + s*v + eps*self.g*u - om*v
        return np.concatenate([r1, r2])

    def jac(self, u: np.ndarray, v: np.ndarray, om: float, eps: float) -> np.ndarray:
        """Real ``(2N, 2N)`` Jacobian of :meth:`res` with respect to ``(u, v)``."""
        n, sg = self.n, self.sigma
        J11 = self.D + np.diag(sg*(3*u**2+v**2) - om)
        J12 = np.diag(2*sg*u*v - eps*self.g)
        J21 = np.diag(2*sg*u*v + eps*self.g)
        J22 = self.D + np.diag(sg*(u**2+3*v**2) - om)
        return np.block([[J11, J12], [J21, J22]])

    def newton(self, phi0: np.ndarray, om: float, eps: float,
               tol: float = 1e-12, itmax: int = 60) -> np.ndarray | None:
        """bordered Newton: kernel direction i*phi removed by a Lagrange multiplier

        The Jacobian is singular along the U(1) orbit, so the system is bordered
        by the unit tangent ``w = (-v, u)`` and solved as a ``(2N+1)``-square
        system.  Returns ``None`` if the linear solve fails, if the orbit
        tangent degenerates, or if the final residual (in the ``sqrt(dx)``-scaled
        norm) exceeds the acceptance threshold ``1e-8``.

        Parameters
        ----------
        phi0 : ndarray, shape (N,), complex
            Initial guess.
        om : float
            Frequency ``omega``.
        eps : float
            Gain-loss continuation parameter.
        tol : float, optional
            Residual norm at which the iteration stops early.
        itmax : int, optional
            Maximum number of Newton steps.

        Returns
        -------
        ndarray or None
            The converged complex profile, or ``None`` on failure.
        """
        u, v = phi0.real.copy(), phi0.imag.copy()
        n = self.n
        for _ in range(itmax):
            r = self.res(u, v, om, eps)
            if norm(r)*np.sqrt(self.dx) < tol:
                break
            J = self.jac(u, v, om, eps)
            w = np.concatenate([-v, u])            # tangent to the U(1) orbit
            nw = norm(w)
            if nw < 1e-14:
                return None
            w = w/nw
            B = np.zeros((2*n+1, 2*n+1))
            B[:2*n, :2*n] = J
            B[:2*n, 2*n] = w
            B[2*n, :2*n] = w
            rhs = np.concatenate([-r, [0.0]])
            try:
                d = np.linalg.solve(B, rhs)
            except np.linalg.LinAlgError:
                return None
            u = u + d[:n]; v = v + d[n:2*n]
        if norm(self.res(u, v, om, eps))*np.sqrt(self.dx) > 1e-8:
            return None
        return u + 1j*v

    def continue_eps(self, phi0: np.ndarray, om: float, eps_target: float,
                     nsteps: int = 25) -> np.ndarray | None:
        """Continue a solution from ``eps = 0`` to ``eps = eps_target``.

        Uses ``nsteps`` equal increments, each solved by :meth:`newton` from the
        previous profile.  Returns ``None`` if any step fails.
        """
        ph = phi0
        for e in np.linspace(0, eps_target, nsteps+1)[1:]:
            ph2 = self.newton(ph, om, e)
            if ph2 is None:
                return None
            ph = ph2
        return ph

    # ---- doubled operators ----
    def Lop(self, phi: np.ndarray, om: float, eps: float) -> np.ndarray:
        """The doubled linearisation ``L(omega)`` of equation (4).

        Acting on ``(p, q)`` with ``q`` playing the role of ``conj(p)``, this is
        the ``(2N, 2N)`` non-self-adjoint matrix whose kernel contains the phase
        mode ``w_0 = (i phi, -i conj(phi))``.
        """
        a = 2*self.sigma*np.abs(phi)**2
        b = self.sigma*phi**2
        A11 = self.D + np.diag(1j*eps*self.g - om + a)
        A12 = np.diag(b)
        A21 = np.diag(-np.conj(b))
        A22 = -(self.D + np.diag(-1j*eps*self.g - om + a))
        return np.block([[A11, A12], [A21, A22]])

    def Lcal(self, phi: np.ndarray, om: float, eps: float) -> np.ndarray:
        """The Chernyavsky--Pelinovsky operator ``Lcal``, their equation (8).

        Related to :meth:`Lop` by ``L = sigma_3 Lcal`` with
        ``sigma_3 = diag(I, -I)``; identity (C1) of Table F.1 checks exactly
        this.
        """
        a = 2*self.sigma*np.abs(phi)**2
        b = self.sigma*phi**2
        A11 = self.D + np.diag(1j*eps*self.g - om + a)
        A12 = np.diag(b)
        A21 = np.diag(np.conj(b))
        A22 = self.D + np.diag(-1j*eps*self.g - om + a)
        return np.block([[A11, A12], [A21, A22]])


def ip(f: np.ndarray, g: np.ndarray, dx: float) -> complex:
    """The paper's pairing ``<f, g> = int conj(f) g dx``.

    Conjugate-linear in the **first** slot.  This is the pairing in which the
    Krein quantity of Lemma 2.3 and the quasi-power ``Q`` of equation (7) are
    written.
    """
    return np.sum(np.conj(f)*g)*dx


def ip_cp(f: np.ndarray, g: np.ndarray, dx: float) -> complex:
    """The Chernyavsky--Pelinovsky pairing ``int f conj(g) dx``.

    Linear in the **first** slot, i.e. the complex conjugate convention of
    :func:`ip`.  Used only for the dictionary check (C6) of Table F.1.
    """
    return np.sum(f*np.conj(g))*dx


def adj_kernel(Lm: np.ndarray, dx: float) -> tuple[np.ndarray, float, float]:
    """Kernel vector of ``L^dagger``, i.e. the adjoint kernel ``chi_0``.

    Computed as the smallest right-singular vector of ``L^dagger``, normalised
    to unit ``L^2(dx)`` norm.  The two smallest singular values are returned as
    a numerical certificate that the kernel is simple: ``s_last`` should be at
    round-off and ``s_prev`` should be an O(1) gap.

    Parameters
    ----------
    Lm : ndarray, shape (2N, 2N)
        The doubled linearisation ``L(omega)``.
    dx : float
        Grid spacing.

    Returns
    -------
    v : ndarray, shape (2N,)
        The adjoint kernel vector, normalised so ``int |v|^2 dx = 1``.
    s_last : float
        Smallest singular value of ``L^dagger`` (the kernel).
    s_prev : float
        Next smallest singular value (the gap).
    """
    U, S, Vh = np.linalg.svd(Lm.conj().T)
    v = Vh[-1].conj()
    return v/np.sqrt(np.sum(np.abs(v)**2)*dx), S[-1], S[-2]


def S_map(chi: np.ndarray, n: int) -> np.ndarray:
    """The antilinear involution ``S(chi_1, chi_2) = (conj(chi_2), conj(chi_1))``.

    ``S`` commutes with the doubled problem and squares to the identity; its
    fixed set is the real ray of Lemma 2.6.
    """
    return np.concatenate([np.conj(chi[n:]), np.conj(chi[:n])])


def real_ray(chi: np.ndarray, n: int, dx: float) -> tuple[np.ndarray, complex]:
    """rotate chi onto the real ray of S (Lemma 2.6): S chi = chi

    If ``S chi = c chi`` with ``|c| = 1``, then ``exp(i arg(c)/2) chi`` is fixed
    by ``S``, because ``S`` is antilinear.  The eigenvalue ``c`` is returned
    alongside the rotated vector so that ``|arg c|`` can be reported.

    Returns
    -------
    chi_real : ndarray
        ``chi`` rotated onto the fixed ray of ``S``.
    c : complex
        The S-eigenvalue of the input ``chi``.
    """
    Sc = S_map(chi, n)
    c = np.sum(np.conj(chi)*Sc)*dx/(np.sum(np.abs(chi)**2)*dx)   # <chi,Schi>
    return chi*np.exp(1j*0.5*np.angle(c)), c


def gaugefix(psi: np.ndarray, ref: np.ndarray, dx: float) -> np.ndarray:
    """Rotate ``psi`` so that ``<ref, psi>`` is real and positive.

    Removes the residual U(1) freedom by aligning the phase of ``psi`` with a
    reference vector, so that finite differences in ``omega`` are taken within
    one gauge.
    """
    z = np.sum(np.conj(ref)*psi)*dx
    return psi*np.exp(-1j*np.angle(z))


def reflect(f: np.ndarray) -> np.ndarray:
    """``x -> -x`` on the periodic Fourier grid ``x_j = -L + 2 L j / N``.

    On this grid the point ``-x_j`` is the grid point with index
    ``(N - j) mod N``, so the reflection is ``np.roll(f[::-1], 1)`` and **not**
    ``f[::-1]``.  The naive reversal is off by one grid spacing: it maps index
    ``j`` to ``N - 1 - j``, i.e. it reflects about ``x = -L + L(N-1)/N`` rather
    than about the origin.  The error is silent -- the result still looks like a
    reflected profile -- but it contaminates every parity residual by a term of
    order ``dx f'(x)``, which is fatal for the ``< 1e-11`` even-direction
    entries of Table 3.
    """
    return np.roll(f[::-1], 1)


def pt_gauge(phi: np.ndarray, dx: float) -> np.ndarray:
    """Rotate ``phi`` onto the PT-covariant ray, ``phi(-x) = conj(phi(x))``.

    For a PT-symmetric base (``V`` even, ``G`` odd) this ray is canonical and
    removes the residual U(1) freedom of the stationary problem, so that any
    drift of ``arg Q`` along a path is attributable to ``chi_0`` alone.
    """
    z = np.sum(np.conj(phi) * np.conj(reflect(phi))) * dx
    return phi * np.exp(-0.5j * np.angle(z))


def S_eigenvalue(chi: np.ndarray, n: int, dx: float) -> complex:
    """The scalar ``c`` with ``S chi = c chi``, ``|c| = 1`` (Lemma 2.6).

    Evaluated as the Rayleigh-type quotient ``<chi, S chi> / <chi, chi>``.  A
    nonzero winding of ``arg c`` along a closed path is exactly a monodromy
    obstruction, which is why Section 5.3 tracks it.
    """
    Sc = S_map(chi, n)
    return np.sum(np.conj(chi) * Sc) * dx / (np.sum(np.abs(chi) ** 2) * dx)


def transport(chi_new: np.ndarray, chi_prev: np.ndarray, dx: float) -> np.ndarray:
    """Fix the phase of a freshly computed kernel vector by continuity.

    The phase of ``chi_new`` is chosen to maximise ``Re <chi_prev, chi_new>``,
    i.e. ``chi_new`` is rotated by ``exp(-i arg <chi_prev, chi_new>)``.  This is
    the continuity tracking analysed in Section 5.3, and it carries a structural
    consequence: if ``chi_prev`` is S-real (``S chi_prev = chi_prev``) and
    ``chi_new`` lies on the S-real line up to a phase, then the overlap
    ``<chi_prev, chi_new>`` is real, so the transport phase is ``0`` or ``pi``.
    Transport therefore preserves S-reality, and the residual freedom left along
    the path is a sign (Z_2), not a phase (U(1)).

    Parameters
    ----------
    chi_new : ndarray
        Freshly computed kernel vector, with arbitrary phase.
    chi_prev : ndarray
        The kernel vector at the previous point of the path.
    dx : float
        Grid spacing.

    Returns
    -------
    ndarray
        ``chi_new`` with its phase transported from ``chi_prev``.
    """
    z = np.sum(np.conj(chi_prev) * chi_new) * dx
    return chi_new * np.exp(-1j * np.angle(z))
