"""
Krein-signature consistency check -- robust version.

Paper convention:  i psi_t = D psi + N(|psi|^2) psi + i G(x) psi,
                   D = -d_xx + V,  N(s) = sigma s,  G = eps*g.
CP convention:     i psi_t = -psi_xx + (V + i gamma W) psi - g_cp |psi|^2 psi
                   => sigma = -g_cp,  G = gamma W,  mu = omega.
"""
import numpy as np
from numpy.linalg import norm, eig

# ---------- grid ----------
def make_grid(L, N):
    x = -L + 2*L*np.arange(N)/N
    k = np.fft.fftfreq(N, d=2*L/N)*2*np.pi
    F = np.fft.fft(np.eye(N), axis=0)
    D2 = np.real(np.fft.ifft(-(k**2)[:, None]*F, axis=0))
    return x, D2, x[1]-x[0]

class Model:
    def __init__(self, V, g, sigma, x, D2, dx):
        self.V, self.g, self.sigma, self.x, self.D2, self.dx = V, g, sigma, x, D2, dx
        self.n = len(x)
        self.D = -D2 + np.diag(V)

    # ---- stationary residual in real coords ----
    def res(self, u, v, om, eps):
        s = self.sigma*(u**2+v**2)
        r1 = self.D@u + s*u - eps*self.g*v - om*u
        r2 = self.D@v + s*v + eps*self.g*u - om*v
        return np.concatenate([r1, r2])

    def jac(self, u, v, om, eps):
        n, sg = self.n, self.sigma
        J11 = self.D + np.diag(sg*(3*u**2+v**2) - om)
        J12 = np.diag(2*sg*u*v - eps*self.g)
        J21 = np.diag(2*sg*u*v + eps*self.g)
        J22 = self.D + np.diag(sg*(u**2+3*v**2) - om)
        return np.block([[J11, J12], [J21, J22]])

    def newton(self, phi0, om, eps, tol=1e-12, itmax=60):
        """bordered Newton: kernel direction i*phi removed by a Lagrange multiplier"""
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

    def continue_eps(self, phi0, om, eps_target, nsteps=25):
        ph = phi0
        for e in np.linspace(0, eps_target, nsteps+1)[1:]:
            ph2 = self.newton(ph, om, e)
            if ph2 is None:
                return None
            ph = ph2
        return ph

    # ---- doubled operators ----
    def Lop(self, phi, om, eps):
        a = 2*self.sigma*np.abs(phi)**2
        b = self.sigma*phi**2
        A11 = self.D + np.diag(1j*eps*self.g - om + a)
        A12 = np.diag(b)
        A21 = np.diag(-np.conj(b))
        A22 = -(self.D + np.diag(-1j*eps*self.g - om + a))
        return np.block([[A11, A12], [A21, A22]])

    def Lcal(self, phi, om, eps):        # CP eq.(8)
        a = 2*self.sigma*np.abs(phi)**2
        b = self.sigma*phi**2
        A11 = self.D + np.diag(1j*eps*self.g - om + a)
        A12 = np.diag(b)
        A21 = np.diag(np.conj(b))
        A22 = self.D + np.diag(-1j*eps*self.g - om + a)
        return np.block([[A11, A12], [A21, A22]])

def ip(f, g, dx):      # paper: conjugate-linear in FIRST slot
    return np.sum(np.conj(f)*g)*dx
def ip_cp(f, g, dx):   # CP: linear in first slot
    return np.sum(f*np.conj(g))*dx

def adj_kernel(Lm, dx):
    U, S, Vh = np.linalg.svd(Lm.conj().T)
    v = Vh[-1].conj()
    return v/np.sqrt(np.sum(np.abs(v)**2)*dx), S[-1], S[-2]

def S_map(chi, n):
    return np.concatenate([np.conj(chi[n:]), np.conj(chi[:n])])

def real_ray(chi, n, dx):
    """rotate chi onto the real ray of S (Lemma 2.6): S chi = chi"""
    Sc = S_map(chi, n)
    c = np.sum(np.conj(chi)*Sc)*dx/(np.sum(np.abs(chi)**2)*dx)   # <chi,Schi>
    return chi*np.exp(1j*0.5*np.angle(c)), c

def gaugefix(psi, ref, dx):
    z = np.sum(np.conj(ref)*psi)*dx
    return psi*np.exp(-1j*np.angle(z))

# ==================================================================
L, N = 18.0, 200
x, D2, dx = make_grid(L, N)
V0, sigma = 2.0, -1.0
V = -V0/np.cosh(x)**2
W = np.tanh(x)/np.cosh(x)                # ODD -> PT
M = Model(V, W, sigma, x, D2, dx)
om = -2.5          # linear bound state of -d_xx - 2 sech^2 sits at E = -1;
                   # the focusing soliton family lives at omega < -1.
seed = np.sqrt(2*(abs(om)-1))/np.cosh(np.sqrt(abs(om))*x) + 0j

print("="*74)
print("PART 0 -- Hermitian anchor")
print("="*74)
def cont_omega(M, seed, om_start, om_end, nst=40):
    ph = M.newton(seed, om_start, 0.0)
    if ph is None: return None
    for o in np.linspace(om_start, om_end, nst+1)[1:]:
        p2 = M.newton(ph, o, 0.0)
        if p2 is None: return None
        ph = p2
    return ph
phi0 = cont_omega(M, seed, -1.05, om)
assert phi0 is not None, "omega-continuation failed" 
phi0 = phi0*np.exp(-1j*np.angle(phi0[N//2]))
print(f"residual              = {norm(M.res(phi0.real,phi0.imag,om,0.0))*np.sqrt(dx):.2e}")
print(f"max |Im phi0|         = {np.max(np.abs(phi0.imag)):.2e}   (should be ~0)")
P = np.sum(np.abs(phi0)**2)*dx
hh = 2e-3
Pp = np.sum(np.abs(M.newton(phi0, om+hh, 0.0))**2)*dx
Pm = np.sum(np.abs(M.newton(phi0, om-hh, 0.0))**2)*dx
Pprime = (Pp-Pm)/(2*hh)
print(f"P(omega)              = {P:.8f}")
print(f"P'(omega)             = {Pprime:.8f}")
Phi0 = np.concatenate([phi0, np.conj(phi0)])

print()
print("="*74)
print("PART 1 -- structural identities, finite eps (PT-symmetric odd g)")
print("="*74)
eps = 0.20
phi = M.continue_eps(phi0, om, eps)
assert phi is not None
print(f"eps={eps}:  residual   = {norm(M.res(phi.real,phi.imag,om,eps))*np.sqrt(dx):.2e}")
print(f"           P(eps)      = {np.sum(np.abs(phi)**2)*dx:.8f}")
Phi = np.concatenate([phi, np.conj(phi)])
w0  = np.concatenate([1j*phi, -1j*np.conj(phi)])
Lm  = M.Lop(phi, om, eps)
Lc  = M.Lcal(phi, om, eps)
s3v = np.concatenate([np.ones(N), -np.ones(N)])

print(f"(C1) max|L - sigma3 Lcal|            = {np.max(np.abs(Lm - s3v[:,None]*Lc)):.2e}")
print(f"(C2) ||L w0||                        = {norm(Lm@w0)*np.sqrt(dx):.2e}")

php = M.newton(phi, om+hh, eps); phm = M.newton(phi, om-hh, eps)
php = gaugefix(php, phi, dx); phm = gaugefix(phm, phi, dx)
dphi = (php-phm)/(2*hh)
dPhi = np.concatenate([dphi, np.conj(dphi)])
print(f"(C3) ||L(i dPhi) - w0||              = {norm(Lm@(1j*dPhi)-w0)*np.sqrt(dx):.2e}")

chi0, s_last, s_prev = adj_kernel(Lm, dx)
chi0 = gaugefix(chi0, Phi, dx)
print(f"     sing.vals of L^dag: {s_last:.2e} (kernel), {s_prev:.2e} (gap)")
print(f"(C4) <chi0,w0> (Lemma2.3 = K_CP(0))  = {ip(chi0,w0,dx):.3e}")

vsharp, _, _ = adj_kernel(Lc, dx)
cand = s3v*vsharp
al = ip(cand, chi0, dx)/ip(cand, cand, dx)
print(f"(C5) ||chi0 - a*sigma3 v#||/||chi0|| = {norm(chi0-al*cand)/norm(chi0):.2e}")

vg = 1j*dPhi
kappa = ip(chi0, dPhi, dx)
chi_cp = chi0/al                       # == sigma3 v#, CP-normalised
cpq = ip_cp(vg, chi_cp, dx)            # <v_g, sigma3 v0#>_CP
print(f"(C6) kappa = <chi0,dPhi>             = {kappa:.6f}")
print(f"     <v_g,sigma3 v0#>_CP             = {cpq:.6f}")
print(f"     <v_g,sigma3 v0#>_CP / (i kappa/conj(a)) = {cpq/(1j*kappa/np.conj(al)):.6f}")
print(f"     |ratio to i*kappa| (gauge-free) = {abs(cpq)/abs(kappa):.6f}, "
      f"1/|a| = {1/abs(al):.6f}")

print()
print("="*74)
print("PART 2 -- Theorem 2 scaling; kappa vs Q' in the canonical gauge")
print("="*74)
nPhi0 = np.sqrt(np.sum(np.abs(Phi0)**2)*dx)
rows = []
for e in [0.16, 0.08, 0.04, 0.02, 0.01]:
    ph = M.continue_eps(phi0, om, e)
    Ph = np.concatenate([ph, np.conj(ph)])
    Lm = M.Lop(ph, om, e)
    ch, _, _ = adj_kernel(Lm, dx)
    ch, cS = real_ray(ch, N, dx)                       # H18 local reality gauge
    ch = ch*nPhi0/np.sqrt(np.sum(np.abs(ch)**2)*dx)    # H13(ii) canonical norm
    Q = 0.5*ip(ch, Ph, dx)
    if Q.real < 0:
        ch, Q = -ch, -Q
    nP = 0.5*np.sum(np.abs(Ph)**2)*dx
    nX = 0.5*np.sum(np.abs(ch)**2)*dx
    K = nP*nX/Q**2
    php = M.newton(ph, om+hh, e); phm = M.newton(ph, om-hh, e)
    php = gaugefix(php, ph, dx); phm = gaugefix(phm, ph, dx)
    dPh = np.concatenate([(php-phm)/(2*hh), np.conj((php-phm)/(2*hh))])
    kap = ip(ch, dPh, dx)
    # Q'(omega) by finite differences at fixed eps, same gauge protocol
    def Qof(omx):
        p = M.continue_eps(M.newton(phi0, omx, 0.0), omx, e)
        Pp_ = np.concatenate([p, np.conj(p)])
        Lp = M.Lop(p, omx, e)
        c_, _, _ = adj_kernel(Lp, dx)
        c_, _ = real_ray(c_, N, dx)
        n0 = np.sqrt(np.sum(np.abs(np.concatenate([M.newton(phi0,omx,0.0),
                     np.conj(M.newton(phi0,omx,0.0))]))**2)*dx)
        c_ = c_*n0/np.sqrt(np.sum(np.abs(c_)**2)*dx)
        q = 0.5*ip(c_, Pp_, dx)
        return q if q.real > 0 else -q
    Qp = (Qof(om+hh)-Qof(om-hh))/(2*hh)
    rows.append((e, (K-1).real, (K-1).real/e**2, Q.real, kap.real, Qp.real,
                 abs(np.angle(cS))))
print(f"{'eps':>7} {'K-1':>12} {'(K-1)/eps^2':>12} {'Q':>12} {'kappa':>12} "
      f"{'Qprime':>12} {'|arg c|':>9}")
for r in rows:
    print(f"{r[0]:7.3f} {r[1]:12.3e} {r[2]:12.5f} {r[3]:12.7f} {r[4]:12.7f} "
          f"{r[5]:12.7f} {r[6]:9.2e}")
print(f"{'0':>7} {'0':>12} {'--':>12} {P:12.7f} {Pprime:12.7f} {Pprime:12.7f} {'0':>9}")
