"""Krein-signature consistency check -- robust version.

Paper convention:  i psi_t = D psi + N(|psi|^2) psi + i G(x) psi,
                   D = -d_xx + V,  N(s) = sigma s,  G = eps*g.
CP convention:     i psi_t = -psi_xx + (V + i gamma W) psi - g_cp |psi|^2 psi
                   => sigma = -g_cp,  G = gamma W,  mu = omega.

Refactor of ``provenance/krein2.py``.  Produces Table F.1 (structural
identities at finite eps) and Table F.2 (the eps -> 0 scaling of the
conditioning factor K and of kappa against Q').

Outputs
-------
data/krein_identities.csv : Table F.1
data/eps_scan.csv         : Table F.2
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numpy.linalg import norm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nhvk.core import Model, adj_kernel, gaugefix, ip, ip_cp, make_grid, real_ray
from nhvk.profiles import pt_well


# ------------------------------------------------------------------ plumbing
def git_hash() -> str:
    """Current commit hash, or ``"unknown"`` outside a git checkout."""
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    return "unknown"


def metadata(params: dict) -> dict:
    """Provenance sidecar recorded next to every CSV."""
    import scipy
    return {
        "script": Path(__file__).name,
        "git_commit": git_hash(),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "parameters": params,
    }


def _cell(v: object) -> object:
    return repr(float(v)) if isinstance(v, (float, np.floating)) else v


def write_table(outdir: Path, stem: str, header: list[str],
                rows: list[tuple], params: dict) -> Path:
    """Write ``<stem>.csv`` and its ``<stem>.json`` metadata sidecar."""
    path = outdir / f"{stem}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow([_cell(v) for v in r])
    with open(outdir / f"{stem}.json", "w") as fh:
        json.dump(metadata(params), fh, indent=2)
        fh.write("\n")
    return path


# ------------------------------------------------------------------ physics
def run(outdir: Path) -> None:
    L, N = 18.0, 200
    x, D2, dx = make_grid(L, N)
    V0, sigma = 2.0, -1.0
    V, W = pt_well(x, V0, 1.0)        # V = -V0 sech^2 (even), W = sech tanh (ODD -> PT)
    M = Model(V, W, sigma, x, D2, dx)
    om = -2.5          # linear bound state of -d_xx - 2 sech^2 sits at E = -1;
                       # the focusing soliton family lives at omega < -1.
    seed = np.sqrt(2*(abs(om)-1))/np.cosh(np.sqrt(abs(om))*x) + 0j

    print("="*74)
    print("PART 0 -- Hermitian anchor")
    print("="*74)

    def cont_omega(M, seed, om_start, om_end, nst=40):
        ph = M.newton(seed, om_start, 0.0)
        if ph is None:
            return None
        for o in np.linspace(om_start, om_end, nst+1)[1:]:
            p2 = M.newton(ph, o, 0.0)
            if p2 is None:
                return None
            ph = p2
        return ph

    phi0 = cont_omega(M, seed, -1.05, om)
    if phi0 is None:
        raise RuntimeError("omega-continuation failed")
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
    if phi is None:
        raise RuntimeError(f"eps-continuation failed at eps = {eps}")
    print(f"eps={eps}:  residual   = {norm(M.res(phi.real,phi.imag,om,eps))*np.sqrt(dx):.2e}")
    print(f"           P(eps)      = {np.sum(np.abs(phi)**2)*dx:.8f}")
    Phi = np.concatenate([phi, np.conj(phi)])
    w0 = np.concatenate([1j*phi, -1j*np.conj(phi)])
    Lm = M.Lop(phi, om, eps)
    Lc = M.Lcal(phi, om, eps)
    s3v = np.concatenate([np.ones(N), -np.ones(N)])

    c1 = np.max(np.abs(Lm - s3v[:, None]*Lc))
    c2 = norm(Lm@w0)*np.sqrt(dx)
    print(f"(C1) max|L - sigma3 Lcal|            = {c1:.2e}")
    print(f"(C2) ||L w0||                        = {c2:.2e}")

    php = M.newton(phi, om+hh, eps); phm = M.newton(phi, om-hh, eps)
    php = gaugefix(php, phi, dx); phm = gaugefix(phm, phi, dx)
    dphi = (php-phm)/(2*hh)
    dPhi = np.concatenate([dphi, np.conj(dphi)])
    c3 = norm(Lm@(1j*dPhi)-w0)*np.sqrt(dx)
    print(f"(C3) ||L(i dPhi) - w0||              = {c3:.2e}")

    chi0, s_last, s_prev = adj_kernel(Lm, dx)
    chi0 = gaugefix(chi0, Phi, dx)
    print(f"     sing.vals of L^dag: {s_last:.2e} (kernel), {s_prev:.2e} (gap)")
    c4 = ip(chi0, w0, dx)
    print(f"(C4) <chi0,w0> (Lemma2.3 = K_CP(0))  = {c4:.3e}")

    vsharp, _, _ = adj_kernel(Lc, dx)
    cand = s3v*vsharp
    al = ip(cand, chi0, dx)/ip(cand, cand, dx)
    c5 = norm(chi0-al*cand)/norm(chi0)
    print(f"(C5) ||chi0 - a*sigma3 v#||/||chi0|| = {c5:.2e}")

    vg = 1j*dPhi
    kappa = ip(chi0, dPhi, dx)
    chi_cp = chi0/al                       # == sigma3 v#, CP-normalised
    cpq = ip_cp(vg, chi_cp, dx)            # <v_g, sigma3 v0#>_CP
    print(f"(C6) kappa = <chi0,dPhi>             = {kappa:.6f}")
    print(f"     <v_g,sigma3 v0#>_CP             = {cpq:.6f}")
    print(f"     <v_g,sigma3 v0#>_CP / (i kappa/conj(a)) = {cpq/(1j*kappa/np.conj(al)):.6f}")
    print(f"     |ratio to i*kappa| (gauge-free) = {abs(cpq)/abs(kappa):.6f}, "
          f"1/|a| = {1/abs(al):.6f}")

    identities = [
        ("max_L_minus_sigma3_Lcal", float(c1)),
        ("norm_L_w0", float(c2)),
        ("norm_L_idPhi_minus_w0", float(c3)),
        ("sv_kernel", float(s_last)),
        ("sv_gap", float(s_prev)),
        ("abs_chi0_w0", float(abs(c4))),
        ("rel_chi0_minus_sigma3_vsharp", float(c5)),
        ("cp_ratio", float(abs(cpq)/abs(kappa))),
        ("kappa", float(kappa.real)),
    ]

    print()
    print("="*74)
    print("PART 2 -- Theorem 2 scaling; kappa vs Q' in the canonical gauge")
    print("="*74)
    nPhi0 = np.sqrt(np.sum(np.abs(Phi0)**2)*dx)
    eps_list = [0.16, 0.08, 0.04, 0.02, 0.01]
    rows = []
    for e in eps_list:
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
            n0 = np.sqrt(np.sum(np.abs(np.concatenate([M.newton(phi0, omx, 0.0),
                         np.conj(M.newton(phi0, omx, 0.0))]))**2)*dx)
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

    params = {
        "L": L, "N": N, "V0": V0, "sigma": sigma, "omega": om,
        "eps_identities": eps, "eps_scan": eps_list, "hh": hh,
        "profile": "pt_well",
    }
    p1 = write_table(outdir, "krein_identities", ["quantity", "value"],
                     identities, params)
    scan_rows = [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows]
    scan_rows.append((0.0, 0.0, float("nan"), float(P), float(Pprime), float(Pprime)))
    p2 = write_table(outdir, "eps_scan",
                     ["eps", "K_minus_1", "K_minus_1_over_eps2", "Q", "kappa",
                      "Q_prime"], scan_rows, params)
    print()
    print(f"wrote {p1}")
    print(f"wrote {p2}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", default=str(REPO_ROOT / "data"),
                    help="directory for the CSV/JSON output (default: data/)")
    args = ap.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run(outdir)
    print("OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:                      # noqa: BLE001 - top-level guard
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
