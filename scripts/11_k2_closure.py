"""Closed form for the O(eps^2) conditioning coefficient K_2.

Theorem ``thm:K2``:

    K(omega; eps) = 1 + K_2(omega) eps^2 + O(eps^3),
    K_2(omega) = (4 / P(omega)) || L_+^{-1} (g phi_0) ||_{L^2}^2 >= 0,

with ``L_+ = D + N(phi_0^2) + 2 N'(phi_0^2) phi_0^2 - omega`` and ``L_+^{-1}``
the bounded inverse supplied by H16.  No second-order datum enters: not
``phi_2``, not ``chi_2``, not ``L_2``, not ``N''``.

The script evaluates the formula (one linear solve) and, independently, the
direct ``eps``-extrapolation of ``K - 1`` from the full nonlinear problem with
``chi_0`` taken from the SVD of ``L^dagger`` in the H18 real-ray gauge -- ``K``
is invariant under a real rescaling of ``chi_0``, so no normalisation is needed.
It also evaluates the equivalent quadrature form ``K_2 = P^{-1} int (r - q)^2``
as an internal cross-check of the two-term collapse ``L_+(r - q) = 2 g phi_0``,
and checks the a priori bound ``K_2 <= 4 int g^2 phi_0^2 / (P d^2)`` with
``d = dist(0, sigma(L_+))``.

Configuration I is the anchor of Appendix ``app:numerics``; Configuration IV has
``N'' != 0`` and confirms that no third-order Taylor datum of ``N`` enters.

Outputs
-------
data/k2_closure.csv  : Table ``tab:K2``
data/k2_eps_scan.csv : the eps ladder of Appendix ``app:numerics``
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
from numpy.linalg import eigvalsh, lstsq, norm, solve

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nhvk.core import adj_kernel, ip, make_grid, real_ray
from nhvk.kappa import QuinticModel, free_seed, newton_robust
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
LX, NGRID, SIGMA = 18.0, 200, -1.0
EPS_LADDER = [0.16, 0.08, 0.04, 0.02, 0.01]
EPS_EXTRAP = [0.08, 0.04, 0.02, 0.01]

# Configuration I is nhvk.profiles.pt_well(x, 2.0, 1.0); the remaining
# configurations are one-off (V, g) pairs used only for this scan.
CONFIGS = [
    ("I", lambda x: pt_well(x, 2.0, 1.0), 0.0, [-1.5, -2.0, -2.5, -3.0, -4.0]),
    ("II", lambda x: (-3.0/np.cosh(x)**2, np.tanh(x)/np.cosh(x)**2), 0.0,
     [-2.5, -3.5]),
    ("III", lambda x: (-2.0/np.cosh(x)**2, x*np.exp(-x**2/2)), 0.0,
     [-2.5, -3.5]),
    ("IV", lambda x: pt_well(x, 2.0, 1.0), 0.02, [-2.5, -3.5]),
    ("V", lambda x: (-1.5*np.exp(-x**2), x*np.exp(-x**2)), 0.0, [-1.0, -2.0]),
]
LABELS = {"I": "V=-2sech^2, g=sech tanh", "II": "V=-3sech^2, g=tanh sech^2",
          "III": "V=-2sech^2, g=x exp(-x^2/2)",
          "IV": "cubic-quintic N=-s+0.02 s^2",
          "V": "V=-1.5 exp(-x^2), g=x exp(-x^2)"}

HEADER = ["configuration", "description", "beta", "omega", "P", "K2_formula",
          "K2_quadrature", "K2_direct", "rel_diff", "min_abs_eig_Lplus",
          "a_priori_bound", "power_balance", "solve_residual"]
SCAN_HEADER = ["configuration", "omega", "eps", "K_minus_1", "scaled",
               "Im_Q_over_absQ", "adjoint_gap"]


def base_state(V, om, x, D2, dx, beta):
    """Free-soliton seed at the target omega, ramp V, then ramp beta."""
    cur, t, st = free_seed(om, x), 0.0, 0.05
    while t < 1.0 - 1e-13:
        st = min(st, 1.0 - t)
        M = QuinticModel((t + st)*V, np.zeros_like(x), SIGMA, x, D2, dx, 0.0)
        p = newton_robust(M, cur, om, 1.0)
        if p is None or np.sum(np.abs(p)**2)*dx < 1e-8:
            st /= 2
            if st < 1e-6:
                raise RuntimeError(f"V-continuation stalled at t = {t:.4f}")
            continue
        cur, t = p, t + st
        st = min(st*1.4, 0.1)
    ph, b, stb = cur, 0.0, 0.01
    while b < beta - 1e-13:
        stb = min(stb, beta - b)
        M = QuinticModel(V, np.zeros_like(x), SIGMA, x, D2, dx, b + stb)
        p = newton_robust(M, ph, om, 1.0)
        if p is None or np.sum(np.abs(p)**2)*dx < 1e-8:
            stb /= 2
            if stb < 1e-7:
                raise RuntimeError("beta-continuation stalled")
            continue
        ph, b = p, b + stb
        stb *= 1.3
    return (ph*np.exp(-1j*np.angle(ph[len(x)//2]))).real.copy()


def k2_formula(V, g, om, x, D2, dx, phi0, beta):
    s = phi0**2
    Nf, Np = SIGMA*s + beta*s**2, SIGMA + 2*beta*s
    Lp = -D2 + np.diag(V + Nf + 2*Np*s - om)
    Lm = -D2 + np.diag(V + Nf - om)
    y = solve(Lp, g*phi0)
    P = float(np.sum(phi0**2)*dx)
    q = lstsq(Lm, -g*phi0, rcond=None)[0]
    r = solve(Lp, (g + 2*Np*phi0*q)*phi0)
    d = float(np.min(np.abs(eigvalsh(Lp))))
    return dict(K2=4.0*float(np.sum(y**2)*dx)/P,
                K2_quad=float(np.sum((r - q)**2)*dx)/P, P=P, gap=d,
                bound=4.0*float(np.sum(g**2*phi0**2)*dx)/(P*d**2),
                res=float(norm(Lp @ y - g*phi0)),
                pbal=float(np.sum(g*phi0**2)*dx))


def k_direct(V, g, om, x, D2, dx, phi0, beta, eps_list):
    n = len(x)
    M = QuinticModel(V, g, SIGMA, x, D2, dx, beta)
    out = []
    for e in eps_list:
        p, cur, st = phi0.astype(complex), 0.0, min(e, 0.01)
        while cur < e - 1e-15:
            st = min(st, e - cur)
            p2 = newton_robust(M, p, om, cur + st)
            if p2 is None:
                st /= 2
                if st < 1e-9:
                    raise RuntimeError("eps-continuation stalled")
                continue
            p, cur = p2, cur + st
            st = min(st*1.4, 0.01)
        Phi = np.concatenate([p, np.conj(p)])
        chi, s1, s2 = adj_kernel(M.Lop(p, om, e), dx)
        chi, _ = real_ray(chi, n, dx)
        Q = 0.5*ip(chi, Phi, dx)
        K = float(((0.5*np.sum(np.abs(Phi)**2)*dx)
                   * (0.5*np.sum(np.abs(chi)**2)*dx)/Q**2).real)
        out.append((e, K - 1.0, (K - 1.0)/e**2, float(abs(Q.imag)/abs(Q)), s2))
    return out


def run(outdir: Path) -> None:
    x, D2, dx = make_grid(LX, NGRID)
    rows, scan = [], []
    print(f"K_2 closure at (L, N) = ({LX:g}, {NGRID}), sigma = {SIGMA:g}")
    print(f"{'cfg':>4} {'description':>32} {'omega':>7} {'K2 formula':>12} "
          f"{'K2 quad':>12} {'K2 direct':>12} {'rel diff':>10}")
    for tag, VG, beta, omegas in CONFIGS:
        V, g = VG(x)
        for om in omegas:
            phi0 = base_state(V, om, x, D2, dx, beta)
            f = k2_formula(V, g, om, x, D2, dx, phi0, beta)
            anchor = (tag == "I" and abs(om + 2.5) < 1e-12)
            ladder = k_direct(V, g, om, x, D2, dx, phi0, beta,
                              EPS_LADDER if anchor else EPS_EXTRAP)
            K2n = (4*ladder[-1][2] - ladder[-2][2])/3.0    # Richardson
            rel = abs(f["K2"] - K2n)/abs(f["K2"])
            rows.append((tag, LABELS[tag], beta, om, f["P"], f["K2"],
                         f["K2_quad"], K2n, rel, f["gap"], f["bound"],
                         f["pbal"], f["res"]))
            print(f"{tag:>4} {LABELS[tag]:>32} {om:7.2f} {f['K2']:12.6f} "
                  f"{f['K2_quad']:12.6f} {K2n:12.6f} {rel:10.1e}")
            if abs(f["K2"] - f["K2_quad"]) > 1e-9*abs(f["K2"]):
                raise RuntimeError("the two forms of K_2 disagree")
            if rel > 1e-6:
                raise RuntimeError("formula and extrapolation disagree")
            if f["K2"] < 0:
                raise RuntimeError("K_2 < 0")
            if f["K2"] > f["bound"]*(1 + 1e-9):
                raise RuntimeError("K_2 exceeds its a priori bound")
            if anchor:
                for e, d, sc, qi, s2 in ladder:
                    scan.append((tag, om, e, d, sc, qi, s2))

    print("\nConfiguration I anchor, eps ladder:")
    for _, _, e, d, sc, qi, _ in scan:
        print(f"    eps = {e:5.3f}   K - 1 = {d:.6e}   "
              f"(K-1)/eps^2 = {sc:.6f}   |Im Q / Q| = {qi:.1e}")

    params = {"L": LX, "N": NGRID, "sigma": SIGMA, "eps_ladder": EPS_LADDER,
              "eps_extrapolation": EPS_EXTRAP,
              "extrapolation": "Richardson on the last two eps",
              "configurations": {t: LABELS[t] for t, _, _, _ in CONFIGS},
              "chi0_gauge": "H18 real ray (K is invariant under real rescaling)"}
    print(f"\nwrote {write_table(outdir, 'k2_closure', HEADER, rows, params)}")
    sparams = {"L": LX, "N": NGRID, "sigma": SIGMA, "configuration": "I",
               "omega": -2.5, "eps_ladder": EPS_LADDER}
    print(f"wrote {write_table(outdir, 'k2_eps_scan', SCAN_HEADER, scan, sparams)}")


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
