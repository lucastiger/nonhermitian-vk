"""Targets A/B and the local exchange of stability at a simple zero of kappa.

Three things are computed on the Theorem A branch, at four resolutions:

* ``omega_kappa``, the zero of ``kappa``, located by a secant iteration on
  ``kappa`` alone -- no eigensolve is involved;
* ``omega_c``, the origin collision, located **independently** by a secant on
  ``nu = (1/2) tr(L_Gamma^2)`` from a Beyn contour projection.  The two agree to
  ``~1e-14``, orders of magnitude finer than the truncation error in either
  separately, which is Theorem ``thm:targetB`` rather than a coincidence;
* the Jordan data at ``omega_kappa``: the origin block has dimension 4, the
  second chain scalar ``kappa_2`` vanishes automatically, ``kappa_3`` is purely
  imaginary, and the predicted exchange constant ``C = -i/kappa_3`` agrees with
  the measured ``nu'/kappa'``.

A Hermitian control (``G = 0``, trapping ``V``, cubic--quintic ``N``) confirms
``kappa = P'`` to round-off and ``C < 0``, so the theorem reduces to the
classical Vakhitov--Kolokolov criterion "stable iff ``P' < 0``" in this
``omega`` convention.

Outputs
-------
data/kappa_local_bifurcation.csv : Table ``tab:partI``
data/kappa_hermitian_limit.csv   : the Hermitian-limit check
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nhvk.kappa import (Branch, QuinticModel, build_wadati, chi0_at,
                        contour_block, jordan_chain, kappa_at, make_grid,
                        move_omega, newton_robust, secant)
from nhvk.profiles import wadati_gaussian

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
GRIDS = [(20.0, 240), (22.0, 260), (22.0, 320), (26.0, 384)]
OM_ANCHOR, A_AMP, SIGMA = -0.5, 1.0, -1.0
BRACKET = (-0.265, -0.255)
FD_H = 2e-3
NU_RADIUS = 0.10
BEYN_SEED, BEYN_NQ, BEYN_M = 0, 48, 8

H_LX, H_N, H_BETA = 20.0, 256, 0.30
H_OMEGAS = [-1.5, -1.4, -1.3, -1.2, -1.1]

HEADER = ["L", "N", "omega_kappa", "omega_c", "omega_diff", "kappa_at_omega_c",
          "abs_kappa_2", "kappa_3_re", "kappa_3_im", "C_predicted",
          "C_predicted_im", "C_measured", "rel_diff_C", "kappa_prime",
          "nu_prime", "block_dim", "chain_res_1", "chain_res_2", "chain_res_3",
          "abs_arg_c"]
H_HEADER = ["omega", "P", "kappa", "P_prime", "abs_kappa_minus_P_prime", "nu",
            "C_measured", "kappa_3_re", "kappa_3_im", "C_from_kappa_3"]


def wadati_rows() -> list[tuple]:
    rows = []
    for (L, N) in GRIDS:
        x, D2, dx = make_grid(L, N)
        M, phi = build_wadati(wadati_gaussian, A_AMP, OM_ANCHOR, x, D2, dx, SIGMA)
        B = Branch(M, phi, OM_ANCHOR, dx)
        ok, _ = secant(lambda o: B.kappa(o)[0], *BRACKET, tol=1e-13)
        oc, _ = secant(lambda o: B.nu(o, r=NU_RADIUS, seed=BEYN_SEED),
                       *BRACKET, tol=1e-13)
        kp = (B.kappa(ok + FD_H)[0] - B.kappa(ok - FD_H)[0])/(2*FD_H)
        nup = (B.nu(ok + FD_H, r=NU_RADIUS, seed=BEYN_SEED)
               - B.nu(ok - FD_H, r=NU_RADIUS, seed=BEYN_SEED))/(2*FD_H)
        k_at_oc = B.kappa(oc)[0]
        ph = B.at(ok)
        chi, Q, _, _, argc, Lm = chi0_at(M, ph, ok, dx)
        kap, k2, k3, res = jordan_chain(M, ph, ok, dx, chi)
        C = -1j/k3
        _, Lc, _ = contour_block(Lm, r=NU_RADIUS, nq=BEYN_NQ, m=BEYN_M,
                                 seed=BEYN_SEED)
        dim = 0 if Lc is None else Lc.shape[0]
        rows.append((L, N, ok, oc, ok - oc, k_at_oc, float(abs(k2)),
                     float(k3.real), float(k3.imag), float(C.real),
                     float(C.imag), float(nup/kp),
                     float(abs(C.real - nup/kp)/abs(C.real)), kp, nup, dim,
                     res[0], res[1], res[2], argc))
        print(f"(L, N) = ({L:g}, {N}): omega_kappa = {ok:.12f}   "
              f"omega_c = {oc:.12f}   difference = {ok-oc:+.2e}")
        print(f"    |kappa_2| = {abs(k2):.2e}   kappa_3 = {k3.real:+.5f}"
              f"{k3.imag:+.5f}i   C = -i/kappa_3 = {C.real:.7f}   "
              f"nu'/kappa' = {nup/kp:.7f}   block dim = {dim}")
        print(f"    chain residuals: {res[0]:.1e}  {res[1]:.1e}  {res[2]:.1e}")
        if dim != 4:
            raise RuntimeError(f"origin block has dimension {dim}, expected 4")
        if abs(k2) > 1e-9:
            raise RuntimeError(f"kappa_2 = {abs(k2):.2e} does not vanish")
        if abs(k3.real) > 1e-6*abs(k3):
            raise RuntimeError("kappa_3 is not purely imaginary")
        if abs(ok - oc) > 1e-10:
            raise RuntimeError(f"omega_kappa != omega_c ({ok-oc:.2e})")
        if abs(C.real - nup/kp) > 1e-3*abs(C.real):
            raise RuntimeError("C = -i/kappa_3 disagrees with nu'/kappa'")
    return rows


def hermitian_rows() -> list[tuple]:
    x, D2, dx = make_grid(H_LX, H_N)
    V = -2.0/np.cosh(x)**2
    G0 = np.zeros_like(x)
    ph = (np.sqrt(3.0)/np.cosh(np.sqrt(1.5)*x)).astype(complex)
    M = QuinticModel(V, G0, SIGMA, x, D2, dx, 0.0)
    ph = newton_robust(M, ph, -1.5, 1.0)
    b, st = 0.0, 0.02
    while b < H_BETA - 1e-13:
        st = min(st, H_BETA - b)
        Mb = QuinticModel(V, G0, SIGMA, x, D2, dx, b + st)
        p = newton_robust(Mb, ph, -1.5, 1.0)
        if p is None:
            st /= 2
            if st < 1e-7:
                raise RuntimeError("beta-continuation stalled")
            continue
        ph, b = p, b + st
        st *= 1.3
    M = QuinticModel(V, G0, SIGMA, x, D2, dx, H_BETA)

    rows, cur, om = [], ph, -1.5
    for tgt in H_OMEGAS:
        p, o = move_omega(M, cur, om, tgt, dom=0.01)
        if p is None:
            raise RuntimeError(f"Hermitian continuation failed at {tgt}")
        cur, om = p, o
        chi, Q, _, _, _, Lm = chi0_at(M, cur, om, dx)
        kap, Pp, _ = kappa_at(M, cur, om, dx, chi)
        _, _, k3, _ = jordan_chain(M, cur, om, dx, chi)
        ev = np.linalg.eigvals(Lm)
        ev = ev[np.abs(ev) < abs(om)*0.999]
        ev = ev[np.abs(ev) > 1e-5]
        nu = float(np.real(ev[np.argmin(np.abs(ev))]**2)) if len(ev) else np.nan
        rows.append((om, float(np.sum(np.abs(cur)**2)*dx), float(kap.real), Pp,
                     float(abs(kap.real - Pp)), nu, float(nu/kap.real),
                     float(k3.real), float(k3.imag), float((-1j/k3).real)))
        print(f"  omega = {om:7.3f}   kappa = {kap.real:10.5f}   "
              f"P' = {Pp:10.5f}   |kappa - P'| = {abs(kap.real-Pp):.1e}   "
              f"nu/kappa = {nu/kap.real:9.5f}   -i/kappa_3 = "
              f"{(-1j/k3).real:9.4f}")
        if abs(kap.real - Pp) > 1e-10:
            raise RuntimeError("kappa != P' in the Hermitian limit")
        if nu/kap.real > 0 or (-1j/k3).real > 0:
            raise RuntimeError("C is not negative in the Hermitian limit")
    return rows


def run(outdir: Path) -> None:
    print("Theorem A branch: omega_kappa, omega_c and the exchange constant")
    rows = wadati_rows()
    params = {"grids": [[g[0], g[1]] for g in GRIDS], "A": A_AMP,
              "sigma": SIGMA, "omega_0": OM_ANCHOR, "bracket": list(BRACKET),
              "fd_step": FD_H, "contour_radius": NU_RADIUS,
              "beyn_seed": BEYN_SEED, "beyn_quadrature_points": BEYN_NQ,
              "beyn_probe_columns": BEYN_M, "profile": "wadati_gaussian"}
    print(f"wrote {write_table(outdir, 'kappa_local_bifurcation', HEADER, rows, params)}")

    print("\nHermitian limit: G = 0, V = -2 sech^2 x, N(s) = -s + 0.30 s^2")
    hrows = hermitian_rows()
    hparams = {"L": H_LX, "N": H_N, "V": "-2 sech^2 x", "G": "0",
               "sigma": SIGMA, "beta": H_BETA, "omegas": H_OMEGAS}
    print(f"wrote {write_table(outdir, 'kappa_hermitian_limit', H_HEADER, hrows, hparams)}")


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