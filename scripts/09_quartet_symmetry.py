"""The second spectral symmetry generator, its class boundary, and the
counterexample that makes the quartet hypothesis necessary.

Three experiments:

* the invariance ``lambda -> conj(lambda)`` on Wadati branches with **cubic**
  ``N``, at four resolutions, for an odd ``g`` (``G`` even, non-PT), for a
  **non-odd** ``g`` (neither ``V`` nor ``G`` even, so no parity operator exists
  at all -- the decisive case), and for an even ``g`` (``G`` odd, the PT control
  covered by Corollary ``cor:symB``);
* the mass-fraction filter: the two members of a quartet decay at different
  rates, so an 85%-mass filter is not conjugation-symmetric and reports a
  spurious Corollary 2 defect.  The gap disk does not;
* Remark ``rem:M3real``: a quintic term breaks the quartet -- Corollary 1
  survives at machine level while the Corollary 2 defect grows smoothly with
  ``beta`` -- and mechanism (M3) then occurs, a pair with a common nonzero
  imaginary part that is never real and never collides.

The defects are reported twice.  ``*_isolated`` is measured on the isolated
complex gap modes; ``*_all`` includes the ``lambda = 0`` Jordan block, whose two
members are resolved only to ``~1e-7`` and which therefore dominates the
maximum.  The paper's ``4e-13`` is the isolated figure.

Outputs
-------
data/quartet_symmetry.csv : Table ``tab:quartet``, Section ``sec:quartet``
data/quartet_m3.csv       : Remark ``rem:M3real``
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
from numpy.linalg import eig

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nhvk.kappa import (build_wadati, gap_modes, make_grid, move_omega,
                        sym_defect)
from nhvk.profiles import wadati_even, wadati_gaussian, wadati_shifted

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
SIGMA, A_AMP = -1.0, 1.0
GRIDS = [(20.0, 240), (26.0, 320), (30.0, 384), (36.0, 448)]
CASES = [("wadati_gaussian", wadati_gaussian, -0.5, -0.5, "odd g, even G (non-PT)"),
         ("wadati_shifted", wadati_shifted, -0.5, -0.7, "g not odd"),
         ("wadati_even", wadati_even, -0.5, -0.5, "even g, odd G (PT control)")]
M3_BETAS = [0.00, 0.02, 0.05, 0.10]
M3_GRID, M3_TRACK_GRID = (30.0, 384), (26.0, 320)
M3_TRACK_BETA = 0.10
M3_TRACK_OMEGAS = [-0.24, -0.26, -0.28, -0.30, -0.32, -0.34, -0.36, -0.40, -0.50]

HEADER = ["profile", "g_parity", "L", "N", "omega", "P",
          "cor1_defect_isolated", "cor2_defect_isolated", "cor1_defect_all",
          "cor2_defect_all", "n_gap_modes", "n_complex", "Re_lambda",
          "Im_lambda", "power_balance"]
M3_HEADER = ["kind", "beta", "omega", "P", "cor1_defect", "cor2_defect",
             "modes"]


def run(outdir: Path) -> None:
    rows = []
    print("Quartet symmetry on Wadati branches with cubic N (gap-disk modes)")
    print(f"{'profile':>18} {'(L,N)':>12} {'omega':>7} {'Cor1_iso':>10} "
          f"{'Cor2_iso':>10} {'Cor2_all':>10} {'power bal.':>11}")
    for name, prof, om0, omr, gpar in CASES:
        for (L, N) in GRIDS:
            x, D2, dx = make_grid(L, N)
            M, phi = build_wadati(prof, A_AMP, om0, x, D2, dx, SIGMA)
            if abs(omr - om0) < 1e-12:
                ph, om = phi, om0
            else:
                ph, om = move_omega(M, phi, om0, omr, dom=0.01)
            if ph is None:
                raise RuntimeError(f"{name}: continuation failed at (L,N)=({L},{N})")
            in_disk, nz = gap_modes(M.Lop(ph, om, 1.0), om)
            cx = np.sort_complex(nz[np.abs(nz.imag) > 1e-4])
            cAx = sym_defect(cx, "A")
            cBx = sym_defect(cx, "B")
            cA, cB = sym_defect(in_disk, "A"), sym_defect(in_disk, "B")
            lam = cx[np.argmax(cx.imag)] if len(cx) else 0j
            pb = float(np.sum(M.g*np.abs(ph)**2)*dx)
            rows.append((name, gpar, L, N, om, float(np.sum(np.abs(ph)**2)*dx),
                         cAx, cBx, cA, cB, len(in_disk), len(cx),
                         float(lam.real), float(lam.imag), pb))
            print(f"{name:>18} {str((L, N)):>12} {om:7.3f} {cAx:10.2e} "
                  f"{cBx:10.2e} {cB:10.2e} {pb:11.2e}")
            if len(cx) and cBx > 1e-10:
                raise RuntimeError(f"{name}: Corollary 2 defect {cBx:.2e} on a "
                                   "cubic Wadati branch")
    params = {"grids": [[g[0], g[1]] for g in GRIDS], "A": A_AMP,
              "sigma": SIGMA, "cases": [c[0] for c in CASES],
              "mode_filter": "gap disk |lambda| < 0.999|omega|",
              "note": "*_isolated excludes the lambda = 0 Jordan block"}
    print(f"wrote {write_table(outdir, 'quartet_symmetry', HEADER, rows, params)}")

    # --- the mass-fraction filter is not conjugation-symmetric ----------
    x, D2, dx = make_grid(30.0, 384)
    M, phi = build_wadati(wadati_gaussian, A_AMP, -0.5, x, D2, dx, SIGMA)
    ev, Vc = eig(M.Lop(phi, -0.5, 1.0))
    n, core = len(x), np.abs(x) < 6.0
    pairs = []
    for j in range(len(ev)):
        if abs(ev[j]) < 0.5*0.999 and abs(ev[j].imag) > 1e-4 and ev[j].real > 0:
            w = np.abs(Vc[:, j])**2
            w = w[:n] + w[n:]
            pairs.append((ev[j], float(w[core].sum()/w.sum())))
    pairs.sort(key=lambda t: -t[0].imag)
    print("\nmass fraction in |x| < 6 of the two members of the quartet "
          "at omega = -0.5:")
    for lam, f in pairs:
        print(f"    lambda = {lam.real:+.6f}{lam.imag:+.6f}i    "
              f"mass fraction = {f:.3f}")
    kept = np.array([lam for lam, f in pairs if f >= 0.85])
    if len(kept):
        print(f"    an 85% filter retains {len(kept)} of {len(pairs)} members "
              f"and reports a spurious Corollary 2 defect of "
              f"{sym_defect(kept, 'B'):.1e}")

    # --- quartet breaking by a quintic term, and (M3) -------------------
    m3 = []
    L, N = M3_GRID
    x, D2, dx = make_grid(L, N)
    print(f"\nQuartet breaking by a quintic term, N(s) = -s + beta s^2, "
          f"omega = -0.5, (L, N) = ({L:g}, {N})")
    for beta in M3_BETAS:
        M, ph = build_wadati(wadati_gaussian, A_AMP, -0.5, x, D2, dx, SIGMA,
                             beta=beta)
        in_disk, nz = gap_modes(M.Lop(ph, -0.5, 1.0), -0.5)
        cx = np.sort_complex(nz[np.abs(nz.imag) > 1e-4])
        cA, cB = sym_defect(cx, "A"), sym_defect(cx, "B")
        txt = " ".join(f"{z.real:+.7f}{z.imag:+.7f}i" for z in cx)
        m3.append(("beta_scan", beta, -0.5, float(np.sum(np.abs(ph)**2)*dx),
                   cA, cB, txt))
        print(f"  beta = {beta:4.2f}   Cor1 = {cA:.2e}   Cor2 = {cB:.2e}   {txt}")
        if beta == 0.0 and cB > 1e-10:
            raise RuntimeError("the quartet is already broken at beta = 0")
        if beta >= 0.02 and not cB > 1e-3:
            raise RuntimeError("the quintic term did not break the quartet")

    L, N = M3_TRACK_GRID
    x, D2, dx = make_grid(L, N)
    M, phi = build_wadati(wadati_gaussian, A_AMP, -0.5, x, D2, dx, SIGMA,
                          beta=M3_TRACK_BETA)
    print(f"\n(M3) tracking at beta = {M3_TRACK_BETA}, (L, N) = ({L:g}, {N})")
    cur, om = phi, -0.5
    for tgt in M3_TRACK_OMEGAS:
        p, o = move_omega(M, cur, om, tgt, dom=0.005)
        if p is None:
            raise RuntimeError(f"(M3) continuation failed at omega = {tgt}")
        cur, om = p, o
        in_disk, nz = gap_modes(M.Lop(cur, om, 1.0), om)
        cxo = nz[np.abs(nz.imag) > 1e-4]
        cA, cB = sym_defect(cxo, "A"), sym_defect(cxo, "B")
        txt = " ".join(f"{z.real:+.6f}{z.imag:+.6f}i"
                       for z in np.sort_complex(nz))
        m3.append(("m3_track", M3_TRACK_BETA, om,
                   float(np.sum(np.abs(cur)**2)*dx), cA, cB, txt))
        print(f"  omega = {om:+.3f}   Cor1 = {cA:.1e}   Cor2 = {cB:.1e}   {txt}")
    m3params = {"beta_scan_grid": list(M3_GRID),
                "track_grid": list(M3_TRACK_GRID), "betas": M3_BETAS,
                "track_beta": M3_TRACK_BETA, "track_omegas": M3_TRACK_OMEGAS,
                "A": A_AMP, "sigma": SIGMA,
                "build_order": "A first, then beta"}
    print(f"wrote {write_table(outdir, 'quartet_m3', M3_HEADER, m3, m3params)}")


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
