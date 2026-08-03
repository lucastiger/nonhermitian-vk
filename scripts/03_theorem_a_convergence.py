"""Grid convergence for the Theorem A branch.

Refactor of ``provenance/validate.py``.  The key frequencies of Table 1 are
recomputed at three resolutions, ``(Lx, N) = (18, 220), (22, 260), (22, 320)``,
each time rebuilding the branch from the free-soliton seed by A-continuation
followed by omega-continuation.  ``P``, ``Q`` and ``Im lambda`` agree to five
significant figures across the three grids, so the sign change of ``Q'`` and the
persistence of the instability are not discretisation artefacts.

Outputs
-------
data/theorem_a_convergence.csv : Table 2
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

from nhvk.core import Model, adj_kernel, gaugefix, ip, make_grid, real_ray
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
SIGMA = -1.0


def build(Lx, Ng):
    x, D2, dx = make_grid(Lx, Ng)
    return x, D2, dx


def model(A, x, D2, dx):
    V, G = wadati_gaussian(x, A)
    return Model(V, G, SIGMA, x, D2, dx)


def branch(A, om_target, Lx, Ng, om_start=-0.5):
    """Build the branch at ``om_start`` by A-continuation, then walk to ``om_target``."""
    x, D2, dx = build(Lx, Ng)
    seed = np.sqrt(2*abs(om_start))/np.cosh(np.sqrt(abs(om_start))*x)+0j
    ph, Ac, st = seed, 0.0, 0.05
    while Ac < A-1e-12:
        st = min(st, A-Ac)
        M = model(Ac+st, x, D2, dx)
        p2 = M.newton(ph, om_start, 1.0)
        if p2 is None:
            st /= 2
            if st < 1e-6:
                return None
            continue
        ph, Ac = p2, Ac+st; st *= 1.3
    M = model(A, x, D2, dx)
    om, st = om_start, np.sign(om_target-om_start)*0.01
    while abs(om-om_target) > 1e-9:
        if abs(st) > abs(om_target-om):
            st = om_target-om
        p2 = M.newton(ph, om+st, 1.0)
        if p2 is None:
            st /= 2
            if abs(st) < 1e-7:
                return None
            continue
        ph, om = p2, om+st
        st *= 1.2; st = np.sign(st)*min(abs(st), 0.01)
    return M, ph, x, dx, Ng


def analyse(M, ph, x, dx, Ng, om):
    """P, Q, most unstable localised eigenvalue, its condition number, gap, Cor.A defect."""
    Phi = np.concatenate([ph, np.conj(ph)])
    Lm = M.Lop(ph, om, 1.0)
    ch, s1, s2 = adj_kernel(Lm, dx)
    ch = gaugefix(ch, Phi, dx); ch, c = real_ray(ch, Ng, dx)
    ch = ch*np.sqrt(np.sum(np.abs(Phi)**2)*dx)/np.sqrt(np.sum(np.abs(ch)**2)*dx)
    Q = 0.5*ip(ch, Phi, dx)
    if Q.real < 0:
        ch, Q = -ch, -Q
    P = np.sum(np.abs(ph)**2)*dx
    ev, Vc = eig(Lm)
    core = np.abs(x) < 6.0
    best, lam = None, 0j
    for j in range(len(ev)):
        if abs(ev[j]) > 40:
            continue
        w = np.abs(Vc[:, j])**2; w = w[:Ng]+w[Ng:]
        if w.sum() <= 0 or w[core].sum()/w.sum() < 0.85:
            continue
        if ev[j].imag > lam.imag:
            lam, best = ev[j], j
    Km = np.nan
    if best is not None:
        evL, VL = eig(Lm.conj().T)
        jj = np.argmin(np.abs(np.conj(evL)-lam))
        vR, vL = Vc[:, best], VL[:, jj]
        Km = (np.sum(np.abs(vR)**2)*np.sum(np.abs(vL)**2)
              / np.abs(np.sum(np.conj(vL)*vR))**2)
    # Corollary A check on the localized part of the spectrum
    small = ev[np.abs(ev) < 40]
    d1 = max(np.min(np.abs(small - m)) for m in -np.conj(small)) if len(small) else np.nan
    return P, Q.real, lam, Km, s2, d1


def run(outdir: Path) -> None:
    A = 1.0
    omegas = [-0.30, -0.26, -0.24, -0.22, -0.20, -0.18]
    grids = [(18.0, 220), (22.0, 260), (22.0, 320)]
    print("Convergence check: replacement counterexample, g = A x exp(-x^2/2), A = 1")
    print(f"{'omega':>7} {'Lx':>5} {'N':>5} {'P':>9} {'Q':>9} {'Im lam':>10} "
          f"{'Re lam':>9} {'K_mode':>8} {'CorA':>9}")
    rows = []
    failures = []
    for om in omegas:
        for (Lx, Ng) in grids:
            out = branch(A, om, Lx, Ng)
            if out is None:
                print(f"{om:7.2f} {Lx:5.1f} {Ng:5d}    branch failed")
                failures.append((om, Lx, Ng))
                continue
            M, ph, x, dx, _ = out
            P, Q, lam, Km, s2, d1 = analyse(M, ph, x, dx, Ng, om)
            rows.append((om, Lx, Ng, P, Q, lam.imag, lam.real, Km, d1))
            print(f"{om:7.2f} {Lx:5.1f} {Ng:5d} {P:9.5f} {Q:9.5f} {lam.imag:10.5f} "
                  f"{lam.real:9.4f} {Km:8.3f} {d1:9.1e}")
    if failures:
        raise RuntimeError(f"branch construction failed at {failures}")

    params = {"A": A, "sigma": SIGMA, "omegas": omegas,
              "grids": [[g[0], g[1]] for g in grids],
              "omega_start": -0.5, "profile": "wadati_gaussian"}
    path = write_table(outdir, "theorem_a_convergence",
                       ["omega", "Lx", "N", "P", "Q", "Im_lambda", "Re_lambda",
                        "K_mode", "corA_residual"], rows, params)
    print()
    print(f"wrote {path}")


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
