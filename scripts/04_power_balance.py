"""Admissibility / power-balance checks (Lemma P and hypothesis H5a).

Refactor of ``provenance/power_balance_check.py``.  Verifies:

 (1) Lemma P on a candidate profile: any decaying stationary state must satisfy
     int G |phi|^2 dx = 0, so a sign-definite G admits no stationary state.
     Demonstrated on G = A sech^2(x), for which the power-balance defect
     evaluated on the free soliton is O(1), not small.

 (2) The Wadati factorisation used in H5a, checked symbolically:
        -d_xx - g^2 + i g' = (d_x + i g)(-d_x + i g).

 (3) That the sign convention matters: with an admissible (odd, non-monotone)
     g the family exists for V = -g^2 and fails to exist for V = +g^2.

 (4) Lemma P along the converged Theorem A branch: int G |phi|^2 is satisfied
     automatically to ~1e-12 without ever being imposed.

Outputs
-------
data/power_balance.csv : Appendix F, Configuration II
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

from nhvk.core import Model, make_grid


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
    L, N = 22.0, 300
    x, D2, dx = make_grid(L, N)
    om, sigma = -0.5, -1.0
    seed = np.sqrt(2*abs(om))/np.cosh(np.sqrt(abs(om))*x) + 0j
    rows = []

    print("=" * 74)
    print("(1) Lemma P: a sign-definite G cannot support a stationary state")
    print("=" * 74)
    print("Power-balance defect int G|phi|^2 dx on the free soliton, G = A sech^2:")
    A_defect = [0.05, 0.2, 0.8]
    for A in A_defect:
        G = A / np.cosh(x) ** 2
        defect = np.sum(G*np.abs(seed)**2)*dx
        print(f"    A = {A:<5}  int G|phi|^2 = {defect:.6f}"
              f"   (must be 0)")
        rows.append(("free_soliton_defect", f"A={A}", float(defect)))

    print()
    print("=" * 74)
    print("(2) Symbolic check of the Wadati factorisation")
    print("=" * 74)
    import sympy as sp
    xx = sp.Symbol('x')
    f = sp.Function('f')
    gg = sp.Function('g')
    lhs = -sp.diff(f(xx), xx, 2) - gg(xx)**2*f(xx) + sp.I*sp.diff(gg(xx), xx)*f(xx)
    inner = -sp.diff(f(xx), xx) + sp.I*gg(xx)*f(xx)
    rhs = sp.diff(inner, xx) + sp.I*gg(xx)*inner
    residual = sp.simplify(sp.expand(lhs - rhs))
    print("    (-d_xx - g^2 + i g') f  -  (d_x + i g)(-d_x + i g) f  simplifies to:",
          residual)
    if residual != 0:
        raise RuntimeError(f"Wadati factorisation did not simplify to 0: {residual}")
    rows.append(("wadati_symbolic_residual", "-", 0.0))

    print()
    print("=" * 74)
    print("(3) Sign convention: g odd and NON-monotone, so that g' changes sign")
    print("=" * 74)
    g_ = lambda A: A*x*np.exp(-x**2/2)
    gp_ = lambda A: A*(1-x**2)*np.exp(-x**2/2)
    for sV, label in [(-1, "V = -g^2  (Wadati factorisable form)"),
                      (+1, "V = +g^2  (non-factorisable)")]:
        ph, ok = seed, True
        for Av in np.linspace(0, 0.6, 25)[1:]:
            M = Model(sV*g_(Av)**2, gp_(Av), sigma, x, D2, dx)
            p2 = M.newton(ph, om, 1.0)
            if p2 is None:
                ok = False
                break
            ph = p2
        if ok:
            r = norm(M.res(ph.real, ph.imag, om, 1.0))*np.sqrt(dx)
            print(f"    {label}: family EXISTS, residual = {r:.1e}")
            rows.append((f"family_exists_V_{'minus' if sV < 0 else 'plus'}_g2",
                         "residual", float(r)))
        else:
            print(f"    {label}: FAILS at A = {Av:.3f}")
            rows.append((f"family_fails_V_{'minus' if sV < 0 else 'plus'}_g2",
                         "A_at_failure", float(Av)))

    print()
    print("=" * 74)
    print("(4) Lemma P along the Theorem A branch (never imposed, only checked)")
    print("=" * 74)
    A_target = 1.0
    ph = seed
    for Av in np.linspace(0, A_target, 41)[1:]:
        M = Model(-g_(Av)**2, gp_(Av), sigma, x, D2, dx)
        p2 = M.newton(ph, om, 1.0)
        if p2 is None:
            raise RuntimeError(f"continuation failed at A = {Av}")
        ph = p2
    omegas = [-0.30, -0.26, -0.24, -0.22, -0.20, -0.18]
    for omv in omegas:
        p = ph
        st = np.sign(omv - om)*0.01
        o = om
        while abs(o - omv) > 1e-9:
            if abs(st) > abs(omv - o):
                st = omv - o
            p2 = M.newton(p, o + st, 1.0)
            if p2 is None:
                st /= 2
                continue
            p, o = p2, o + st
        pb = np.sum(gp_(A_target)*np.abs(p)**2)*dx
        print(f"    omega = {omv:6.2f}   int G|phi|^2 = {pb:+.3e}")
        rows.append(("branch_power_balance", f"omega={omv:.2f}", float(pb)))

    params = {"L": L, "N": N, "omega": om, "sigma": sigma,
              "A_free_soliton": A_defect, "A_target": A_target,
              "omegas": omegas, "g": "A*x*exp(-x^2/2)"}
    path = write_table(outdir, "power_balance", ["check", "parameter", "value"],
                       rows, params)
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
