"""Gauge monodromy / amplitude sweep (Section 5.3).

Refactor of ``provenance/gauge_monodromy.py``.  Tracks the biorthogonal
quasi-power

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
therefore attributable to chi_0 alone, which is the point of the test.  No
re-gauging of chi_0 is applied anywhere along the sweep.

Also reported is the S-eigenvalue c, defined by S chi_0 = c chi_0 (Lemma 2.6),
together with its cumulative unwrapped phase, since a monodromy obstruction is
precisely a nonzero winding of arg c around the path, and the residual
S_residual = ||S chi_0 - chi_0|| / ||chi_0||, which is the direct evidence that
transport alone keeps chi_0 on the real ray.

Outputs
-------
data/gauge_monodromy.csv : Table 7
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

from nhvk.core import (Model, S_eigenvalue, S_map, adj_kernel, ip, make_grid,
                       pt_gauge, transport)
from nhvk.profiles import pt_barrier


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
    # ------------------------------------------------------------- setup
    Lx, Ng = 20.0, 240
    x, D2, dx = make_grid(Lx, Ng)
    sigma = -1.0
    V0 = 0.3
    V, g = pt_barrier(x, V0, 1.0)   # V = +0.3 sech^2 (even), g = -sech tanh (odd -> PT)
    om = -1.0
    A_max = 0.5
    dA = 0.005                                # fine steps: phase tracking must
                                              # be unambiguous between steps
    M = Model(V, g, sigma, x, D2, dx)

    # -------------------------------------------- Hermitian anchor, A = 0
    seed = np.sqrt(2 * abs(om)) / np.cosh(np.sqrt(abs(om)) * x) + 0j
    ph = M.newton(seed, om, 0.0)
    if ph is None:
        raise RuntimeError("Hermitian anchor failed")
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
            if p2 is None:
                raise RuntimeError(f"continuation failed at A = {A}")
            ph = pt_gauge(p2, dx)
            Phi = np.concatenate([ph, np.conj(ph)])
            Lm = M.Lop(ph, om, A)
            v, sv, gap = adj_kernel(Lm, dx)
            # ---- naive continuity transport: maximise overlap with previous chi_0
            v = transport(v, chi_prev, dx)
            # ---- canonical norm ||chi_0|| = ||Phi||  (H13(ii))
            v = v * np.sqrt(np.sum(np.abs(Phi) ** 2) * dx) / np.sqrt(
                np.sum(np.abs(v) ** 2) * dx)
            chi_prev = v
        else:
            sv, gap = 0.0, np.nan

        Q = 0.5 * ip(chi_prev, Phi, dx)
        P = np.sum(np.abs(ph) ** 2) * dx
        c = S_eigenvalue(chi_prev, Ng, dx)
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

    params = {"Lx": Lx, "N": Ng, "V0": V0, "sigma": sigma, "omega": om,
              "A_max": A_max, "dA": dA, "report_at": report_at,
              "profile": "pt_barrier", "regauging": "none",
              "argc_unwrapped_total": float(argc_unwrapped)}
    path = write_table(outdir, "gauge_monodromy",
                       ["A", "P", "Re_Q", "Im_Q", "Im_Q_over_absQ", "K", "arg_c",
                        "S_residual"], rows, params)
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
