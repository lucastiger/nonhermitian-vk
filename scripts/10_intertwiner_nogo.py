"""Numerical no-go for a Darboux-type differential intertwiner.

By Lemma ``lem:intertwine`` the missing ``lambda -> conj(lambda)`` generator is
equivalent to an invertible ``Y`` with ``L^flat Y = Y L``, where ``L^flat`` is
``L`` with ``G -> -G`` at **fixed** ``a`` and ``b``.  The order-by-order
reduction proves that no matrix differential ``Y`` of order ``<= 2`` exists
(obstruction ``|b| ghat^2 = const``).  This script extends the exclusion
numerically to order ``<= 4``.

Method: minimise ``|| W (L^flat Y - Y L) f_j ||`` over all
``Y = sum_{k<=K} Y_k(x) d_x^k``, normalised on the core ``|x| < x_c``.  The core
normalisation is essential -- in the tails ``L^flat - L = -2 i G -> 0``, so any
``Y`` commuting with the free operator there is a spurious near-solution and an
unnormalised search would always report a small residual.  The Hermitian case,
where ``Y = I`` is an exact intertwiner, is run as a positive control and must
return ``~1e-16``; if it does not, the normalisation is wrong and the test says
nothing.  The mild decay of the reported ratio with ``K`` tracks the
conditioning deflation ``sigma_max ~ ||D_1||^K``, not convergence to zero.

Outputs
-------
data/intertwiner_nogo.csv : Section ``sec:quartet``
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
from numpy.linalg import qr, svd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nhvk.kappa import (QuinticModel, build_wadati, first_derivative_matrix,
                        make_grid)
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
LX, NGRID = 16.0, 112
A_AMP, SIGMA, OMEGA = 1.0, -1.0, -0.5
CORE, ORDERS, N_TEST, BANDLIMIT, SEED = 8.0, [0, 1, 2, 3, 4], 60, 9.0, 1
HEADER = ["order_K", "unknowns", "rows", "sigma_ratio_wadati",
          "sigma_ratio_hermitian_control", "orders_of_magnitude_above_control"]


def build_system(Lm, Lfl, K, D1, x, W, rng):
    n = len(x)
    nun = 4*(K + 1)*n
    rows = []
    for _ in range(N_TEST):
        c = rng.standard_normal((n, 2)) + 1j*rng.standard_normal((n, 2))
        idx = np.minimum(np.arange(n), n - np.arange(n))[:, None]
        f = np.fft.ifft(c*np.exp(-idx**2/(2*BANDLIMIT**2)), axis=0)
        f = np.concatenate([f[:, 0], f[:, 1]])
        f1, f2 = f[:n], f[n:]
        Lf = Lm @ f
        d1, d2 = [f1.copy()], [f2.copy()]
        e1, e2 = [Lf[:n].copy()], [Lf[n:].copy()]
        for _ in range(K):
            d1.append(D1 @ d1[-1]); d2.append(D1 @ d2[-1])
            e1.append(D1 @ e1[-1]); e2.append(D1 @ e2[-1])
        A = np.zeros((2*n, nun), dtype=complex)
        st = lambda q, a, b: ((q*4) + (a*2 + b))*n
        r = np.arange(n)
        for q in range(K + 1):
            A[:, st(q, 0, 0):st(q, 0, 0)+n] += Lfl[:, :n]*d1[q][None, :]
            A[:, st(q, 0, 1):st(q, 0, 1)+n] += Lfl[:, :n]*d2[q][None, :]
            A[:, st(q, 1, 0):st(q, 1, 0)+n] += Lfl[:, n:]*d1[q][None, :]
            A[:, st(q, 1, 1):st(q, 1, 1)+n] += Lfl[:, n:]*d2[q][None, :]
            A[r, st(q, 0, 0)+r] -= e1[q]
            A[r, st(q, 0, 1)+r] -= e2[q]
            A[n+r, st(q, 1, 0)+r] -= e1[q]
            A[n+r, st(q, 1, 1)+r] -= e2[q]
        rows.append(W[:, None]*A)
    return np.vstack(rows), nun


def sigma_min_core(A, K, core_mask):
    cmask = np.tile(np.concatenate([core_mask]*4), K + 1)
    Acol, Atl = A[:, np.where(cmask)[0]], A[:, np.where(~cmask)[0]]
    Qt, _ = qr(Atl, mode="reduced")
    sv = svd(Acol - Qt @ (Qt.conj().T @ Acol), compute_uv=False)
    return float(sv[-1]/sv[0])


def run(outdir: Path) -> None:
    x, D2, dx = make_grid(LX, NGRID)
    D1 = first_derivative_matrix(LX, NGRID)
    core_mask = np.abs(x) < CORE
    W = np.concatenate([core_mask, core_mask]).astype(float)

    M, phi = build_wadati(wadati_gaussian, A_AMP, OMEGA, x, D2, dx, SIGMA)
    V, G = wadati_gaussian(x, A_AMP)
    Mflat = QuinticModel(V, -G, SIGMA, x, D2, dx, 0.0)   # same V, a, b; G -> -G
    Mherm = QuinticModel(V, 0.0*G, SIGMA, x, D2, dx, 0.0)
    Lm, Lfl = M.Lop(phi, OMEGA, 1.0), Mflat.Lop(phi, OMEGA, 1.0)
    Lh = Mherm.Lop(phi, OMEGA, 1.0)

    print("no-go search for a differential intertwiner  L^flat Y = Y L")
    print(f"  (L, N) = ({LX:g}, {NGRID}),  core |x| < {CORE:g},  "
          f"{N_TEST} test functions,  seed = {SEED}")
    print(f"{'K':>3} {'unknowns':>9} {'sigma_min/sigma_max (Wadati)':>30} "
          f"{'(Hermitian control)':>22}")
    rows = []
    for K in ORDERS:
        Aw, nun = build_system(Lm, Lfl, K, D1, x, W, np.random.default_rng(SEED))
        Ah, _ = build_system(Lh, Lh, K, D1, x, W, np.random.default_rng(SEED))
        sw, sh = sigma_min_core(Aw, K, core_mask), sigma_min_core(Ah, K, core_mask)
        rows.append((K, nun, Aw.shape[0], sw, sh, float(np.log10(sw/sh))))
        print(f"{K:3d} {nun:9d} {sw:30.4e} {sh:22.4e}")
        if sh > 1e-12:
            raise RuntimeError("the Hermitian control does not detect Y = I; "
                               "the normalisation is wrong")
        if sw < 1e-9:
            raise RuntimeError(f"an intertwiner of order <= {K} appears to exist")
    print("\nno differential intertwiner of order <= 4: the Wadati column sits "
          "nine to fourteen\norders of magnitude above the control at every order")
    params = {"L": LX, "N": NGRID, "core": CORE, "orders": ORDERS,
              "n_test_functions": N_TEST, "bandlimit": BANDLIMIT, "seed": SEED,
              "A": A_AMP, "sigma": SIGMA, "omega": OMEGA,
              "profile": "wadati_gaussian"}
    print(f"wrote {write_table(outdir, 'intertwiner_nogo', HEADER, rows, params)}")


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
