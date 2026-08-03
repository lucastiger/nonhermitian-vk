"""Parity obstruction (Proposition 2, Section 3.7).

Refactor of ``provenance/parity_gamma.py``.  Computes the exceptional-point
splitting coefficient

    gamma = (-i / kappa(omega)) * < chi_0 , i h(x) w_0 >                (Eq. 13-14)

for even, odd and generic (neither-parity) perturbation directions h, on the
PT-symmetric base

    D  = -d_xx + 0.3 sech^2(x)      (V even)
    G0 = -A sech(x) tanh(x)         (odd  ->  PT-symmetric),  A = 0.3
    N(s) = sigma s,  sigma = -1     (focusing cubic)

at omega in {-0.8, -1.5, -2.5}.

Proposition 2 asserts gamma == 0 identically for every EVEN h.

Note on well-posedness of the reported quantity.  Both the bracket
B = <chi_0, i h w_0> and kappa = <chi_0, d_omega Phi_0> are linear in chi_0, so
the ratio |gamma| = |B| / |kappa| is independent of the normalisation of chi_0.
It is likewise invariant under the residual U(1) phase of phi_0, because chi_0
and w_0 both carry the same Lambda = diag(e^{i th}, e^{-i th}) factor
(Lemma 2.4).  |gamma| is therefore a gauge-invariant number and needs no gauge
fixing; the gauges are nevertheless imposed below so that the parity structure
of Im(conj(eta_0) phi_0) can be checked explicitly.

Normalisation of h: each test direction is scaled to sup-norm 1, so that the
three magnitudes are directly comparable.  This choice is a convention and is
stated in the paper alongside the numbers.

Outputs
-------
data/parity_gamma.csv : Table 3
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

from nhvk.core import (Model, adj_kernel, gaugefix, ip, make_grid, pt_gauge,
                       real_ray, reflect)
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
def test_directions(xg):
    """The three perturbation directions h, each scaled to sup-norm 1."""
    h_even = 1.0 / np.cosh(xg) ** 2
    h_odd = np.tanh(xg) / np.cosh(xg)
    h_gen = 1.0 / np.cosh(xg - 1.0) ** 2          # neither even nor odd
    out = {}
    for name, key, h in [("even   h=sech^2(x)", "even", h_even),
                         ("odd    h=sech(x)tanh(x)", "odd", h_odd),
                         ("generic h=sech^2(x-1)", "generic", h_gen)]:
        out[name] = (key, h / np.max(np.abs(h)))
    return out


def run(outdir: Path) -> None:
    # ------------------------------------------------------------- setup
    Lx, Ng = 20.0, 240
    x, D2, dx = make_grid(Lx, Ng)
    sigma = -1.0
    V0 = 0.3
    V, g = pt_barrier(x, V0, 1.0)   # V = +0.3 sech^2 (even), g = -sech tanh (odd -> PT)
    A = 0.3
    hh = 2e-3                                     # finite-difference step in omega

    M = Model(V, g, sigma, x, D2, dx)

    def solve_base(om, A_target, nsteps=30):
        """Free-soliton seed, then continuation in the gain-loss amplitude."""
        seed = np.sqrt(2 * abs(om)) / np.cosh(np.sqrt(abs(om)) * x) + 0j
        ph = M.newton(seed, om, 0.0)
        if ph is None:
            return None
        for e in np.linspace(0, A_target, nsteps + 1)[1:]:
            p2 = M.newton(ph, om, e)
            if p2 is None:
                return None
            ph = p2
        return pt_gauge(ph, dx)

    print("=" * 78)
    print("Proposition 2 -- parity obstruction:  |gamma| = |<chi_0, i h w_0>| / |kappa|")
    print(f"base: V = 0.3 sech^2(x),  G0 = -A sech(x)tanh(x),  A = {A},  sigma = {sigma}")
    print(f"grid: Lx = {Lx}, N = {Ng}")
    print("=" * 78)
    print(f"{'omega':>7} {'kernel sv':>11} {'gap':>9} {'|kappa|':>10} "
          f"{'parity resid':>13} {'direction':>26} {'|gamma|':>12}")

    omegas = [-0.8, -1.5, -2.5]
    rows = []
    csv_rows = []
    for om in omegas:
        ph = solve_base(om, A)
        if ph is None:
            raise RuntimeError(f"base continuation failed at omega={om}")

        Phi = np.concatenate([ph, np.conj(ph)])
        w0 = np.concatenate([1j * ph, -1j * np.conj(ph)])
        Lm = M.Lop(ph, om, A)

        chi0, sv_kernel, sv_gap = adj_kernel(Lm, dx)
        chi0 = gaugefix(chi0, Phi, dx)
        chi0, _ = real_ray(chi0, Ng, dx)           # H18 local reality gauge

        # kappa = <chi_0, d_omega Phi>, unrescaled pairing (Eq. 12)
        php = M.newton(ph, om + hh, A)
        phm = M.newton(ph, om - hh, A)
        php = pt_gauge(php, dx)
        phm = pt_gauge(phm, dx)
        dphi = (php - phm) / (2 * hh)
        dPhi = np.concatenate([dphi, np.conj(dphi)])
        kappa = ip(chi0, dPhi, dx)

        # explicit check of the parity structure used in the proof:
        # Im(conj(eta_0) phi_0) must be an odd function of x
        eta0 = chi0[:Ng]
        prod_im = np.imag(np.conj(eta0) * ph)
        parity_resid = (np.max(np.abs(prod_im + reflect(prod_im)))
                        / np.max(np.abs(prod_im)))

        for name, (key, h) in test_directions(x).items():
            hw0 = np.concatenate([1j * h * w0[:Ng], 1j * h * w0[Ng:]])
            B = ip(chi0, hw0, dx)
            # |gamma| = |B| / |kappa|: both slots are linear in chi_0, so the
            # ratio is independent of how chi_0 is normalised -- no convention
            # is needed here.
            gam = -1j * B / kappa
            rows.append((om, name, abs(gam)))
            csv_rows.append((om, key, float(abs(gam)), float(abs(kappa)),
                             float(parity_resid)))
            tag = (f"{om:7.2f} {sv_kernel:11.2e} {sv_gap:9.2e} {abs(kappa):10.5f} "
                   f"{parity_resid:13.2e}") if name.startswith("even") else " " * 53
            print(f"{tag} {name:>26} {abs(gam):12.4e}")

    print()
    print("Summary  (Proposition 2 predicts |gamma| = 0 exactly for even h):")
    print(f"{'direction':>26} " + "".join(f"{f'omega={o}':>16}" for o in omegas))
    for name in test_directions(x):
        vals = [v for (o, n, v) in rows if n == name]
        print(f"{name:>26} " + "".join(f"{v:16.4e}" for v in vals))

    params = {"Lx": Lx, "N": Ng, "V0": V0, "A": A, "sigma": sigma,
              "omegas": omegas, "hh": hh, "profile": "pt_barrier",
              "directions": ["sech^2(x)", "sech(x)tanh(x)", "sech^2(x-1)"],
              "h_normalisation": "sup-norm 1"}
    path = write_table(outdir, "parity_gamma",
                       ["omega", "direction", "abs_gamma", "abs_kappa",
                        "parity_residual"], csv_rows, params)
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
