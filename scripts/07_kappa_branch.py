"""kappa-VK branch sweep, parity index, and the two Hamiltonian--Hopf collisions.

The Theorem A branch (``g = A x exp(-x^2/2)``, ``V = -g^2``, ``G = g'``,
``A = 1``) is built at ``omega_0 = -0.5`` by continuation in ``A`` and swept in
``omega`` in both directions, with ``chi_0`` transported by **continuity** from
the anchor.  At each frequency the script records ``kappa``, ``Q``, ``P``, the
enclosed discrete spectrum, the unstable count ``n_u``, and the parity product
``(-1)^{n_u} sign kappa``, which Theorem ``thm:parity`` asserts is a constant.

It then locates the two Hamiltonian--Hopf collisions by a secant iteration on
the discriminant of the colliding positive-real pair, and reports ``kappa``
there: ``kappa`` is smooth and nonzero across both, which is why the
unrestricted criterion is false and hypothesis (A2) is not removable.

Two distinct mode sets are used and are not interchangeable: ``n_u`` is counted
on the enclosed spectrum, the Corollary 1/2 defects are measured on the
gap disk only.

Outputs
-------
data/kappa_branch.csv     : Table ``tab:parity``
data/hopf_collisions.csv  : Section ``sec:necessity``
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

from nhvk.kappa import (Branch, build_wadati, chi0_at, classify_gap,
                        enclosed_modes, kappa_at, make_grid, move_omega,
                        n_unstable, secant, sym_defect)
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
LX, NGRID = 22.0, 320
OM_ANCHOR, A_AMP, SIGMA = -0.5, 1.0, -1.0
DOM = 0.01
IM_THRESH = 1e-4
OM_TABLE = [-1.600, -1.000, -0.920, -0.880, -0.600, -0.350, -0.310, -0.270,
            -0.230, -0.150]
OM_SWEEP = sorted(set(OM_TABLE)
                  | set(np.round(np.arange(-1.60, -0.119, 0.08), 3))
                  | {-0.90, -0.86, -0.34, -0.32, -0.30, -0.26, -0.24})

HEADER = ["omega", "P", "Q", "kappa", "Im_kappa_over_abskappa", "P_prime",
          "n_unstable", "sign_kappa", "parity_eps", "n_real_pairs",
          "n_imag_pairs", "n_quartets", "n_enclosed", "n_gap_disk",
          "cor1_defect", "cor2_defect", "power_balance", "in_table_parity"]


def run(outdir: Path) -> None:
    x, D2, dx = make_grid(LX, NGRID)
    M, phi = build_wadati(wadati_gaussian, A_AMP, OM_ANCHOR, x, D2, dx, SIGMA)
    print(f"branch built at (L, N) = ({LX:g}, {NGRID}), A = {A_AMP}, "
          f"omega_0 = {OM_ANCHOR}")

    res = {}
    for direction in ("down", "up"):
        targets = [o for o in OM_SWEEP
                   if (o < OM_ANCHOR if direction == "down" else o > OM_ANCHOR)]
        targets.sort(reverse=(direction == "down"))
        ph, om, chi_prev = phi.copy(), OM_ANCHOR, None
        for tgt in targets:
            p, o = move_omega(M, ph, om, tgt, dom=DOM)
            if p is None:
                print(f"  {direction}: stopped at omega = {o:.4f}")
                break
            ph, om = p, o
            chi, Q, _, _, _, Lm = chi0_at(M, ph, om, dx, chi_prev=chi_prev)
            chi_prev = chi
            kap, Pp, _ = kappa_at(M, ph, om, dx, chi)
            enc, in_disk = enclosed_modes(Lm, om, x, LX)
            nz = enc[np.abs(enc) > 1e-5]
            nu_ = n_unstable(enc, IM_THRESH)
            nr, ni, nq = classify_gap(nz)
            sk = 1 if kap.real > 0 else -1
            res[round(om, 6)] = (
                om, float(np.sum(np.abs(ph)**2)*dx), float(Q.real),
                float(kap.real), float(abs(kap.imag)/max(abs(kap), 1e-300)),
                Pp, nu_, sk, ((-1)**nu_)*sk, nr, ni, nq, len(enc),
                len(in_disk), sym_defect(in_disk, "A"), sym_defect(in_disk, "B"),
                float(np.sum(M.g*np.abs(ph)**2)*dx),
                int(round(om, 3) in [round(t, 3) for t in OM_TABLE]))

    rows = [res[k] for k in sorted(res)]
    if not rows:
        raise RuntimeError("omega-continuation produced no points")
    show = ["omega", "P", "Q", "kappa", "n_unstable", "sign_kappa",
            "parity_eps", "n_real_pairs", "n_imag_pairs", "n_quartets",
            "cor1_defect", "cor2_defect"]
    idx = [HEADER.index(c) for c in show]
    print()
    print(" ".join(f"{c:>12}" for c in show))
    for r in rows:
        print(" ".join(
            (f"{r[i]:12.6f}" if isinstance(r[i], float) and abs(r[i]) > 1e-4
             else (f"{r[i]:12.2e}" if isinstance(r[i], float) else f"{r[i]:12d}"))
            for i in idx))

    eps_set = {r[HEADER.index("parity_eps")] for r in rows}
    if eps_set != {1}:
        raise RuntimeError(f"parity constant is not +1 everywhere: {eps_set}")
    print(f"\nparity identity (-1)^n_u = eps sign(kappa) holds with eps = +1 "
          f"at all {len(rows)} sampled frequencies")

    params = {"L": LX, "N": NGRID, "A": A_AMP, "sigma": SIGMA,
              "omega_0": OM_ANCHOR, "domega_max": DOM,
              "im_threshold": IM_THRESH, "profile": "wadati_gaussian",
              "chi0_gauge": "H18 real ray, H13(ii) norm, continuity transport",
              "n_u_mode_set": "enclosed: gap disk + localised modes above 3/(2L)",
              "symmetry_mode_set": "gap disk |lambda| < 0.999|omega| only"}
    print(f"\nwrote {write_table(outdir, 'kappa_branch', HEADER, rows, params)}")

    # --- the two Hamiltonian--Hopf collisions ---------------------------
    B = Branch(M, phi.copy(), OM_ANCHOR, dx, dom=DOM)

    def disc(om: float) -> float:
        """``(lambda_1 - lambda_2)^2`` of the colliding positive-real pair."""
        ph = B.at(om)
        ev = np.linalg.eigvals(M.Lop(ph, om, 1.0))
        sel = [z for z in ev if z.real > 0.05 and abs(z) < abs(om)*0.998]
        if len(sel) < 2:
            return float("nan")
        by_im = sorted(sel, key=lambda z: -abs(z.imag))
        if abs(by_im[0].imag) > 1e-7:
            return -(2*by_im[0].imag)**2
        by_re = sorted(sel, key=lambda z: z.real)
        return (by_re[-1].real - by_re[-2].real)**2

    hdr = ["name", "omega", "Re_lambda", "abs_kappa", "abs_Q", "discriminant"]
    hopf = []
    for name, a, b in (("HH1", -0.322, -0.330), ("HH2", -0.895, -0.910)):
        r, _ = secant(disc, a, b, tol=1e-9)
        ph = B.at(r)
        chi, Q, _, _, _, Lm = chi0_at(M, ph, r, dx)
        kap, _, _ = kappa_at(M, ph, r, dx, chi)
        ev = np.linalg.eigvals(Lm)
        sel = sorted([z for z in ev if z.real > 0.05 and abs(z) < abs(r)*0.998],
                     key=lambda z: z.real)[-2:]
        hopf.append((name, r, float(sel[-1].real), float(abs(kap)),
                     float(abs(Q.real)), disc(r)))
        print(f"{name}: omega = {r:.7f}   collision at Re lambda = "
              f"{sel[-1].real:+.6f}   |kappa| = {abs(kap):.5f} "
              f"(smooth and nonzero: invisible to kappa)")
    hparams = {"L": LX, "N": NGRID, "A": A_AMP, "sigma": SIGMA,
               "bracket_HH1": [-0.322, -0.330],
               "bracket_HH2": [-0.895, -0.910], "secant_tol": 1e-9}
    print(f"wrote {write_table(outdir, 'hopf_collisions', hdr, hopf, hparams)}")


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