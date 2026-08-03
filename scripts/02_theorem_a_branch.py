"""Theorem A branch: omega-sweep of a non-PT Wadati soliton family.

Refactor of ``provenance/sweep2.py``.  The branch is built at ``omega_0 = -0.5``
by continuation in the gain-loss amplitude ``A`` up to ``A = 1``, then swept in
``omega`` in both directions at fixed ``A``.  At each frequency the quasi-power
``Q``, the power ``P`` and the most unstable localised eigenvalue of ``L(omega)``
are recorded.  ``Q'(omega)`` changes sign while ``max Im lambda`` stays strictly
positive: this is the falsification on which Theorem A rests.

Outputs
-------
data/theorem_a_branch.csv : Table 1
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
def run(outdir: Path) -> None:
    L, N = 20.0, 240
    x, D2, dx = make_grid(L, N)
    sigma = -1.0

    def model(A):
        V, G = wadati_gaussian(x, A)
        return Model(V, G, sigma, x, D2, dx)

    # --- stage 1: build the branch at omega0 by A-continuation ---
    OM = -0.5
    A_target = 1.0
    seed = np.sqrt(2*abs(OM))/np.cosh(np.sqrt(abs(OM))*x)+0j
    ph, Acur, step = seed, 0.0, 0.05
    while Acur < A_target - 1e-12:
        step = min(step, A_target-Acur)
        M = model(Acur+step)
        p2 = M.newton(ph, OM, 1.0)
        if p2 is None:
            step /= 2
            if step < 1e-5:
                break
            continue
        ph, Acur = p2, Acur+step
        step *= 1.3
    print(f"A-continuation at omega={OM} reached A={Acur:.4f}")
    if abs(Acur - A_target) > 1e-9:
        raise RuntimeError(f"A-continuation stalled at A = {Acur}")
    A = Acur
    M = model(A)

    def loc_eigs(Lm, N, xg, maxabs=60.0, locfrac=0.85):
        """Eigenpairs of L that are bounded and concentrated on |x| < 6."""
        ev, Vc = eig(Lm)
        keep = []
        core = np.abs(xg) < 6.0
        for j in range(len(ev)):
            if abs(ev[j]) > maxabs:
                continue
            w = np.abs(Vc[:, j])**2
            w = w[:N] + w[N:]
            if w.sum() <= 0:
                continue
            if w[core].sum()/w.sum() < locfrac:
                continue
            keep.append(j)
        return ev[keep], Vc[:, keep]

    def mode_condition(Lm, lam, vR):
        """Condition number K of the isolated eigenvalue lam of L."""
        evL, VL = eig(Lm.conj().T)
        jj = np.argmin(np.abs(np.conj(evL)-lam))
        vL = VL[:, jj]
        return (np.sum(np.abs(vR)**2)*np.sum(np.abs(vL)**2)
                / np.abs(np.sum(np.conj(vL)*vR))**2)

    def observe(M, ph, om):
        Phi = np.concatenate([ph, np.conj(ph)])
        Lm = M.Lop(ph, om, 1.0)
        ch, s1, s2 = adj_kernel(Lm, dx)
        ch = gaugefix(ch, Phi, dx)
        ch, c = real_ray(ch, N, dx)
        ch = ch*np.sqrt(np.sum(np.abs(Phi)**2)*dx)/np.sqrt(np.sum(np.abs(ch)**2)*dx)
        Q = 0.5*ip(ch, Phi, dx)
        if Q.real < 0:
            Q = -Q
        ev, Vc = loc_eigs(Lm, N, x)
        if len(ev):
            j = np.argmax(ev.imag); lam = ev[j]
            Km = mode_condition(Lm, lam, Vc[:, j])
        else:
            lam = 0j
            Km = np.nan
        P = np.sum(np.abs(ph)**2)*dx
        return P, Q, lam, np.angle(c), s1, s2, Km

    # --- stage 2: omega-continuation in both directions at fixed A ---
    # Note (behaviour preserved from provenance/sweep2.py): the "down" loop
    # guard is `om > lo` with lo = OM, so it is false at the first test and the
    # downward leg contributes no points.  The reported branch is the upward
    # leg, omega in [-0.48, -0.14], which is the range covered by Table 1.
    res = {}
    for direction, lo, hi in [("down", OM, -2.6), ("up", OM, -0.15)]:
        p = ph.copy(); om = OM
        stp = -0.02 if direction == "down" else 0.02
        while (om > lo if direction == "down" else om < hi):
            om_new = om + stp
            p2 = M.newton(p, om_new, 1.0)
            if p2 is None:
                stp /= 2
                if abs(stp) < 1e-4:
                    break
                continue
            p, om = p2, om_new
            stp *= 1.2
            stp = np.sign(stp)*min(abs(stp), 0.02)
            if abs(round(om, 3)*100 % 5) < 1e-6 or True:
                res[round(om, 4)] = observe(M, p, om)
        print(f"  {direction}: stopped at omega = {om:.4f}")

    oms = sorted(res)
    if not oms:
        raise RuntimeError("omega-continuation produced no points")
    print(f"\nA = {A:.3f};  {len(oms)} omega points from {oms[0]:.3f} to {oms[-1]:.3f}")
    print(f"{'omega':>8} {'P':>9} {'Q':>9} {'ImQ/|Q|':>9} {'maxIm lam':>10} {'Re lam':>9} {'gap':>9}")
    sel = oms[::max(1, len(oms)//24)]
    for o in sel:
        P, Q, lam, argc, s1, s2, Km = res[o]
        print(f"{o:8.3f} {P:9.5f} {Q.real:9.5f} {Q.imag/abs(Q):9.1e} "
              f"{lam.imag:10.5f} {lam.real:9.4f} {s2:9.2e}")

    # find sign changes of Q' and P'
    Qs = np.array([res[o][1].real for o in oms]); Ps = np.array([res[o][0] for o in oms])
    Ims = np.array([res[o][2].imag for o in oms]); O = np.array(oms)
    dQ = np.gradient(Qs, O); dP = np.gradient(Ps, O)
    print("\nsign changes:")
    for nm, d in [("Q'", dQ), ("P'", dP)]:
        s = np.sign(d); idx = np.where(np.diff(s) != 0)[0]
        for i in idx:
            print(f"   {nm} changes sign near omega = {0.5*(O[i]+O[i+1]):.4f}"
                  f"   (maxIm lambda there = {Ims[i]:+.5f})")
    print(f"\nmax Im(lambda) over the whole branch: {Ims.max():+.5f}; min: {Ims.min():+.5f}")

    rows = []
    for o in oms:
        P, Q, lam, argc, s1, s2, Km = res[o]
        rows.append((o, P, Q.real, Q.imag/abs(Q), lam.imag, lam.real, Km))
    params = {
        "L": L, "N": N, "A": A, "sigma": sigma, "omega_0": OM,
        "omega_range_down": -2.6, "omega_range_up": -0.15,
        "profile": "wadati_gaussian",
    }
    path = write_table(outdir, "theorem_a_branch",
                       ["omega", "P", "Q", "Im_Q_over_absQ", "max_Im_lambda",
                        "Re_lambda", "K_mode"], rows, params)
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
