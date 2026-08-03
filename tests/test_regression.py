"""Regression tests pinning the scripts' output to the numbers in the paper.

Each test runs the relevant script in ``scripts/`` into a temporary directory
and checks the resulting CSV against the published table.  Tolerances follow the
precision to which the paper quotes each quantity:

* eigenvalues        ``rtol = 1e-4``  (plus ``atol = 5e-6``: the tables quote
  ``Im lambda`` to five decimals, so half of the last quoted digit is 5e-6, and
  a relative tolerance is meaningless for the entries reported as 0.0)
* ``P`` and ``Q``    ``rtol = 1e-5``
* ``|gamma|``        ``rtol = 1e-2``  (quoted to three significant figures; the
  magnitudes move in the third under grid refinement)

Scripts 02, 03 and 06 are marked ``slow``.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"

RTOL_EIG = 1e-4
ATOL_EIG = 5e-6          # half of the last decimal quoted in the paper's tables
RTOL_PQ = 1e-5
RTOL_GAMMA = 1e-2


# ---------------------------------------------------------------- helpers
def run_script(name: str, outdir: Path) -> str:
    """Run ``scripts/<name>`` into ``outdir``; assert it prints OK and exits 0."""
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / name), "--outdir", str(outdir)],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, f"{name} failed:\n{proc.stdout}\n{proc.stderr}"
    assert proc.stdout.rstrip().endswith("OK"), f"{name} did not report OK"
    return proc.stdout


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def keyed(rows: list[dict], key: str) -> dict[str, float]:
    return {r[key]: float(r["value"]) for r in rows}


def sidecar_ok(outdir: Path, stem: str) -> None:
    """The JSON sidecar exists and carries the provenance fields."""
    import json
    meta = json.loads((outdir / f"{stem}.json").read_text())
    for field in ("script", "git_commit", "timestamp_utc", "python_version",
                  "numpy_version", "scipy_version", "parameters"):
        assert field in meta, f"{stem}.json missing {field}"


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def krein(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("krein")
    run_script("01_krein_reduction.py", d)
    return d


@pytest.fixture(scope="session")
def power(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("power")
    run_script("04_power_balance.py", d)
    return d


@pytest.fixture(scope="session")
def parity(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("parity")
    run_script("05_parity_gamma.py", d)
    return d


@pytest.fixture(scope="session")
def branch(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("branch")
    run_script("02_theorem_a_branch.py", d)
    return d


@pytest.fixture(scope="session")
def convergence(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("convergence")
    run_script("03_theorem_a_convergence.py", d)
    return d


@pytest.fixture(scope="session")
def monodromy(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("monodromy")
    run_script("06_gauge_monodromy.py", d)
    return d


# ================================================================ Table F.1
def test_table_f1_krein_quantity_vanishes(krein):
    """Lemma 2.3: the Krein quantity vanishes identically at the phase mode."""
    q = keyed(read_csv(krein / "krein_identities.csv"), "quantity")
    assert q["abs_chi0_w0"] < 1e-12


def test_table_f1_adjoint_identification(krein):
    """chi_0 is the CP adjoint eigenvector: chi_0 = alpha sigma_3 v^#."""
    q = keyed(read_csv(krein / "krein_identities.csv"), "quantity")
    assert q["rel_chi0_minus_sigma3_vsharp"] < 1e-12


def test_table_f1_structural_identities(krein):
    """L = sigma_3 Lcal, L w_0 = 0, and the kernel of L^dagger is simple."""
    q = keyed(read_csv(krein / "krein_identities.csv"), "quantity")
    assert q["max_L_minus_sigma3_Lcal"] < 1e-12
    assert q["norm_L_w0"] < 1e-10
    assert q["sv_kernel"] < 1e-10
    assert q["sv_gap"] > 1e-2
    assert_allclose(q["cp_ratio"], 1.0, rtol=1e-6)
    sidecar_ok(krein, "krein_identities")


# ================================================================ Table F.2
def _eps_scan(krein):
    rows = read_csv(krein / "eps_scan.csv")
    scan = [r for r in rows if float(r["eps"]) > 0]
    herm = [r for r in rows if float(r["eps"]) == 0]
    assert len(herm) == 1, "eps_scan.csv must end with the exact eps = 0 row"
    return scan, herm[0]


def test_table_f2_conditioning_scaling(krein):
    """(K-1)/eps^2 converges to 0.3046 as eps -> 0."""
    scan, _ = _eps_scan(krein)
    eps = np.array([float(r["eps"]) for r in scan])
    ratio = np.array([float(r["K_minus_1_over_eps2"]) for r in scan])
    order = np.argsort(eps)
    assert abs(ratio[order][0] - 0.3046) < 5e-4
    assert abs(ratio[order][1] - 0.3046) < 5e-4


def test_table_f2_hermitian_limits(krein):
    """Q -> P and kappa -> P' as eps -> 0."""
    scan, herm = _eps_scan(krein)
    assert_allclose(float(herm["Q"]), 3.4077038, rtol=RTOL_PQ)
    assert_allclose(float(herm["kappa"]), -1.7808671, rtol=RTOL_PQ)
    assert_allclose(float(herm["Q_prime"]), -1.7808671, rtol=RTOL_PQ)

    eps = np.array([float(r["eps"]) for r in scan])
    Q = np.array([float(r["Q"]) for r in scan])
    kap = np.array([float(r["kappa"]) for r in scan])
    order = np.argsort(eps)
    dQ = np.abs(Q[order] - float(herm["Q"]))
    dk = np.abs(kap[order] - float(herm["kappa"]))
    assert dQ[0] < 1e-4 and dk[0] < 1e-4
    assert np.all(np.diff(dQ) > 0), "|Q - P| must shrink as eps -> 0"
    assert np.all(np.diff(dk) > 0), "|kappa - P'| must shrink as eps -> 0"


def test_table_f2_kappa_vs_Qprime_is_second_order(krein):
    """|Q' - kappa| = O(eps^2): successive differences fall by a factor 4."""
    scan, _ = _eps_scan(krein)
    eps = np.array([float(r["eps"]) for r in scan])
    diff = np.abs(np.array([float(r["Q_prime"]) - float(r["kappa"]) for r in scan]))
    order = np.argsort(eps)[::-1]                 # eps halving downwards
    eps, diff = eps[order], diff[order]
    assert_allclose(eps[:-1]/eps[1:], 2.0, rtol=1e-12, err_msg="eps must halve")
    ratios = diff[:-1]/diff[1:]
    assert np.all(np.abs(ratios - 4.0) < 0.3), f"ratios = {ratios}"


# ================================================================== Table 1
TABLE_1 = {
    #  omega:  (P,       Q,       max Im lambda)
    -0.30: (2.45452, 0.62595, 0.0),
    -0.26: (2.40276, 0.69878, 0.00400),
    -0.24: (2.37419, 0.72304, 0.13068),
    -0.22: (2.33526, 0.73056, 0.17711),
    -0.20: (2.27525, 0.71547, 0.20354),
    -0.18: (2.18708, 0.67831, 0.21522),
}


def _finest(convergence) -> dict[float, dict]:
    """Rows of the convergence table at the finest grid, (Lx, N) = (22, 320)."""
    rows = read_csv(convergence / "theorem_a_convergence.csv")
    out = {}
    for r in rows:
        if float(r["Lx"]) == 22.0 and int(r["N"]) == 320:
            out[round(float(r["omega"]), 2)] = r
    return out


@pytest.mark.slow
def test_table_1_values(convergence):
    """Table 1: P, Q and max Im lambda at A = 1, (Lx, N) = (22, 320)."""
    got = _finest(convergence)
    assert set(got) == set(TABLE_1), f"missing frequencies: {set(TABLE_1) - set(got)}"
    for om, (P, Q, im) in TABLE_1.items():
        assert_allclose(float(got[om]["P"]), P, rtol=RTOL_PQ, err_msg=f"P at {om}")
        assert_allclose(float(got[om]["Q"]), Q, rtol=RTOL_PQ, err_msg=f"Q at {om}")
        assert_allclose(float(got[om]["Im_lambda"]), im,
                        rtol=RTOL_EIG, atol=ATOL_EIG, err_msg=f"Im lambda at {om}")
    sidecar_ok(convergence, "theorem_a_convergence")


@pytest.mark.slow
def test_table_1_falsification(convergence):
    """Q has an interior maximum where the instability is alive.

    This is the falsification Theorem A rests on: a three-point parabolic fit
    through omega = -0.24, -0.22, -0.20 must place the vertex of Q strictly
    inside the window, while max Im lambda > 0 at all three points.
    """
    got = _finest(convergence)
    oms = np.array([-0.24, -0.22, -0.20])
    Q = np.array([float(got[o]["Q"]) for o in oms])
    im = np.array([float(got[o]["Im_lambda"]) for o in oms])

    assert Q[1] > Q[0] and Q[1] > Q[2], "Q has no interior maximum in [-0.24, -0.20]"
    h = 0.02
    denom = Q[0] - 2*Q[1] + Q[2]
    assert denom < 0, "parabolic fit is not concave"
    vertex = oms[1] + 0.5*h*(Q[0] - Q[2])/denom
    assert abs(vertex - (-0.223)) < 0.005, f"vertex at omega* = {vertex}"
    assert np.all(im > 0.1), f"instability must persist across the window: {im}"


@pytest.mark.slow
def test_theorem_a_branch_sign_change(branch):
    """Script 02: Q' changes sign on the branch while the instability persists."""
    rows = read_csv(branch / "theorem_a_branch.csv")
    om = np.array([float(r["omega"]) for r in rows])
    Q = np.array([float(r["Q"]) for r in rows])
    im = np.array([float(r["max_Im_lambda"]) for r in rows])
    order = np.argsort(om)
    om, Q, im = om[order], Q[order], im[order]

    dQ = np.gradient(Q, om)
    sign_changes = np.where(np.diff(np.sign(dQ)) != 0)[0]
    assert len(sign_changes) >= 1, "Q' does not change sign on the branch"
    i = sign_changes[0]
    om_star = 0.5*(om[i] + om[i+1])
    assert -0.26 < om_star < -0.19, f"Q' turns at omega = {om_star}"
    assert im[i] > 0.0, "the instability must already be present where Q' turns"

    # Q real to round-off along the whole branch (Lemma 2.6 real ray)
    imq = np.abs(np.array([float(r["Im_Q_over_absQ"]) for r in rows]))
    assert imq.max() < 1e-10, f"max |Im Q|/|Q| = {imq.max()}"
    sidecar_ok(branch, "theorem_a_branch")


# ================================================================== Table 2
@pytest.mark.slow
def test_table_2_grid_convergence(convergence):
    """P, Q and Im lambda are grid-converged at omega = -0.24 and -0.22.

    The two Lx = 22 resolutions agree to machine precision, which is the
    5-significant-figure claim of Table 2.  The Lx = 18 domain is a genuinely
    different truncation and deviates by at most ~3e-5 relative, i.e. by one
    unit in the fifth significant figure.
    """
    rows = read_csv(convergence / "theorem_a_convergence.csv")
    for om in (-0.24, -0.22):
        sel = {(float(r["Lx"]), int(r["N"])): r
               for r in rows if abs(float(r["omega"]) - om) < 1e-9}
        assert set(sel) == {(18.0, 220), (22.0, 260), (22.0, 320)}
        for key in ("P", "Q", "Im_lambda"):
            v22a = float(sel[(22.0, 260)][key])
            v22b = float(sel[(22.0, 320)][key])
            v18 = float(sel[(18.0, 220)][key])
            assert_allclose(v22a, v22b, rtol=RTOL_PQ,
                            err_msg=f"{key} at omega={om}, Lx=22 resolutions")
            assert_allclose(v18, v22b, rtol=5e-5,
                            err_msg=f"{key} at omega={om}, Lx=18 vs Lx=22")


# ================================================================== Table 3
TABLE_3 = {
    #  omega:  (odd |gamma|, generic |gamma|)
    -0.8: (3.24, 1.40),
    -1.5: (5.93, 2.48),
    -2.5: (9.96, 4.01),
}


def _parity(parity) -> dict[tuple[float, str], dict]:
    rows = read_csv(parity / "parity_gamma.csv")
    return {(round(float(r["omega"]), 2), r["direction"]): r for r in rows}


def test_table_3_even_directions_vanish(parity):
    """Proposition 2: |gamma| = 0 for every even h."""
    got = _parity(parity)
    for om in TABLE_3:
        g_even = float(got[(om, "even")]["abs_gamma"])
        g_odd = float(got[(om, "odd")]["abs_gamma"])
        assert g_even < 1e-11, f"|gamma| for even h at omega={om} is {g_even}"
        assert g_even < 1e-10 * g_odd, (
            f"even/odd separation at omega={om} is only {g_odd/g_even:.2e}")


def test_table_3_odd_and_generic_magnitudes(parity):
    """The odd and generic directions reproduce the published magnitudes."""
    got = _parity(parity)
    for om, (odd, gen) in TABLE_3.items():
        assert_allclose(float(got[(om, "odd")]["abs_gamma"]), odd,
                        rtol=RTOL_GAMMA, err_msg=f"odd h at omega={om}")
        assert_allclose(float(got[(om, "generic")]["abs_gamma"]), gen,
                        rtol=RTOL_GAMMA, err_msg=f"generic h at omega={om}")
    sidecar_ok(parity, "parity_gamma")


def test_table_3_parity_residual(parity):
    """Im(conj(eta_0) phi_0) is odd -- the mechanism behind Proposition 2."""
    got = _parity(parity)
    for (om, _direction), row in got.items():
        resid = float(row["parity_residual"])
        assert resid < 1e-10, f"parity residual at omega={om} is {resid}"


# ================================================================== Table 7
TABLE_7 = {
    #  A:      (P,        Re Q,      K)
    0.00: (4.600000, 4.600000, 1.000000),
    0.10: (4.602222, 4.434420, 1.077114),
    0.20: (4.608889, 4.050401, 1.294782),
    0.30: (4.620000, 3.633918, 1.616343),
    0.40: (4.635556, 3.283545, 1.993047),
    0.50: (4.655556, 3.019986, 2.376475),
}


@pytest.mark.slow
def test_table_7_values(monodromy):
    """Table 7: P, Re Q and K under continuity transport of chi_0."""
    rows = read_csv(monodromy / "gauge_monodromy.csv")
    got = {round(float(r["A"]), 2): r for r in rows}
    for A, (P, ReQ, K) in TABLE_7.items():
        assert_allclose(float(got[A]["P"]), P, rtol=RTOL_PQ, err_msg=f"P at A={A}")
        assert_allclose(float(got[A]["Re_Q"]), ReQ, rtol=RTOL_PQ,
                        err_msg=f"Re Q at A={A}")
        assert_allclose(float(got[A]["K"]), K, rtol=RTOL_PQ, err_msg=f"K at A={A}")
    sidecar_ok(monodromy, "gauge_monodromy")


@pytest.mark.slow
def test_table_7_transport_keeps_chi0_real(monodromy):
    """Section 5.3: no re-gauging is needed -- Q stays real, S chi_0 = chi_0.

    These three columns are the content of the transport result, not decoration:
    Im Q vanishes, arg c does not wind, and chi_0 stays on the S-real ray.
    """
    rows = read_csv(monodromy / "gauge_monodromy.csv")
    for r in rows:
        A = float(r["A"])
        assert abs(float(r["Im_Q"])) < 1e-10, f"Im Q at A={A}"
        assert abs(float(r["arg_c"])) < 1e-10, f"arg c at A={A}"
        assert float(r["S_residual"]) < 1e-10, f"S residual at A={A}"


@pytest.mark.slow
def test_table_7_conditioning_grows(monodromy):
    """K is strictly increasing in the gain-loss amplitude."""
    rows = sorted(read_csv(monodromy / "gauge_monodromy.csv"),
                  key=lambda r: float(r["A"]))
    K = [float(r["K"]) for r in rows]
    assert all(b > a for a, b in zip(K, K[1:])), f"K not increasing: {K}"


# ============================================================ Power balance
def test_power_balance_on_branch(power):
    """Lemma P holds to round-off at every omega on the Theorem A branch."""
    rows = read_csv(power / "power_balance.csv")
    branch_rows = [r for r in rows if r["check"] == "branch_power_balance"]
    assert len(branch_rows) == 6
    for r in branch_rows:
        assert abs(float(r["value"])) < 1e-10, f"{r['parameter']}: {r['value']}"


def test_power_balance_excludes_sign_definite_G(power):
    """A sign-definite G = A sech^2 has an O(1) power-balance defect."""
    rows = read_csv(power / "power_balance.csv")
    defect = {r["parameter"]: float(r["value"])
              for r in rows if r["check"] == "free_soliton_defect"}
    assert defect["A=0.05"] > 0.07
    assert all(v > 0.07 for v in defect.values())


def test_power_balance_wadati_and_sign_convention(power):
    """The Wadati factorisation is exact and only V = -g^2 supports the family."""
    rows = read_csv(power / "power_balance.csv")
    checks = {r["check"]: float(r["value"]) for r in rows}
    assert checks["wadati_symbolic_residual"] == 0.0
    assert checks["family_exists_V_minus_g2"] < 1e-10
    assert "family_fails_V_plus_g2" in checks
    sidecar_ok(power, "power_balance")
