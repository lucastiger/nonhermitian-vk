"""Regression tests for the kappa-VK scripts (07-11).

Companion to ``test_regression.py``; the helpers and tolerance philosophy are
the same.  Each test runs the relevant script into a temporary directory and
checks the CSV against the numbers printed in the paper, at the precision to
which the paper quotes them.

Epistemic note.  These tests pin *numerical* results.  Where the manuscript
tags a claim PROVEN (the master identity, Target B, the exchange law, the K_2
closure), the numerics corroborate a theorem and a failure means the
implementation drifted.  Where it tags a claim NUMERICALLY SUPPORTED (the
quartet on Wadati branches, the order <= 4 no-go, mechanism M3), the test is the
evidence itself and must not be read as more than that.

Script 10 is fast; 07, 08, 09 and 11 are marked ``slow``.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"

RTOL_KAPPA = 1e-4          # kappa is quoted to five decimals in tab:parity
ATOL_OMEGA = 5e-6          # omega_HH is quoted to seven decimals; grids differ
RTOL_K2 = 1e-5             # K_2 is quoted to six decimals in tab:K2


def run_script(name: str, outdir: Path) -> str:
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


def sidecar_ok(outdir: Path, stem: str) -> None:
    meta = json.loads((outdir / f"{stem}.json").read_text())
    for field in ("script", "git_commit", "timestamp_utc", "python_version",
                  "numpy_version", "scipy_version", "parameters"):
        assert field in meta, f"{stem}.json missing {field}"


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def nogo(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("nogo")
    run_script("10_intertwiner_nogo.py", d)
    return d


@pytest.fixture(scope="session")
def kbranch(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("kbranch")
    run_script("07_kappa_branch.py", d)
    return d


@pytest.fixture(scope="session")
def localbif(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("localbif")
    run_script("08_kappa_local_bifurcation.py", d)
    return d


@pytest.fixture(scope="session")
def quartet(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("quartet")
    run_script("09_quartet_symmetry.py", d)
    return d


@pytest.fixture(scope="session")
def k2(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("k2")
    run_script("11_k2_closure.py", d)
    return d


# ================================================ script 10 (fast, no-go)
def test_nogo_hermitian_control_detects_the_identity(nogo):
    """The control must find Y = I, or the normalisation is wrong."""
    rows = read_csv(nogo / "intertwiner_nogo.csv")
    for r in rows:
        assert float(r["sigma_ratio_hermitian_control"]) < 1e-12


def test_nogo_no_intertwiner_up_to_order_four(nogo):
    """NUMERICALLY SUPPORTED: no differential Y of order <= 4."""
    rows = read_csv(nogo / "intertwiner_nogo.csv")
    assert {int(r["order_K"]) for r in rows} == {0, 1, 2, 3, 4}
    for r in rows:
        assert float(r["sigma_ratio_wadati"]) > 1e-9
        assert float(r["orders_of_magnitude_above_control"]) > 8.0
    sidecar_ok(nogo, "intertwiner_nogo")


# ================================================ script 08 (tab:partI)
@pytest.mark.slow
def test_target_b_omega_kappa_equals_omega_c(localbif):
    """PROVEN (thm:targetB): the two roots coincide, not merely agree."""
    rows = read_csv(localbif / "kappa_local_bifurcation.csv")
    assert len(rows) == 4
    for r in rows:
        assert abs(float(r["omega_diff"])) < 1e-10
        assert_allclose(float(r["omega_kappa"]), -0.2600177, atol=5e-6)


@pytest.mark.slow
def test_local_bifurcation_jordan_data(localbif):
    """tab:partI: J_4 block, kappa_2 = 0, kappa_3 imaginary, C = -i/kappa_3."""
    rows = read_csv(localbif / "kappa_local_bifurcation.csv")
    for r in rows:
        assert int(r["block_dim"]) == 4
        assert float(r["abs_kappa_2"]) < 1e-9
        assert abs(float(r["kappa_3_re"])) < 1e-6
        assert_allclose(float(r["kappa_3_im"]), -36.163, rtol=1e-3)
        assert_allclose(float(r["C_predicted"]), float(r["C_measured"]),
                        rtol=1e-3)
        assert_allclose(float(r["C_predicted"]), 0.02765, rtol=1e-3)
    sidecar_ok(localbif, "kappa_local_bifurcation")


@pytest.mark.slow
def test_hermitian_limit_recovers_classical_vk(localbif):
    """kappa = P' to round-off and C < 0, i.e. stable iff P' < 0."""
    rows = read_csv(localbif / "kappa_hermitian_limit.csv")
    for r in rows:
        assert float(r["abs_kappa_minus_P_prime"]) < 1e-10
        assert float(r["C_measured"]) < 0
        assert float(r["C_from_kappa_3"]) < 0
    sidecar_ok(localbif, "kappa_hermitian_limit")


# ================================================ script 07 (tab:parity)
@pytest.mark.slow
def test_parity_index_is_constant(kbranch):
    """PROVEN CONDITIONAL (thm:parity): (-1)^{n_u} sign kappa is constant."""
    rows = read_csv(kbranch / "kappa_branch.csv")
    assert len(rows) > 20
    assert {int(r["parity_eps"]) for r in rows} == {1}
    sidecar_ok(kbranch, "kappa_branch")


@pytest.mark.slow
def test_parity_table_rows(kbranch):
    """The ten frequencies quoted in tab:parity, with their n_u."""
    rows = {round(float(r["omega"]), 3): r for r in
            read_csv(kbranch / "kappa_branch.csv")}
    expected = {-1.600: (1.08841, 0), -1.000: (1.33213, 0),
                -0.920: (1.36770, 0), -0.880: (1.38535, 2),
                -0.600: (1.48666, 2), -0.350: (1.24151, 2),
                -0.310: (0.96058, 0), -0.270: (0.29077, 0),
                -0.230: (-1.36714, 1), -0.150: (-5.21205, 1)}
    for om, (kap, nu) in expected.items():
        assert om in rows, f"omega = {om} missing from the sweep"
        assert_allclose(float(rows[om]["kappa"]), kap, rtol=RTOL_KAPPA)
        assert int(rows[om]["n_unstable"]) == nu


@pytest.mark.slow
def test_hopf_collisions_are_invisible_to_kappa(kbranch):
    """sec:necessity: n_u jumps by 2 with kappa smooth and positive."""
    rows = {r["name"]: r for r in read_csv(kbranch / "hopf_collisions.csv")}
    assert_allclose(float(rows["HH1"]["omega"]), -0.3253331, atol=ATOL_OMEGA)
    assert_allclose(float(rows["HH2"]["omega"]), -0.9081568, atol=ATOL_OMEGA)
    assert_allclose(float(rows["HH1"]["abs_kappa"]), 1.09543, rtol=1e-3)
    assert_allclose(float(rows["HH2"]["abs_kappa"]), 1.37294, rtol=1e-3)
    sidecar_ok(kbranch, "hopf_collisions")


# ================================================ script 09 (tab:quartet)
@pytest.mark.slow
def test_quartet_exact_on_cubic_wadati_branches(quartet):
    """NUMERICALLY SUPPORTED: Corollary 2 holds, including for non-odd g."""
    rows = read_csv(quartet / "quartet_symmetry.csv")
    profiles = {r["profile"] for r in rows}
    assert {"wadati_gaussian", "wadati_shifted", "wadati_even"} <= profiles
    for r in rows:
        if int(r["n_complex"]) == 0:
            continue
        assert float(r["cor1_defect_isolated"]) < 1e-10
        assert float(r["cor2_defect_isolated"]) < 1e-10
        assert abs(float(r["power_balance"])) < 1e-11
    sidecar_ok(quartet, "quartet_symmetry")


@pytest.mark.slow
def test_quintic_term_breaks_the_quartet_and_m3_occurs(quartet):
    """rem:M3real: Corollary 1 survives, Corollary 2 fails, M3 appears."""
    rows = [r for r in read_csv(quartet / "quartet_m3.csv")
            if r["kind"] == "beta_scan"]
    by_beta = {round(float(r["beta"]), 3): r for r in rows}
    assert float(by_beta[0.0]["cor2_defect"]) < 1e-10
    assert float(by_beta[0.02]["cor2_defect"]) > 1e-2
    assert float(by_beta[0.1]["cor2_defect"]) > 1e-1
    for r in rows:
        assert float(r["cor1_defect"]) < 1e-6
    track = [r for r in read_csv(quartet / "quartet_m3.csv")
             if r["kind"] == "m3_track"]
    assert len(track) >= 5
    for r in track:
        assert float(r["cor2_defect"]) > 1e-3      # quartet genuinely absent
    sidecar_ok(quartet, "quartet_m3")


# ================================================ script 11 (tab:K2)
@pytest.mark.slow
def test_k2_formula_matches_direct_extrapolation(k2):
    """PROVEN (thm:K2): the closed form reproduces the eps-extrapolation."""
    rows = read_csv(k2 / "k2_closure.csv")
    assert len(rows) == 13
    for r in rows:
        assert float(r["K2_formula"]) >= 0.0
        assert_allclose(float(r["K2_formula"]), float(r["K2_quadrature"]),
                        rtol=1e-9)
        assert_allclose(float(r["K2_formula"]), float(r["K2_direct"]),
                        rtol=1e-6)
        assert float(r["K2_formula"]) <= float(r["a_priori_bound"])*(1 + 1e-9)
    sidecar_ok(k2, "k2_closure")


@pytest.mark.slow
def test_k2_anchor_value(k2):
    """app:numerics: K_2(-2.5) = 0.304619 in Configuration I."""
    rows = [r for r in read_csv(k2 / "k2_closure.csv")
            if r["configuration"] == "I" and abs(float(r["omega"]) + 2.5) < 1e-9]
    assert len(rows) == 1
    assert_allclose(float(rows[0]["K2_formula"]), 0.304619, rtol=RTOL_K2)
    scan = read_csv(k2 / "k2_eps_scan.csv")
    assert len(scan) == 5
    finest = min(scan, key=lambda r: float(r["eps"]))
    assert_allclose(float(finest["scaled"]), 0.304623, rtol=1e-5)
    sidecar_ok(k2, "k2_eps_scan")