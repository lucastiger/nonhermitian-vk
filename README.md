# nonhermitian-vk

Reproducibility code for *The Vakhitov–Kolokolov criterion beyond Hermiticity:
a master identity and a parity index for
non-Hermitian solitons*.

The paper studies the nonlinear Schrödinger equation with local gain and loss,
`i ∂_t ψ = D ψ + N(|ψ|²) ψ + i G(x) ψ` with `D = -∂_xx + V(x)` and `N(s) = σ s`,
and asks what survives of the Vakhitov–Kolokolov criterion when the linearisation
`L(ω)` is no longer self-adjoint. The Hermitian slope condition `P'(ω)` is
replaced by the biorthogonal quasi-power `Q(ω) = ⟨⟨χ₀, Φ⟩⟩` built from the
adjoint kernel `χ₀`, corrected by a conditioning factor `K` that measures how far
the mode is from orthogonality.

The paper's negative result is a soliton branch on which `Q'(ω)` changes sign
while the linear instability persists on both sides of the turning point, so no
slope criterion in `Q` alone can be sufficient. Its positive result is that a
different scalar does work: the Jordan-chain quantity `κ(ω)`, which reduces to
`P'(ω)` in the Hermitian limit, equals the product of the nonzero eigenvalues
enclosed by any contour around the origin (the *master identity*), so its zeros
are exactly the frequencies at which spectrum crosses `λ = 0`, and
`(-1)^{n_unstable} = ε · sign κ` is constant along a branch. `κ` (the
Jordan-chain scalar) and `K` (the Petermann conditioning factor) are unrelated
objects, and the code keeps them apart as the paper does.

This repository contains the code that produced every number in the paper, the
generated data files, and regression tests that pin those numbers.

## Paper ↔ code map

| Paper | Script | Output | Runtime |
|---|---|---|---|
| Table 1 (branch: `P`, `Q`, `max Im λ` vs `ω`) | `scripts/02_theorem_a_branch.py`, values quoted at the finest grid of `scripts/03_theorem_a_convergence.py` | `data/theorem_a_branch.csv` (`data/theorem_a_convergence.csv` for the quoted digits) | 15 s (02) |
| Table 2 (grid convergence) | `scripts/03_theorem_a_convergence.py` | `data/theorem_a_convergence.csv` | 65 s |
| Table 3 (parity selection rule for `γ`) | `scripts/05_parity_gamma.py` | `data/parity_gamma.csv` | 5 s |
| Table 7 (gauge transport of `χ₀`) | `scripts/06_gauge_monodromy.py` | `data/gauge_monodromy.csv` | 25 s |
| Table F.1 (structural identities) | `scripts/01_krein_reduction.py` | `data/krein_identities.csv` | 10 s |
| Table F.2 (`ε → 0` scaling of `K`, `κ` vs `Q'`) | `scripts/01_krein_reduction.py` | `data/eps_scan.csv` | (same run) |
| Appendix F, Configuration II (power balance) | `scripts/04_power_balance.py` | `data/power_balance.csv` | 20 s |
| Table `tab:partI` (local bifurcation: `ω_κ`, `κ₂`, `κ₃`, `C`) | `scripts/08_kappa_local_bifurcation.py` | `data/kappa_local_bifurcation.csv` | 5 min |
| Hermitian limit of the exchange law (`κ = P'`, `C < 0`) | `scripts/08_kappa_local_bifurcation.py` | `data/kappa_hermitian_limit.csv` | (same run) |
| Table `tab:parity` (parity index along the branch) | `scripts/07_kappa_branch.py` | `data/kappa_branch.csv` | 4 min |
| Section `sec:necessity` (the two Hamiltonian–Hopf collisions) | `scripts/07_kappa_branch.py` | `data/hopf_collisions.csv` | (same run) |
| Table `tab:quartet`, Section `sec:quartet` (quartet symmetry, class boundary) | `scripts/09_quartet_symmetry.py` | `data/quartet_symmetry.csv` | 12 min |
| Remark `rem:M3real` (quintic term breaks the quartet; mechanism M3) | `scripts/09_quartet_symmetry.py` | `data/quartet_m3.csv` | (same run) |
| Section `sec:quartet` (no differential intertwiner of order ≤ 4) | `scripts/10_intertwiner_nogo.py` | `data/intertwiner_nogo.csv` | 90 s |
| Table `tab:K2` (closed form for `K₂` vs direct extrapolation) | `scripts/11_k2_closure.py` | `data/k2_closure.csv` | 5 min |
| Appendix `app:numerics` (the `ε` ladder for `K₂`) | `scripts/11_k2_closure.py` | `data/k2_eps_scan.csv` | (same run) |

Runtimes are wall-clock on one core of an ordinary laptop-class machine with
`OMP_NUM_THREADS=1`. Scripts 01–06 total about two and a half minutes; scripts
07–11 add roughly half an hour, dominated by the dense `eig` calls of script 09
at `(L_x, N) = (36, 448)`. Table 1 is quoted in the paper at `(L_x, N) = (22, 320)`,
which is the finest grid of script 03; script 02 computes the same branch at
`(L_x, N) = (20, 240)` over the full range of `ω` and agrees to four to five
significant figures.

## Installation

```
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

The package is not installed by the requirements file. Either put it on the path
for a single run,

```
PYTHONPATH=src python scripts/01_krein_reduction.py
```

or install it in editable form if you prefer (`pip install -e .` once a
`pyproject.toml` is added; the scripts do not need it — each one inserts `src/`
on `sys.path` itself, so `python scripts/01_krein_reduction.py` works from a
clean checkout).

Only NumPy is needed for `nhvk` itself. SciPy is recorded in the metadata
sidecars, and SymPy is used by script 04 for one symbolic identity.

## Quick start

```
make all
```

runs scripts 01–11 in order, writes the CSV tables and their JSON metadata
sidecars into `data/`, and prints a summary with the row count of each file. The
whole sequence takes about half an hour on one core. The individual
scripts accept `--outdir` if you want the output somewhere else:

```
python scripts/03_theorem_a_convergence.py --outdir /tmp/run
```

Every script prints the same human-readable table it printed when the paper was
written, then `OK`, and exits 0. A failure prints a diagnostic on stderr and
exits non-zero.

Files written by `make all`:

```
data/krein_identities.csv        data/krein_identities.json
data/eps_scan.csv                data/eps_scan.json
data/theorem_a_branch.csv        data/theorem_a_branch.json
data/theorem_a_convergence.csv   data/theorem_a_convergence.json
data/power_balance.csv           data/power_balance.json
data/parity_gamma.csv            data/parity_gamma.json
data/gauge_monodromy.csv         data/gauge_monodromy.json
data/kappa_branch.csv            data/kappa_branch.json
data/hopf_collisions.csv         data/hopf_collisions.json
data/kappa_local_bifurcation.csv data/kappa_local_bifurcation.json
data/kappa_hermitian_limit.csv   data/kappa_hermitian_limit.json
data/quartet_symmetry.csv        data/quartet_symmetry.json
data/quartet_m3.csv              data/quartet_m3.json
data/intertwiner_nogo.csv        data/intertwiner_nogo.json
data/k2_closure.csv              data/k2_closure.json
data/k2_eps_scan.csv             data/k2_eps_scan.json
```

Each JSON sidecar records the script name, the git commit, a UTC timestamp, the
Python, NumPy and SciPy versions, and the full parameter set used.

## What each script does

### `01_krein_reduction.py`

Builds the PT-symmetric Configuration I soliton (`V = -2 sech²x`,
`G = ε sech x tanh x`, `ω = -2.5`) and verifies that the paper's `χ₀` is exactly
the Chernyavsky–Pelinovsky adjoint eigenvector: `L = σ₃ 𝓛`, the kernel of
`L†` is simple, and `χ₀ = α σ₃ v#` to round-off. It also checks that the Krein
quantity `⟨χ₀, w₀⟩` vanishes identically at the phase mode (Lemma 2.3), and then
scans `ε → 0` to show `K - 1 = O(ε²)` and `Q' - κ = O(ε²)`.

### `02_theorem_a_branch.py`

Constructs the non-PT Wadati branch `g = A x e^{-x²/2}`, `V = -g²`, `G = g'` at
`A = 1` by continuation in `A`, then sweeps `ω`. At each frequency it records
`P`, `Q` and the most unstable localised eigenvalue of `L(ω)`. `Q'(ω)` changes
sign near `ω ≈ -0.23` while `max Im λ` stays strictly positive on both sides:
this is the falsification that Theorem A rests on.

### `03_theorem_a_convergence.py`

Repeats the six key frequencies of script 02 at three resolutions,
`(L_x, N) = (18, 220), (22, 260), (22, 320)`, rebuilding the branch from scratch
each time. It confirms that the sign change of `Q'` and the persistence of the
instability are properties of the continuous problem, not of the discretisation.

### `04_power_balance.py`

Checks the admissibility conditions. Lemma P requires `∫G|φ|² dx = 0` for any
decaying stationary state, which excludes sign-definite `G`; the script exhibits
an O(1) defect for `G = A sech²x` and confirms the identity holds to `~1e-14`
along the Theorem A branch, where it was never imposed. It also verifies the
Wadati factorisation `-∂_xx - g² + i g' = (∂_x + i g)(-∂_x + i g)` symbolically
with SymPy, and shows that the family exists for `V = -g²` and fails for
`V = +g²`.

### `05_parity_gamma.py`

Evaluates the exceptional-point splitting coefficient
`γ = (-i/κ) ⟨χ₀, i h w₀⟩` on the PT-symmetric barrier base
(`V = 0.3 sech²x`, `G = -0.3 sech x tanh x`) for an even, an odd and a generic
perturbation direction `h`, at three frequencies. The even directions give
`|γ| < 1e-13` while the odd and generic ones give O(1) values, which is the
parity selection rule of Proposition 2. The script also reports the oddness
residual of `Im(conj(η₀) φ₀)`, the mechanism behind the proposition.

### `06_gauge_monodromy.py`

Transports `χ₀` from the exact Hermitian anchor at `A = 0` up to `A = 0.5` in
steps of `dA = 0.005`, fixing its phase at each step only by maximising the
overlap with the previous step — no re-gauging is applied anywhere. `Q` stays
real to `1e-13`, `arg c` does not wind, and the S-residual
`‖Sχ₀ - χ₀‖/‖χ₀‖` stays at round-off, so continuity transport alone keeps `χ₀`
on the real ray and the only residual freedom is a sign.

### `07_kappa_branch.py`

Sweeps the Theorem A branch in `ω` with `χ₀` transported by **continuity** from
the anchor at `ω = -0.5`, recording at each frequency `κ`, `Q`, `P`, the enclosed
discrete spectrum, the unstable count `n_u`, and the parity product
`(-1)^{n_u} · sign κ`. That product is `+1` at every sampled frequency, across
both Hamiltonian–Hopf collisions and the zero of `κ`. The script then locates the
two collisions by a secant iteration on the discriminant of the colliding
positive-real pair and reports `κ` there: `κ` is smooth and nonzero across both,
which is why the unrestricted criterion is false and hypothesis (A2) cannot be
dropped.

Two mode sets are used and are **not** interchangeable. `n_u` is counted on the
*enclosed* spectrum — the gap disk plus strongly localised modes above the
`3/(2 L_x)` finite-domain artefact floor — because as `ω → 0⁻` the gap disk
shrinks and no longer encloses the whole discrete spectrum. The Corollary 1 and 2
defects are measured on the gap disk alone, the only set invariant under both
symmetries.

### `08_kappa_local_bifurcation.py`

Locates `ω_κ` from `κ` alone by secant — no eigensolve — and `ω_c` independently
from `ν = ½ tr(L_Γ²)` via a Beyn contour projection, at four resolutions. The two
agree to `~1e-14`, orders of magnitude finer than the truncation error in either
separately: that is Theorem `thm:targetB`, not a coincidence. At `ω_κ` it extracts
the Jordan data — the origin block has dimension 4, `κ₂` vanishes automatically,
`κ₃` is purely imaginary, and the predicted exchange constant `C = -i/κ₃` matches
the measured `ν'/κ'` to four significant figures. A Hermitian control (`G = 0`,
trapping `V`, cubic–quintic `N`) confirms `κ = P'` to round-off and `C < 0`, so
the theorem reduces to classical VK in this `ω` convention.

`ν` comes from the compressed block rather than from individual eigenvalues: at
the `J₄` point the latter are accurate only to `ε^{1/4}`, the former to `~1e-10`.

### `09_quartet_symmetry.py`

Measures the `λ → conj(λ)` defect on Wadati branches with **cubic** `N` at four
resolutions: for an odd `g` (`G` even, non-PT), for a **non-odd** `g` — the
decisive case, where neither `V` nor `G` is even and no parity operator exists at
all — and for an even `g` (the PT control). It also shows why an 85 %-mass filter
is unusable here: the two members of a quartet decay at different rates, so the
filter keeps one and drops the other and reports a spurious defect. Finally it
adds a quintic term: Corollary 1 survives at machine level while the Corollary 2
defect grows smoothly with `β`, and mechanism M3 then occurs — a pair with a
common nonzero imaginary part that is never real and never collides.

Defects are reported twice. `*_isolated` is measured on the isolated complex gap
modes and is the `4e-13` quoted in the paper; `*_all` also includes the `λ = 0`
Jordan block, whose two members are resolved only to `~1e-7` and which therefore
dominates the maximum.

### `10_intertwiner_nogo.py`

Least-squares search for a matrix differential `Y` with `L♭ Y = Y L` over all
orders `K ≤ 4`, normalised on the core `|x| < 8`. The core normalisation is
essential: in the tails `L♭ - L = -2iG → 0`, so any `Y` commuting with the free
operator there is a spurious near-solution. The Hermitian case, where `Y = I` is
an exact intertwiner, is the positive control and returns `~1e-16`; if it did
not, the normalisation would be wrong and the test would say nothing. The mild
decay of the reported ratio with `K` tracks the conditioning deflation
`σ_max ~ ‖D₁‖^K`, not convergence to zero.

### `11_k2_closure.py`

Evaluates `K₂ = (4/P) ‖L₊⁻¹(g φ₀)‖²` by one linear solve and compares it with an
independent `ε`-extrapolation of `K - 1` from the full nonlinear problem, over
thirteen `(V, g, N, ω)` points. It also evaluates the equivalent quadrature form
`K₂ = P⁻¹ ∫ (r - q)²` as an internal cross-check of the two-term collapse
`L₊(r - q) = 2 g φ₀`, and verifies the a priori bound `K₂ ≤ 4 ∫ g²φ₀² / (P d²)`
with `d = dist(0, σ(L₊))`. Configuration IV has `N'' ≠ 0` and confirms directly
that no third-order Taylor datum of `N` enters.

## Reproducing the paper's tables

```
make all                       # regenerate every CSV in data/
PYTHONPATH=src pytest          # fast regression tests (Tables F.1, F.2, 3, power balance)
PYTHONPATH=src pytest -m slow  # slow regression tests (Tables 1, 2, 7)
```

or, equivalently, `make test` and `make test-slow`. The slow tests rebuild full
branches and sweeps and take a few minutes.

What agreement to expect. `P` and `Q` reproduce to five or six significant
figures. Eigenvalues may differ in the last printed digit across BLAS
implementations; the tests allow `rtol = 1e-4` on `Im λ`, together with an
absolute tolerance of `5e-6` because the tables quote five decimals and one
entry is reported as exactly zero. The `|γ|` magnitudes of Table 3 are quoted to
three significant figures and move in the third under grid refinement, so they
are tested at `rtol = 1e-2`; the even-direction entries, which are zero by
Proposition 2, are tested against an absolute threshold of `1e-11` instead. Set
`OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` — the scripts do this themselves —
so that BLAS reduction order cannot vary between runs.

For the `κ` scripts: `ω_κ` and `ω_c` agree to `1e-14` and are tested against
`-0.2600177` at `atol = 5e-6`, the domain-truncation spread across the four
grids; `C = -i/κ₃` is tested at `rtol = 1e-3`, the precision to which the
centred-difference `ν'/κ'` resolves it; the collision frequencies are tested at
`atol = 5e-6` and `κ` there at `rtol = 1e-3`. The quartet defects are tested as
thresholds (`< 1e-10` on the isolated modes) rather than as values, because they
sit at the discretisation floor and carry no significant digits. `K₂` reproduces
to six significant figures and is tested at `rtol = 1e-5`. The two seeded
computations — the Beyn probe block in script 08 and the test functions in
script 10 — use fixed seeds recorded in their JSON sidecars, so both are
deterministic.

`provenance/` holds the six original scripts exactly as they were executed for
the paper. They are kept for the record and are not the supported entry points;
`scripts/` reproduces their printed output digit for digit.

## Licence

Code (`src/`, `scripts/`, `tests/`, `provenance/`) is MIT — see `LICENSE`. The
generated data files in `data/` are CC-BY-4.0 — see `LICENSE-DATA`.

## Citation

See `CITATION.cff`, which carries both the software record and a
`preferred-citation` block for the article. The DOI and ORCID fields are
placeholders and must be filled in before the repository is archived.