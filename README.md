# nonhermitian-vk

Reproducibility code for *The Vakhitov–Kolokolov criterion beyond Hermiticity: a
biorthogonal, conditioning-corrected stability theory for non-Hermitian
solitons* (submitted to *Physica D: Nonlinear Phenomena*).

The paper studies the nonlinear Schrödinger equation with local gain and loss,
`i ∂_t ψ = D ψ + N(|ψ|²) ψ + i G(x) ψ` with `D = -∂_xx + V(x)` and `N(s) = σ s`,
and asks what survives of the Vakhitov–Kolokolov criterion when the linearisation
`L(ω)` is no longer self-adjoint. The Hermitian slope condition `P'(ω)` is
replaced by the biorthogonal quasi-power `Q(ω) = ⟨⟨χ₀, Φ⟩⟩` built from the
adjoint kernel `χ₀`, corrected by a conditioning factor `K` that measures how far
the mode is from orthogonality. The main negative result is a soliton branch on
which `Q'(ω)` changes sign while the linear instability persists on both sides of
the turning point, so no slope criterion in `Q` alone can be sufficient. This
repository contains the code that produced every number in the paper, the
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

Runtimes are wall-clock on one core of an ordinary laptop-class machine with
`OMP_NUM_THREADS=1`. Table 1 is quoted in the paper at `(L_x, N) = (22, 320)`,
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

runs scripts 01–06 in order, writes the CSV tables and their JSON metadata
sidecars into `data/`, and prints a summary with the row count of each file. The
whole sequence takes about two and a half minutes on one core. The individual
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
