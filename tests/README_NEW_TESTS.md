# Unit tests: `tests/test_unit_*.py`

These tests check PyDESeq2's numerical core (normalization, dispersions, GLM fits,
Cook's distances, Wald tests, shrinkage) against expected values derived
independently of PyDESeq2. They are written to keep passing when the computation is
reorganized (batched linear algebra, vectorized solvers) and to fail when an output
changes by more than the precision of the method that produced it.

| File | Contents |
|---|---|
| `tests/helpers.py` | simulator, hand-worked inputs, reference implementations (likelihood, fits, Wald test, apeglm, BH, lowess, independent filtering), comparison helper |
| `tests/conftest.py` | `n_cpus` and `inference` fixtures; a report header naming the PyDESeq2 under test |
| `tests/TARGETS.md` | every public target, its area and test file, and whether the existing tests reach it |
| `tests/test_unit_harness.py` | self-tests of `helpers.py` |
| `tests/test_unit_data.py` | area A: size factors, preprocessing, `misc` |
| `tests/test_unit_data_dataset.py` | area A: `DeseqDataSet` constructor, `Inference` interface |
| `tests/test_unit_data_vst.py` | area A: variance-stabilizing transform |
| `tests/test_unit_dispersions.py` | area B: dispersion functions (gene-wise MLE, trend, moments, grid) |
| `tests/test_unit_dispersions_dds.py` | area B: `DeseqDataSet` dispersion methods, R fixtures in `tests/data/unit_b/` |
| `tests/test_unit_glm.py` | area C: GLM fitting, Cook's distances, refit |
| `tests/test_unit_stats.py` | area D: `DeseqStats` (Wald tests, Cook's filtering, p-value adjustment, shrinkage) |
| `tests/test_unit_stats_functions.py` | area D: `glm.wald_test`, `nbinomFn`, `nbinomGLM`, `grid_fit_shrink_beta`, `lowess` |

## Running

From the repository root:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=agg
python -m pytest tests/test_unit_*.py                     # serial
python -m pytest -n 4 tests/test_unit_*.py                # pytest-xdist
PYDESEQ2_TEST_CPUS=4 python -m pytest tests/test_unit_*.py
```

- `pytest` and `python -m pytest` both work (CI uses `coverage run -m pytest`).
- Wall time on 2026-10-28 (Python 3.12, pandas 3.0.6, one BLAS thread): the new
  tests take about 15 s serially at `PYDESEQ2_TEST_CPUS=1`, and the whole suite
  about 18 s with `-n 4` and the CPU count Python reports capped at 2 (see below).
- The report header names the PyDESeq2 under test and its directory, and the
  `PYDESEQ2_TEST_CPUS` in effect. To test a source tree without installing it, add
  `PYTHONPATH=src`.
- `PYDESEQ2_TEST_CPUS` (default 1) is the `n_cpus` every new test passes to
  PyDESeq2. With 1, joblib runs the per-gene solvers in the test process; with more,
  it starts that many loky workers per test process. PyDESeq2's own default,
  `n_cpus=None`, is one worker per core. The existing tests use that default, so on
  a large machine cap them (for example `-n 4` and Python 3.13+'s
  `PYTHON_CPU_COUNT`).
- The hatch test environments install `pytest-randomly`, and CI runs
  `hatch run hatch-test.<env>:run-cov -n auto`, which does not pass
  `-p no:randomly`: CI shuffles the test order and reseeds NumPy's global generator
  before each test with a per-session seed. Tests must not depend on order or on the
  global generator (the simulator uses its own). Locally, `-p no:randomly` keeps
  file order and `-p randomly --randomly-seed=N` replays an order.

## Where the suite passes (measured 2026-10-28)

Each environment was built the way hatch builds `hatch-test.*` (the package's
dependencies, the `dev` and `test` groups and hatch's own test dependencies). The
existing suite is the 65 tests in `test_pydeseq2.py`, `test_edge_cases.py` and
`test_stats.py`.

| Environment | Python | pandas | numpy | scipy | anndata | Existing suite | New tests |
|---|---|---|---|---|---|---|---|
| CI `py3.12-stable`, resolved today | 3.12.13 | 3.1.0 | 2.5.4 | 1.18.1 | 0.13.4 | 49 failed, 2 errors | pass |
| CI `py3.14-stable`, resolved today | 3.14.3 | 3.1.0 | 2.5.4 | 1.18.1 | 0.13.4 | 49 failed, 2 errors | pass |
| CI `py3.14-pre`, resolved today | 3.14.3 | 3.1.0 | 2.5.4 | 1.18.1 | 0.13.4 | 49 failed, 2 errors | pass |
| `py3.12-stable` as of 2026-10-15 05:00 UTC | 3.12.13 | 3.0.6 | 2.5.4 | 1.18.1 | 0.13.4 | pass | pass |
| `py3.14-pre` as of 2026-10-15 05:00 UTC | 3.14.3 | 3.1.0rc0 | 2.5.4 | 1.18.1 | 0.13.4 | 49 failed, 2 errors | pass |
| today, pandas pinned to 3.0.6 | 3.12.13 | 3.0.6 | 2.5.4 | 1.18.1 | 0.13.4 | pass | pass |
| today, pandas pinned to 3.0.6 | 3.14.3 | 3.0.6 | 2.5.4 | 1.18.1 | 0.13.4 | pass | pass |
| today, pandas pinned below 3 | 3.12.13 | 2.3.3 | 2.5.4 | 1.18.1 | 0.13.4 | pass | pass |
| today, pandas below 3, formulaic 1.0.2 (the lowest allowed) | 3.12.13 | 2.3.3 | 2.5.4 | 1.18.1 | 0.13.4 | pass | pass |
| today, scipy pinned to 1.15.0 (the lowest allowed) | 3.12.13 | 3.1.0 | 2.4.6 | 1.15.0 | 0.13.4 | 49 failed, 2 errors | pass |

Why: pandas 3.1.0 (released 2026-10-22, after the last passing scheduled CI run on
2026-10-15) emits two `Pandas4Warning`s (a `DeprecationWarning` subclass), which the
repository's `filterwarnings = error` turns into failures:

1. anndata 0.13.4 reads the deprecated option `pd.options.future.infer_string`
   whenever an `AnnData` is built (`anndata.compat.pandas_as_str`), so every test that
   builds a `DeseqDataSet` fails: 50 of the 51 failures and errors. The other one is
   `test_design_matrix_init`, which itself calls `DataFrame.rename(...,
   inplace=True)`.
2. With (1) ignored, 34 tests fail: 33 because
   `DeseqDataSet._fit_parametric_dispersion_trend` calls `Series.drop(...,
   inplace=True)` (deprecated by PDEP-8 in pandas 3.1), which every parametric trend
   fit reaches, and `test_design_matrix_init`.

An environment resolved as the `pre` job (continue-on-error) would have been on
2026-10-15 gets pandas 3.1.0rc0 and fails the same way. Pinning pandas to 3.0.6
makes the existing suite pass, and so does ignoring the two messages on pandas 3.1.0
(`-W "ignore:The 'future.infer_string' option is deprecated:DeprecationWarning"
-W "ignore:The inplace keyword:DeprecationWarning"`: 65 passed in `py3.12-stable`).

formulaic 1.0.2 does not recognise pandas 3's `str` dtype as categorical: with
pandas 3.1.0 it passes a string column such as `condition` into the model matrix
unencoded, so `DeseqDataSet` cannot be built (`Cannot cast ufunc 'svd' input from
dtype('O')`) and the existing suite fails (45 failed, 2 errors, with the two pandas
messages above ignored), as do the new tests. With pandas 2.3.3 both pass.

With the lowest versions the direct dependencies allow (`uv pip install
--resolution lowest-direct`: numpy 2.2.0, scipy 1.15.0, pandas 2.3.0, anndata
0.11.0, matplotlib 3.10.0), importing matplotlib 3.10.0 under pyparsing 3.3.3 raises
a `PyparsingDeprecationWarning`, so neither suite starts there; that environment
appears only in the determinism table below.

Every new test module sets `pytestmark = helpers.PANDAS_DEPRECATION_FILTERS`, which
ignores exactly these two messages (matched by text, so the filters are valid on
pandas 2.x too), so the new tests pass in every configuration above, including those
where the existing suite fails.

## Determinism of PyDESeq2's outputs (measured 2026-10-28 on 2ca3b74)

`deseq2()`, `summary()` and `lfc_shrink()` were run on three datasets (the bundled
synthetic data with `~group + condition`; 2,015 simulated genes x 24 samples with
`~condition`, outliers, all-zero, single-sample and large-count genes; 1,000 simulated
genes x 18 samples with `~group + condition + age`), and 103 output fields were
compared (`obs`, every numeric `var` column, `varm["LFC"]`, `layers`, `uns`, and both
results tables).

| Varied | Values | Result |
|---|---|---|
| repeated runs | same process twice; separate processes | bitwise identical |
| `n_cpus` | 1, 4, 8 | bitwise identical |
| `OPENBLAS_NUM_THREADS` | 1, 4 (with `n_cpus` 1 and 4) | bitwise identical |
| `batch_size` | 1, 16, 128, 1000 (`n_cpus` 4); 16 (`n_cpus` 1) | bitwise identical |
| CI environment | `py3.12-stable`, `py3.14-stable`, `py3.14-pre` | bitwise identical |
| pandas | 3.1.0 vs 2.3.3, numpy/scipy unchanged | simulated data identical; bundled data up to 1.4e-13 relative in p-values, 1.2e-12 in Cook's distances |
| scipy and numpy | 1.18.1/2.5.4 vs 1.15.0/2.4.6 | bundled and two-group data within 2e-12 (Cook's) and 2e-13 (p-values). Multi-factor data: each build's MAP L-BFGS-B fails on three genes, not the same three, and those genes move by up to 28% in MAP dispersion and 89% in p-value; over the 994 genes where every L-BFGS-B fit converged under both builds, MAP dispersions agree to 1.8e-6, LFCs to 1.6e-7, p-values to 1.1e-4 and `padj` to 1.6e-2 (a gene's `padj` depends on every other gene's p-value) |
| lowest direct dependencies | numpy 2.2.0, scipy 1.15.0, pandas 2.3.0, anndata 0.11.0, ... | the scipy row's differences plus the pandas row's |

The three CI environments agree bitwise today; they share numpy and scipy versions,
and other numpy and scipy builds give different bits. Expected values must not rely
on bitwise agreement: compare at the precision of the method, and keep genes where
an optimizer fails or falls back out of tight comparisons.

## Precision of PyDESeq2's solvers (for choosing tolerances)

Measured against the reference minimizers in `helpers.py` on simulated genes:

- `dispersions.fit_alpha_mle` (L-BFGS-B, `pgtol` 1e-5 on log dispersion) with the
  true means and Cox-Reid on: relative error up to 9.5e-5 over 450 genes with 6 to
  40 samples, objective gap up to 3e-10. The error scales as `pgtol` over the
  curvature of the objective in log dispersion: the error in log dispersion times
  that curvature stayed below 8.2e-5.
  `test_unit_dispersions.py` uses `rtol=1e-3`. Without Cox-Reid, near-Poisson genes
  have objectives so flat that L-BFGS-B stops far from the minimizer (relative
  error up to 350, objective gap up to 6.5e-6): compare objective values there.
- When L-BFGS-B reports failure, `fit_alpha_mle` returns `grid_fit_alpha`'s value,
  which is within half a step of its fine grid of the minimizer (up to 2.1e-3
  relative at `max_disp` 10). L-BFGS-B reports failure (an abnormal line search) on
  a few gene-wise fits that end at `pgtol`, and which genes do moves with changes of
  1e-11 in their means, so a gene can take the fallback on one platform only (gene30
  of `test_vst_with_continuous_covariate` on some Windows runners).
  `test_unit_data_vst.py` compares at most two such genes per comparison at one
  fine step.
- `glm.irls_solver` (stops when the relative deviance change is below
  `beta_tol=1e-8`) with the true dispersion: absolute coefficient error (natural
  log) up to 2.2e-5 with two coefficients and 3.4e-4 with five (two factors and a
  continuous covariate); relative objective gap up to 1.3e-8. Use `atol=1e-3` on
  natural-log coefficients, or compare objective values at 1e-7 relative. Genes
  with a fitted mean below `min_mu=0.5` are not comparable to the reference: IRLS
  clamps the mean there, which changes its fixed point. On flat objectives (high
  dispersions) IRLS stops further from the minimizer, up to 4e-4 at dispersion 2.5,
  while the objective gap stays near 1e-9.
- Dispersions on `min_disp` or `max_disp`, and LFCs on `min_beta` or `max_beta`:
  compare with the bound at the solver's precision (1e-4 relative for dispersions,
  `atol=1e-3` for LFCs), not to rounding. L-BFGS-B projects onto the bound, but a
  solver that stops on a projected gradient may stop just inside it.
- `nbinomGLM` (apeglm shrinkage) stops L-BFGS-B at `ftol=gtol=1e-8` on its objective
  divided by the objective at beta = 0: against the reference MAP, absolute errors
  up to 3.5e-3 on the shrunk natural-log coefficient (6.4e-3 on any coefficient)
  and relative objective gaps up to 8.5e-6, over 820 simulated genes. The shrinkage
  tests use `atol=2e-2` on coefficients and 1e-4 on objective gaps, for dispersions
  well above 1e-6 (see the known issues).
- The Wald test, Cook's distances and the hat diagonal are closed-form given the
  fitted LFCs and dispersions; recomputed from a fit's public outputs they agree to
  rounding (relative 1e-8 covers condition numbers of `X^T W X + ridge` up to about
  1e6 and was checked against a batched and a closed-form 2 x 2 rewrite).

Two separate fits (gene order, `n_cpus`, `batch_size`, another parametrization of
the same design) agree to the solvers' precision, not to rounding: a reorganized
solver whose stopping point depends on which genes share a batch is still correct.
Compare them with the tolerances above, and leave out genes whose MAP dispersion
took the grid fallback in one fit only (`helpers.map_fallback_flips`): where
L-BFGS-B's convergence on the MAP objective is marginal, rounding-level changes
decide whether it fails, and the fallback, which drops the prior, moves the gene
by up to 30%.

## Writing a test

1. Find the target in `TARGETS.md` and work in your area's file. Changes to
   `helpers.py` or `conftest.py` are shared; make them in a separate commit and
   rerun `test_unit_harness.py`.
2. Start a new module like the existing ones:

   ```python
   import numpy as np
   import pytest

   from tests import helpers

   pytestmark = helpers.PANDAS_DEPRECATION_FILTERS
   ```

3. Build datasets with `helpers.make_dds(counts, metadata, design=..., **kwargs)`
   (sets `quiet=True` and `n_cpus` from `PYDESEQ2_TEST_CPUS`), or use the `n_cpus`
   and `inference` fixtures. Never leave `n_cpus` at its default.
4. Derive the expected value independently (rules below), write the tolerance and
   its reason next to the assertion, and use `helpers.assert_fields_close` for
   per-gene outputs so a failure names the genes and fields that differ.
5. Run the file serially and with `-n 4`, at `PYDESEQ2_TEST_CPUS=1` and `4`, in a
   pandas 3.1 environment and a pandas 2.x or 3.0 one, and in random order
   (`-p randomly --randomly-seed=N`). Keep all new tests together under about 60 s
   serial at `PYDESEQ2_TEST_CPUS=1`.

### Helpers (`tests/helpers.py`)

- `get_test_cpus()`, `make_dds(...)`; fixtures `n_cpus`, `inference`.
- `map_fallback_flips(first, second)`: genes whose MAP dispersion took L-BFGS-B in
  one fit and the grid fallback in the other (see the precision section).
- `trend_coeffs(dds.uns["trend_coeffs"])`: `a0` and `a1` as a float Series, read by
  key, whether PyDESeq2 stores them as a Series or a dict.
- `simulate_counts(n_genes=200, n_samples=12, *, seed=0, factors={"condition":
  ["A", "B"]}, continuous=(), design=None, mean_range=(10, 5000), frac_de=0.3,
  lfc_sd=1.0, trend=(0.1, 4.0), disp_scatter=0.0, size_factor_sd=0.3, n_all_zero=0,
  n_single_sample=0, n_large=0, large_mean=1e7, n_outliers=0, outlier_factor=100)`
  returns a `SimulatedCounts` with `counts`, `metadata`, `design`, `design_matrix`,
  `beta` (natural log, columns named as in `varm["LFC"]`), `log2_fold_changes`,
  `dispersions`, `size_factors` (geometric mean 1), `mu`, `normed_means`,
  `gene_kind`, `outliers` and `trend`. Draws come from legacy `RandomState`
  streams, which NumPy keeps fixed across versions, one stream per component.
- `size_factor_example()`: a 3 x 7 matrix with exact median-of-ratios size factors
  (with and without control genes); `nb_example()`: an NB likelihood worked by hand.
- `nb_logpmf`, `nb_loglik`: the NB log-likelihood (mean `mu`, variance
  `mu + alpha mu^2`). Equal to `scipy.stats.nbinom` where scipy is accurate, and
  written so that it stays accurate as `alpha` goes to 0 (checked against the
  Poisson limit), where scipy's form is off by up to 2e-7 per sample at 1e-8. Its
  values are accurate to about 1e-11 absolute on typical genes (more for counts near
  1e7, whose terms reach 1e8), which limits finite differences of it: a central
  difference with step `h` carries an error near `1e-11 / h`. Differentiating the
  dispersion objective needed Richardson extrapolation to check `dnb_nll`.
- `dispersion_objective(counts, mu, design_matrix, *, cr_reg=True,
  prior_disp_var=None, prior_dispersion=None)` and `reference_dispersion_mle(...,
  min_disp=1e-8, max_disp=10.0, grid_size=2001)`: the Cox-Reid adjusted (and
  optionally prior-penalized) dispersion objective, minimized by a dense grid plus a
  bounded Brent search.
- `glm_objective(counts, size_factors, design_matrix, disp, *, ridge=1e-6)` and
  `reference_glm_mle(...)`: the ridge-penalized NB GLM objective PyDESeq2's IRLS
  solves, minimized by Newton's method. The stop rule and the step halving allow for
  rounding (of the gradient and of the objective), so it converges on counts near
  1e7, where the objective cannot resolve the last steps.
- `median_of_ratios(counts, control=None)`, `lin_reg_mu(...)`,
  `rough_dispersions(...)`, `moments_dispersions(...)`: DESeq2's ratio size factors,
  linear-model means (floored at `min_mu`) and starting dispersion estimates.
  `gamma_identity_glm(x, y, start)` and `reference_trend(normed_means, dispersions,
  *, min_disp=None)`: DESeq2's `parametricDispersionFit` (gamma GLM with identity
  link by IRLS, refit on the genes within `(1e-4, 15)` times the current curve);
  with `min_disp`, genes not above `100 * min_disp` are left out first.
- `wald_reference(x, disp, beta, mu, ridge, contrast)`: DESeq2's Wald test of one
  gene (sandwich covariance of the ridge-penalized estimate).
- `apeglm_objective`, `apeglm_grad_hess`, `reference_map(x, counts, size, offset,
  shrink_index, prior_scale, starts)`: apeglm's negative log posterior (NB
  likelihood, N(0, 15^2) prior on the coefficients not shrunk, Cauchy prior on the
  shrunk one), its gradient and Hessian, and its minimizer by trust-region Newton
  from several starts; `reference_map` also returns the spread of the starts' minima,
  and a spread above `BIMODAL_SPREAD` means two local minima. `apeglm_prior_var`:
  apeglm's adaptive prior variance.
- `bh_adjust`, `reference_lowess`, `reference_independent_filtering`:
  Benjamini-Hochberg by hand, Cleveland's robust lowess (as Biopython's, which
  `pydeseq2.stats.lowess` follows) and DESeq2's independent filtering
  (`pvalueAdjustment` in DESeq2's `results.R`).
- `compare_fields(actual, expected, *, rtol, atol=0.0, name)` and
  `assert_fields_close(...)`: per-field tolerance comparison; NaN and infinity
  positions must match exactly; the failure message lists failing genes and fields.

### Gotchas

- `filterwarnings = error` applies: a test that triggers any other warning fails.
  Expect warnings with `pytest.warns` (for example the "residual degrees of freedom
  is less than 3" warning of `fit_dispersion_prior` with few samples, or "Every gene
  contains at least one zero" from `fit_size_factors`).
- Importing a function named `test_*` into a test module makes pytest collect it
  (`pydeseq2.misc.test_valid_counts`): import the module instead.
- `varm["LFC"]` is in natural log; `results_df["log2FoldChange"]` and `lfcSE` are in
  log2.
- The ratio method's size factors are not rescaled to a geometric mean of 1
  (`poscounts` and `iterative` are); compare against the simulator's truth after
  dividing both by their geometric mean.
- `DeseqDataSet` clips dispersions at `max(max_disp, n_samples)`, not `max_disp`.
- `DeseqStats` requires `refit` to have run when `refit_cooks=True`.
- Importing `pydeseq2.dds` sets `warnings.simplefilter("ignore", FutureWarning)` for
  the whole process.

## Rules for expected values

- Derive expected values independently wherever practical: hand-worked small
  inputs, a short reference implementation inside the test (median-of-ratios by
  hand, the NB likelihood from `helpers.nb_loglik`, brute-force minimization of the
  exact objective), the simulator's recorded truth with a statistical tolerance, or
  R DESeq2's output. A test that only records the current output says so in its
  docstring ("pins the current output"); keep these few.
- No test of behaviour the current code gets wrong (see the known issues below).
- Tolerances, not bits: no exact float comparison unless the quantity is exact by
  construction (integer counts, masks, booleans, NaN positions, shapes, index labels,
  dtypes). Use a relative tolerance justified by the reference's own precision,
  written next to the assertion. Compare optimizer outputs (dispersions, LFCs) at
  the precision the optimizer's own tolerance supports, not tighter.
- Do not pin implementation details a correct reimplementation could change: private
  attributes, the order of internal calls, joblib usage, intermediate arrays not
  exposed in `.obs`, `.var`, `.varm`, `.obsm`, `.uns`, `.layers` or a results
  DataFrame. Test module-level functions through their documented signatures and
  return values.
- No dependence on the platform, the Python version, the dependency versions in the
  CI matrix, `n_cpus` or test order.
- Tests write only under `tmp_path`.

## Reference outputs from R

The existing fixtures in `tests/data/` come from DESeq2 1.34.0 (see
`tests/data/README.md`). On 2026-10-28, DESeq2 1.50.2 (R 4.5.3, apeglm 1.32.0, from
bioconda) reproduced the single-factor fixtures (`results`, dispersions and apeglm
`lfcShrink` on `datasets/synthetic`, `~condition`) to within 6e-13 relative, so new
fixtures from 1.50.2 are consistent with the old ones. An environment with both
packages:

```bash
micromamba create -p <env> --override-channels -c conda-forge -c bioconda \
    bioconductor-deseq2 bioconductor-apeglm
```

A new fixture is a small CSV (kilobytes) under `tests/data/<case>/`, committed with
the R script that generates it and the input counts it read; the script prints
`packageVersion("DESeq2")` and the R version into a comment at the top of the
fixture or into the case's README.

## Known issues in the current code (not tested)

Each was compared with DESeq2 1.50.2 (and apeglm 1.32.0) or with the documented
behaviour; the tests avoid the inputs that reach them.

Size factors and normalization:

- `fit_size_factors(fit_type="poscounts")` drops genes whose positive counts are all
  1 (`logmeans > 0`); DESeq2 keeps every gene that is not all zero.
- `deseq2_norm_transform` (and `vst_transform` on new samples) takes the median of
  `log(count) - logmean` including zero counts, which enter as `-inf`: a new sample
  with zeros in at least half of the fitted genes gets size factor 0 and NaN or
  infinite normalized counts. DESeq2 leaves zero counts out of that median.

Dispersions:

- Gene-wise dispersions whose starting value (the minimum of the rough and moments
  estimates) is at `min_disp` stay at `min_disp`, even when the Cox-Reid objective
  is lower elsewhere. DESeq2 1.50.2 does the same.
- When L-BFGS-B fails on the MAP dispersion objective, `fit_alpha_mle` falls back to
  `grid_fit_alpha` without passing `prior_disp_var`, `cr_reg` or `prior_reg`, so
  those genes get the gene-wise MLE instead of the MAP estimate. DESeq2's grid
  fallback keeps the prior.
- The parametric dispersion trend (DEA and VST) fits its first gamma GLM to every
  non-zero gene and never re-admits genes it drops; DESeq2 starts from the curve
  `0.1 + 1 / mean` and reselects genes within `(1e-4, 15)` times the current curve
  before every fit. A few genes with a count in one sample (dispersion on the upper
  bound) are enough to move `a1` two- to threefold.
- With three or fewer residual degrees of freedom, `fit_dispersion_prior` warns and
  still uses `mad^2 - trigamma((n - p) / 2)`; DESeq2 1.50.2 replaces that formula
  with a simulation-based estimate there (`estimateDispersionsPriorVar`). On 300
  simulated genes with five samples and `~condition` the two gave 1.99 and 2.01.
  The warning documents the simplification; the tests check the warning and the
  formula.

GLM fitting, Cook's distances and refit:

- `irls_solver` uses `min_beta` only as the bound of its fallback optimizer, so with
  asymmetric bounds a converged IRLS fit can return an LFC below `min_beta`; the
  `max_beta` docstring gives its default as -30 (it is 30).
- After a refit, refit genes keep their first-pass `MAP_dispersions`, while
  `dispersions`, `genewise_dispersions`, `fitted_dispersions` and `LFC` hold the
  refit values; DESeq2 copies the refit `dispMAP` back too.
- `dispersions.trimmed_variance(..., axis=1)` subtracts the trimmed means without
  restoring the reduced axis, so it centres the rows on the wrong means (or raises
  for non-square input). PyDESeq2 itself only uses `axis=0`.

Testing and shrinkage:

- With `alt_hypothesis` `"greater"`, `"less"` or `"lessAbs"`, genes whose statistic
  is clipped to zero get p-value 0.5; DESeq2 computes these p-values from the
  unclipped normal tail (between 0.5 and 1).
- `lfc_null` and `alt_hypothesis` are applied to each coefficient before the contrast
  is taken, which equals DESeq2's test of the contrast only for unit contrast
  vectors (a reversed or level-vs-level contrast tests a different null).
- `prior_LFC_var` enters the Wald test's ridge as `1 / prior_LFC_var**2`; a normal
  prior of that variance (and DESeq2's `betaPriorVar`) gives `1 / prior_LFC_var`.
- The shrunk `lfcSE` comes from a Hessian in `nbinomGLM` to which the prior's
  curvature is added on every row (`np.diag` of a diagonal matrix is its diagonal
  vector), so off-diagonal entries are wrong: up to 18.6% from DESeq2's `lfcSE`
  on the bundled synthetic data.
- `nbinomGLM` reports convergence but stops far from the MAP for dispersions of 1e-6
  and below (reachable, as `min_disp` is 1e-8); apeglm's `nbinomCR` recovers it by
  refitting from a second start.
- Genes that become all zero after outlier replacement get p-value 1 from
  `run_wald_test` and then NaN from Cook's filtering, because `replace_cooks` zeroes
  the distances of refitted genes only; DESeq2 reports 1.
- The "three or more larger counts" exception to Cook's filtering is applied to every
  design; DESeq2 applies it only when the design has a single two-level factor.
- `stats.lowess` with robustness iterations returns 0 at points of an exactly fitted
  neighbourhood: the residuals are rounding noise, the bisquare weights become 0 and
  the weighted normal equations vanish. Which points depends on the numpy build.

Other:

- `import pydeseq2.grid_search` as the first PyDESeq2 import raises an `ImportError`
  (circular import with `distributions`). Inside the suite `conftest.py` imports
  `pydeseq2.default_inference` first, so tests are not affected.
