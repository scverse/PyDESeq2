# Unit-test targets

Every public class, method and module-level function of `src/pydeseq2`, found by walking the source with `ast`. Columns:

- **Done**: `[x]` when the new tests cover the target (from the area writers' coverage maps and the reviewers' changes), with what they leave out.
- **Docs**: listed in the Sphinx API reference. `docs/api.md` autosummarizes the classes `DeseqDataSet`, `DeseqStats`, `Inference` and `DefaultInference` (their own public methods and properties, through `docs/_templates/autosummary/class.rst`) and the public functions defined in the modules `datasets`, `dispersions`, `distributions`, `glm`, `grid_search`, `misc`, `plotting`, `preprocessing` and `stats`. Read from those rules, not from a docs build.
- **Existing tests reach it**: at least one line of the body runs during the 65 existing tests (coverage of the base suite at 2ca3b74, pandas 3.0.6, `n_cpus=1` so the per-gene solvers run in-process). Reached is not the same as checked; `-` means not applicable (abstract method, or an alias whose function row says it).

## Areas

Each area owns its own test files, so writers can work in parallel without editing the same file. All of them share `tests/helpers.py` and `tests/conftest.py`; changes to those go through the harness owner.

The brief's starting split is kept, with these adjustments made from the source:

- Fallback solvers sit with the solver they back up: `grid_fit_alpha` and `vec_nb_nll` in B (dispersion MLE), `grid_fit_beta` in C (IRLS), `grid_fit_shrink_beta` in D (`nbinomGLM`).
- The robust dispersion estimators `trimmed_variance`, `trimmed_cell_variance` and `robust_method_of_moments_disp` are used only for Cook's distances, so they are in C, not B. `stats.trimmed_mean` (used there and in outlier replacement) is in C; `stats.mean_absolute_deviation` (dispersion prior) is in B; `stats.lowess` (independent filtering) is in D.
- `distributions.dnb_nll` is the gradient of the dispersion objective, so it is in B; `nb_nll` stays in C with the GLM likelihood; `nbinomFn` and `nbinomGLM` are shrinkage, so D.
- `DeseqDataSet.cooks_outlier`, `cond` and `contrast` feed `DeseqStats` (p-value filtering and contrast vectors), so they are in D. `DeseqDataSet.deseq2()` is in C, which owns the end-to-end fit (C's tests already need the full pipeline for Cook's distances and refit).
- `DefaultInference` methods follow the module function they dispatch to; the class itself, `n_cpus` and the `Inference` interface are in A with the `DeseqDataSet` constructor that consumes them.

## A. Data handling and normalization

Files: `tests/test_unit_data.py`, `tests/test_unit_data_dataset.py`, `tests/test_unit_data_vst.py`.

| Done | Target | Kind | Docs | Existing tests reach it |
|---|---|---|---|---|
| [x] | `dds.DeseqDataSet.fit_size_factors` | method | yes | yes |
| [x] | `dds.DeseqDataSet.to_picklable_anndata` | method | yes | **no** |
| [x] | `dds.DeseqDataSet.variables` | property | yes | **no** |
| [x] | `dds.DeseqDataSet.vst_fit` | method | yes | yes |
| [x] | `dds.DeseqDataSet.vst_transform` | method | yes | yes |
| [x] | `dds.DeseqDataSet.vst` | method | yes | yes |
| [x] | `dds.DeseqDataSet` | class | yes | yes |
| [x] | `default_inference.DefaultInference.n_cpus` | property | yes | yes |
| [x] | `default_inference.DefaultInference` | class | yes | yes |
| [x] | `inference.Inference` | class | yes | yes |
| [x] | `misc.get_num_processes` | function | yes | yes |
| [x] | `misc.n_or_more_replicates` | function | yes | yes |
| [x] | `misc.test_valid_counts` | function | yes | yes |
| [x] | `preprocessing.deseq2_norm_fit` | function | yes | yes |
| [x] | `preprocessing.deseq2_norm_transform` | function | yes | yes |
| [x] | `preprocessing.deseq2_norm` | function | yes | yes |

Private methods this area covers through the public ones above: `DeseqDataSet._check_full_rank_design` (reached), `DeseqDataSet._fit_iterate_size_factors` (reached).

## B. Dispersions

Files: `tests/test_unit_dispersions.py`, `tests/test_unit_dispersions_dds.py`.

| Done | Target | Kind | Docs | Existing tests reach it |
|---|---|---|---|---|
| [x] | `dds.DeseqDataSet.disp_function` | method | yes | yes |
| [x] | `dds.DeseqDataSet.fit_dispersion_prior` | method | yes | yes |
| [x] | `dds.DeseqDataSet.fit_dispersion_trend` | method | yes | yes |
| [x] | `dds.DeseqDataSet.fit_genewise_dispersions` | method | yes | yes |
| [x] | `dds.DeseqDataSet.fit_MAP_dispersions` | method | yes | yes |
| [x] | `default_inference.DefaultInference.alpha_mle` | method | yes | yes |
| [x] | `default_inference.DefaultInference.dispersion_trend_gamma_glm` | method | yes | yes |
| [x] | `default_inference.DefaultInference.fit_moments_dispersions` | static method (alias) | yes | yes |
| [x] | `default_inference.DefaultInference.fit_rough_dispersions` | static method (alias) | yes | yes |
| [x] | `dispersions.dispersion_trend` | function | yes | yes |
| [x] | `dispersions.fit_alpha_mle` | function | yes | yes |
| [x] | `dispersions.fit_moments_dispersions` | function | yes | yes |
| [x] | `dispersions.fit_rough_dispersions` | function | yes | yes |
| [x] | `distributions.dnb_nll` | function | yes | yes |
| [x] directly; whether `fit_alpha_mle` takes it is not visible in outputs | `grid_search.grid_fit_alpha` | function | yes | yes |
| [x] | `grid_search.vec_nb_nll` | function | yes | yes |
| [x] | `inference.Inference.alpha_mle` | abstract method | yes | - |
| [x] | `inference.Inference.dispersion_trend_gamma_glm` | abstract method | yes | - |
| [x] | `inference.Inference.fit_moments_dispersions` | abstract method | yes | - |
| [x] | `inference.Inference.fit_rough_dispersions` | abstract method | yes | - |
| [x] | `stats.mean_absolute_deviation` | function | yes | yes |

Private methods this area covers through the public ones above: `DeseqDataSet._fit_MoM_dispersions` (reached), `DeseqDataSet._fit_mean_dispersion_trend` (reached), `DeseqDataSet._fit_parametric_dispersion_trend` (reached).

## C. GLM fitting, Cook's distances and refit

File: `tests/test_unit_glm.py`.

| Done | Target | Kind | Docs | Existing tests reach it |
|---|---|---|---|---|
| [x] | `dds.DeseqDataSet.calculate_cooks` | method | yes | yes |
| [x] | `dds.DeseqDataSet.deseq2` | method | yes | yes |
| [x] | `dds.DeseqDataSet.fit_LFC` | method | yes | yes |
| [x] | `dds.DeseqDataSet.refit` | method | yes | yes |
| [x] | `default_inference.DefaultInference.irls` | method | yes | yes |
| [x] | `default_inference.DefaultInference.lin_reg_mu` | method | yes | yes |
| [x] | `dispersions.robust_method_of_moments_disp` | function | yes | yes |
| [x] | `dispersions.trimmed_cell_variance` | function | yes | yes |
| [x] `axis=1` untested (known issue) | `dispersions.trimmed_variance` | function | yes | yes |
| [x] | `distributions.nb_nll` | function | yes | yes |
| [x] | `glm.fit_lin_mu` | function | yes | yes |
| [x] | `glm.irls_solver` | function | yes | yes |
| [x] | `grid_search.grid_fit_beta` | function | yes | **no** |
| [x] | `inference.Inference.irls` | abstract method | yes | - |
| [x] | `inference.Inference.lin_reg_mu` | abstract method | yes | - |
| [x] | `stats.trimmed_mean` | function | yes | yes |

Private methods this area covers through the public ones above: `DeseqDataSet._refit_without_outliers` (reached), `DeseqDataSet._replace_outliers` (reached).

## D. Testing and shrinkage

Files: `tests/test_unit_stats.py`, `tests/test_unit_stats_functions.py`.

| Done | Target | Kind | Docs | Existing tests reach it |
|---|---|---|---|---|
| [x] | `dds.DeseqDataSet.cond` | method | yes | **no** |
| [x] | `dds.DeseqDataSet.contrast` | method | yes | yes |
| [x] two-group designs only (known issue) | `dds.DeseqDataSet.cooks_outlier` | method | yes | yes |
| [x] | `default_inference.DefaultInference.lfc_shrink_nbinom_glm` | method | yes | yes |
| [x] | `default_inference.DefaultInference.wald_test` | method | yes | yes |
| [x] | `distributions.nbinomFn` | function | yes | yes |
| [x] grid fallback not reached; `inv_hessian` only under a weak prior (known issue) | `distributions.nbinomGLM` | function | yes | yes |
| [x] shrunk `lfcSE` not compared (known issue) | `ds.DeseqStats.lfc_shrink` | method | yes | yes |
| [x] | `ds.DeseqStats.run_wald_test` | method | yes | yes |
| [x] | `ds.DeseqStats.summary` | method | yes | yes |
| [x] | `ds.DeseqStats.variables` | property | yes | **no** |
| [x] | `ds.DeseqStats` | class | yes | yes |
| [x] | `glm.wald_test` | function | yes | yes |
| [x] directly; not reached through `nbinomGLM` | `grid_search.grid_fit_shrink_beta` | function | yes | **no** |
| [x] | `inference.Inference.lfc_shrink_nbinom_glm` | abstract method | yes | - |
| [x] | `inference.Inference.wald_test` | abstract method | yes | - |
| [x] | `stats.lowess` | function | yes | yes |

Private methods this area covers through the public ones above: `DeseqStats._build_contrast_vector` (reached), `DeseqStats._cooks_filtering` (reached), `DeseqStats._fit_prior_var` (reached), `DeseqStats._independent_filtering` (reached), `DeseqStats._p_value_adjustment` (reached).

## Out of scope (plotting, example data); reached only as callers

| Done | Target | Kind | Docs | Existing tests reach it |
|---|---|---|---|---|
| - | `datasets.load_example_data` | function | yes | yes |
| - | `dds.DeseqDataSet.plot_dispersions` | method | yes | **no** |
| - | `ds.DeseqStats.plot_MA` | method | yes | yes |
| - | `plotting.make_MA_plot` | function | yes | yes |
| - | `plotting.make_scatter` | function | yes | **no** |

Totals: A 16, B 21, C 16, D 17 targets; 5 out of scope.
