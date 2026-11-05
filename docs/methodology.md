# Statistical methods and interpretation

The raw score matrix `X` (rows: documents; columns: concept IDs) is thresholded once into `Z_ic = 1[X_ic > t]`. All prevalence, enrichment, and prediction analyses operate on `Z`. Raw activation strengths are retained **only to rank examples that pass threshold**; they do not influence statistical estimates.

## Prevalence and uncertainty

`p̂_c = (1/n) Σ_i Z_ic`. Each prevalence is accompanied by a two-sided **95% Wilson interval**; concepts are sorted by the lower Wilson bound. Documents are treated as independent, which may not hold for repeated tweets by one author, paragraphs from one book, or clustered observations.

## Additive and multiplicative enrichment

Additive: `p̂_group - p̂_reference`. Multiplicative: `p̂_group / p̂_reference`. The latter is undefined when the reference prevalence is zero; the toolkit excludes undefined point estimates from ranking rather than inventing pseudocounts.

For group-versus-rest, a conservative interval is formed from the Wilson endpoints: `[lo_group - hi_rest, hi_group - lo_rest]` or `[lo_group / hi_rest, hi_group / lo_rest]` when denominators are positive. Ratios with a zero lower reference endpoint have unbounded upper intervals. Sorting uses the lower endpoint. This interval construction is not a jointly calibrated 95% procedure and does not account for selecting features after looking at the results.

For general-corpus reference rates, the package uses UCE's baseline point estimate and its one-sided **99% upper bound** for conservative ranking. These reference bounds are not symmetrical: reported intervals are provisional, not calibrated two-sample confidence intervals.

### Cross-model calibration and reference thresholds

Reference prevalences were collected with a historical Gemini teacher model, rather than UCE-Gemma itself. However, **UCE activation thresholds were tuned so that prevalence rates transfer reasonably well** to the supplied reference statistics. They can therefore serve as a meaningful approximate baseline for exploratory comparisons.

The UCE activation threshold and reference threshold may have *different numerical values*. Use `baseline_threshold` to specify the documented calibrated correspondence; the package does not infer or interpolate it. Cross-model calibration error and domain shift are not included in the supplied reference confidence bound. Thus, the reported intervals are not fully calibrated two-sample or post-selection intervals. Users should validate consequential findings on independent data.

## Predictive concepts

For categorical targets, each concept is compared against each class versus all other classes, recording the highest conservative lower bound on the absolute difference in prevalence, its signed estimate, and associated class. For continuous targets, univariate Pearson `r²` (equivalent to a one-predictor linear fit with intercept) is accompanied by **approximate** Fisher-z confidence endpoints. These rely on assumptions that are imperfect for binary features and should not be construed as rigorous post-selection bounds. Avoid causal interpretations.

## L1 models

Categorical targets: L1-penalized logistic regression (`LogisticRegressionCV`) with internal cross-validation on training rows. Continuous targets: `LassoCV`. Reported metrics use a held-out random split. This is suitable for demos, but repeated authors/poems or temporal dependence can make a random row-level split optimistic. Use group- or time-aware split for research claims. Nonzero selected coefficients are model-dependent and not individually statistically significant by default.

## Fidelity

The model card describes `fidelities.csv` precision, recall, F1, and other automated metrics across 80,513 concepts. The evaluation uses *up to* 30 positive examples and 30 sparse zeros per feature with model-generated judgments; the sample is conditionally balanced and does not estimate corpus-population performance. Empty F1 cells indicate missing positives and are excluded from filtering by `f1`. Filtering is applied to every method, including feature availability for the L1 model.

## Multiple testing

Sorting ~80,000 concepts by their confidence bounds is exploratory; reported confidence bounds are **pointwise and not corrected for multiple comparisons or selection**. Confirm findings on independent data, especially when selecting features with rare activations.


## Fixed-budget supervised models

`classifier()` and `regressor()` screen out concepts appearing in fewer than
`min_count` training documents *within each CV fold*, select up to
`n_concepts` with an L1-penalized model, and refit a regularized model on
the selected binary features. Cross-validation tunes both selection and refit
regularization. A separate holdout provides a final score. No claim of exact
best-subset optimality is made. Feature selection may identify rare-but-specific
concepts, so use minimum counts and inspect selected examples.
For repeated tweets or other clustered observations, grouped holdout and CV
are recommended. Do not reuse the final test set to select n_concepts.
