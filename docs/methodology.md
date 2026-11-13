# Statistical methods and interpretation

The full activation matrix `Z` (rows: documents; columns: concept IDs) is
retained in `E.Z`. Initialization constructs `E.binary` using
`B_ic = 1[Z_ic > binary_threshold]`. Prevalence operates on `B`; enrichment, risk, and joint selection can use
either `B` or the retained raw scores. Full activation values remain available and
are used to rank examples that pass the binary threshold.

## Prevalence

`p̂_c = (1/n) Σ_i B_ic`. `prevalent()` ranks concepts by descending activation
count, breaking ties by concept ID, and reports counts and prevalence without
confidence intervals. Category-comparison intervals treat documents as
independent, which may not hold for repeated tweets by one author, paragraphs
from one book, or clustered observations.

## Additive and multiplicative enrichment

`enrichment(labels, metric='difference')` compares each category with the remaining documents in
the supplied dataset. The default `min_count=10` counts binary activations across all documents
with valid targets, consistently with risk and joint selection. Both category and remaining samples must be nonempty.

`metric='difference'`: `p̂_group - p̂_reference`. `metric='ratio'`: `p̂_group / p̂_reference`. The latter is undefined when the reference prevalence is zero; the toolkit excludes undefined point estimates from ranking rather than inventing pseudocounts.

Confidence intervals are computed for the enrichment quantity directly. Let `v_group = p̂_group(1 - p̂_group)/n_group`, and similarly for an independently sampled reference. The additive interval is the two-sided **99% Wald interval** `difference ± z_0.995 sqrt(v_group + v_reference)`, clipped to `[-1, 1]`. The multiplicative interval uses the delta method on the log ratio: `exp(log(ratio) ± z_0.995 sqrt(v_group/p̂_group² + v_reference/p̂_reference²))`. Sorting uses the lower endpoint (the `low` endpoint). These normal approximations can be inaccurate for small samples and prevalences near zero or one; additive intervals can collapse at the boundaries. For zero numerator prevalence, the log approximation is undefined and the ratio interval is left uninformative at `[0, infinity]`. Group-versus-rest comparisons require nonempty group and reference samples.

With `enrichment(..., activations='raw')`, the estimates instead compare mean
scores from `E.Z`. Each group mean includes implicit sparse zeros. Scores must
be finite and in `[0, 1]`, and each category and its remaining sample must have
at least two documents. For each sample of size `n`, replace the binomial mean
variance above with `s²/n`, where `s² = (sum(z²) - sum(z)²/n)/(n-1)` is the
unbiased sample variance. The difference interval remains a normal approximation
clipped to `[-1, 1]`; the ratio interval uses the same log-scale delta method
with these observed mean variances. These bounds can be inaccurate in small
samples or near-zero denominator means. Raw scores are not calibrated
probabilities. `min_count` counts binary threshold crossings across all valid-target documents.
Results expose `category_mean`, `reference_mean`, and `effect`.

`enrichment` and `risk` expose `confidence_level=0.99`. For a requested level `q`,
replace `z_0.995` by `Phi^-1((1+q)/2)`. Results report the endpoints as `low`
and `high`.

## Conditioning on concept activation

`risk(labels, metric='difference')` reverses the conditioning used
by the default category comparison. For each category `c`, let
`p1 = P(Y=c | B=1)` and `p0 = P(Y=c | B=0)`, estimated among concept-active
and concept-inactive documents respectively. Concepts must have both active
and inactive documents and at least `min_count=10` activations among all
valid-target documents by default.

The additive estimate is `p1 - p0`. Compute separate Wilson intervals at `confidence_level` (99% by default)
`[L1, U1]` and `[L0, U0]` using the active and inactive sample sizes, then return
`[L1 - U0, U1 - L0]`. This conservative endpoint subtraction is not an exactly
calibrated interval at the requested level for the difference. Ranking uses
`max(L1 - U0, L0 - U1, 0)`, the smallest absolute difference
allowed by the interval. Thus, strong negative associations can also rank highly.

With `metric='ratio'`, the estimate is `p1 / p0`. For active and inactive
sample sizes `n1` and `n0`, the log-ratio delta-method standard error is
`sqrt((1-p1)/(n1*p1) + (1-p0)/(n0*p0))`. Exponentiating
`log(p1/p0) ± z_0.995 * SE` gives the approximate 99% interval. Ratios rank by
the lower endpoint, favoring positive enrichment. Zero denominator probabilities
are excluded; zero numerator probabilities receive `[0, infinity]`.

### Raw-score risk contrasts

With `risk(..., activations='raw')`, fit a separate unpenalized logistic model
for each concept and each one-versus-rest category: `p(z) = sigmoid(a + b*z)`.
Implicit sparse zeros are observations. Define `p0 = p(0)` and `p1 = p(1)`;
the risk difference is `p1-p0` and the risk ratio is `p1/p0`. These are
model-based endpoint contrasts and may extrapolate beyond a concept's observed
score range. `b` measures log odds per unit score; `exp(b)` is an odds ratio
and differs from the risk ratio.

Optimization uses centered, scaled raw scores for numerical stability; returned
coefficients and intercepts are transformed back to the original score units.
Covariance comes from the inverse observed logistic information matrix. In
original coordinates, with `d0 = [1, 0]` and `d1 = [1, 1]`, the difference
has gradient `p1*(1-p1)*d1 - p0*(1-p0)*d0`. Its standard error is
`sqrt(g.T @ covariance @ g)`. Normal bounds are clipped to `[-1, 1]`.
The log risk ratio has gradient `(1-p1)*d1 - (1-p0)*d0`; its normal bounds
are exponentiated. These are approximate model-based intervals, assuming
independent documents and the logistic mean relationship. They are pointwise
and do not account for ranking or model misspecification.

Raw mode requires finite scores in `[0, 1]` and at least three valid-target documents.
Missing and nonfinite targets are omitted before fitting. `min_count` counts binary activations, and
`E.allowed` restricts candidates. Constant raw scores and complete or
quasi-separated data have no identified finite slope; other failed or
ill-conditioned fits are also excluded, with reasons recorded in
`result['metadata']['excluded_concepts']`. See the
[discussion of logistic estimation failures](https://www.statsmodels.org/stable/pitfalls.html#incomplete-convergence-in-maximum-likelihood-estimation).
Differences rank by the conservative absolute bound and ratios by their lower
endpoint, as in binary mode. The raw scores themselves are not calibrated
probabilities; the fitted logistic response supplies the category probabilities.

## Joint concept selection

`joint(y, kind='categorical')` fits L1 logistic regression, while
`joint(y, kind='continuous')` fits Lasso regression. The goal is to select
`n_concepts` nonzero coefficients (20 by default), rather than maximize cross-validation
prediction scores. All valid supplied documents are used. There is no
holdout, CV, or post-selection refit.

`activations='binary'` uses `E.binary`; `activations='raw'` uses full values
in `E.Z`. `E.allowed` restricts candidates. In both modes, `min_count=10`
counts binary activations among documents with valid targets. Features that
are constant in the chosen representation are excluded. Remaining features
are scaled to unit variance without centering the sparse matrix; the model
fits an intercept. Reported coefficients are divided by the feature scales
so they are in the original activation units, while retaining L1 shrinkage.

For continuous targets, the initial penalty scale is the largest absolute
feature covariance with the centered target. For categorical targets, it is
the largest absolute covariance with the binary-coded target. A coarse grid
samples 25 log-spaced strengths from 100 times that scale down to 1e-8 times
the scale. Continuous fits use `alpha = strength`; categorical fits use
`C = 1 / (n * strength)`. The scale is floored at `1e-12`.

The search stops when it finds the requested support size. Otherwise, it
refines brackets whose counts straddle the requested count, using up to 40
additional fits. Counts use absolute standardized-feature coefficients above
`1e-8`. Supports can jump or vary nonmonotonically, so an exact count is not
guaranteed, and a bounded search need not find the globally closest attainable
count. The closest support found is returned; ties prefer fewer concepts,
then stronger regularization. Features are never padded or truncated to
force a particular count. Inspect the returned dictionary's `metadata` for the
actual count, exact-match status, chosen penalty, and number of fits.

Missing and nonfinite targets are removed, as in enrichment and risk.
At least two valid observations are required; categorical targets must have
exactly two classes, and continuous targets must vary. Positive logistic
coefficients favor the second class in sorted class order, recorded as
`positive_class` in the result's `metadata`.

This is exploratory feature selection, not best-subset optimization or an
estimate of predictive performance. Correlated concepts can substitute for
one another, and coefficients are not individually statistically significant.
Use independent data or a separate evaluation procedure to assess predictive
performance and selection stability.

## Fidelity

The model card describes `fidelities.csv` precision, recall, F1, and other automated metrics across 80,513 concepts. The evaluation uses *up to* 30 positive examples and 30 sparse zeros per feature with model-generated judgments; the sample is conditionally balanced and does not estimate corpus-population performance. Empty F1 cells indicate missing positives and are excluded from filtering by `f1`. Filtering is applied to every method, including feature availability for `joint()`.

## Multiple testing

Sorting ~80,000 concepts by their confidence bounds is exploratory; reported confidence bounds are **pointwise and not corrected for multiple comparisons or selection**. Confirm findings on independent data, especially when selecting features with rare activations.

To filter by fidelity, construct a Boolean mask from `E.concept_dict` and assign it
to `E.allowed` before analysis or fitting. Missing or nonnumeric metric values
should be excluded explicitly; see the example in the API reference.

## Result interpretation

All comparisons return the selected difference or ratio as `effect`, with
`low` and `high` confidence endpoints. Differences have a null value of 0;
ratios have a null value of 1. Ordering uses the bounds described above, but
no derived ranking score is returned. For fields, counts, and sample sizes,
see the [API reference](api.md).
