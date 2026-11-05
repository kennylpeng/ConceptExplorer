# API reference

```python
from uce_concept_explorer import ConceptExplorer, load_metadata
```

## Metadata and construction

### `load_metadata(repo_id="klpeng/uce-gemma-80k", revision=None)`

Loads concept metadata without downloading encoder weights. `repo_id` can be a
Hugging Face repository ID or a local model directory; `revision` selects a Hub
branch, tag, or commit.

Returns the UCE package's `ConceptMetadata.from_pretrained()` result when
available. Otherwise, returns a `pandas.DataFrame` assembled from `labels.csv`
and any available `prevalences.csv` and `fidelities.csv` files.

```python
metadata = load_metadata('klpeng/uce-gemma-80k')
# Reuse an already-downloaded model directory:
metadata = load_metadata(model_dir)
```

### `ConceptExplorer(X, texts, metadata, threshold=0.92, min_fidelity=None, fidelity_field="f1", baseline_threshold=None)`

Creates an explorer from an already-encoded score matrix. `X` can be a dense
NumPy array or SciPy sparse matrix with shape `(n_documents, n_concepts)`.
`texts` contains one plain string per row, in the same order. `metadata` accepts
a DataFrame or an object exposing `.to_dataframe()` or `.describe(ids)`;
metadata concept ID *i* must describe matrix column *i*.

The explorer stores raw scores in `E.raw` and binary activations in `E.Z`, using
the strict comparison `X > threshold`. `threshold` must be between 0 and 1.
Analyses use binary activations; raw scores determine the ordering of examples.
`min_fidelity` applies an initial filter using `fidelity_field` (see
`set_fidelity_filter()` below). `E.allowed` is the Boolean mask of eligible
concept columns.

`baseline_threshold` selects the reference-corpus prevalence columns used by
`enriched()` and `by_category(reference='corpus')`. It defaults to `threshold`.
Supply the calibrated UCE-to-reference threshold mapping when the numerical
values differ; the explorer does not infer or interpolate that mapping.

```python
E = ConceptExplorer(X, texts, metadata, threshold=0.92)
# If the documented calibration maps UCE 0.92 to reference 0.05:
E = ConceptExplorer(X, texts, metadata, threshold=0.92, baseline_threshold=0.05)
```

## Methods

Table results identify concepts with `concept_id` and `concept_label`. Search
and descriptive ranking tables also include `fidelity` once a fidelity metric
has been loaded. See [Statistical methods and interpretation](methodology.md)
for uncertainty estimates, reference calibration, and modeling assumptions.

### `E.name(concept_id)`

Returns the full concept label as a string. This lookup works even when the
concept is excluded by the fidelity filter. Raises `IndexError` for an ID outside
the matrix's column range.

```python
label = E.name(42)
```

### `E.set_fidelity_filter(min_fidelity=None, field="f1")`

Updates `E.allowed` to keep concepts whose numeric `field` value is finite and
at least `min_fidelity`. Missing or nonnumeric values are excluded. Raises
`ValueError` if the requested metadata column is absent. Calling with no minimum
clears the eligibility filter. Returns `E`, so calls can be chained.

The filter affects search, examples, descriptive rankings, and modeling feature
availability. UCE fidelity F1 comes from a limited, conditionally balanced
evaluation sample; it does not estimate corpus-population performance.

```python
E.set_fidelity_filter(min_fidelity=0.8, field='f1')
E.set_fidelity_filter()  # Allow all concepts again.
```

### `E.search(query, max_results=30, all_active=False)`

Searches full concept labels using a case-insensitive, literal substring.
Returns up to `max_results` eligible concepts, sorted by descending activation
count and then label, with a `count` column. Set `all_active=True` to require at
least one activation in this dataset; otherwise, matching inactive concepts can
also appear.

```python
matches = E.search('poetry', max_results=30, all_active=True)
```

### `E.examples(concept_id, n=10)`

Returns up to `n` texts whose concept score exceeds the explorer's threshold,
ordered by descending raw activation. The DataFrame contains `concept_id`,
`concept_label`, `activation`, `text_id` (the original matrix row index), and
the full `text`. Raises `ValueError` if the concept fails the fidelity filter.

```python
examples = E.examples(42, n=10)
```

### `E.prevalent(top_n=20, min_count=1)`

Ranks eligible concepts by the lower endpoint of their 95% Wilson prevalence
interval. Requires at least `min_count` activating documents and returns up to
`top_n` rows with `count`, `prevalence`, `low`, and `high`. Prevalence is the
fraction of all dataset documents activating the concept.

```python
common = E.prevalent(top_n=20, min_count=5)
```

### `E.enriched(top_n=20, multiplicative=False, min_count=5)`

Compares the full dataset's concept prevalence with the UCE general-corpus
baseline. By default, `enrichment` is the prevalence difference; with
`multiplicative=True`, it is the prevalence ratio. Concepts need at least
`min_count` dataset activations. Undefined ratios with zero reference prevalence
are excluded.

Returns up to `top_n` rows, sorted by descending conservative lower bound, with
`count`, `prevalence`, `reference_prevalence`, `enrichment`, `low`, `high`, and
`rank_bound`. The corpus baseline uses its supplied one-sided 99% upper bound
for conservative ranking; the displayed bounds are provisional rather than
calibrated two-sample confidence intervals.

The metadata must contain baseline columns matching `E.baseline_threshold`,
such as `prevalence_gt_0p05` and `prevalence_gt_0p05_upper_99` for `0.05`.
Raises `ValueError` when the required columns are missing. Configure the
calibrated threshold at construction as described above.

```python
enrichment = E.enriched(top_n=20, multiplicative=False, min_count=5)
```

### `E.by_category(labels, top_n=15, multiplicative=False, min_count=5, reference='rest')`

Ranks enriched concepts separately for each category. `labels` supplies one
category per document in matrix-row order. With `reference='rest'`, each
category is compared with all other documents, so each comparison needs both
category and reference documents. With `reference='corpus'`, each category is
compared with the general-corpus baseline and requires the baseline metadata
described under `enriched()`.

Returns a dictionary mapping category values to enrichment DataFrames with the
same columns as `enriched()`. `top_n` applies separately to each category, and
`min_count` counts activations within that category. `multiplicative=True`
selects ratios instead of differences. Both modes rank by the lower bound.

```python
by_party = E.by_category(labels, top_n=15, multiplicative=True, reference='rest')
```

### `E.predictive(y, kind='categorical', top_n=20, min_count=5)`

Ranks individual concepts by their association with a target containing one
value per document. This is a descriptive, univariate ranking computed on the
supplied data; it does not fit a held-out predictive model.

For `kind='categorical'`, compares each class with all remaining classes and
keeps the comparison with the largest conservative lower bound on absolute
prevalence separation. Returns `best_class`, signed `prevalence_difference`,
`low`, `high`, and `separation_LCB`, sorted by the last column. Use at least two
classes. `min_count` is the minimum activation count over the dataset.

For `kind='continuous'`, drops nonfinite numeric targets and ranks by the lower
bound on squared Pearson correlation. Requires more than four valid
observations and a variable target. Returns `count`, signed `correlation`,
`R2`, `low`, `high`, and `R2_LCB`. Bounds use an approximate Fisher-z interval;
`min_count` is applied only to rows with valid targets, while the returned
`count` reports activations over the full dataset.

```python
categorical = E.predictive(labels, kind='categorical', top_n=20, min_count=5)
continuous = E.predictive(y_numeric, kind='continuous', top_n=20, min_count=5)
```

### `E.lasso(y, kind='categorical', top_n=20, test_size=0.25, random_state=42)`

Fits an L1 model on eligible binary concepts, tuning regularization on training
rows and evaluating on a random holdout of size `test_size`. `y` must align
with matrix rows. Categorical targets use stratified splitting and
`LogisticRegressionCV` with three-fold CV; continuous targets use `LassoCV`
with five-fold CV. Supply complete targets and enough observations for the
split and CV folds.

Returns `(fitted_model, metrics_dict, selected_concepts_dataframe)`:

- Categorical metrics: `heldout_accuracy`, `heldout_balanced_accuracy`, and
  `n_eligible_concepts`.
- Continuous metrics: `heldout_R2`, `heldout_RMSE`, and `n_eligible_concepts`.
- The table contains `target_class`, `concept_id`, `concept_label`, and
  `coefficient`, ordered by absolute coefficient. `top_n` limits the displayed
  nonzero coefficients; it does not limit how many concepts the model uses.

The fitted model expects `E.Z[:, E.allowed]` using the same eligible columns as
at fitting time. For grouped evaluation or a limit on selected concepts, use
`classifier()` or `regressor()`.

```python
model, metrics, concepts = E.lasso(labels, kind='categorical', top_n=20)
```

### `E.classifier(y, n_concepts=5, min_count=10, test_size=0.2, scoring='balanced_accuracy', random_state=42, groups=None)`

Fits a binary classifier using up to `n_concepts`. A training-fold frequency
filter keeps concepts activating in at least `min_count` documents, L1 logistic
regression selects features, and regularized logistic regression refits on
those features. Three-fold CV tunes both selection and refit regularization
using `scoring`. The final test data are held out from tuning.

`y` must have one value per document and exactly two nonmissing classes;
missing targets are removed. At least 12 valid observations are required, with
enough examples of each class for the holdout and CV folds. Pass a document-aligned
`groups` array, such as author IDs, for grouped holdout and grouped CV. Otherwise,
the method uses a stratified random holdout. `test_size` controls the holdout
fraction (the fraction of groups when grouped), and `random_state` controls
randomized operations.

Returns `(fitted_pipeline, metrics, selected_concepts_dataframe)`. Metrics include
`cv_score`, `best_params`, `scoring`, `n_train`, `n_test`, `n_selected`,
`n_eligible_concepts`, `test_balanced_accuracy`, and `test_auc`. The selected
table contains `concept_id`, `concept_label`, `train_count`, and the refitted
`coefficient`, sorted by absolute coefficient, plus `fidelity` when available.

The pipeline consumes `E.Z[:, E.allowed]` with the same eligible columns and
ordering used at fitting time.

```python
model, metrics, concepts = E.classifier(y, n_concepts=5, groups=author_ids)
predictions = model.predict(E.Z[:, E.allowed])
```

### `E.regressor(y, n_concepts=5, min_count=10, test_size=0.2, scoring='r2', random_state=42, groups=None)`

Fits a numerical-target model using up to `n_concepts`, with training-fold
frequency filtering, Lasso-based feature selection, and Ridge refitting.
Three-fold CV tunes selection and refit regularization using `scoring`, and a
separate holdout provides the final evaluation. Filtering and selection occur
within CV folds.

`y` must contain one numeric value per document. Nonfinite targets are removed;
at least 12 valid observations and a variable target are required. `min_count`,
`test_size`, `random_state`, and `groups` work as described under `classifier()`,
except that the ungrouped holdout and CV splits are unstratified.

Returns `(fitted_pipeline, metrics, selected_concepts_dataframe)`. Metrics include
`cv_score`, `best_params`, `scoring`, `n_train`, `n_test`, `n_selected`,
`n_eligible_concepts`, `test_r2`, and `test_mae`. The selected table has the same
columns and ordering as `classifier()`; coefficients are in the target's units.
The pipeline expects the same eligible binary columns used during fitting.

```python
model, metrics, concepts = E.regressor(y_numeric, n_concepts=5, groups=author_ids)
predictions = model.predict(E.Z[:, E.allowed])
```
