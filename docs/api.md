# API reference

## Metadata

```python
from uce_concept_explorer import load_metadata
metadata = load_metadata('klpeng/uce-gemma-80k')
```

Calls the UCE package's `ConceptMetadata.from_pretrained()` if available; otherwise downloads `labels.csv`, `prevalences.csv`, and `fidelities.csv` without model weights. Accepts an existing `pandas.DataFrame` or an object exposing `.to_dataframe()` / `.describe(ids)` to initialize the explorer.

```python
E = ConceptExplorer(X, texts, metadata, threshold=0.92)
```

`X` may be a dense NumPy array or SciPy sparse matrix. `texts` should be a list of plain strings corresponding in order to matrix rows. Column ID *i* must correspond to metadata concept ID *i*.

## Methods

```python
E.search('poetry', max_results=30, all_active=False)
E.examples(42, n=10)
E.prevalent(top_n=20, min_count=1)
E.enriched(top_n=20, multiplicative=False, min_count=5)
E.by_category(labels, top_n=15, multiplicative=True, min_count=5, reference='rest')
E.predictive(labels, kind='categorical', top_n=20, min_count=5)
E.predictive(y_numeric, kind='continuous', top_n=20)
E.lasso(labels, kind='categorical', top_n=20, test_size=0.25, random_state=42)
E.set_fidelity_filter(min_fidelity=0.8, field='f1')
E.set_fidelity_filter()  # clear
```

For `enriched` and `reference='corpus'`, specify `baseline_threshold` at construction to select the reference-corpus threshold calibrated to the UCE score threshold. Example: `ConceptExplorer(X, texts, metadata, threshold=0.92, baseline_threshold=0.05)` (illustrative only: confirm the mapping in UCE documentation). Without an explicit mapping, the function looks for the numerically identical threshold and errors if absent. No threshold interpolation is performed.

`E.examples()` returns one row per activating text, with concept ID, label, raw activation, row index (`text_id`), and full text.

`E.lasso()` returns `(fitted_model, metrics_dict, selected_concepts_dataframe)`.

Some users will want to pass group identifiers to a custom grouped cross-validation procedure rather than using the built-in simple random split.


### `E.classifier(y, n_concepts=5, min_count=10, groups=None, ...)`
For binary targets, trains a max-`n_concepts` L1-selected logistic classifier,
with minimum frequency screening and inner CV, then refits regularized logistic
regression. Returns `(fitted_pipeline, metrics, selected_concepts_dataframe)`.
Pass `groups` for grouped holdout and CV.

### `E.regressor(y, n_concepts=5, min_count=10, groups=None, ...)`
For numerical targets, uses Lasso-based selection and Ridge refitting with CV.
Returns `(fitted_pipeline, metrics, selected_concepts_dataframe)`.
Missing numeric targets are removed, and `groups` is optional.

For both methods, the fitted pipeline consumes the same eligible concept columns
as `E.Z[:, E.allowed]`, not the original raw matrix. Feature filtering and
selection occur within CV folds. The final test data are held out from tuning.
