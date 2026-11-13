# API reference

```python
from uce_concept_explorer import ConceptExplorer
```

For a runnable example, see the [README](../README.md). For model loading, see
[encoding](encoding.md). Statistical formulas are in [methodology](methodology.md).

## Construction

```python
E = ConceptExplorer(Z, texts, concept_dict, binary_threshold=0.92)
```

| Parameter | Meaning |
| --- | --- |
| `Z` | Dense array or SciPy sparse matrix with shape `(n_documents, n_concepts)`. |
| `texts` | One document text per matrix row, in the same order. |
| `concept_dict` | Dictionary keyed by integer concept IDs `0` through `n_concepts - 1`; each entry must contain a nonempty string `label`. Additional fields are retained. |
| `binary_threshold` | Number in `[0, 1]`. Binary activation uses the strict comparison `Z > binary_threshold`. |

Construction copies the scores into a CSR matrix and copies the concept
dictionary. At least one document is required. Obtain an aligned dictionary
from `model.concept_dict` or UCE's dictionary-only loader; see the encoding guide.

## Attributes and filtering

| Attribute | Meaning |
| --- | --- |
| `E.Z` | Original scores as a CSR matrix. |
| `E.binary` | CSR matrix of binary activations computed at construction. |
| `E.texts` | Document texts in matrix-row order. |
| `E.concept_dict` | Concept metadata keyed by matrix column ID. |
| `E.counts` | Binary activation counts over the full dataset, in concept ID order. |
| `E.allowed` | Boolean eligibility mask in concept ID order; initially all true. Used by every method. |

To restrict concepts using optional fidelity metadata:

```python
import numpy as np
import pandas as pd

f1 = pd.to_numeric(
    [E.concept_dict[i].get('f1') for i in range(E.Z.shape[1])], errors='coerce',
)
E.allowed = np.isfinite(f1) & (f1 >= .8)
# E.allowed[:] = True  # Clear the filter.
```

Fidelity metrics do not change statistical calculations; they only affect
eligibility when you use them to set `E.allowed`.

## Result format

Every individual result is a plain dictionary:

```python
{
    'data': {
        'concept_id': [0, 2],
        'concept_label': ['Politics', 'Food'],
        'activation_count': [12, 8],
        # Additional columns depend on the method.
    },
    'metadata': {'n_observations': 20},
}
```

`data` contains equal-length Python lists in result order. An empty result
retains its column names with empty lists. `metadata` holds sample sizes and
analysis details, separate from the columns.

```python
result = E.prevalent()
table = pd.DataFrame(result['data'])
# For a bar chart:
# plt.barh(result['data']['concept_label'], result['data']['prevalence'])
```

`enrichment()` and `risk()` return **category → result** dictionaries:

```python
results = E.enrichment(labels)
category_result = results['Democrat']  # Use a category present in labels.
table = pd.DataFrame(category_result['data'])
```

Every data dictionary includes `concept_id` and the full `concept_label`.
Concept IDs identify columns of the original activation matrix. Ranking scores
are used internally and are not returned.

## Shared input rules

- Categorical targets must be strings or numbers with comparable class values
  (use all strings or all numbers). Targets must have one entry per document.
- Missing targets (`None`, `NaN`, `pd.NA`) and nonfinite numeric targets are
  omitted with their matrix rows, consistently across comparisons and selection.
  Metadata reports `n_observations` and `n_dropped`.
- `min_count` is a nonnegative integer counting binary threshold crossings
  among the documents used by that method. It applies in raw mode too.
- Result limits (`max_results`, `n`, `top_n`) are nonnegative integers; zero
  returns empty columns. `n_concepts` must be a positive integer.
- Raw comparisons require finite scores in `[0, 1]` on valid-target rows.
- Invalid arguments raise `ValueError`; an invalid dictionary structure can
  raise `TypeError`. No eligible comparison concepts produces an empty result.

## `E.search(query, max_results=30, all_active=False)`

Find a case-insensitive literal substring in eligible concept labels.
`query` must be a string. `all_active=True` requires at least one binary
activation; otherwise concepts with zero activations can match.

**Data:** `concept_id`, `concept_label`, `activation_count` (full dataset).
**Metadata:** `n_observations` (full dataset size).
**Order:** descending activation count, then label. Up to `max_results` rows.

## `E.examples(concept_id, n=10)`

Retrieve documents whose raw concept score exceeds the construction threshold.
`concept_id` must be an integer within the matrix columns and eligible under
`E.allowed`. Fractional, negative, and out-of-range IDs raise `ValueError`.

| Data field | Meaning |
| --- | --- |
| `concept_id`, `concept_label` | Requested concept repeated on every row. |
| `activation` | Raw activation score for the document and concept. |
| `text_id` | Zero-based original matrix row index, for alignment with external metadata. |
| `text` | Full document text. |

**Metadata:** empty dictionary.
**Order:** descending raw activation, ties by original row index. Up to `n` rows.

## `E.prevalent(top_n=20, min_count=10)`

Rank eligible concepts by frequency over the full dataset.

| Data field | Meaning |
| --- | --- |
| `activation_count` | Number of activating documents. |
| `prevalence` | `activation_count / n_observations`, in `[0, 1]`. |

**Metadata:** `n_observations` (full dataset size).
**Order:** descending count, ties by concept ID. Up to `top_n` rows.

## `E.enrichment(labels, metric='difference', top_n=15, min_count=10, confidence_level=0.99, activations='binary')`

For each category, compare mean concept activation in that category with the
remaining valid documents. Requires at least two categories.

| Parameter | Meaning |
| --- | --- |
| `labels` | One category per original document. |
| `metric` | `'difference'` computes `category_mean - reference_mean`; `'ratio'` computes `category_mean / reference_mean`. |
| `top_n` | Maximum concepts per category. |
| `min_count` | Minimum binary activations across **all valid-target documents**. |
| `confidence_level` | Number strictly between 0 and 1; controls `low` and `high`. |
| `activations` | `'binary'` compares prevalences; `'raw'` compares mean raw scores and requires at least two documents in each category and its reference sample. |

| Data field | Meaning |
| --- | --- |
| `activation_count` | Binary activations across all valid-target documents. |
| `category_count` | Binary activations within this category. |
| `category_mean` | Mean binary activation (prevalence) or raw score within this category. |
| `reference_mean` | Corresponding mean outside this category. |
| `effect` | Requested difference or ratio. Null value: 0 for differences, 1 for ratios. |
| `low`, `high` | Confidence endpoints for `effect`. |

**Metadata:** `metric`, `activations`, `confidence_level`, `n_observations`,
`n_dropped`, `category_size`, `reference_size`.
**Order:** descending `low`, ties by concept ID. Up to `top_n` rows per category.
Undefined ratios with zero reference means are excluded; a zero numerator
receives an uninformative `[0, infinity]` interval.

## `E.risk(labels, metric='difference', top_n=15, min_count=10, confidence_level=0.99, activations='binary')`

For each category, compare its probability at concept activation 1 versus 0.
In binary mode, these are observed category frequencies among active versus
inactive documents. Raw mode fits a separate logistic regression per concept,
then evaluates the fitted probabilities at raw scores 1 and 0.

Parameters have the same defaults and validation as `enrichment()`. Raw risk
requires at least three valid documents. Always-active and always-inactive
concepts are excluded in binary mode; raw mode requires varying raw scores.

| Data field | Meaning |
| --- | --- |
| `activation_count` | Binary activations across all valid-target documents. |
| `inactive_count` | `n_observations - activation_count`. In raw mode, this is descriptive; the fit uses every raw score. |
| `risk_at_0` | Observed category probability among inactive documents, or fitted probability at raw score 0. |
| `risk_at_1` | Observed category probability among active documents, or fitted probability at raw score 1. |
| `effect` | `risk_at_1 - risk_at_0` or `risk_at_1 / risk_at_0`, according to `metric`. |
| `low`, `high` | Confidence endpoints for `effect`. |
| `coefficient` | Raw mode only: logistic slope in original score units (log-odds per unit score). |
| `intercept` | Raw mode only: fitted category log-odds at score 0. |

**Metadata:** same fields as enrichment; raw mode also includes
`excluded_concepts`, a dictionary mapping excluded concept IDs to `constant`,
`separated`, `fit_failed`, `singular_information`, or `nonfinite_estimate`.
Concepts failing `E.allowed` or `min_count` are not included in this dictionary.

**Order:** differences sort by `max(low, -high, 0)`, so negative associations
can rank highly. Ratios sort by `low`. Ties use concept ID. Up to `top_n` rows
per category. Binary ratios with zero reference probability are excluded.

Raw endpoint probabilities can extrapolate beyond the observed scores.
Exponentiating the logistic coefficient gives an odds ratio, distinct from
`effect` with `metric='ratio'`.

## `E.joint(y, kind='categorical', n_concepts=20, min_count=10, activations='binary', random_state=42)`

Select concepts jointly by tuning an L1 penalty toward `n_concepts` nonzero
coefficients. Uses all valid-target documents and eligible, variable features.
It returns selection results; it does not estimate held-out performance.

| Parameter | Meaning |
| --- | --- |
| `y` | One target per original document. |
| `kind` | `'categorical'`: L1 logistic regression, requiring exactly two classes. `'continuous'`: Lasso, requiring numeric targets that vary. |
| `n_concepts` | Positive integer requested support size. Returns the closest count found if an exact match is unavailable. |
| `min_count` | Minimum binary activations among valid-target documents. |
| `activations` | `'binary'` or `'raw'`. Features are scaled to unit variance during selection. Raw selection requires finite scores; unlike raw comparisons, it does not require `[0, 1]`. |
| `random_state` | Solver random seed. |

| Data field | Meaning |
| --- | --- |
| `activation_count` | Binary activations among valid-target documents. |
| `coefficient` | Selected L1 coefficient converted to original activation units, retaining shrinkage. Positive categorical coefficients favor `positive_class`. Continuous coefficients measure target change per unit activation, holding other selected features fixed. |

**Order:** descending absolute coefficient, ties by concept ID. The result may
contain fewer or more than `n_concepts`, including zero. No features are padded
or truncated. Raises `ValueError` if no eligible variable features remain.

| Metadata field | Meaning |
| --- | --- |
| `kind`, `activations` | Requested target kind and activation mode. |
| `n_observations`, `n_dropped` | Documents used and omitted because of invalid targets. |
| `n_requested`, `n_selected` | Requested and actual selected count. |
| `exact_match` | Whether the counts match. |
| `penalty` | Selected `{'C': value}` for categorical or `{'alpha': value}` for continuous targets. |
| `n_fits` | Number of penalty values evaluated. |
| `n_eligible_concepts` | Candidate count after filtering constant features. |
| `coefficient_tolerance` | Threshold on standardized coefficients used to define a nonzero. |
| `positive_class` | Categorical only: second class in sorted class order. |

```python
selected = E.joint(y, n_concepts=5)
print(selected['metadata']['n_selected'], selected['metadata']['exact_match'])
table = pd.DataFrame(selected['data'])
```
