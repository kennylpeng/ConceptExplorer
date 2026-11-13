# UCE Concept Explorer

Explore sparse concept activations: find concepts, inspect documents, compare
categories, and select a small set of concepts with L1 regression.

The analysis package runs on CPU and takes an existing activation matrix.
[Encoding texts with UCE-Gemma-80k](docs/encoding.md) is a separate step that can
use CUDA or Apple MPS.

## Install

```bash
pip install -e .
```

## Try it

This example runs without downloading a model:

```python
import numpy as np
import pandas as pd
from uce_concept_explorer import ConceptExplorer

texts = [f'Document {i}' for i in range(8)]
labels = ['A'] * 4 + ['B'] * 4
scores = np.array([
    [.99, 0, .98], [.95, .99, 0], [.98, 0, 0], [0, .96, 0],
    [0, .99, .97], [0, .95, .99], [.96, 0, .95], [0, .98, 0],
])
concepts = {
    0: {'label': 'Politics'},
    1: {'label': 'Poetry'},
    2: {'label': 'Food'},
}
E = ConceptExplorer(scores, texts, concepts, binary_threshold=.92)

# Every method returns plain dictionaries containing data and metadata.
common = E.prevalent(min_count=1)
print(common['data'])
display_table = pd.DataFrame(common['data'])

matches = E.search('politics')
examples = E.examples(0)
comparison = E.enrichment(labels, min_count=1)['A']
risks = E.risk(labels, min_count=1)['A']
selected = E.joint(labels, n_concepts=2, min_count=1)
print(selected['metadata']['n_selected'])
```

`data` contains equal-length lists, one per column. Use them directly in a chart:

```python
# Optional: pip install matplotlib
import matplotlib.pyplot as plt

plt.barh(common['data']['concept_label'], common['data']['prevalence'])
plt.xlabel('Fraction of documents activating the concept')
plt.show()
```

## Choose a method

| Method | Question |
| --- | --- |
| `search(query)` | Which concept labels match this text? |
| `examples(concept_id)` | Which documents activate this concept most strongly? |
| `prevalent()` | Which concepts occur most often? |
| `enrichment(labels)` | Is a concept more active in this category than in the rest? |
| `risk(labels)` | Is this category more likely when the concept is active? |
| `joint(y)` | Which concepts jointly explain a categorical or continuous target? |

`enrichment()` and `risk()` return a dictionary of results keyed by category.
Their `effect` field is a difference by default; use `metric='ratio'` for ratios.
Both also support `activations='raw'`. `joint()` supports raw scores and
`kind='continuous'` for numeric targets.

Counts and `min_count` always refer to binary threshold crossings. Comparisons
and selection omit documents with missing or nonfinite targets. Selection uses
all supplied valid targets; assess predictive performance on separate data.

## Documentation

- [API reference](docs/api.md): parameters, output fields, ordering, and validation.
- [Statistical methodology](docs/methodology.md): formulas and interpretation.
- [Encoding guide](docs/encoding.md): model loading, cached scores, and acceleration.
- [Evaluation scripts](evaluation/README.md): dataset splits and sparse encoding.
- `notebooks/01_political_tweets.ipynb`, `02_poetry.ipynb`, and `03_recipes.ipynb`:
  worked examples. Install their dependencies with `pip install -e '.[notebooks]'`.

## Changes in 0.3.0

Results now use `{'data': {column: list, ...}, 'metadata': {...}}` instead of
DataFrames. Convert with `pd.DataFrame(result['data'])`; former DataFrame
attributes are in `result['metadata']`. Comparison estimates use `effect`,
counts use `activation_count`, and enrichment adds `category_count`.
Ranking scores are no longer returned. `min_count` for enrichment now counts
activations across all valid-target documents, consistent with other methods.
See the API reference for the complete schemas.

## Licensing

Package code is MIT-licensed. Model weights and datasets have their own terms.
When saving activations, keep the corresponding texts, concept dictionary,
model revision, and preprocessing settings together.
