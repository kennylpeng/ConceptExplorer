# UCE Concept Explorer

**Encode once, explore many times.** A small, CPU-only analysis toolkit for the sparse concept activations produced by [UCE-Gemma-80k](https://huggingface.co/klpeng/uce-gemma-80k).

It supports concept-label search, example retrieval, prevalence, additive and multiplicative enrichment, categorical comparisons, concept-level prediction, and L1-regularized prediction. Concept tables always include **concept ID and full label**. Analyses use **binary activations at a user-chosen threshold**.

**Status:** v0.1 release candidate. The packaged tests pass on synthetic matrices. The example notebooks have been syntax-checked, but the full GPU/Hugging Face pipeline has not been executed in this environment.

## Try in Google Colab

Open `notebooks/01_political_tweets.ipynb` in Colab and run with a T4 GPU. The notebook is self-contained (including model loading, data download, and analysis). Two other runnable examples cover poetry and concept-based recipe search.

## Install locally

```bash
pip install -e .
# For notebook datasets and visualization
pip install -e '.[notebooks]'
```

## Quickstart (after encoding)

```python
from scipy import sparse
from uce_concept_explorer import ConceptExplorer, load_metadata

# X: sparse UCE scores, shape (n_documents, 80513); texts: list[str]
metadata = load_metadata('klpeng/uce-gemma-80k')  # no encoder weights/GPU
E = ConceptExplorer(X, texts, metadata, threshold=0.92)

E.search('democracy')
E.examples(79008, n=10)
E.prevalent(top_n=20)
E.by_category(labels, top_n=20, multiplicative=True)
E.predictive(labels, kind='categorical', top_n=20)
clf, metrics, concepts = E.lasso(labels, kind='categorical')
```

Optional fidelity filtering:

```python
E.set_fidelity_filter(min_fidelity=0.8, field='f1')
```

The `f1` here is an automated, conditionally sampled fidelity metric, *not* a prevalence-weighted population F1; see [methods](docs/methodology.md).

## Methods

| Method | Output |
| --- | --- |
| `search(query)` | Case-insensitive label substring search over all concepts |
| `examples(id, n=10)` | Original texts above threshold, ordered by activation strength |
| `prevalent(top_n=20)` | Concept frequency and 95% Wilson bounds; sorted by lower bound |
| `enriched(multiplicative=False)` | Enrichment vs. historical reference baseline, if available |
| `by_category(labels, reference='rest')` | One-vs-rest prevalence differences or ratios, with bounds |
| `predictive(y, kind='categorical')` | Best one-vs-rest separation and conservative lower bound |
| `predictive(y, kind='continuous')` | Univariate R² with approximate Fisher-transformed bounds |
| `lasso(y, kind='categorical')` | Sparse L1 logistic classifier, cross-validated penalty, held-out metrics |
| `lasso(y, kind='continuous')` | Lasso regression, cross-validated penalty, held-out metrics |
| `set_fidelity_filter(min_fidelity, field='f1')` | Exclude concepts below selected fidelity measure |

`labels` can contain strings or numerical category IDs. For predictive analyses the returned `concept_label` and `concept_id` identify the feature. `lasso` returns a fitted estimator, held-out metrics, and a coefficient table.

## Calibrated reference prevalences

The package's general-corpus reference prevalences originate from a Gemini teacher model. **UCE-Gemma activation thresholds were tuned so that concept prevalence rates translate reasonably well to those reference rates.** Therefore, the reference corpus is useful for exploratory enrichment comparisons, despite the different encoders.

A UCE score threshold need not equal the reference corpus's numerical threshold. Specify the calibrated pairing explicitly (the example below is illustrative, **not a verified calibration mapping**):

```python
E = ConceptExplorer(X, texts, metadata, threshold=0.92, baseline_threshold=0.05)
E.enriched(multiplicative=True)
```

Consult the model's calibration guidance for the appropriate threshold pairing. The library never guesses or interpolates threshold mappings. The reference 99% upper bound concerns reference-corpus sampling and does **not** include error introduced by cross-model calibration, domain shift, or concept-specific differences. The ranking remains exploratory, not a calibrated hypothesis test.

## Repository contents

- `src/uce_concept_explorer/`: reusable CPU-only toolkit
- `notebooks/01_political_tweets.ipynb`: categorical labels, party enrichment and sparse classification
- `notebooks/02_poetry.ipynb`: short poems, author comparisons, fidelity filtering
- `notebooks/03_recipes.ipynb`: interpretable binary sparse retrieval
- `docs/methodology.md`: definitions, uncertainty, evaluation caveats
- `docs/api.md`: API reference and usage patterns
- `tests/`: small synthetic-data tests

## Data and model licensing

The package's own code is MIT-licensed. The pretrained UCE-Gemma weights, underlying model, and third-party datasets may have separate licenses, terms, and access requirements. Verify these before redistributing outputs or data.

## Reproducibility

Save and reload the sparse scores with `scipy.sparse.save_npz()` / `load_npz()`, and keep text ordering, threshold, dataset version, and **model revision/vocabulary ordering** together. Split by politician/author where practical when evaluating out-of-sample generalization.


## Fixed-budget predictive models

`E.classifier()` (binary categorical target) and `E.regressor()` (continuous target)
select **at most** `n_concepts` jointly using L1 regression, then refit
regularized logistic or Ridge regression. A minimum activation-count filter and
feature selection are fitted *inside each training fold* to avoid CV leakage.
Hyperparameters are chosen by cross-validation on the training partition;
a disjoint test set is used once for evaluation. The output contains the fitted
pipeline, metrics, and a table with concept IDs, labels, counts, and coefficients.

```python
clf, scores, features = E.classifier(party_labels, n_concepts=5, min_count=10)
reg, scores, features = E.regressor(numeric_ratings, n_concepts=5, min_count=10)
```

If observations are grouped (e.g., tweets by politician), pass
`groups=politician_ids` to prevent the same group appearing in train and test
and within cross-validation folds. Models returned by these methods expect the
binary matrix `E.Z[:, E.allowed]` (and the same columns/order as during training).
The selected concepts are an approximate best subset, **not** a guaranteed
globally optimal subset, and `n_concepts` is a maximum rather than a requirement.

## Install from GitHub (including a monorepo subfolder)

If this package lives at the repository root:

```bash
pip install 'uce-concept-explorer @ git+https://github.com/YOUR_ORG/YOUR_REPO.git@main'
```

If the package lives in a subfolder `packages/uce-concept-explorer` containing
`pyproject.toml`:

```bash
pip install 'uce-concept-explorer @ git+https://github.com/YOUR_ORG/YOUR_REPO.git@main#subdirectory=packages/uce-concept-explorer'
```

Replace the URL and branch/tag with your own. For reproducible public notebooks,
prefer a release tag (e.g. `@v0.2.0`) to `@main`. The `subdirectory` option
is supported by pip, and refers to the folder containing `pyproject.toml`.
