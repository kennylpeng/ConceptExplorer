import numpy as np
import pandas as pd
from scipy import sparse
from uce_concept_explorer import ConceptExplorer


def make_explorer():
    rng = np.random.default_rng(20)
    n, p = 90, 14
    y = np.r_[np.zeros(n//2, dtype=int), np.ones(n//2, dtype=int)]
    X = (rng.random((n, p)) < .3).astype(float)
    X[:, 0] = (rng.random(n) < (.15 + .65*y)).astype(float)
    X[:, 1] = (rng.random(n) < (.7 - .45*y)).astype(float)
    meta = pd.DataFrame({'concept_id':np.arange(p),
                         'label':[f'Concept {j}' for j in range(p)],
                         'f1':[.95]*p})
    E = ConceptExplorer(sparse.csr_matrix(X), [f'text{i}' for i in range(n)], meta, threshold=.92)
    return E, y


def test_classifier_budget_and_fidelity():
    E, y = make_explorer()
    E.set_fidelity_filter(.9)
    model, metrics, table = E.classifier(y, n_concepts=3, min_count=3)
    assert 0 < len(table) <= 3
    assert all(table.train_count >= 3)
    assert table.concept_id.is_unique
    assert 0 <= metrics['test_balanced_accuracy'] <= 1
    assert len(model.predict(E.Z[:, E.allowed])) == len(y)


def test_regressor_budget():
    E, _ = make_explorer()
    rng = np.random.default_rng(7)
    y = 4*E.Z[:,0].toarray().ravel() + rng.normal(0,.3,E.Z.shape[0])
    model, metrics, table = E.regressor(y, n_concepts=4, min_count=3)
    assert 0 < len(table) <= 4
    assert np.isfinite(metrics['test_r2'])
    assert len(model.predict(E.Z[:, E.allowed])) == len(y)
