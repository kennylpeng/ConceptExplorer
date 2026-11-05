import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from uce_concept_explorer import ConceptExplorer

@pytest.fixture
def explorer():
    X = sparse.csr_matrix([[.99, .1, 0], [.95, .8, 0], [0, .98, .99], [0, 0, .96], [0, 0, .1], [0, 0, .1]])
    meta = pd.DataFrame({
        'concept_id': [0, 1, 2], 'label': ['Politics', 'Poetry', 'Food'],
        'f1': [.9, .4, .95], 'prevalence_gt_0p05': [.01, .2, .1],
        'prevalence_gt_0p05_upper_99': [.02, .22, .11]})
    return ConceptExplorer(X, [f'text {i}' for i in range(X.shape[0])], meta, threshold=.92)

def test_binary_prevalence(explorer):
    assert explorer.counts.tolist() == [2,1,2]
    result = explorer.prevalent(top_n=3, min_count=1)
    assert set(result.concept_id) == {0,1,2}
    assert result.iloc[0]['prevalence'] == pytest.approx(2 / 6)

def test_search_examples(explorer):
    assert explorer.search('poe').iloc[0].concept_id == 1
    assert explorer.examples(0, n=10).text_id.tolist() == [0, 1]

def test_fidelity(explorer):
    explorer.set_fidelity_filter(.8, field='f1')
    assert explorer.allowed.tolist() == [True, False, True]
    assert len(explorer.search('Poetry')) == 0
    explorer.set_fidelity_filter()
    assert len(explorer.search('Poetry')) == 1

def test_reference_raises_on_threshold_mismatch(explorer):
    with pytest.raises(ValueError, match='No reference baseline'):
        explorer.enriched()

def test_reference_additive_and_ratio(explorer):
    explorer.baseline_threshold = .05  # explicitly select calibrated reference threshold
    for multiplicative in (False, True):
        r = explorer.enriched(top_n=3, min_count=1, multiplicative=multiplicative)
        assert {'concept_id','concept_label','enrichment','rank_bound'} <= set(r.columns)

def test_categories_predictive(explorer):
    y = np.array(['a','a','b','b','b','b'])
    out = explorer.by_category(y, top_n=3, min_count=1)
    assert set(out) == {'a','b'}
    r = explorer.predictive(y, top_n=3, min_count=1)
    assert 'separation_LCB' in r

def test_continuous(explorer):
    y = np.array([0.,1.,3.,4.,5.,6.])
    r = explorer.predictive(y, kind='continuous', top_n=3, min_count=1)
    assert 'R2' in r

def test_shape_validation(explorer):
    with pytest.raises(ValueError, match='texts length'):
        ConceptExplorer(explorer.raw, [], explorer.table)


def test_explicit_threshold_mapping(explorer):
    explorer.baseline_threshold = .05
    result = explorer.enriched(top_n=3, min_count=1)
    assert len(result) > 0
    assert 'reference_prevalence' in result
