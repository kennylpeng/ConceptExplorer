from conftest import frame, categories
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.stats import norm
from uce_concept_explorer import ConceptExplorer

@pytest.fixture
def explorer():
    X = sparse.csr_matrix([[.99, .1, 0], [.95, .8, 0], [0, .98, .99], [0, 0, .96], [0, 0, .1], [0, 0, .1]])
    meta = pd.DataFrame({
        'concept_id': [0, 1, 2], 'label': ['Politics', 'Poetry', 'Food'],
        'f1': [.9, .4, .95]}).set_index('concept_id').to_dict(orient='index')
    return ConceptExplorer(X, [f'text {i}' for i in range(X.shape[0])], meta, binary_threshold=.92)

def test_binary_prevalence(explorer):
    assert explorer.counts.tolist() == [2,1,2]
    result = frame(explorer.prevalent(top_n=3, min_count=1))
    assert result.concept_id.tolist() == [0, 2, 1]
    assert result['activation_count'].tolist() == [2, 2, 1]
    assert set(result.columns) == {'concept_id', 'concept_label', 'activation_count', 'prevalence'}
    assert result.iloc[0]['prevalence'] == pytest.approx(2 / 6)


@pytest.mark.parametrize('as_sparse', [False, True])
def test_full_activations_and_binary_threshold(as_sparse):
    values = np.array([[.9, .5], [.1, .500000001]], dtype=np.float64)
    expected = values.copy()
    Z = sparse.csr_matrix(values) if as_sparse else values
    meta = pd.DataFrame({'concept_id': [0, 1], 'label': ['First', 'Second']}).set_index('concept_id').to_dict(orient='index')
    E = ConceptExplorer(Z=Z, texts=['a', 'b'], concept_dict=meta, binary_threshold=.5)
    np.testing.assert_array_equal(E.Z.toarray(), values)
    np.testing.assert_array_equal(E.binary.toarray(), [[1, 0], [0, 1]])
    assert not hasattr(E, 'binary_threshold')
    assert frame(E.examples(1)).activation.tolist() == [.500000001]
    # The retained matrix is independent of the caller's input storage.
    if as_sparse:
        Z.data[:] = 0
    else:
        Z[:] = 0
    np.testing.assert_array_equal(E.Z.toarray(), expected)


@pytest.mark.parametrize('binary_threshold', [-.1, 1.1, np.nan])
def test_binary_threshold_validation(explorer, binary_threshold):
    with pytest.raises(ValueError, match='binary_threshold'):
        ConceptExplorer(explorer.Z, explorer.texts, explorer.concept_dict,
                        binary_threshold=binary_threshold)


def test_wilson_default_is_99_percent():
    from uce_concept_explorer.explorer import wilson

    low, high = wilson(40, 100)
    # Independently computed Wilson endpoints for z = Phi^-1(0.995).
    z = norm.ppf(.995)
    center = (.4 + z*z / 200) / (1 + z*z / 100)
    half = z * np.sqrt(.4 * .6 / 100 + z*z / 40000) / (1 + z*z / 100)
    np.testing.assert_allclose([low, high], [center - half, center + half])

def test_search_examples(explorer):
    assert frame(explorer.search('poe')).iloc[0].concept_id == 1
    assert frame(explorer.examples(0, n=10)).text_id.tolist() == [0, 1]

def test_fidelity(explorer):
    explorer.allowed = np.array([explorer.concept_dict[i]['f1'] >= .8 for i in range(3)])
    assert explorer.allowed.tolist() == [True, False, True]
    assert len(frame(explorer.search('Poetry'))) == 0
    explorer.allowed[:] = True
    assert len(frame(explorer.search('Poetry'))) == 1


def test_concept_dict_alignment_and_numeric_filter(explorer):
    concept_dict = {2: {'label': 'Food', 'f1': 'not available'},
                0: {'label': 'Politics', 'f1': '.9'},
                1: {'label': 'Poetry', 'f1': None}}
    E = ConceptExplorer(explorer.Z, explorer.texts, concept_dict)
    assert list(E.concept_dict) == [0, 1, 2]
    f1 = np.asarray(pd.to_numeric([E.concept_dict[i].get('f1') for i in range(3)],
                                  errors='coerce'), dtype=float)
    E.allowed = np.isfinite(f1) & (f1 >= .8)
    assert frame(E.search('')).concept_id.tolist() == [0]
    with pytest.raises(ValueError, match='excluded'):
        frame(E.examples(2))
    E.allowed[:] = True
    assert set(frame(E.search('')).concept_id) == {0, 1, 2}


def test_normalize_concept_dictionary_is_copied_and_used_for_labels(explorer):
    concept_dict = {i: dict(record) for i, record in explorer.concept_dict.items()}
    E = ConceptExplorer(explorer.Z, explorer.texts, concept_dict)
    concept_dict[0]['label'] = 'Changed input'
    assert E.concept_dict[0]["label"] == 'Politics'
    E.concept_dict[0]['label'] = 'New label'
    assert E.concept_dict[0]["label"] == 'New label'
    assert frame(E.search('NEW')).concept_id.tolist() == [0]
    assert frame(E.prevalent(min_count=1)).iloc[0].concept_label == 'New label'


@pytest.mark.parametrize('concept_dict, error', [
    ({0: {'label': 'Only one'}}, ValueError),
    ({0: {'f1': .9}}, ValueError),
    ({'0': {'label': 'String key'}}, ValueError),
    ({0: 'Bare label'}, TypeError),
    ({0: {'label': '   '}}, ValueError),
    ({i: {'label': str(i)} for i in range(4)}, ValueError),
    (pd.DataFrame({'concept_id': [0], 'label': ['Table']}), TypeError),
])
def test_normalize_concept_dictionary_validation(explorer, concept_dict, error):
    with pytest.raises(error):
        ConceptExplorer(explorer.Z, explorer.texts, concept_dict)


def test_uce_concept_dict_contract(explorer):
    # UCE stores missing metrics as None and source feature IDs as optional data.
    concepts = {0: {'label': 'Politics', 'f1': .9, 'feature_id': 900},
                1: {'label': 'Poetry', 'f1': None, 'feature_id': 300},
                2: {'label': 'Food'}}
    E = ConceptExplorer(explorer.Z, explorer.texts, concepts)
    assert E.concept_dict == concepts
    assert E.concept_dict[0]["label"] == 'Politics'
    assert E.concept_dict[1]['f1'] is None
    f1 = np.asarray(pd.to_numeric([E.concept_dict[i].get('f1') for i in range(3)],
                                  errors='coerce'), dtype=float)
    E.allowed = np.isfinite(f1) & (f1 >= .8)
    assert frame(E.search('')).concept_id.tolist() == [0]


def test_categories(explorer):
    y = np.array(['a','a','b','b','b','b'])
    out = categories(explorer.enrichment(y, top_n=3, min_count=1))
    assert set(out) == {'a','b'}


@pytest.mark.parametrize('labels', [
    ['a', 'a', 'b', 'b', 'b', 'b'],
    ['a', 'a', 'b', 'b', 'c', 'c'],
])
@pytest.mark.parametrize('confidence_level', [.95, .99])
def test_category_conditions_on_activation(explorer, labels, confidence_level):
    from uce_concept_explorer.explorer import wilson

    y = np.array(labels)
    results = categories(explorer.risk(y, top_n=3, min_count=1,
                                   confidence_level=confidence_level))
    z = norm.ppf((1 + confidence_level) / 2)
    Z = explorer.binary.toarray().astype(bool)
    assert set(results) == set(labels)
    for category, result in results.items():
        for row in result.itertuples():
            active = Z[:, row.concept_id]
            target = y == category
            expected = target[active].mean() - target[~active].mean()
            assert row.effect == pytest.approx(expected)
            la, ua = wilson(target[active].sum(), active.sum(), z=z)
            lb, ub = wilson(target[~active].sum(), (~active).sum(), z=z)
            np.testing.assert_allclose([row.low, row.high], [la - ub, ua - lb])
        scores = np.maximum(np.maximum(result.low, -result.high), 0)
        assert pd.Series(scores).is_monotonic_decreasing
    assert results['a'].set_index('concept_id').loc[1].effect == pytest.approx(-.4)
    assert categories(explorer.enrichment(y, min_count=0))['a'].set_index('concept_id').loc[1].effect == pytest.approx(-.25)


@pytest.mark.parametrize('metric', ['difference', 'ratio'])
def test_risk_filters(metric):
    X = np.array([[0., 1., 1.], [0., 1., 0.], [0., 1., 0.]])
    meta = pd.DataFrame({'concept_id': [0, 1, 2], 'label': ['Never', 'Always', 'Some']}).set_index('concept_id').to_dict(orient='index')
    E = ConceptExplorer(X, ['text'] * 3, meta)
    # Class b has zero probability among active documents, but a finite ratio.
    options = dict(metric=metric)
    with np.errstate(divide='raise', invalid='raise'):
        assert categories(E.risk(['a', 'b', 'b'], min_count=0, **options))['b'].concept_id.tolist() == [2]
    assert categories(E.risk(['a', 'b', 'b'], min_count=2, **options))['b'].empty
    E.allowed[2] = False
    assert categories(E.risk(['a', 'b', 'b'], min_count=0, **options))['b'].empty


@pytest.mark.parametrize('confidence_level', [.95, .99])
def test_category_concept_ratio(confidence_level):
    # Among 60 active documents 40 are a; among 140 inactive, 60 are a.
    X = np.array([1.] * 40 + [0.] * 60 + [1.] * 20 + [0.] * 80)[:, None]
    E = ConceptExplorer(X, ['text'] * 200, pd.DataFrame({'concept_id': [0], 'label': ['Example']}).set_index('concept_id').to_dict(orient='index'))
    result = categories(E.risk(['a'] * 100 + ['b'] * 100,
                           metric='ratio',
                           confidence_level=confidence_level))
    row = result['a'].iloc[0]
    estimate = (40 / 60) / (60 / 140)
    se = np.sqrt((1 - 40 / 60) / 40 + (1 - 60 / 140) / 60)
    assert row.effect == pytest.approx(estimate)
    assert row.risk_at_1 == pytest.approx(40 / 60)
    assert row.risk_at_0 == pytest.approx(60 / 140)
    assert row['activation_count'] == 60
    np.testing.assert_allclose([row.low, row.high], estimate * np.exp(np.array([-1, 1]) * norm.ppf((1 + confidence_level) / 2) * se))


def test_category_concept_zero_ratio(explorer):
    result = categories(explorer.risk(['a', 'a', 'b', 'b', 'b', 'b'],
                                  metric='ratio', min_count=0))
    row = result['a'].set_index('concept_id').loc[2]
    assert row.effect == 0
    assert row.low == 0
    assert np.isposinf(row.high)
    assert 0 not in result['a'].concept_id.values  # Zero denominator is undefined.


@pytest.mark.parametrize('method', ['enrichment', 'risk'])
def test_category_metric_validation(explorer, method):
    compare = getattr(explorer, method)
    labels = ['a', 'a', 'b', 'b', 'b', 'b']
    with pytest.raises(ValueError, match='metric must'):
        compare(labels, metric='invalid')
    with pytest.raises(ValueError, match='at least two'):
        compare(['a'] * 6)
    with pytest.raises(ValueError, match='one value per document'):
        compare(['a', 'b'])


@pytest.mark.parametrize('method', ['enrichment', 'risk'])
@pytest.mark.parametrize('confidence_level', [0, 1, -.1, 1.1, np.nan, np.inf])
def test_category_confidence_validation(explorer, method, confidence_level):
    with pytest.raises(ValueError, match='confidence_level'):
        getattr(explorer, method)(['a', 'a', 'b', 'b', 'b', 'b'],
                                  confidence_level=confidence_level)


def test_shape_validation(explorer):
    with pytest.raises(ValueError, match='texts length'):
        ConceptExplorer(explorer.Z, [], explorer.concept_dict)


@pytest.mark.parametrize('metric', ['difference', 'ratio'])
@pytest.mark.parametrize('confidence_level', [.95, .99])
def test_enrichment_ci(metric, confidence_level):
    # Independent binomial samples: 40/100 versus 20/100.
    X = np.array([1.] * 40 + [0.] * 60 + [1.] * 20 + [0.] * 80)[:, None]
    meta = pd.DataFrame({'concept_id': [0], 'label': ['Example']}).set_index('concept_id').to_dict(orient='index')
    E = ConceptExplorer(X, ['text'] * 200, meta)
    labels = np.array(['a'] * 100 + ['b'] * 100)
    row = categories(E.enrichment(labels, metric=metric,
                        confidence_level=confidence_level))['a'].iloc[0]
    z = norm.ppf((1 + confidence_level) / 2)
    if metric == 'ratio':
        expected = 2 * np.exp(np.array([-1, 1]) * z * np.sqrt(.055))
        assert row.effect == pytest.approx(2)
    else:
        expected = .2 + np.array([-1, 1]) * z * np.sqrt(.004)
        assert row.effect == pytest.approx(.2)
    np.testing.assert_allclose([row.low, row.high], expected)


@pytest.mark.parametrize('metric', ['difference', 'ratio'])
@pytest.mark.parametrize('confidence_level', [.95, .99])
def test_raw_enrichment_mean_intervals(metric, confidence_level):
    values = np.array([.99, .95, .7, 0, .98, .6, .4, 0])
    E = ConceptExplorer(sparse.csr_matrix(values[:, None]), ['text'] * 8,
                        {0: {'label': 'Example'}})
    labels = ['a'] * 4 + ['b'] * 4
    row = categories(E.enrichment(labels, metric=metric, activations='raw', min_count=1,
                        confidence_level=confidence_level))['a'].iloc[0]
    pa, pb = values[:4].mean(), values[4:].mean()
    va, vb = values[:4].var(ddof=1) / 4, values[4:].var(ddof=1) / 4
    z = norm.ppf((1 + confidence_level) / 2)
    if metric == 'difference':
        estimate = pa - pb
        expected = np.clip(estimate + np.array([-1, 1]) * z * np.sqrt(va + vb), -1, 1)
    else:
        estimate = pa / pb
        expected = estimate * np.exp(np.array([-1, 1]) * z * np.sqrt(va/pa**2 + vb/pb**2))
    assert row.category_mean == pytest.approx(pa)
    assert row.reference_mean == pytest.approx(pb)
    assert row.effect == pytest.approx(estimate)
    assert row['activation_count'] == 3
    assert row['category_count'] == 2  # Threshold crossings within category a.
    np.testing.assert_allclose([row.low, row.high], expected)


def test_raw_enrichment_filters_by_binary_count(explorer):
    labels = ['a', 'a', 'b', 'b', 'b', 'b']
    result = categories(explorer.enrichment(labels, activations='raw', min_count=2))
    assert result['a'].concept_id.tolist() == [0, 2]
    assert result['b'].concept_id.tolist() == [2, 0]
    explorer.allowed[2] = False
    assert categories(explorer.enrichment(labels, activations='raw', min_count=2))['b'].concept_id.tolist() == [0]


def test_raw_enrichment_zero_denominator(explorer):
    result = categories(explorer.enrichment(['a', 'a', 'b', 'b', 'b', 'b'],
                                  metric='ratio', activations='raw', min_count=0))
    assert 0 not in result['a'].concept_id.values
    row = result['a'].set_index('concept_id').loc[2]
    assert row.effect == 0
    assert row.low == 0
    assert np.isposinf(row.high)


@pytest.mark.parametrize('score', [np.nan, np.inf, -.1, 1.1])
def test_raw_enrichment_rejects_invalid_scores(score):
    E = ConceptExplorer(np.array([[score], [.3], [.5], [.9]]), ['text'] * 4,
                        {0: {'label': 'Example'}})
    with pytest.raises(ValueError, match='finite activation scores'):
        categories(E.enrichment(['a', 'a', 'b', 'b'], activations='raw', min_count=0))


def test_enrichment_activation_validation(explorer):
    with pytest.raises(ValueError, match='activations must'):
        categories(explorer.enrichment(['a', 'a', 'b', 'b', 'b', 'b'], activations='invalid'))
    with pytest.raises(ValueError, match='at least two documents'):
        categories(explorer.enrichment(['a', 'b', 'b', 'b', 'b', 'b'], activations='raw'))


def test_enrichment_zero_prevalences(explorer):
    result = categories(explorer.enrichment(['a', 'a', 'b', 'b', 'b', 'b'],
                                 metric='ratio', min_count=0, top_n=3))
    # Concept 2 has zero numerator in a, and zero denominator in b.
    row = result['a'].set_index('concept_id').loc[2]
    assert row.effect == 0
    assert row.low == 0
    assert np.isposinf(row.high)
    assert 2 not in result['b'].concept_id.values


def test_enrichment_requires_reference_documents(explorer):
    with pytest.raises(ValueError, match='at least two'):
        categories(explorer.enrichment(['a'] * 6))



def test_enrichment_filters(explorer):
    explorer.allowed = np.array([explorer.concept_dict[i]['f1'] >= .8 for i in range(3)])
    labels = ['a', 'a', 'b', 'b', 'b', 'b']
    result = categories(explorer.enrichment(labels, min_count=2))
    assert result['a'].concept_id.tolist() == [0, 2]
    assert result['b'].concept_id.tolist() == [2, 0]
    assert all(table.empty for table in categories(explorer.enrichment(labels, min_count=3)).values())
