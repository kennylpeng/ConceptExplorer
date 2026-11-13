from conftest import frame, categories
import numpy as np
import pytest
from scipy import sparse
from scipy.stats import norm
from uce_concept_explorer import ConceptExplorer


def logistic_sample():
    x = np.repeat([0., .5, 1.], 100)
    y = np.concatenate([np.r_[np.ones(k), np.zeros(100-k)] for k in [20, 50, 80]])
    return x, y


@pytest.mark.parametrize('metric', ['difference', 'ratio'])
@pytest.mark.parametrize('confidence_level', [.95, .99])
def test_raw_risk_logistic_contrasts_and_bounds(metric, confidence_level):
    x, y = logistic_sample()
    E = ConceptExplorer(sparse.csr_matrix(x[:, None]), ['text'] * len(y),
                        {0: {'label': 'Example'}})
    result = categories(E.risk(y, metric=metric, activations='raw', confidence_level=confidence_level))
    row = result[1].iloc[0]
    assert row.coefficient == pytest.approx(np.log(16), abs=1e-5)
    assert row.intercept == pytest.approx(-np.log(4), abs=1e-5)
    assert row.risk_at_0 == pytest.approx(.2, abs=1e-6)
    assert row.risk_at_1 == pytest.approx(.8, abs=1e-6)
    assert row['activation_count'] == 100
    design = np.column_stack([np.ones(len(x)), x])
    probabilities = np.repeat([.2, .5, .8], 100)
    covariance = np.linalg.inv(design.T @ ((probabilities*(1-probabilities))[:, None] * design))
    z = norm.ppf((1+confidence_level)/2)
    if metric == 'difference':
        estimate = .6
        gradient = np.array([0., .16])
        se = np.sqrt(gradient @ covariance @ gradient)
        expected = estimate + np.array([-1, 1]) * z * se
        assert row.effect == pytest.approx(estimate, abs=1e-6)
    else:
        estimate = 4.
        gradient = np.array([-.6, .2])
        se = np.sqrt(gradient @ covariance @ gradient)
        expected = estimate * np.exp(np.array([-1, 1]) * z * se)
        assert row.effect == pytest.approx(estimate, rel=1e-6)
    np.testing.assert_allclose([row.low, row.high], expected, rtol=1e-5)
    assert not result[1].attrs['excluded_concepts']


def test_raw_risk_negative_association_and_filtering():
    x, y = logistic_sample()
    E = ConceptExplorer(np.column_stack([x, 1-x]), ['text'] * len(y),
                        {0: {'label': 'Positive'}, 1: {'label': 'Negative'}})
    table = categories(E.risk(y, activations='raw'))[1].set_index('concept_id')
    assert table.loc[0].effect == pytest.approx(.6, abs=1e-6)
    assert table.loc[1].effect == pytest.approx(-.6, abs=1e-6)
    E.allowed[0] = False
    assert categories(E.risk(y, activations='raw'))[1].concept_id.tolist() == [1]
    assert categories(E.risk(y, activations='raw', min_count=101))[1].empty


def test_raw_risk_can_use_variable_scores_when_binary_is_constant():
    x, y = logistic_sample()
    E = ConceptExplorer((.93 + .07*x)[:, None], ['text'] * len(y), {0: {'label': 'Signal'}})
    assert categories(E.risk(y))[1].empty
    table = categories(E.risk(y, activations='raw'))[1]
    assert table.concept_id.tolist() == [0]
    assert table.iloc[0].coefficient == pytest.approx(np.log(16)/.07, rel=1e-5)


def test_raw_risk_excludes_unidentifiable_concepts():
    X = np.array([[.5, 0, 0], [.5, .2, 1], [.5, .8, 0], [.5, 1, 1]])
    E = ConceptExplorer(X, ['text'] * 4, {i: {'label': str(i)} for i in range(3)})
    table = categories(E.risk(['a', 'a', 'b', 'b'], activations='raw', min_count=0))['a']
    assert table.concept_id.tolist() == [2]
    assert table.attrs['excluded_concepts'] == {0: 'constant', 1: 'separated'}
    assert table.iloc[0].effect == pytest.approx(0, abs=1e-8)


@pytest.mark.parametrize('score', [np.nan, np.inf, -.1, 1.1])
def test_raw_risk_invalid_scores(score):
    E = ConceptExplorer(np.array([[score], [.3], [.5], [.9]]), ['text'] * 4,
                        {0: {'label': 'Example'}})
    with pytest.raises(ValueError, match='finite activation scores'):
        categories(E.risk(['a', 'a', 'b', 'b'], activations='raw', min_count=0))


def test_raw_risk_input_validation():
    E = ConceptExplorer(np.array([[0], [.4], [.6], [1.]]), ['text'] * 4,
                        {0: {'label': 'Example'}})
    with pytest.raises(ValueError, match='activations must'):
        categories(E.risk(['a', 'a', 'b', 'b'], activations='invalid'))
    output = E.risk([0, 0, 1, np.nan], activations='raw')
    assert output[0]['metadata']['n_observations'] == 3
    assert output[0]['metadata']['n_dropped'] == 1
    E = ConceptExplorer(np.array([[0], [1.]]), ['a', 'b'], {0: {'label': 'Example'}})
    with pytest.raises(ValueError, match='at least three'):
        categories(E.risk(['a', 'b'], activations='raw'))
