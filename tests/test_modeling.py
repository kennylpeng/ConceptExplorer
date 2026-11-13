from conftest import frame, categories
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from uce_concept_explorer import ConceptExplorer


def make_explorer():
    rng = np.random.default_rng(20)
    n, p = 90, 14
    y = np.r_[np.zeros(n//2, dtype=int), np.ones(n//2, dtype=int)]
    X = (rng.random((n, p)) < .3).astype(float)
    X[:, 0] = (rng.random(n) < (.15 + .65*y)).astype(float)
    X[:, 1] = (rng.random(n) < (.7 - .45*y)).astype(float)
    meta = pd.DataFrame({'concept_id': np.arange(p),
                         'label': [f'Concept {j}' for j in range(p)], 'f1': [.95]*p}).set_index('concept_id').to_dict(orient='index')
    return ConceptExplorer(sparse.csr_matrix(X), [f'text{i}' for i in range(n)], meta), y


@pytest.mark.parametrize('kind', ['categorical', 'continuous'])
@pytest.mark.parametrize('activations', ['binary', 'raw'])
@pytest.mark.filterwarnings('error:.*penalty.*')
def test_joint_support_size_and_eligibility(kind, activations):
    E, y = make_explorer()
    E.concept_dict[2]['f1'] = np.nan
    E.concept_dict[4]['f1'] = .3
    E.allowed = np.array([E.concept_dict[i]['f1'] >= .9 for i in range(len(E.concept_dict))])
    if kind == 'continuous':
        y = 4 * E.binary[:, 0].toarray().ravel() + np.random.default_rng(7).normal(0, .3, len(y))
    table = frame(E.joint(y, kind=kind, activations=activations, n_concepts=3, min_count=3))
    assert len(table) == 3
    assert table.attrs['exact_match']
    assert table.attrs['n_selected'] == 3
    assert table.attrs['activations'] == activations
    assert table.attrs['n_observations'] == len(y)
    assert (table.coefficient != 0).all()
    assert (table['activation_count'] >= 3).all()
    assert E.allowed[table.concept_id.to_numpy()].all()
    assert table.concept_id.is_unique
    assert np.abs(table.coefficient).is_monotonic_decreasing
    if kind == 'categorical':
        assert table.attrs['positive_class'] == 1


def test_joint_raw_retains_signal_lost_by_binarization():
    # All values exceed the binary threshold, but raw activations carry a signal.
    values = np.linspace(.93, 1, 80)[:, None]
    E = ConceptExplorer(values, ['text'] * len(values),
                        pd.DataFrame({'concept_id': [0], 'label': ['Signal']}).set_index('concept_id').to_dict(orient='index'))
    y = 3 * values[:, 0]
    table = frame(E.joint(y, kind='continuous', activations='raw', n_concepts=1))
    assert table.concept_id.tolist() == [0]
    # L1 coefficients remain shrunk; conversion restores the input feature units.
    expected = 3 - table.attrs['penalty']['alpha'] / values.std()
    assert table.coefficient.iloc[0] == pytest.approx(expected)
    with pytest.raises(ValueError, match='variable activations'):
        frame(E.joint(y, kind='continuous', activations='binary', n_concepts=1))


def test_joint_closest_count_without_padding():
    E, y = make_explorer()
    E.allowed[:] = False
    E.allowed[0] = True
    table = frame(E.joint(y, n_concepts=5))
    assert table.concept_id.tolist() == [0]
    assert table.attrs['n_requested'] == 5
    assert table.attrs['n_selected'] == 1
    assert not table.attrs['exact_match']
    assert (table.coefficient != 0).all()


def test_joint_does_not_truncate_simultaneously_entering_features():
    # Equal orthogonal signals enter together, so the path jumps from 0 to 2.
    Z = np.tile([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], (20, 1))
    E = ConceptExplorer(Z, ['text'] * len(Z),
                        pd.DataFrame({'concept_id': [0, 1], 'label': ['A', 'B']}).set_index('concept_id').to_dict(orient='index'))
    table = frame(E.joint(Z.sum(axis=1), kind='continuous', n_concepts=1))
    # Both are one away; the documented tie rule prefers the smaller support.
    assert table.empty
    assert table.attrs['n_selected'] == 0
    assert not table.attrs['exact_match']


@pytest.mark.parametrize('kind', ['categorical', 'continuous'])
def test_joint_drops_invalid_targets(kind):
    E, y = make_explorer()
    y = y.astype(float)
    y[:10] = np.nan
    y[10] = np.inf
    table = frame(E.joint(y, kind=kind, n_concepts=2, min_count=3))
    valid = np.isfinite(y)
    assert table.attrs['n_observations'] == valid.sum()
    counts = np.asarray(E.binary[valid].sum(axis=0)).ravel()
    np.testing.assert_array_equal(table['activation_count'], counts[table.concept_id])


@pytest.mark.parametrize('kwargs, message', [
    ({'kind': 'invalid'}, 'kind must'),
    ({'activations': 'invalid'}, 'activations must'),
    ({'n_concepts': 0}, 'positive integer'),
    ({'n_concepts': 1.5}, 'positive integer'),
    ({'n_concepts': True}, 'positive integer'),
    ({'min_count': -1}, 'min_count must'),
    ({'min_count': 100}, 'No eligible concepts'),
])
def test_joint_validation(kwargs, message):
    E, y = make_explorer()
    with pytest.raises(ValueError, match=message):
        frame(E.joint(y, **kwargs))


def test_joint_requires_variable_targets():
    E, y = make_explorer()
    with pytest.raises(ValueError, match='exactly two classes'):
        frame(E.joint(np.zeros_like(y)))
    with pytest.raises(ValueError, match='must vary'):
        frame(E.joint(np.zeros_like(y), kind='continuous'))
