"""Joint concept selection by tuning L1 regularization to a requested support size."""
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.preprocessing import StandardScaler
from ._utils import integer, targets, result


def _select_joint(explorer, y, *, kind='categorical', n_concepts=20, min_count=10,
                  activations='binary', random_state=42):
    if kind not in ('categorical', 'continuous'):
        raise ValueError("kind must be 'categorical' or 'continuous'")
    if activations not in ('binary', 'raw'):
        raise ValueError("activations must be 'binary' or 'raw'")
    n_concepts = integer(n_concepts, 'n_concepts', minimum=1)
    min_count = integer(min_count, 'min_count')
    y, valid = targets(y, explorer.Z.shape[0], 'y', continuous=kind == 'continuous')
    if len(y) < 2:
        raise ValueError('Need at least two valid observations')
    classes = None
    if kind == 'categorical':
        classes, target = np.unique(y, return_inverse=True)
        if len(classes) != 2:
            raise ValueError('Categorical joint selection supports exactly two classes')
        target = target.astype(float)
    else:
        target = y
        if np.std(target) == 0:
            raise ValueError('Continuous target must vary')

    # Frequency always means threshold-crossing documents, even for raw fits.
    counts = np.asarray(explorer.binary[valid].sum(axis=0)).ravel()
    ids = np.flatnonzero(explorer.allowed & (counts >= min_count))
    if not len(ids):
        raise ValueError('No eligible concepts satisfy min_count')
    source = explorer.binary if activations == 'binary' else explorer.Z
    X = sparse.csr_matrix(source[valid][:, ids], dtype=float)
    if not np.isfinite(X.data).all():
        raise ValueError('Activations must be finite on valid-target rows')
    scaler = StandardScaler(with_mean=False).fit(X)
    varying = scaler.var_ > 0
    ids = ids[varying]
    if not len(ids):
        raise ValueError('No eligible concepts have variable activations')
    scales = scaler.scale_[varying]
    X = scaler.transform(X)[:, varying].tocsr()
    centered = target - target.mean()
    max_strength = max(float(np.max(np.abs(X.T @ centered))) / len(target), 1e-12)
    candidates = {}
    tolerance = 1e-8
    # New sklearn versions select the penalty through l1_ratio. Older versions
    # require penalty='l1'; detect the constructor default without version pins.
    l1_options = {}
    if kind == 'categorical':
        default_penalty = LogisticRegression().get_params().get('penalty', 'deprecated')
        l1_options = {'l1_ratio': 1.0} if default_penalty == 'deprecated' else {'penalty': 'l1'}

    def evaluate(log_strength):
        strength = float(np.exp(log_strength))
        if kind == 'categorical':
            estimator = LogisticRegression(solver='liblinear', **l1_options,
                                           C=1 / (len(target) * strength),
                                           max_iter=10000, tol=1e-8,
                                           random_state=random_state)
        else:
            estimator = Lasso(alpha=strength, max_iter=10000, tol=1e-8,
                              random_state=random_state)
        estimator.fit(X, target)
        coef = np.asarray(estimator.coef_).ravel().copy()
        count = int(np.count_nonzero(np.abs(coef) > tolerance))
        candidates[log_strength] = (count, coef, strength)
        return count

    # Sample a broad path first, then refine every bracket crossing the requested
    # count. Supports can jump or be nonmonotone; no exact-size guarantee is made.
    grid = np.linspace(np.log(max_strength) + np.log(100),
                       np.log(max_strength) + np.log(1e-8), 25)
    exact = False
    for value in grid:
        if evaluate(float(value)) == n_concepts:
            exact = True
            break
    for _ in range(40):
        if exact:
            break
        values = sorted(candidates)
        brackets = [(a, b) for a, b in zip(values, values[1:])
                    if (candidates[a][0] - n_concepts) * (candidates[b][0] - n_concepts) < 0
                    and b - a > 1e-10]
        if not brackets:
            break
        a, b = max(brackets, key=lambda pair: pair[1] - pair[0])
        exact = evaluate((a + b) / 2) == n_concepts

    # Prefer fewer concepts on equal distance, then stronger regularization.
    chosen = min(candidates, key=lambda value: (
        abs(candidates[value][0] - n_concepts),
        candidates[value][0] > n_concepts, -value))
    count, coef, strength = candidates[chosen]
    selected = np.flatnonzero(np.abs(coef) > tolerance)
    coefficients = coef[selected] / scales[selected]
    order = np.argsort(-np.abs(coefficients), kind='stable')
    selected_ids = ids[selected][order]
    data = {
        'concept_id': selected_ids,
        'concept_label': [explorer.concept_dict[i]['label'] for i in selected_ids],
        'activation_count': counts[selected_ids].astype(int),
        'coefficient': coefficients[order],
    }
    metadata = dict(kind=kind, activations=activations, n_observations=len(y),
                    n_dropped=int((~valid).sum()), n_requested=n_concepts,
                    n_selected=count, exact_match=count == n_concepts,
                    n_eligible_concepts=len(ids), coefficient_tolerance=tolerance,
                    penalty={'C': 1 / (len(target) * strength)} if kind == 'categorical'
                    else {'alpha': strength}, n_fits=len(candidates))
    if classes is not None:
        value = classes[1]
        metadata['positive_class'] = value.item() if isinstance(value, np.generic) else value
    return result(data, **metadata)
