"""Univariate logistic risk contrasts for raw concept scores."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from ._utils import result


def _raw_risk(explorer, labels, source, counts, metric, top_n, min_count, z, metadata):
    source = source.tocsc()
    if len(labels) < 3:
        raise ValueError('Raw risk requires at least three documents')
    eligible = np.flatnonzero(explorer.allowed & (counts >= min_count))
    columns = ['concept_id', 'concept_label', 'activation_count', 'inactive_count',
               'risk_at_0', 'risk_at_1', 'effect', 'low', 'high', 'coefficient', 'intercept']
    out = {}
    for group in np.unique(labels):
        y = (labels == group).astype(float)
        rows, excluded = [], {}
        for concept_id in eligible:
            x = source[:, concept_id].toarray().ravel()
            scale = x.std()
            if scale == 0:
                excluded[int(concept_id)] = 'constant'
                continue
            negative, positive = x[y == 0], x[y == 1]
            # Complete or quasi separation has no finite unpenalized slope.
            if negative.max() <= positive.min() or positive.max() <= negative.min():
                excluded[int(concept_id)] = 'separated'
                continue
            center = x.mean()
            design = np.column_stack([np.ones(len(x)), (x - center) / scale])
            initial = [np.log(y.mean() / (1 - y.mean())), 0.]
            fit = minimize(
                lambda beta: np.mean(np.logaddexp(0, design @ beta) - y * (design @ beta)),
                initial, jac=lambda beta: design.T @ (expit(design @ beta) - y) / len(y),
                method='BFGS', options={'gtol': 1e-8, 'maxiter': 200},
            )
            if not np.isfinite(fit.x).all() or (not fit.success and np.max(np.abs(fit.jac)) > 1e-6):
                excluded[int(concept_id)] = 'fit_failed'
                continue
            fitted = expit(design @ fit.x)
            information = design.T @ ((fitted * (1 - fitted))[:, None] * design)
            if not np.isfinite(information).all() or np.linalg.cond(information) > 1e12:
                excluded[int(concept_id)] = 'singular_information'
                continue
            covariance = np.linalg.inv(information)
            d0 = np.array([1., -center / scale])
            d1 = np.array([1., (1 - center) / scale])
            eta0, eta1 = d0 @ fit.x, d1 @ fit.x
            p0, p1 = expit(eta0), expit(eta1)
            if metric == 'difference':
                estimate = p1 - p0
                gradient = p1 * (1 - p1) * d1 - p0 * (1 - p0) * d0
                se = np.sqrt(max(0., gradient @ covariance @ gradient))
                low, high = np.clip([estimate - z * se, estimate + z * se], -1, 1)
                rank = max(low, -high, 0.)
            else:
                # Work on the log scale to avoid dividing tiny probabilities.
                log_ratio = -np.logaddexp(0, -eta1) + np.logaddexp(0, -eta0)
                gradient = (1 - p1) * d1 - (1 - p0) * d0
                se = np.sqrt(max(0., gradient @ covariance @ gradient))
                with np.errstate(over='ignore'):
                    estimate = np.exp(log_ratio)
                    low, high = np.exp([log_ratio - z * se, log_ratio + z * se])
                rank = low
            if not np.isfinite(estimate) or not np.isfinite(rank) or np.isnan(high):
                excluded[int(concept_id)] = 'nonfinite_estimate'
                continue
            rows.append((rank, {
                'concept_id': int(concept_id),
                'concept_label': explorer.concept_dict[concept_id]['label'],
                'activation_count': int(counts[concept_id]),
                'inactive_count': int(len(labels) - counts[concept_id]),
                'coefficient': fit.x[1] / scale,
                'intercept': fit.x[0] - fit.x[1] * center / scale,
                'risk_at_0': p0, 'risk_at_1': p1, 'effect': estimate,
                'low': low, 'high': high,
            }))
        rows.sort(key=lambda item: -item[0])
        data = {column: [row[column] for _, row in rows[:top_n]] for column in columns}
        out[group] = result(data, **metadata,
                            category_size=int(y.sum()), reference_size=int(len(y) - y.sum()),
                            excluded_concepts=excluded)
    return out
