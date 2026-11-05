"""Feature-budgeted predictive models for binary UCE concept activations."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, r2_score, mean_absolute_error
from sklearn.model_selection import (train_test_split, GridSearchCV, StratifiedKFold, KFold,
                                     GroupShuffleSplit, GroupKFold)
from sklearn.pipeline import Pipeline


class MinCountFilter(BaseEstimator, TransformerMixin):
    """Discard features occurring in fewer than min_count training rows (fit within CV)."""
    def __init__(self, min_count=10):
        self.min_count = min_count

    def fit(self, X, y=None):
        if self.min_count < 1:
            raise ValueError('min_count must be positive')
        self.indices_ = np.flatnonzero(np.asarray((X > 0).sum(axis=0)).ravel() >= self.min_count)
        if not len(self.indices_):
            raise ValueError('No features satisfy min_count in this training fold')
        return self

    def transform(self, X):
        return X[:, self.indices_]


def _fit_budgeted(self, y, *, kind, n_concepts=5, min_count=10, test_size=0.2,
                  scoring=None, random_state=42, groups=None):
    if n_concepts < 1:
        raise ValueError('n_concepts must be >= 1')
    X = self.Z.tocsr().astype(float)[:, np.flatnonzero(self.allowed)]
    allowed_ids = np.flatnonzero(self.allowed)
    if not len(allowed_ids):
        raise ValueError('No concepts pass the fidelity filter')
    y = np.asarray(y)
    if y.ndim != 1 or len(y) != X.shape[0]:
        raise ValueError('y must have one value per document')
    if kind == 'continuous':
        y = y.astype(float)
        valid = np.isfinite(y)
    else:
        valid = ~pd.isna(y)
    X, y = X[valid], y[valid]
    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != len(valid):
            raise ValueError('groups length mismatch')
        groups = groups[valid]
    if len(y) < 12:
        raise ValueError('Need at least 12 observations')
    if kind == 'categorical':
        if len(np.unique(y)) != 2:
            raise ValueError('classifier currently supports exactly two classes')
    elif kind == 'continuous':
        if np.std(y) == 0:
            raise ValueError('Continuous target must vary')
    else:
        raise ValueError("kind must be 'categorical' or 'continuous'")

    idx = np.arange(len(y))
    if groups is None:
        itr, ite = train_test_split(idx, test_size=test_size, random_state=random_state,
                                    stratify=y if kind == 'categorical' else None)
        cv = (StratifiedKFold(3, shuffle=True, random_state=random_state)
              if kind == 'categorical' else KFold(3, shuffle=True, random_state=random_state))
        fit_groups = None
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        itr, ite = next(splitter.split(X, y, groups))
        cv = GroupKFold(n_splits=3)
        fit_groups = groups[itr]
    if kind == 'categorical' and (len(np.unique(y[itr])) < 2 or len(np.unique(y[ite])) < 2):
        raise ValueError('Both classes must appear in train and test; try a different split')

    if kind == 'categorical':
        selector = LogisticRegression(penalty='l1', solver='liblinear', max_iter=4000,
                                      random_state=random_state)
        final = LogisticRegression(max_iter=4000, random_state=random_state)
        params = {'select__estimator__C': [0.1, 1, 10], 'model__C': [0.1, 1, 10]}
        scoring = scoring or 'balanced_accuracy'
    else:
        # Standardize target for Lasso alpha stability, then Ridge refit in original units.
        # Alpha grid is rescaled to target std (Lasso's alpha has units of target).
        selector = Lasso(max_iter=8000, random_state=random_state)
        final = Ridge(solver='lsqr')
        scale = max(float(np.std(y[itr])), 1e-8)
        params = {'select__estimator__alpha': [0.001*scale, 0.01*scale, 0.1*scale],
                  'model__alpha': [0.1, 1, 10]}
        scoring = scoring or 'r2'

    pipe = Pipeline([
        ('filter', MinCountFilter(min_count=min_count)),
        ('select', SelectFromModel(selector, max_features=n_concepts, threshold=-np.inf)),
        ('model', final)
    ])
    search = GridSearchCV(pipe, params, cv=cv, scoring=scoring, n_jobs=1, error_score='raise')
    search.fit(X[itr], y[itr], groups=fit_groups)
    best = search.best_estimator_
    predictions = best.predict(X[ite])
    metrics = {'cv_score': float(search.best_score_), 'best_params': search.best_params_,
               'n_train': len(itr), 'n_test': len(ite), 'scoring': scoring}
    if kind == 'categorical':
        metrics['test_balanced_accuracy'] = float(balanced_accuracy_score(y[ite], predictions))
        probabilities = best.predict_proba(X[ite])[:, 1]
        metrics['test_auc'] = float(roc_auc_score((y[ite] == best.classes_[1]).astype(int), probabilities))
    else:
        metrics['test_r2'] = float(r2_score(y[ite], predictions))
        metrics['test_mae'] = float(mean_absolute_error(y[ite], predictions))

    selected_local = best.named_steps['filter'].indices_[best.named_steps['select'].get_support(indices=True)]
    concept_ids = allowed_ids[selected_local]
    train_counts = np.asarray((X[itr] > 0).sum(axis=0)).ravel()
    coefs = np.asarray(best.named_steps['model'].coef_).ravel()
    table = pd.DataFrame({
        'concept_id': concept_ids,
        'concept_label': [self.name(i) for i in concept_ids],
        'train_count': train_counts[selected_local],
        'coefficient': coefs,
    }).sort_values('coefficient', key=lambda x: x.abs(), ascending=False).reset_index(drop=True)
    if self.fidelity is not None:
        table['fidelity'] = self.fidelity[table.concept_id.to_numpy()]
    metrics['n_selected'] = len(concept_ids)
    metrics['n_eligible_concepts'] = len(allowed_ids)
    # best pipeline expects self.Z restricted to allowed_ids (same ordering as used in fit).
    return best, metrics, table


def classifier(self, y, n_concepts=5, min_count=10, test_size=0.2,
               scoring='balanced_accuracy', random_state=42, groups=None):
    """L1-select up to n_concepts, refit logistic regression, test on held-out rows.

    Optional groups avoids group overlap in train/test and in cross-validation.
    Returned pipeline expects `self.Z[:, self.allowed]` as input.
    """
    return _fit_budgeted(self, y, kind='categorical', n_concepts=n_concepts,
                         min_count=min_count, test_size=test_size, scoring=scoring,
                         random_state=random_state, groups=groups)


def regressor(self, y, n_concepts=5, min_count=10, test_size=0.2,
              scoring='r2', random_state=42, groups=None):
    """Lasso-select up to n_concepts, refit Ridge regression, evaluate held-out R²/MAE.

    Optional groups avoids group overlap in train/test and in cross-validation.
    Returned pipeline expects `self.Z[:, self.allowed]` as input.
    """
    return _fit_budgeted(self, y, kind='continuous', n_concepts=n_concepts,
                         min_count=min_count, test_size=test_size, scoring=scoring,
                         random_state=random_state, groups=groups)
