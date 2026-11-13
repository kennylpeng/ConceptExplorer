"""Shared validation and plain-dictionary results."""
from numbers import Real

import numpy as np
import pandas as pd


def integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        qualifier = 'positive' if minimum == 1 else 'nonnegative'
        raise ValueError(f'{name} must be a {qualifier} integer')
    return int(value)


def targets(values, n_documents, name, continuous=False):
    """Omit missing/nonfinite targets, preserving matrix-row alignment."""
    values = np.asarray(values, dtype=object)
    if values.ndim != 1 or len(values) != n_documents:
        raise ValueError(f'{name} must have one value per document')
    valid = ~pd.isna(values)
    if continuous:
        numeric = np.full(len(values), np.nan)
        numeric[valid] = np.asarray(values[valid], dtype=float)
        valid &= np.isfinite(numeric)
        return numeric[valid], valid
    for i in np.flatnonzero(valid):
        value = values[i]
        if isinstance(value, Real):
            valid[i] = np.isfinite(value)
        elif not isinstance(value, str):
            raise ValueError(f'{name} must contain strings or numbers')
    values = values[valid]
    try:
        np.unique(values)
    except TypeError as exc:
        raise ValueError(f'{name} must use comparable categories; use all strings or all numbers') from exc
    return values, valid


def result(data, **metadata):
    """Keep equal-length data columns separate from analysis metadata."""
    return {'data': {key: np.asarray(value).tolist() for key, value in data.items()},
            'metadata': metadata}
