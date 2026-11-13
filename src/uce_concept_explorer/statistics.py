"""Confidence intervals and ordering, separate from result construction."""
import numpy as np
from scipy.stats import norm

DEFAULT_CI_Z = norm.ppf(.995)


def wilson(k, n, z=DEFAULT_CI_Z):
    """Wilson prevalence bounds (99% by default)."""
    p = np.asarray(k, float) / n
    a = z*z / n
    center = (p + a/2) / (1+a)
    half = z*np.sqrt(p*(1-p)/n + a/(4*n)) / (1+a)
    return np.maximum(0, center-half), np.minimum(1, center+half)


def contrast(p1, p0, variance1, variance0, metric, z):
    """Difference or ratio of independent means, with delta-method bounds."""
    p1, p0 = np.asarray(p1, float), np.asarray(p0, float)
    if metric == 'difference':
        effect = p1 - p0
        half = z * np.sqrt(variance1 + variance0)
        return effect, np.maximum(-1, effect - half), np.minimum(1, effect + half)
    effect = np.divide(p1, p0, out=np.full_like(p1, np.nan), where=p0 > 0)
    positive = (p1 > 0) & (p0 > 0)
    low, high = np.zeros_like(p1), np.full_like(p1, np.inf)
    se = np.sqrt(variance1[positive] / p1[positive]**2 + variance0[positive] / p0[positive]**2)
    with np.errstate(over='ignore'):
        low[positive] = np.exp(np.log(effect[positive]) - z * se)
        high[positive] = np.exp(np.log(effect[positive]) + z * se)
    return effect, low, high


def separation(low, high):
    """Conservative lower bound on the absolute difference."""
    return np.maximum(np.maximum(low, -np.asarray(high)), 0)


def ranked_ids(score, eligible, top_n):
    """Select finite scores in descending order; break ties by input position."""
    score = np.where(eligible, score, -np.inf)
    ids = np.argsort(-score, kind='stable')
    return ids[np.isfinite(score[ids])][:top_n]
