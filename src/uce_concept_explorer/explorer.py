"""CPU-only analysis of thresholded Universal Concept Encoder activations."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from numbers import Real

import numpy as np
from scipy import sparse
from scipy.stats import norm
from ._utils import integer, targets, result
from .statistics import contrast, ranked_ids, separation, wilson
from .modeling import _select_joint
from .risk import _raw_risk


def _normalize_concept_dict(concept_dict, n_concepts):
    """Validate and copy the concept dictionary supplied by the encoder."""
    if not isinstance(concept_dict, Mapping):
        raise TypeError('concept_dict must be a dictionary keyed by integer concept IDs')
    normalized = {}
    for concept_id, record in concept_dict.items():
        if isinstance(concept_id, bool) or not isinstance(concept_id, (int, np.integer)) or concept_id < 0:
            raise ValueError('concept_dict keys must be nonnegative integer concept IDs')
        if not isinstance(record, Mapping):
            raise TypeError('Each concept_dict entry must be a dictionary containing label')
        if ('label' not in record or not isinstance(record['label'], str)
                or not record['label'].strip()):
            raise ValueError('Each concept_dict entry must contain a nonempty string label')
        normalized[int(concept_id)] = deepcopy(dict(record))
    ids = range(n_concepts)
    if set(normalized) != set(ids):
        raise ValueError('Concept dictionary does not align with matrix columns')
    return {i: normalized[i] for i in ids}


class ConceptExplorer:
    def __init__(self, Z, texts, concept_dict, binary_threshold=0.92):
        """Retain sparse scores and build binary activations using Z > threshold."""
        self._binary_threshold = float(binary_threshold)
        if not 0 <= self._binary_threshold <= 1:
            raise ValueError('binary_threshold must lie between 0 and 1')
        self.Z = sparse.csr_matrix(Z, copy=True)
        self.Z.eliminate_zeros()
        self.binary = (self.Z > self._binary_threshold).astype(np.float32).tocsr()
        self.texts = list(texts)
        if len(self.texts) != self.binary.shape[0]:
            raise ValueError('texts length does not match matrix rows')
        if not self.texts:
            raise ValueError('Z must have at least one document')
        self.counts = np.asarray(self.binary.sum(axis=0)).ravel().astype(int)
        self.concept_dict = _normalize_concept_dict(concept_dict, self.binary.shape[1])
        self.allowed = np.ones(self.binary.shape[1], dtype=bool)

    def _result(self, ids, columns, **metadata):
        """Construct column lists after calculation, filtering, and ordering."""
        ids = np.asarray(ids, dtype=int)
        data = {'concept_id': ids,
                'concept_label': [self.concept_dict[i]['label'] for i in ids]}
        data.update({key: np.asarray(value)[ids] for key, value in columns.items()})
        return result(data, **metadata)

    def search(self, query, max_results=30, all_active=False):
        """Find eligible labels by substring; order by activation count, then label."""
        max_results = integer(max_results, 'max_results')
        if not isinstance(query, str):
            raise ValueError('query must be a string')
        query = query.casefold()
        ids = [i for i in range(self.Z.shape[1]) if self.allowed[i]
               and query in self.concept_dict[i]['label'].casefold()
               and (not all_active or self.counts[i] > 0)]
        ids.sort(key=lambda i: (-self.counts[i], self.concept_dict[i]['label']))
        return self._result(ids[:max_results], {'activation_count': self.counts},
                            n_observations=len(self.texts))

    def examples(self, concept_id, n=10):
        """Retrieve threshold-crossing documents in descending raw-score order."""
        i = integer(concept_id, 'concept_id')
        n = integer(n, 'n')
        if i >= self.Z.shape[1]:
            raise ValueError('concept_id is outside the activation matrix')
        if not self.allowed[i]:
            raise ValueError(f'Concept {i} is excluded by E.allowed')
        idx = self.binary.getcol(i).nonzero()[0]
        strengths = self.Z[idx, i].toarray().ravel()
        idx = idx[np.argsort(-strengths, kind='stable')[:n]]
        return result({'concept_id': [i] * len(idx),
                       'concept_label': [self.concept_dict[i]['label']] * len(idx),
                       'activation': self.Z[idx, i].toarray().ravel(),
                       'text_id': idx, 'text': [self.texts[j] for j in idx]})

    def prevalent(self, top_n=20, min_count=10):
        """Rank eligible concepts by activating-document count."""
        top_n = integer(top_n, 'top_n')
        min_count = integer(min_count, 'min_count')
        ids = ranked_ids(self.counts, self.allowed & (self.counts >= min_count), top_n)
        return self._result(ids, {'activation_count': self.counts,
                                 'prevalence': self.counts / len(self.texts)},
                            n_observations=len(self.texts))

    def _comparison_inputs(self, labels, metric, confidence_level, activations, top_n, min_count):
        integer(top_n, 'top_n')
        integer(min_count, 'min_count')
        if metric not in ('difference', 'ratio'):
            raise ValueError("metric must be 'difference' or 'ratio'")
        if activations not in ('binary', 'raw'):
            raise ValueError("activations must be 'binary' or 'raw'")
        if not isinstance(confidence_level, Real) or not 0 < confidence_level < 1:
            raise ValueError('confidence_level must be between 0 and 1')
        labels, valid = targets(labels, len(self.texts), 'labels')
        if len(np.unique(labels)) < 2:
            raise ValueError('Need at least two target classes')
        binary = self.binary[valid]
        source = self.Z[valid].astype(float) if activations == 'raw' else binary.astype(float)
        if activations == 'raw' and (not np.isfinite(source.data).all()
                                     or np.any((source.data < 0) | (source.data > 1))):
            raise ValueError('Raw comparisons require finite activation scores between 0 and 1')
        counts = np.asarray(binary.sum(axis=0)).ravel().astype(int)
        metadata = dict(metric=metric, activations=activations, confidence_level=float(confidence_level),
                        n_observations=len(labels), n_dropped=int((~valid).sum()))
        return labels, source, binary, counts, norm.ppf((1 + confidence_level) / 2), metadata

    def enrichment(self, labels, metric='difference', top_n=15, min_count=10,
                   confidence_level=0.99, activations='binary'):
        """Compare mean concept activation within each category versus the rest.

        Returns category -> {data: column lists, metadata: analysis details}.
        Binary means are prevalences. Results sort by the lower effect bound.
        min_count counts binary activations among all valid-target documents.
        """
        labels, source, binary, counts, z, metadata = self._comparison_inputs(
            labels, metric, confidence_level, activations, top_n, min_count)
        total = np.asarray(source.sum(axis=0)).ravel()
        squared = np.asarray(source.power(2).sum(axis=0)).ravel() if activations == 'raw' else None
        out = {}
        for group in np.unique(labels):
            mask = labels == group
            n1, n0 = int(mask.sum()), int((~mask).sum())
            a = np.asarray(source[mask].sum(axis=0)).ravel()
            b = total - a
            p1, p0 = a / n1, b / n0
            if activations == 'raw':
                if min(n1, n0) < 2:
                    raise ValueError('Raw enrichment requires at least two documents per group and rest')
                squared1 = np.asarray(source[mask].power(2).sum(axis=0)).ravel()
                variance1 = np.maximum(0, squared1 - a*a/n1) / ((n1 - 1)*n1)
                variance0 = np.maximum(0, squared - squared1 - b*b/n0) / ((n0 - 1)*n0)
            else:
                variance1, variance0 = p1*(1-p1)/n1, p0*(1-p0)/n0
            effect, low, high = contrast(p1, p0, variance1, variance0, metric, z)
            ids = ranked_ids(low, self.allowed & (counts >= min_count) & np.isfinite(effect), top_n)
            out[group] = self._result(ids, {
                'activation_count': counts,
                'category_count': np.asarray(binary[mask].sum(axis=0)).ravel().astype(int),
                'category_mean': p1, 'reference_mean': p0,
                'effect': effect, 'low': low, 'high': high,
            }, **metadata, category_size=n1, reference_size=n0)
        return out

    def risk(self, labels, metric='difference', top_n=15, min_count=10,
             confidence_level=0.99, activations='binary'):
        """Compare category probabilities at concept scores 1 and 0.

        Binary mode uses observed active/inactive samples; raw mode fits a
        separate logistic model per concept. Differences sort by conservative
        absolute separation; ratios sort by the lower effect bound.
        """
        labels, source, binary, counts, z, metadata = self._comparison_inputs(
            labels, metric, confidence_level, activations, top_n, min_count)
        if activations == 'raw':
            return _raw_risk(self, labels, source, counts, metric, top_n, min_count, z, metadata)
        inactive = len(labels) - counts
        eligible = self.allowed & (counts >= min_count) & (counts > 0) & (inactive > 0)
        n1, n0 = np.maximum(counts, 1), np.maximum(inactive, 1)
        out = {}
        for group in np.unique(labels):
            mask = labels == group
            a = np.asarray(binary[mask].sum(axis=0)).ravel()
            b = mask.sum() - a
            p1, p0 = a / n1, b / n0
            if metric == 'difference':
                la, ua = wilson(a, n1, z)
                lb, ub = wilson(b, n0, z)
                effect, low, high = p1 - p0, la - ub, ua - lb
            else:
                effect, low, high = contrast(p1, p0, p1*(1-p1)/n1, p0*(1-p0)/n0, metric, z)
            score = separation(low, high) if metric == 'difference' else low
            ids = ranked_ids(score, eligible & np.isfinite(effect), top_n)
            out[group] = self._result(ids, {
                'activation_count': counts, 'inactive_count': inactive,
                'risk_at_0': p0, 'risk_at_1': p1,
                'effect': effect, 'low': low, 'high': high,
            }, **metadata, category_size=int(mask.sum()), reference_size=int((~mask).sum()))
        return out

    def joint(self, y, kind='categorical', n_concepts=20, min_count=10,
              activations='binary', random_state=42):
        """Select concepts with L1 regression, aiming for n_concepts nonzeros.

        Returns data columns and metadata with the actual selection count,
        penalty, and search details. Fits all valid-target documents.
        """
        return _select_joint(self, y, kind=kind, n_concepts=n_concepts,
                             min_count=min_count, activations=activations,
                             random_state=random_state)
