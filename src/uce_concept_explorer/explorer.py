"""CPU-only analysis of thresholded Universal Concept Encoder activations."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


def load_metadata(repo_id="klpeng/uce-gemma-80k", revision=None):
    """Load UCE ConceptMetadata without downloading encoder weights.

    Uses the metadata class provided by the UCE package, when available.
    Falls back to fetching only small CSV metadata files via huggingface_hub.
    """
    try:
        from uce_gemma import ConceptMetadata
        if hasattr(ConceptMetadata, "from_pretrained"):
            try:
                return ConceptMetadata.from_pretrained(repo_id, revision=revision)
            except TypeError:
                if revision is None:
                    return ConceptMetadata.from_pretrained(repo_id)
    except ImportError:
        pass
    from huggingface_hub import hf_hub_download
    from pathlib import Path
    if Path(repo_id).is_dir():
        folder = Path(repo_id)
        fetch = lambda file: folder / file
    else:
        fetch = lambda file: hf_hub_download(repo_id=repo_id, filename=file, revision=revision)
    data = pd.read_csv(fetch("labels.csv"))
    for filename in ("prevalences.csv", "fidelities.csv"):
        try:
            extra = pd.read_csv(fetch(filename))
        except (FileNotFoundError, OSError):
            continue
        extra = extra.drop(columns=[c for c in ("label",) if c in extra], errors="ignore")
        data = data.merge(extra, on="concept_id", how="left", validate="one_to_one")
    return data


def _metadata_frame(metadata, n_concepts):
    if isinstance(metadata, pd.DataFrame):
        frame = metadata.copy()
    elif hasattr(metadata, "to_dataframe"):
        frame = metadata.to_dataframe()
    elif hasattr(metadata, "describe"):
        frame = metadata.describe(list(range(n_concepts)))
    else:
        raise TypeError("metadata must be a DataFrame or expose to_dataframe()/describe()")
    frame = pd.DataFrame(frame)
    if "concept_id" not in frame and frame.index.name == "concept_id":
        frame = frame.reset_index()
    if "concept_label" in frame and "label" not in frame:
        frame = frame.rename(columns={"concept_label":"label"})
    if not {"concept_id", "label"}.issubset(frame.columns):
        raise ValueError("Metadata must contain concept_id and label")
    frame["concept_id"] = frame["concept_id"].astype(int)
    if frame.concept_id.duplicated().any():
        raise ValueError("Duplicate concept IDs in metadata")
    frame = frame.set_index("concept_id").reindex(range(n_concepts))
    if frame.label.isna().any():
        raise ValueError("Concept metadata does not align with matrix columns")
    return frame.reset_index().rename(columns={"index":"concept_id"})

from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV, LassoCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score, mean_squared_error


def wilson(k, n, z=1.96):
    """Wilson prevalence bounds; supports arrays of concept counts."""
    p = np.asarray(k, float) / n
    a = z*z / n
    center = (p + a/2) / (1+a)
    half = z*np.sqrt(p*(1-p)/n + a/(4*n)) / (1+a)
    return np.maximum(0, center-half), np.minimum(1, center+half)


class ConceptExplorer:
    def __init__(self, X, texts, metadata, threshold=0.92, min_fidelity=None,
                 fidelity_field="f1", baseline_threshold=None):
        """Explore already-encoded scores without loading the neural model.

        X: SciPy sparse or dense (n_documents, n_concepts) score matrix.
        metadata: ConceptMetadata instance or compatible metadata table/provider.
        """
        self.raw = sparse.csr_matrix(X, dtype=np.float32)
        self.raw.eliminate_zeros()
        self.Z = (self.raw > threshold).astype(np.float32).tocsr()
        self.texts = list(texts)
        self.threshold = float(threshold)
        # Explicit calibration mapping: UCE threshold -> reference-corpus threshold.
        self.baseline_threshold = float(baseline_threshold) if baseline_threshold is not None else self.threshold
        if len(self.texts) != self.Z.shape[0]:
            raise ValueError("texts length does not match matrix rows")
        if not (0 <= threshold <= 1):
            raise ValueError("threshold must lie between 0 and 1")
        if self.Z.shape[0] == 0:
            raise ValueError("X must have at least one document")
        self.metadata = metadata
        self.counts = np.asarray(self.Z.sum(axis=0)).ravel()
        self.table = _metadata_frame(metadata, self.Z.shape[1])
        self._names = self.table.set_index("concept_id")["label"].to_dict()
        self.fidelity = None
        self.fidelity_field = fidelity_field
        self.allowed = np.ones(self.Z.shape[1], dtype=bool)
        self._baseline = None
        if min_fidelity is not None:
            self.set_fidelity_filter(min_fidelity, fidelity_field)

    def set_fidelity_filter(self, min_fidelity=None, field="f1"):
        """Keep concepts with an observed fidelity metric >= min_fidelity.

        The UCE-Gemma fidelity F1 uses a limited, balanced evaluation sample;
        it is NOT a corpus-prevalence-adjusted measure.
        """
        if min_fidelity is None:
            self.allowed[:] = True
            return self
        if field not in self.table:
            raise ValueError(f"No fidelity column {field!r}; available: {list(self.table.columns)}")
        values = pd.to_numeric(self.table[field], errors="coerce").to_numpy()
        self.fidelity, self.fidelity_field = values, field
        self.allowed = np.isfinite(values) & (values >= min_fidelity)
        return self

    def name(self, concept_id):
        i = int(concept_id)
        if not 0 <= i < self.Z.shape[1]:
            raise IndexError(f"Invalid concept ID: {i}")
        return str(self._names[i])

    def _table(self, ids, **columns):
        ids = np.asarray(ids, dtype=int)
        data = {'concept_id': ids, 'concept_label': [self.name(i) for i in ids]}
        if self.fidelity is not None:
            data['fidelity'] = self.fidelity[ids]
        data.update({key: np.asarray(val)[ids] for key,val in columns.items()})
        return pd.DataFrame(data).reset_index(drop=True)

    def search(self, query, max_results=30, all_active=False):
        """Search full concept labels by case-insensitive substring.

        Set all_active=True to limit to concepts present in this dataset.
        """
        hits = self.table["label"].astype(str).str.contains(query, case=False, regex=False)
        eligible = self.allowed.copy()
        if all_active:
            eligible &= self.counts > 0
        ids = np.flatnonzero(hits.to_numpy() & eligible)
        ids = sorted(ids, key=lambda i: (-self.counts[i], self.name(i)))[:max_results]
        return self._table(ids, count=self.counts)

    def examples(self, concept_id, n=10):
        """Top texts activating concept above chosen binary threshold."""
        i = int(concept_id)
        if not self.allowed[i]:
            raise ValueError(f'Concept {i} does not pass the fidelity filter')
        idx = self.Z.getcol(i).nonzero()[0]
        # Within thresholded hits, order by original strength, for inspection only.
        strengths = self.raw[idx, i].toarray().ravel()
        idx = idx[np.argsort(-strengths)[:n]]
        return pd.DataFrame({'concept_id':i, 'concept_label':self.name(i),
                             'activation':self.raw[idx, i].toarray().ravel(),
                             'text_id':idx, 'text':[self.texts[j] for j in idx]})

    def prevalent(self, top_n=20, min_count=1):
        n = self.Z.shape[0]
        lo,hi = wilson(self.counts,n)
        order = np.argsort(-lo)
        order = [i for i in order if self.allowed[i] and self.counts[i] >= min_count][:top_n]
        return self._table(order, count=self.counts, prevalence=self.counts/n, low=lo, high=hi)

    def _corpus_baseline(self):
        """Reference prevalences, using an explicitly calibrated threshold mapping."""
        if self._baseline is None:
            key = "prevalence_gt_" + format(self.baseline_threshold, ".2f").replace(".", "p")
            upper_key = key + "_upper_99"
            if key not in self.table or upper_key not in self.table:
                raise ValueError(
                    f"No reference baseline for baseline_threshold={self.baseline_threshold:g} "
                    f"(UCE activation threshold={self.threshold:g}). "
                    f"Expected {key!r}, {upper_key!r}; available: "
                    f"{[c for c in self.table if 'prevalence_gt_' in c]}. "
                    "Supply baseline_threshold according to the calibrated UCE-to-reference threshold mapping."
                )
            b = pd.to_numeric(self.table[key], errors="coerce").to_numpy()
            upper = pd.to_numeric(self.table[upper_key], errors="coerce").to_numpy()
            self._baseline = (b, upper)
        return self._baseline

    def _rank_enrichment(self, a, n_a, b, n_b=None, multiplicative=False, top_n=20, min_count=5,
                         corpus_upper=None):
        pa = a / n_a
        la,ua = wilson(a,n_a)
        if n_b is not None:
            pb = b / n_b
            lb,ub = wilson(b,n_b)
        else:
            pb, ub = b, corpus_upper
            lb = pb   # upper displayed endpoint conditional on fixed baseline point estimate
        if multiplicative:
            estimate = np.divide(pa,pb, out=np.full_like(pa,np.nan),where=pb>0)
            low = np.divide(la,ub,out=np.zeros_like(pa),where=ub>0)
            high = np.divide(ua,lb,out=np.full_like(pa,np.inf),where=lb>0)
            rank = low.copy()
            rank[~np.isfinite(estimate)] = -np.inf
        else:
            estimate, low, high = pa-pb, la-ub, ua-lb
            rank = low.copy()
        rank[(a < min_count) | (~self.allowed)] = -np.inf
        ids = np.argsort(-rank)[:top_n]
        ids = ids[np.isfinite(rank[ids])]
        return self._table(ids, count=a, prevalence=pa, reference_prevalence=pb,
                           enrichment=estimate, low=low, high=high, rank_bound=rank)

    def enriched(self, top_n=20, multiplicative=False, min_count=5):
        """Full dataset vs UCE general-corpus baseline."""
        b,bu = self._corpus_baseline()
        return self._rank_enrichment(self.counts,self.Z.shape[0],b,None,
                                     multiplicative,top_n,min_count,corpus_upper=bu)

    def by_category(self, labels, top_n=15, multiplicative=False, min_count=5,
                    reference='rest'):
        """One-versus-rest categories; reference='corpus' uses UCE general corpus."""
        labels = np.asarray(labels)
        assert len(labels) == self.Z.shape[0]
        out = {}
        if reference == 'corpus':
            base, upper = self._corpus_baseline()
        elif reference != 'rest':
            raise ValueError("reference must be 'rest' or 'corpus'")
        for group in np.unique(labels):
            mask = labels == group
            a = np.asarray(self.Z[mask].sum(axis=0)).ravel()
            if reference == 'rest':
                b = self.counts - a
                result = self._rank_enrichment(a,mask.sum(),b,(~mask).sum(),
                                               multiplicative,top_n,min_count)
            else:
                result = self._rank_enrichment(a,mask.sum(),base,None,
                                               multiplicative,top_n,min_count,corpus_upper=upper)
            out[group] = result
        return out

    def predictive(self, y, kind='categorical', top_n=20, min_count=5):
        """Categorical: best class-vs-rest absolute separation; continuous: univariate R²."""
        y = np.asarray(y)
        assert len(y) == self.Z.shape[0]
        if kind == 'categorical':
            best = np.full(self.Z.shape[1], -np.inf)
            effect = np.zeros(self.Z.shape[1]); best_class = np.empty(self.Z.shape[1],dtype=object)
            lo_best = np.zeros(self.Z.shape[1]); hi_best = np.zeros(self.Z.shape[1])
            for c in np.unique(y):
                mask = y == c
                a = np.asarray(self.Z[mask].sum(axis=0)).ravel()
                b = self.counts-a
                pa,pb = a/mask.sum(), b/(~mask).sum()
                la,ua=wilson(a,mask.sum()); lb,ub=wilson(b,(~mask).sum())
                diff=pa-pb
                # Conservative lower bound on |difference|, zero if interval intersects zero
                lower_abs=np.maximum(np.maximum(la-ub,lb-ua),0)
                update=lower_abs>best
                best[update]=lower_abs[update]
                effect[update]=diff[update]
                lo_best[update]=(la-ub)[update]
                hi_best[update]=(ua-lb)[update]
                best_class[update]=c
            best[(self.counts<min_count) | (~self.allowed)]=-np.inf
            ids=np.argsort(-best)[:top_n]
            ids=ids[np.isfinite(best[ids])]
            result=self._table(ids, best_class=best_class, prevalence_difference=effect,
                               low=lo_best, high=hi_best, separation_LCB=best)
        elif kind == 'continuous':
            y=y.astype(float)
            valid=np.isfinite(y)
            Y=y[valid]; V=self.Z[valid]
            n=len(Y)
            if n <= 4 or np.std(Y)==0: raise ValueError('Need >4 observations and variable target')
            p=np.asarray(V.mean(axis=0)).ravel()
            std_z=np.sqrt(p*(1-p)); std_y=Y.std()
            covariance=np.asarray(V.T @ (Y-Y.mean())).ravel()/n
            r=np.divide(covariance, std_z*std_y,out=np.zeros_like(p),where=std_z>0)
            r=np.clip(r,-0.999999,0.999999)
            delta=1.96/np.sqrt(n-3)
            rlo=np.tanh(np.arctanh(r)-delta)
            rhi=np.tanh(np.arctanh(r)+delta)
            # If CI contains zero, lower bound on R² is zero.
            lower=np.where((rlo<=0)&(rhi>=0),0,np.minimum(rlo**2,rhi**2))
            upper=np.maximum(rlo**2,rhi**2)
            lower[(np.asarray(V.sum(axis=0)).ravel()<min_count) | (~self.allowed)]=-np.inf
            ids=np.argsort(-lower)[:top_n]
            ids=ids[np.isfinite(lower[ids])]
            result=self._table(ids, count=self.counts, correlation=r, R2=r**2,
                               low=lower, high=upper, R2_LCB=lower)
        else:
            raise ValueError("kind must be 'categorical' or 'continuous'")
        return result

    def lasso(self, y, kind='categorical', top_n=20, test_size=0.25, random_state=42):
        """L1 logistic regression or LassoCV; hyperparameters tuned on training data."""
        y=np.asarray(y)
        if len(y)!=self.Z.shape[0]: raise ValueError('Target length mismatch')
        strat=y if kind=='categorical' else None
        itr,ite=train_test_split(np.arange(len(y)),test_size=test_size,
                                 random_state=random_state,stratify=strat)
        feature_ids = np.flatnonzero(self.allowed)
        if len(feature_ids) == 0:
            raise ValueError('No concepts pass the fidelity filter')
        Xtr,Xte=self.Z[itr][:,feature_ids],self.Z[ite][:,feature_ids]
        if kind=='categorical':
            clf=LogisticRegressionCV(Cs=8,cv=3,penalty='l1',solver='saga',
                max_iter=3000,scoring='balanced_accuracy',random_state=random_state,n_jobs=-1)
            clf.fit(Xtr,y[itr]); pred=clf.predict(Xte)
            metrics={'heldout_accuracy':accuracy_score(y[ite],pred),
                     'heldout_balanced_accuracy':balanced_accuracy_score(y[ite],pred)}
            coef=np.atleast_2d(clf.coef_)
            class_names=list(clf.classes_) if coef.shape[0]>1 else [clf.classes_[1]]
        elif kind=='continuous':
            clf=LassoCV(cv=5,max_iter=10000,random_state=random_state,n_jobs=-1)
            clf.fit(Xtr,y[itr].astype(float)); pred=clf.predict(Xte)
            metrics={'heldout_R2':r2_score(y[ite].astype(float),pred),
                     'heldout_RMSE':np.sqrt(mean_squared_error(y[ite].astype(float),pred))}
            coef=np.atleast_2d(clf.coef_)
            class_names=['target']
        else:
            raise ValueError("kind must be 'categorical' or 'continuous'")
        rows=[]
        for c,coef_row in zip(class_names,coef):
            for j in np.argsort(-np.abs(coef_row)):
                if coef_row[j]==0: break
                original_id = int(feature_ids[j])
                rows.append({'target_class':c,'concept_id':original_id,'concept_label':self.name(original_id),
                             'coefficient':float(coef_row[j])})
        table=pd.DataFrame(rows,columns=['target_class','concept_id','concept_label','coefficient'])
        if not table.empty:
            table=table.assign(abs_coefficient=lambda d:abs(d.coefficient)) \
                       .sort_values('abs_coefficient',ascending=False).drop(columns='abs_coefficient').head(top_n)
        metrics['n_eligible_concepts'] = len(feature_ids)
        return clf, metrics, table.reset_index(drop=True)
# Feature-budgeted supervised models (kept separately from descriptive analysis).
from .modeling import classifier, regressor
ConceptExplorer.classifier = classifier
ConceptExplorer.regressor = regressor
