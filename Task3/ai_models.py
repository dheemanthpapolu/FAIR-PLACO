import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


def _hgb(**kw):
    return HistGradientBoostingClassifier(
        max_iter=kw.pop("max_iter", 100),
        max_depth=kw.pop("max_depth", 6),
        learning_rate=kw.pop("learning_rate", 0.1),
        random_state=kw.pop("random_state", 0),
        **kw,
    )


class BasicAI(BaseEstimator, ClassifierMixin):
    def __init__(self, **kw):
        self.kw = kw
        self.clf = None

    def fit(self, X, y, A=None):
        self.clf = _hgb(**self.kw)
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def predict(self, X):
        return self.clf.predict(X)


class FairAI(BaseEstimator, ClassifierMixin):

    CONSTRAINT_NAMES = ("equalized_odds", "equal_opportunity", "demographic_parity")

    def __init__(self, constraint="equalized_odds", eps=0.05, max_iter=20, **kw):
        if constraint not in self.CONSTRAINT_NAMES:
            raise ValueError(f"constraint must be one of {self.CONSTRAINT_NAMES}")
        self.constraint = constraint
        self.eps = eps
        self.max_iter = max_iter
        self.kw = kw
        self.mitigator = None

    def _make_constraint(self):
        from fairlearn.reductions import (
            EqualizedOdds, TruePositiveRateParity, DemographicParity
        )
        return {
            "equalized_odds": EqualizedOdds,
            "equal_opportunity": TruePositiveRateParity,
            "demographic_parity": DemographicParity,
        }[self.constraint](difference_bound=self.eps)

    def fit(self, X, y, A):
        from fairlearn.reductions import ExponentiatedGradient
        base = _hgb(**self.kw)
        self.mitigator = ExponentiatedGradient(
            base, constraints=self._make_constraint(),
            max_iter=self.max_iter, eps=self.eps,
        )
        self.mitigator.fit(X, y, sensitive_features=np.asarray(A))
        return self

    def predict_proba(self, X):
        weights = np.asarray(self.mitigator.weights_, dtype=float)
        weights = weights / max(weights.sum(), 1e-12)
        predictors = self.mitigator.predictors_
        n = X.shape[0]
        p1 = np.zeros(n, dtype=float)
        for w, est in zip(weights, predictors):
            if w <= 0:
                continue
            try:
                pp = est.predict_proba(X)[:, 1]
            except Exception:
                pp = est.predict(X).astype(float)
            p1 += w * pp
        p1 = np.clip(p1, 1e-6, 1 - 1e-6)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_ai_model(kind, constraint=None, **kw):
    if kind == "basic":
        return BasicAI(**kw)
    if kind == "fair":
        return FairAI(constraint=constraint or "equalized_odds", **kw)
    raise ValueError(f"unknown AI kind {kind!r}")
