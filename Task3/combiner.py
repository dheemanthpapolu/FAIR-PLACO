import numpy as np
from sklearn.metrics import confusion_matrix


class BinaryMAPCombiner:
    def __init__(self, num_humans, diag_acc=0.75, strength=1.0, num_classes=2):
        self.num_humans = num_humans
        self.diag_acc = diag_acc
        self.strength = strength
        self.n_cls = num_classes
        self.confusion_matrix = None

    @staticmethod
    def _dirichlet(acc, strength, n_cls):
        beta = 0.1
        alpha = beta * (n_cls - 1) * acc / (1.0 - acc)
        alpha *= strength
        beta *= strength
        return alpha + 1, beta + 1

    def fit(self, y_h_train, y_train):
        cms = []
        alpha, beta = self._dirichlet(self.diag_acc, self.strength, self.n_cls)
        prior = np.eye(self.n_cls) * alpha + (np.ones(self.n_cls) - np.eye(self.n_cls)) * beta
        for h in range(self.num_humans):
            obs = confusion_matrix(y_train, y_h_train[:, h], labels=list(range(self.n_cls))).astype(float)
            post = obs + prior
            post = post.T
            denom = np.sum(post, axis=0, keepdims=True) - self.n_cls
            denom = np.where(denom <= 0, 1.0, denom)
            post = (post - 1.0) / denom
            post = np.clip(post, 1e-6, 1.0)
            cms.append(post)
        self.confusion_matrix = cms
        return self

    def combine_proba(self, model_probs, y_h, humans_per_instance):
        n = model_probs.shape[0]
        out = np.empty((n, self.n_cls))
        for i in range(n):
            p = model_probs[i].copy()
            for h in humans_per_instance[i]:
                p = p * self.confusion_matrix[h][y_h[i, h]]
            if not np.any(p > 0):
                p = np.ones(self.n_cls) / self.n_cls
            out[i] = p / p.sum()
        return out

    def combine(self, model_probs, y_h, humans_per_instance):
        return np.argmax(self.combine_proba(model_probs, y_h, humans_per_instance), axis=1)
