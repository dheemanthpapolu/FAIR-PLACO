import numpy as np


def _confusion(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp, tn, fp, fn


def _safe_div(a, b):
    return a / b if b > 0 else 0.0


def positive_rate(y_pred):
    return float(np.mean(np.asarray(y_pred) == 1))


def true_positive_rate(y_true, y_pred):
    tp, _, _, fn = _confusion(y_true, y_pred)
    return _safe_div(tp, tp + fn)


def false_positive_rate(y_true, y_pred):
    _, tn, fp, _ = _confusion(y_true, y_pred)
    return _safe_div(fp, fp + tn)


def demographic_parity_difference(y_pred, A):
    pr1 = positive_rate(np.asarray(y_pred)[np.asarray(A) == 1])
    pr0 = positive_rate(np.asarray(y_pred)[np.asarray(A) == 0])
    return abs(pr1 - pr0)


def equal_opportunity_difference(y_true, y_pred, A):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); A = np.asarray(A)
    tpr1 = true_positive_rate(y_true[A == 1], y_pred[A == 1])
    tpr0 = true_positive_rate(y_true[A == 0], y_pred[A == 0])
    return abs(tpr1 - tpr0)


def equalized_odds_difference(y_true, y_pred, A):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); A = np.asarray(A)
    tpr1 = true_positive_rate(y_true[A == 1], y_pred[A == 1])
    tpr0 = true_positive_rate(y_true[A == 0], y_pred[A == 0])
    fpr1 = false_positive_rate(y_true[A == 1], y_pred[A == 1])
    fpr0 = false_positive_rate(y_true[A == 0], y_pred[A == 0])
    return max(abs(tpr1 - tpr0), abs(fpr1 - fpr0))


def treatment_equality_difference(y_true, y_pred, A, eps=1.0):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); A = np.asarray(A)
    out = []
    for a in (0, 1):
        _, _, fp, fn = _confusion(y_true[A == a], y_pred[A == a])
        out.append(np.log((fn + eps) / (fp + eps)))
    return abs(out[0] - out[1])


def subgroup_fairness(y_true, y_pred, A, X, n_quantiles=2):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); A = np.asarray(A); X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    tprs = []
    for f in range(X.shape[1]):
        x = X[:, f]
        try:
            bins = np.unique(np.quantile(x, np.linspace(0, 1, n_quantiles + 1)))
            if len(bins) < 2:
                continue
            buckets = np.digitize(x, bins[1:-1])
        except Exception:
            continue
        for q in np.unique(buckets):
            for a in (0, 1):
                m = (buckets == q) & (A == a)
                if m.sum() < 10:
                    continue
                tprs.append(true_positive_rate(y_true[m], y_pred[m]))
    if len(tprs) < 2:
        return 0.0
    return float(max(tprs) - min(tprs))


def individual_fairness(X, y_pred, k=5):
    from sklearn.neighbors import NearestNeighbors
    X = np.asarray(X); y_pred = np.asarray(y_pred)
    n = len(y_pred)
    if n < k + 1:
        return 0.0
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)
    cons = 0.0
    for i in range(n):
        cons += float(np.mean(y_pred[idx[i, 1:]] == y_pred[i]))
    return cons / n


def all_metrics(y_true, y_pred, A, X=None):
    from sklearn.metrics import accuracy_score
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "demographic_parity_diff": demographic_parity_difference(y_pred, A),
        "equalized_odds_diff": equalized_odds_difference(y_true, y_pred, A),
        "equal_opportunity_diff": equal_opportunity_difference(y_true, y_pred, A),
        "treatment_equality_diff": treatment_equality_difference(y_true, y_pred, A),
    }
    if X is not None:
        out["subgroup_fairness"] = subgroup_fairness(y_true, y_pred, A, X)
        out["individual_fairness"] = individual_fairness(X, y_pred)
    return out
