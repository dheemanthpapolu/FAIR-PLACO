import numpy as np

def _func(p, phi, acc):
    a = float(np.min(np.diag(phi)))
    if acc >= 0.5:
        if (p + 2 * a) <= 1:
            return 1e-9
        if (p + 2 * a) >= 2:
            return 1e9
        return (p + 2 * a - 1) / (2 - (p + 2 * a))
    return 1e-9


def placo_value(x, hcm_list, mpv, y_cap, est, acc_list):
    subset = [i for i, b in enumerate(x) if b == 1]
    if len(subset) == 0:
        return 1.0
    vals = np.array([_func(hcm_list[i][est[i]][y_cap], hcm_list[i], acc_list[i])
                     for i in subset])
    return float(np.prod(vals))


class FairnessState:

    def __init__(self, eps=0.05):
        self.eps = eps
        self.reset()

    def reset(self):
        self.n     = np.zeros(2, dtype=float)
        self.sum_q = np.zeros(2, dtype=float)
        self.P     = np.zeros(2, dtype=float)
        self.N     = np.zeros(2, dtype=float)
        self.TP    = np.zeros(2, dtype=float)
        self.FP    = np.zeros(2, dtype=float)
        self.FN    = np.zeros(2, dtype=float)

    @staticmethod
    def _safe(num, denom):
        return num / denom if denom > 1e-9 else 0.0

    def _dp_violation(self, n, sum_q):
        sr0 = self._safe(sum_q[0], n[0])
        sr1 = self._safe(sum_q[1], n[1])
        return max(0.0, abs(sr0 - sr1) - self.eps) ** 2

    def _eopp_violation(self, P, TP):
        tpr0 = self._safe(TP[0], P[0])
        tpr1 = self._safe(TP[1], P[1])
        return max(0.0, abs(tpr0 - tpr1) - self.eps) ** 2

    def _eo_violation(self, P, TP, N, FP):
        tpr0 = self._safe(TP[0], P[0])
        tpr1 = self._safe(TP[1], P[1])
        fpr0 = self._safe(FP[0], N[0])
        fpr1 = self._safe(FP[1], N[1])
        return (max(0.0, abs(tpr0 - tpr1) - self.eps) ** 2 +
                max(0.0, abs(fpr0 - fpr1) - self.eps) ** 2)

    def _te_violation(self, FN, FP, n):
        n0 = max(n[0], 1.0)
        n1 = max(n[1], 1.0)
        fnr0 = FN[0] / n0
        fnr1 = FN[1] / n1
        fpr0 = FP[0] / n0
        fpr1 = FP[1] / n1
        return (max(0.0, abs(fnr0 - fnr1) - self.eps) ** 2 +
                max(0.0, abs(fpr0 - fpr1) - self.eps) ** 2)

    def violations(self):
        return {
            "dp":   self._dp_violation(self.n, self.sum_q),
            "eopp": self._eopp_violation(self.P, self.TP),
            "eo":   self._eo_violation(self.P, self.TP, self.N, self.FP),
            "te":   self._te_violation(self.FN, self.FP, self.n),
        }

    def delta(self, a, p_y1, q_pred):
        a = int(a)
        n     = self.n.copy()
        sum_q = self.sum_q.copy()
        P     = self.P.copy()
        N     = self.N.copy()
        TP    = self.TP.copy()
        FP    = self.FP.copy()
        FN    = self.FN.copy()

        n[a]     += 1.0
        sum_q[a] += q_pred
        P[a]     += p_y1
        N[a]     += 1.0 - p_y1
        TP[a]    += p_y1 * q_pred
        FP[a]    += (1.0 - p_y1) * q_pred
        FN[a]    += p_y1 * (1.0 - q_pred)

        old = self.violations()
        new = {
            "dp":   self._dp_violation(n, sum_q),
            "eopp": self._eopp_violation(P, TP),
            "eo":   self._eo_violation(P, TP, N, FP),
            "te":   self._te_violation(FN, FP, n),
        }
        return {k: new[k] - old[k] for k in old}

    def update(self, a, p_y1, q_pred):
        a = int(a)
        self.n[a]     += 1.0
        self.sum_q[a] += q_pred
        self.P[a]     += p_y1
        self.N[a]     += 1.0 - p_y1
        self.TP[a]    += p_y1 * q_pred
        self.FP[a]    += (1.0 - p_y1) * q_pred
        self.FN[a]    += p_y1 * (1.0 - q_pred)


def fair_placo_value(x, hcm_list, mpv, y_cap, est, acc_list,
                     fair_state, a_i, p_y1, q_subset, lambdas, beta=0.0,
                     subset_cost=0.0, primary_metric="eopp",
                     penalty_strength=200.0):
    base  = placo_value(x, hcm_list, mpv, y_cap, est, acc_list)
    delta = fair_state.delta(a_i, p_y1, q_subset)

    pen_terms = sum(lambdas.get(k, 0.0) * max(0.0, delta[k]) for k in delta)

    cur = fair_state.violations()
    pen_terms += 0.1 * sum(lambdas.get(k, 0.0) * cur[k] for k in cur)

    fairness_pen = penalty_strength * pen_terms
    log_score = np.log(max(base, 1e-12)) - fairness_pen - beta * subset_cost
    return float(np.exp(np.clip(log_score, -50.0, 50.0)))
