import numpy as np
import pulp
from typing import Dict

from value_function import (
    placo_value,
    fair_placo_value,
    FairnessState,
)
from cost_function import subset_cost
from dependencies import (
    LAMBDA_INIT,
    LAMBDA_LR,
    LAMBDA_MAX,
    EPSILON_FAIRNESS,
    BETA_COST,
)


def _estimate_q(combiner, mpv, y_h_row, subset):
    p = mpv.copy()
    for h in subset:
        p = p * combiner.confusion_matrix[h][y_h_row[h]]
    s = p.sum()
    if s <= 0:
        return 0.5
    return float(p[1] / s)


def placo_lp_budget(combiner, model_probs, y_h, num_humans,
                    h_costs, est_true_labels, est_human_labels,
                    acc_list, budget):
    n = len(model_probs)
    per_instance_budget = budget / n
    chosen = []
    costs = []
    hcm_list = combiner.confusion_matrix

    for i in range(n):
        mpv = model_probs[i]
        y_cap = est_true_labels[i]
        est = est_human_labels[i]

        h_vals = [
            placo_value(
                [1 if k == h else 0 for k in range(num_humans)],
                hcm_list, mpv, y_cap, est, acc_list,
            )
            for h in range(num_humans)
        ]

        prob = pulp.LpProblem(f"PLACO_LP_{i}", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("x", range(num_humans), cat=pulp.LpBinary)
        prob += pulp.lpSum(h_vals[h] * x[h] for h in range(num_humans))
        prob += (
            pulp.lpSum(h_costs[i, h] * x[h] for h in range(num_humans))
            <= per_instance_budget
        )
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        subset = [h for h in range(num_humans)
                  if x[h].varValue is not None and x[h].varValue > 0.5]
        cur_cost = sum(float(h_costs[i, h]) for h in subset) if subset else 0.0
        chosen.append(subset)
        costs.append(cur_cost)

    return chosen, costs


def fair_placo_lp(combiner, model_probs, y_h, num_humans,
                  h_costs, est_true_labels, est_human_labels,
                  acc_list, budget, A_test,
                  lambda_init=LAMBDA_INIT,
                  lambda_lr=LAMBDA_LR,
                  lambda_max=LAMBDA_MAX,
                  eps=EPSILON_FAIRNESS,
                  beta=BETA_COST,
                  batch_size=64):
    n = len(model_probs)
    per_instance_budget = budget / n
    hcm_list = combiner.confusion_matrix
    chosen = []
    costs = []
    fair_state = FairnessState(eps=eps)
    lambdas: Dict[str, float] = {
        "dp":   lambda_init,
        "eopp": lambda_init,
        "eo":   lambda_init,
        "te":   lambda_init,
    }
    history = {"lambdas": [], "violations": []}

    for i in range(n):
        mpv = model_probs[i]
        y_cap = est_true_labels[i]
        est = est_human_labels[i]
        a_i = int(A_test[i])
        p_y1 = float(mpv[1])

        h_vals = []
        for h in range(num_humans):
            ind = [1 if k == h else 0 for k in range(num_humans)]
            q = _estimate_q(combiner, mpv, y_h[i], [h])
            v = fair_placo_value(
                ind, hcm_list, mpv, y_cap, est, acc_list,
                fair_state, a_i, p_y1, q_subset=q,
                lambdas=lambdas, beta=beta,
                subset_cost=float(h_costs[i, h]),
            )
            h_vals.append(v)

        prob = pulp.LpProblem(f"FAIR_LP_{i}", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("x", range(num_humans), cat=pulp.LpBinary)
        prob += pulp.lpSum(h_vals[h] * x[h] for h in range(num_humans))
        prob += (
            pulp.lpSum(h_costs[i, h] * x[h] for h in range(num_humans))
            <= per_instance_budget
        )
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        subset = [h for h in range(num_humans)
                  if x[h].varValue is not None and x[h].varValue > 0.5]
        cur_cost = sum(float(h_costs[i, h]) for h in subset) if subset else 0.0

        q_final = (_estimate_q(combiner, mpv, y_h[i], subset)
                   if subset else float(mpv[1]))
        fair_state.update(a_i, p_y1, q_final)

        chosen.append(subset)
        costs.append(cur_cost)

        if (i + 1) % batch_size == 0:
            v = fair_state.violations()
            for k in lambdas:
                lambdas[k] = float(min(lambda_max,
                                       max(0.0, lambdas[k] + lambda_lr * (v[k] - 0.0))))
            history["lambdas"].append(dict(lambdas))
            history["violations"].append(dict(v))

    return chosen, costs, history


def placo_greedy_budget(combiner, model_probs, y_h, num_humans,
                        h_costs, est_true_labels, est_human_labels,
                        acc_list, budget):
    n = len(model_probs)
    chosen = []
    costs = []
    spent = 0.0
    hcm_list = combiner.confusion_matrix
    for i in range(n):
        mpv = model_probs[i]
        y_cap = est_true_labels[i]
        est = est_human_labels[i]
        candidates = []
        for h in range(num_humans):
            ind = [0] * num_humans
            ind[h] = 1
            v = placo_value(ind, hcm_list, mpv, y_cap, est, acc_list)
            candidates.append((v, h))
        candidates.sort(reverse=True)
        subset = []
        cur_cost = 0.0
        for v, h in candidates:
            if v > 1.0:
                if spent + cur_cost + h_costs[i, h] <= budget:
                    subset.append(h)
                    cur_cost += h_costs[i, h]
        if not subset and candidates:
            v, h = candidates[0]
            if spent + h_costs[i, h] <= budget:
                subset.append(h)
                cur_cost += h_costs[i, h]
        chosen.append(subset)
        costs.append(cur_cost)
        spent += cur_cost
    return chosen, costs


def fair_placo_greedy(combiner, model_probs, y_h, num_humans,
                      h_costs, est_true_labels, est_human_labels,
                      acc_list, budget, A_test,
                      lambda_init=LAMBDA_INIT,
                      lambda_lr=LAMBDA_LR,
                      lambda_max=LAMBDA_MAX,
                      eps=EPSILON_FAIRNESS,
                      beta=BETA_COST,
                      batch_size=64):
    n = len(model_probs)
    hcm_list = combiner.confusion_matrix
    chosen = []
    costs = []
    spent = 0.0
    fair_state = FairnessState(eps=eps)
    lambdas: Dict[str, float] = {
        "dp":   lambda_init,
        "eopp": lambda_init,
        "eo":   lambda_init,
        "te":   lambda_init,
    }
    history = {"lambdas": [], "violations": []}

    for i in range(n):
        mpv = model_probs[i]
        y_cap = est_true_labels[i]
        est = est_human_labels[i]
        a_i = int(A_test[i])
        p_y1 = float(mpv[1])
        best_subset = []
        best_score = fair_placo_value(
            [0] * num_humans, hcm_list, mpv, y_cap, est, acc_list,
            fair_state, a_i, p_y1,
            q_subset=float(mpv[1]),
            lambdas=lambdas, beta=beta, subset_cost=0.0,
        )
        improved = True
        while improved:
            improved = False
            best_h = None
            for h in range(num_humans):
                if h in best_subset:
                    continue
                trial = best_subset + [h]
                trial_cost = sum(h_costs[i, hh] for hh in trial)
                if spent + trial_cost > budget:
                    continue
                ind = [1 if k in trial else 0 for k in range(num_humans)]
                q = _estimate_q(combiner, mpv, y_h[i], trial)
                score = fair_placo_value(
                    ind, hcm_list, mpv, y_cap, est, acc_list,
                    fair_state, a_i, p_y1, q_subset=q,
                    lambdas=lambdas, beta=beta, subset_cost=trial_cost,
                )
                if score > best_score:
                    best_score = score
                    best_h = h
            if best_h is not None:
                best_subset.append(best_h)
                improved = True

        cur_cost = sum(h_costs[i, h] for h in best_subset)
        q_final = _estimate_q(combiner, mpv, y_h[i], best_subset)
        fair_state.update(a_i, p_y1, q_final)

        chosen.append(best_subset)
        costs.append(float(cur_cost))
        spent += cur_cost

        if (i + 1) % batch_size == 0:
            v = fair_state.violations()
            for k in lambdas:
                lambdas[k] = float(min(lambda_max,
                                       max(0.0, lambdas[k] + lambda_lr * (v[k] - 0.0))))
            history["lambdas"].append(dict(lambdas))
            history["violations"].append(dict(v))

    return chosen, costs, history


def estimate_run_data(combiner, model_probs, num_humans, acc_list):
    from estimation_methods import posterior_estimation
    n = len(model_probs)
    hcm_list = combiner.confusion_matrix
    n_cls = model_probs.shape[1]
    est_human_labels = []
    est_true_labels = []
    for mpv in model_probs:
        _, est = posterior_estimation(hcm_list, mpv)
        est_human_labels.append(est)
    for i, mpv in enumerate(model_probs):
        best_v = -1.0
        best_y = 0
        for j in range(n_cls):
            v = placo_value([1] * num_humans, hcm_list, mpv, j,
                            est_human_labels[i], acc_list)
            if v > best_v:
                best_v = v
                best_y = j
        est_true_labels.append(best_y)
    return est_true_labels, est_human_labels
