import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS, ".."))
PLACO_DATASET_DIR = os.path.join(ROOT, "PLACO", "dataset")
sys.path.insert(0, THIS)

from cost_function import make_costs
from dependencies import (
    BETA_COST,
    BUDGET_FRACTION,
    CSV_DIR,
    EPSILON_FAIRNESS,
    HUMAN_ACCURACIES,
    LAMBDA_INIT,
    LAMBDA_LR,
    LAMBDA_MAX,
    NUM_HUMANS,
    RNG_SEED,
)
from estimation_methods import posterior_estimation
from policy import placo_lp_budget
import pulp
from value_function import placo_value


SETTINGS = [
    ("a_placo_basicAI_unbiased",          dict(ai="basic", policy="placo",      humans="unbiased")),
    ("b_placo_basicAI_explicit",          dict(ai="basic", policy="placo",      humans="explicit_biased")),
    ("c_placo_basicAI_automation",        dict(ai="basic", policy="placo",      humans="automation_biased")),
    ("d_placo_fairAI_unbiased",           dict(ai="fair",  policy="placo",      humans="unbiased")),
    ("e_placo_fairAI_explicit",           dict(ai="fair",  policy="placo",      humans="explicit_biased")),
    ("f_placo_fairAI_automation",         dict(ai="fair",  policy="placo",      humans="automation_biased")),
    ("g_FAIRPLACO_fairAI_unbiased",       dict(ai="fair",  policy="fair_placo", humans="unbiased")),
    ("h_FAIRPLACO_fairAI_explicit",       dict(ai="fair",  policy="fair_placo", humans="explicit_biased")),
    ("i_FAIRPLACO_fairAI_automation",     dict(ai="fair",  policy="fair_placo", humans="automation_biased")),
    ("j_FAIRPLACO_basicAI_unbiased",      dict(ai="basic", policy="fair_placo", humans="unbiased")),
    ("k_FAIRPLACO_basicAI_explicit",      dict(ai="basic", policy="fair_placo", humans="explicit_biased")),
    ("l_FAIRPLACO_basicAI_automation",    dict(ai="basic", policy="fair_placo", humans="automation_biased")),
]

DATASET_SPECS = {
    "cifar10h": {"model_name": "cnn_data"},
    "imagenet": {"model_name": "imagenet_data"},
}


def load_image_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model_name = DATASET_SPECS[name]["model_name"]
    path = os.path.join(PLACO_DATASET_DIR, name, f"{model_name}.csv")
    data = np.genfromtxt(path, delimiter=",")

    if model_name == "cnn_data":
        y_true = data[:, 0].astype(int)
        human_counts = data[:, 1:11]
        model_probs = data[:, 11:]
    else:
        y_true = data[:, 164].astype(int)
        human_counts = data[:, 165:181]
        model_probs = data[:, 148:164]
    return human_counts, model_probs, y_true


def build_groups(y_true: np.ndarray, n_classes: int, groups_path: str = None) -> np.ndarray:
    if groups_path:
        arr = np.genfromtxt(groups_path, delimiter=",").astype(int).reshape(-1)
        if len(arr) != len(y_true):
            raise ValueError(f"group file length {len(arr)} != dataset length {len(y_true)}")
        return arr
    return (y_true < (n_classes // 2)).astype(int)


def maybe_subsample(
    human_counts: np.ndarray,
    model_probs: np.ndarray,
    y_true: np.ndarray,
    groups: np.ndarray,
    max_samples: int,
    seed: int,
):
    if max_samples is None or max_samples == 0 or len(y_true) <= max_samples:
        return human_counts, model_probs, y_true, groups
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_true), size=max_samples, replace=False)
    return human_counts[idx], model_probs[idx], y_true[idx], groups[idx]


def sample_wrong_label(rng, probs_row, true_label):
    prob = probs_row.copy().astype(float)
    prob[true_label] = 0.0
    if prob.sum() <= 0:
        prob = np.ones_like(prob, dtype=float)
        prob[true_label] = 0.0
    prob = prob / prob.sum()
    return int(rng.choice(len(prob), p=prob))


def simulate_unbiased_multiclass(human_counts, y_true, accuracies=HUMAN_ACCURACIES, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    H = np.zeros((len(y_true), len(accuracies)), dtype=int)
    for j, acc in enumerate(accuracies):
        for i in range(len(y_true)):
            if rng.random() < acc:
                H[i, j] = int(y_true[i])
            else:
                H[i, j] = sample_wrong_label(rng, human_counts[i], int(y_true[i]))
    return H


def simulate_explicit_biased_multiclass(
    human_counts,
    y_true,
    A,
    ai_probs,
    accuracies=HUMAN_ACCURACIES,
    minority_penalty=0.20,
    margin_threshold=0.15,
    seed=RNG_SEED + 1,
):
    rng = np.random.default_rng(seed)
    H = np.zeros((len(y_true), len(accuracies)), dtype=int)
    sorted_probs = np.sort(ai_probs, axis=1)
    uncertain = (sorted_probs[:, -1] - sorted_probs[:, -2]) < margin_threshold

    for j, acc in enumerate(accuracies):
        for i in range(len(y_true)):
            if A[i] == 0 and uncertain[i]:
                if j % 2 == 0:
                    H[i, j] = sample_wrong_label(rng, ai_probs[i], int(y_true[i]))
                else:
                    eff_acc = max(0.0, acc - minority_penalty)
                    if rng.random() < eff_acc:
                        H[i, j] = int(y_true[i])
                    else:
                        H[i, j] = sample_wrong_label(rng, human_counts[i], int(y_true[i]))
            else:
                if rng.random() < acc:
                    H[i, j] = int(y_true[i])
                else:
                    H[i, j] = sample_wrong_label(rng, human_counts[i], int(y_true[i]))
    return H


def simulate_automation_biased_multiclass(
    human_counts,
    y_true,
    ai_probs,
    accuracies=HUMAN_ACCURACIES,
    trust=0.85,
    seed=RNG_SEED + 2,
):
    rng = np.random.default_rng(seed)
    H = np.zeros((len(y_true), len(accuracies)), dtype=int)
    ai_label = np.argmax(ai_probs, axis=1)
    for j, acc in enumerate(accuracies):
        for i in range(len(y_true)):
            if rng.random() < trust:
                H[i, j] = int(ai_label[i])
            else:
                if rng.random() < acc:
                    H[i, j] = int(y_true[i])
                else:
                    H[i, j] = sample_wrong_label(rng, human_counts[i], int(y_true[i]))
    return H


def simulate_humans(kind, human_counts, y_true, A, ai_probs):
    if kind == "unbiased":
        return simulate_unbiased_multiclass(human_counts, y_true)
    if kind == "explicit_biased":
        return simulate_explicit_biased_multiclass(human_counts, y_true, A, ai_probs)
    if kind == "automation_biased":
        return simulate_automation_biased_multiclass(human_counts, y_true, ai_probs)
    raise KeyError(kind)


class MultiClassMAPCombiner:
    def __init__(self, num_humans, num_classes, diag_acc=0.75, strength=1.0):
        self.num_humans = num_humans
        self.num_classes = num_classes
        self.diag_acc = diag_acc
        self.strength = strength
        self.confusion_matrix = None

    @staticmethod
    def _dirichlet(acc, strength, n_cls):
        beta = 0.1
        alpha = beta * (n_cls - 1) * acc / (1.0 - acc)
        alpha *= strength
        beta *= strength
        return alpha + 1, beta + 1

    def fit(self, y_h_train, y_train):
        from sklearn.metrics import confusion_matrix

        alpha, beta = self._dirichlet(self.diag_acc, self.strength, self.num_classes)
        prior = np.eye(self.num_classes) * alpha + (np.ones((self.num_classes, self.num_classes)) - np.eye(self.num_classes)) * beta
        cms = []
        for h in range(self.num_humans):
            obs = confusion_matrix(y_train, y_h_train[:, h], labels=list(range(self.num_classes))).astype(float)
            post = obs + prior
            post = post.T
            denom = np.sum(post, axis=0, keepdims=True) - self.num_classes
            denom = np.where(denom <= 0, 1.0, denom)
            post = (post - 1.0) / denom
            post = np.clip(post, 1e-6, 1.0)
            cms.append(post)
        self.confusion_matrix = cms
        return self

    def combine_proba(self, model_probs, y_h, humans_per_instance):
        n = model_probs.shape[0]
        out = np.empty_like(model_probs, dtype=float)
        for i in range(n):
            p = model_probs[i].copy()
            for h in humans_per_instance[i]:
                p = p * self.confusion_matrix[h][y_h[i, h]]
            if p.sum() <= 0:
                p = np.ones(self.num_classes, dtype=float) / self.num_classes
            out[i] = p / p.sum()
        return out

    def combine(self, model_probs, y_h, humans_per_instance):
        return np.argmax(self.combine_proba(model_probs, y_h, humans_per_instance), axis=1)


def estimate_run_data(combiner, model_probs, acc_list):
    est_human_labels = []
    est_true_labels = []
    num_humans = len(acc_list)
    for mpv in model_probs:
        _, est = posterior_estimation(combiner.confusion_matrix, mpv)
        est_human_labels.append(est)
    for i, mpv in enumerate(model_probs):
        best_v = -1.0
        best_y = 0
        for j in range(model_probs.shape[1]):
            v = placo_value([1] * num_humans, combiner.confusion_matrix, mpv, j, est_human_labels[i], acc_list)
            if v > best_v:
                best_v = v
                best_y = j
        est_true_labels.append(best_y)
    return est_true_labels, est_human_labels


class ImageFairnessState:
    def __init__(self, eps=EPSILON_FAIRNESS):
        self.eps = eps
        self.counts = np.zeros(2, dtype=float)
        self.correct_mass = np.zeros(2, dtype=float)

    @staticmethod
    def _safe(num, den):
        return num / den if den > 1e-9 else 0.0

    def violation(self, counts=None, correct_mass=None):
        counts = self.counts if counts is None else counts
        correct_mass = self.correct_mass if correct_mass is None else correct_mass
        acc0 = self._safe(correct_mass[0], counts[0])
        acc1 = self._safe(correct_mass[1], counts[1])
        return max(0.0, abs(acc0 - acc1) - self.eps) ** 2

    def delta(self, a_i, q_correct):
        counts = self.counts.copy()
        correct_mass = self.correct_mass.copy()
        old = self.violation()
        counts[a_i] += 1.0
        correct_mass[a_i] += q_correct
        new = self.violation(counts, correct_mass)
        return new - old

    def update(self, a_i, q_correct):
        self.counts[a_i] += 1.0
        self.correct_mass[a_i] += q_correct


def estimate_q_correct(combiner, mpv, y_h_row, subset, y_cap):
    probs = combiner.combine_proba(
        mpv.reshape(1, -1),
        y_h_row.reshape(1, -1),
        [subset],
    )[0]
    return float(probs[y_cap])


def fair_placo_value_image(
    x,
    hcm_list,
    mpv,
    y_cap,
    est,
    acc_list,
    fair_state,
    a_i,
    q_correct,
    lambda_acc_gap,
    beta=0.0,
    subset_cost=0.0,
    penalty_strength=200.0,
):
    base = placo_value(x, hcm_list, mpv, y_cap, est, acc_list)
    delta = max(0.0, fair_state.delta(a_i, q_correct))
    cur = fair_state.violation()
    fairness_pen = penalty_strength * (lambda_acc_gap * delta + 0.1 * lambda_acc_gap * cur)
    log_score = np.log(max(base, 1e-12)) - fairness_pen - beta * subset_cost
    return float(np.exp(np.clip(log_score, -50.0, 50.0)))


def fair_placo_greedy_image(
    combiner,
    model_probs,
    y_h,
    num_humans,
    h_costs,
    est_true_labels,
    est_human_labels,
    acc_list,
    budget,
    A_test,
    lambda_init=LAMBDA_INIT,
    lambda_lr=LAMBDA_LR,
    lambda_max=LAMBDA_MAX,
    eps=EPSILON_FAIRNESS,
    beta=BETA_COST,
    batch_size=64,
):
    hcm_list = combiner.confusion_matrix
    chosen = []
    costs = []
    spent = 0.0
    fair_state = ImageFairnessState(eps=eps)
    lambda_acc_gap = float(lambda_init)
    history = {"lambda_acc_gap": [], "violation": []}

    for i in range(len(model_probs)):
        mpv = model_probs[i]
        y_cap = est_true_labels[i]
        est = est_human_labels[i]
        a_i = int(A_test[i])

        best_subset = []
        best_score = fair_placo_value_image(
            [0] * num_humans,
            hcm_list,
            mpv,
            y_cap,
            est,
            acc_list,
            fair_state,
            a_i,
            q_correct=float(mpv[y_cap]),
            lambda_acc_gap=lambda_acc_gap,
            beta=beta,
            subset_cost=0.0,
        )

        improved = True
        while improved:
            improved = False
            best_h = None
            for h in range(num_humans):
                if h in best_subset:
                    continue
                trial = best_subset + [h]
                trial_cost = float(np.sum([h_costs[i, hh] for hh in trial]))
                if spent + trial_cost > budget:
                    continue
                q_correct = estimate_q_correct(combiner, mpv, y_h[i], trial, y_cap)
                ind = [1 if k in trial else 0 for k in range(num_humans)]
                score = fair_placo_value_image(
                    ind,
                    hcm_list,
                    mpv,
                    y_cap,
                    est,
                    acc_list,
                    fair_state,
                    a_i,
                    q_correct=q_correct,
                    lambda_acc_gap=lambda_acc_gap,
                    beta=beta,
                    subset_cost=trial_cost,
                )
                if score > best_score:
                    best_score = score
                    best_h = h
            if best_h is not None:
                best_subset.append(best_h)
                improved = True

        cur_cost = float(np.sum([h_costs[i, h] for h in best_subset])) if best_subset else 0.0
        q_final = estimate_q_correct(combiner, mpv, y_h[i], best_subset, y_cap) if best_subset else float(mpv[y_cap])
        fair_state.update(a_i, q_final)
        chosen.append(best_subset)
        costs.append(cur_cost)
        spent += cur_cost

        if (i + 1) % batch_size == 0:
            v = fair_state.violation()
            lambda_acc_gap = float(min(lambda_max, max(0.0, lambda_acc_gap + lambda_lr * v)))
            history["lambda_acc_gap"].append(lambda_acc_gap)
            history["violation"].append(v)

    return chosen, costs, history


def fair_placo_lp_image(
    combiner,
    model_probs,
    y_h,
    num_humans,
    h_costs,
    est_true_labels,
    est_human_labels,
    acc_list,
    budget,
    A_test,
    lambda_init=LAMBDA_INIT,
    lambda_lr=LAMBDA_LR,
    lambda_max=LAMBDA_MAX,
    eps=EPSILON_FAIRNESS,
    beta=BETA_COST,
    batch_size=64,
):
    hcm_list = combiner.confusion_matrix
    n = len(model_probs)
    per_instance_budget = budget / n
    chosen = []
    costs = []
    fair_state = ImageFairnessState(eps=eps)
    lambda_acc_gap = float(lambda_init)
    history = {"lambda_acc_gap": [], "violation": []}

    for i in range(n):
        mpv = model_probs[i]
        y_cap = est_true_labels[i]
        est = est_human_labels[i]
        a_i = int(A_test[i])

        h_vals = []
        for h in range(num_humans):
            ind = [1 if k == h else 0 for k in range(num_humans)]
            q_correct = estimate_q_correct(combiner, mpv, y_h[i], [h], y_cap)
            v = fair_placo_value_image(
                ind, hcm_list, mpv, y_cap, est, acc_list,
                fair_state, a_i, q_correct=q_correct,
                lambda_acc_gap=lambda_acc_gap, beta=beta,
                subset_cost=float(h_costs[i, h]),
            )
            h_vals.append(v)

        prob = pulp.LpProblem(f"FAIR_LP_IMG_{i}", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("x", range(num_humans), cat=pulp.LpBinary)
        prob += pulp.lpSum(h_vals[h] * x[h] for h in range(num_humans))
        prob += (
            pulp.lpSum(h_costs[i, h] * x[h] for h in range(num_humans))
            <= per_instance_budget
        )
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        best_subset = [
            h for h in range(num_humans)
            if x[h].varValue is not None and x[h].varValue > 0.5
        ]
        cur_cost = float(np.sum([h_costs[i, h] for h in best_subset])) if best_subset else 0.0
        q_final = (
            estimate_q_correct(combiner, mpv, y_h[i], best_subset, y_cap)
            if best_subset else float(mpv[y_cap])
        )
        fair_state.update(a_i, q_final)
        chosen.append(best_subset)
        costs.append(cur_cost)

        if (i + 1) % batch_size == 0:
            v = fair_state.violation()
            lambda_acc_gap = float(min(lambda_max, max(0.0, lambda_acc_gap + lambda_lr * v)))
            history["lambda_acc_gap"].append(lambda_acc_gap)
            history["violation"].append(v)

    return chosen, costs, history


def ai_probs_for_kind(kind, model_probs):
    return model_probs.copy()


def accuracy_gap(y_true, y_pred, A):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    A = np.asarray(A)
    if np.sum(A == 0) == 0 or np.sum(A == 1) == 0:
        return 0.0
    acc0 = float(np.mean(y_pred[A == 0] == y_true[A == 0]))
    acc1 = float(np.mean(y_pred[A == 1] == y_true[A == 1]))
    return abs(acc0 - acc1)


def run_one(dataset_name, setting_label, cfg, human_counts, model_probs, y_true, A, seed=RNG_SEED):
    y_h = simulate_humans(cfg["humans"], human_counts, y_true, A, model_probs)
    ai_probs = ai_probs_for_kind(cfg["ai"], model_probs)

    y_h_tr, y_h_te, mp_tr, mp_te, y_true_tr, y_true_te, A_tr, A_te = train_test_split(
        y_h, ai_probs, y_true, A, test_size=0.4, random_state=seed, stratify=y_true
    )

    comb = MultiClassMAPCombiner(num_humans=NUM_HUMANS, num_classes=mp_te.shape[1]).fit(y_h_tr, y_true_tr)
    h_costs = make_costs(len(y_true_te), NUM_HUMANS, num_classes=mp_te.shape[1], seed=seed + 7)
    budget = NUM_HUMANS * mp_te.shape[1] * BUDGET_FRACTION * len(y_true_te)
    est_y, est_h = estimate_run_data(comb, mp_te, HUMAN_ACCURACIES)

    if cfg["policy"] == "placo":
        subsets, costs = placo_lp_budget(
            comb,
            mp_te,
            y_h_te,
            NUM_HUMANS,
            h_costs,
            est_y,
            est_h,
            HUMAN_ACCURACIES,
            budget,
        )
        history = None
    else:
        subsets, costs, history = fair_placo_lp_image(
            comb,
            mp_te,
            y_h_te,
            NUM_HUMANS,
            h_costs,
            est_y,
            est_h,
            HUMAN_ACCURACIES,
            budget,
            A_te,
        )

    ai_only_pred = np.argmax(mp_te, axis=1)
    final_pred = comb.combine(mp_te, y_h_te, subsets)

    return {
        "dataset": dataset_name,
        "setting": setting_label,
        "ai_kind": cfg["ai"],
        "policy_kind": cfg["policy"],
        "human_kind": cfg["humans"],
        "n_test": len(y_true_te),
        "group_source": "external" if np.any(A != build_groups(y_true, mp_te.shape[1])) else "proxy_class_bucket",
        "ai_only_accuracy": float(accuracy_score(y_true_te, ai_only_pred)),
        "ai_only_acc_gap": accuracy_gap(y_true_te, ai_only_pred, A_te),
        "best_human_accuracy": float(np.mean(y_h_te[:, 0] == y_true_te)),
        "final_accuracy": float(accuracy_score(y_true_te, final_pred)),
        "final_acc_gap": accuracy_gap(y_true_te, final_pred, A_te),
        "avg_subset_size": float(np.mean([len(s) for s in subsets])),
        "pct_human_used": float(np.mean([len(s) > 0 for s in subsets])) * 100.0,
        "spent_cost": float(np.sum(costs)),
        "budget": float(budget),
        "lambda_trace_len": 0 if history is None else len(history["lambda_acc_gap"]),
    }


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict]):
    grouped: Dict[Tuple[str, str], List[Dict]] = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["setting"]), []).append(row)
    out = []
    for (dataset, setting), group in grouped.items():
        first = group[0]
        out.append(
            {
                "dataset": dataset,
                "setting": setting,
                "ai_kind": first["ai_kind"],
                "policy_kind": first["policy_kind"],
                "human_kind": first["human_kind"],
                "runs": len(group),
                "ai_only_accuracy_mean": float(np.mean([r["ai_only_accuracy"] for r in group])),
                "best_human_accuracy_mean": float(np.mean([r["best_human_accuracy"] for r in group])),
                "final_accuracy_mean": float(np.mean([r["final_accuracy"] for r in group])),
                "ai_only_acc_gap_mean": float(np.mean([r["ai_only_acc_gap"] for r in group])),
                "final_acc_gap_mean": float(np.mean([r["final_acc_gap"] for r in group])),
                "avg_subset_size_mean": float(np.mean([r["avg_subset_size"] for r in group])),
                "spent_cost_mean": float(np.mean([r["spent_cost"] for r in group])),
            }
        )
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Run Task-3 a–l settings on PLACO CIFAR-10H / ImageNet datasets.")
    p.add_argument("--datasets", nargs="*", default=list(DATASET_SPECS.keys()), choices=list(DATASET_SPECS.keys()))
    p.add_argument("--runs", type=int, default=3, help="Number of repeated random splits.")
    p.add_argument("--max-samples", type=int, default=1500, help="Optional subsample size for faster runs; 0 disables.")
    p.add_argument("--seed", type=int, default=RNG_SEED)
    p.add_argument(
        "--groups-dir",
        default=None,
        help="Optional directory containing {dataset}_groups.csv with one 0/1 group per row.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    rows = []

    for dataset_name in args.datasets:
        print(f"\n=== {dataset_name} ===")
        human_counts, model_probs, y_true = load_image_dataset(dataset_name)
        groups_path = None
        if args.groups_dir:
            candidate = os.path.join(args.groups_dir, f"{dataset_name}_groups.csv")
            if os.path.exists(candidate):
                groups_path = candidate
        A = build_groups(y_true, model_probs.shape[1], groups_path=groups_path)
        human_counts, model_probs, y_true, A = maybe_subsample(
            human_counts, model_probs, y_true, A, args.max_samples, args.seed
        )

        for run_idx in range(args.runs):
            seed = args.seed + run_idx
            for setting_label, cfg in SETTINGS:
                print(f"  -> run={run_idx} | {setting_label}")
                row = run_one(dataset_name, setting_label, cfg, human_counts, model_probs, y_true, A, seed=seed)
                row["run"] = run_idx
                rows.append(row)
                print(
                    f"     final_acc={row['final_accuracy']:.3f} "
                    f"gap={row['final_acc_gap']:.3f} "
                    f"subset={row['avg_subset_size']:.2f} "
                    f"cost={row['spent_cost']:.1f}"
                )

    long_path = os.path.join(CSV_DIR, "image_results_long.csv")
    summary_path = os.path.join(CSV_DIR, "image_results_summary.csv")

    long_fields = [
        "dataset", "run", "setting", "ai_kind", "policy_kind", "human_kind",
        "n_test", "group_source", "ai_only_accuracy", "ai_only_acc_gap",
        "best_human_accuracy", "final_accuracy", "final_acc_gap",
        "avg_subset_size", "pct_human_used", "spent_cost", "budget",
        "lambda_trace_len",
    ]
    write_csv(long_path, rows, long_fields)

    summary_rows = summarize(rows)
    summary_fields = [
        "dataset", "setting", "ai_kind", "policy_kind", "human_kind", "runs",
        "ai_only_accuracy_mean", "best_human_accuracy_mean", "final_accuracy_mean",
        "ai_only_acc_gap_mean", "final_acc_gap_mean", "avg_subset_size_mean",
        "spent_cost_mean",
    ]
    write_csv(summary_path, summary_rows, summary_fields)

    print(f"\nWrote {long_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
