import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dependencies import (
    HUMAN_ACCURACIES, NUM_HUMANS, BUDGET_FRACTION, CSV_DIR,
    EPSILON_FAIRNESS,
)
from data_loaders import load_dataset
from ai_models import make_ai_model
from human_simulators import simulate_humans
from combiner import BinaryMAPCombiner
from cost_function import make_costs
from policy import placo_lp_budget, fair_placo_lp, estimate_run_data
from fairness_metrics import all_metrics


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

DATASETS = ["compas", "german", "adult", "acs_income", "cifar10h", "imagenet"]


def run_one(dataset_name, setting_label, cfg, fair_constraint="equalized_odds",
            quick=False, seed=42):
    t0 = time.time()
    Xt, Xe, yt, ye, At, Ae, feats = load_dataset(dataset_name)

    if quick and len(yt) > 1500:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(yt), size=1500, replace=False)
        Xt, yt, At = Xt[idx], yt[idx], At[idx]
    if quick and len(ye) > 800:
        rng = np.random.default_rng(seed + 1)
        idx = rng.choice(len(ye), size=800, replace=False)
        Xe, ye, Ae = Xe[idx], ye[idx], Ae[idx]

    if cfg["ai"] == "basic":
        ai = make_ai_model("basic", random_state=seed).fit(Xt, yt)
    else:
        ai = make_ai_model("fair", constraint=fair_constraint,
                           max_iter=8, random_state=seed).fit(Xt, yt, A=At)
    mp_tr = np.clip(ai.predict_proba(Xt), 1e-6, 1 - 1e-6)
    mp_te = np.clip(ai.predict_proba(Xe), 1e-6, 1 - 1e-6)
    ai_pred_tr = (mp_tr[:, 1] >= 0.5).astype(int)
    ai_pred_te = (mp_te[:, 1] >= 0.5).astype(int)
    ai_only_metrics = all_metrics(ye, ai_pred_te, Ae, Xe)

    H_tr = simulate_humans(cfg["humans"], yt, At, mp_tr[:, 1])
    H_te = simulate_humans(cfg["humans"], ye, Ae, mp_te[:, 1])

    comb = BinaryMAPCombiner(num_humans=NUM_HUMANS).fit(H_tr, yt)

    n_te = len(ye)
    h_costs = make_costs(n_te, NUM_HUMANS, num_classes=2, seed=seed + 7)
    budget = NUM_HUMANS * 2.0 * BUDGET_FRACTION * n_te
    est_y, est_h = estimate_run_data(comb, mp_te, NUM_HUMANS, HUMAN_ACCURACIES)

    if cfg["policy"] == "placo":
        subsets, costs = placo_lp_budget(
            comb, mp_te, H_te, NUM_HUMANS, h_costs,
            est_y, est_h, HUMAN_ACCURACIES, budget,
        )
        history = None
    else:
        subsets, costs, history = fair_placo_lp(
            comb, mp_te, H_te, NUM_HUMANS, h_costs,
            est_y, est_h, HUMAN_ACCURACIES, budget, Ae,
            eps=EPSILON_FAIRNESS,
        )

    final_pred = comb.combine(mp_te, H_te, subsets)
    metrics = all_metrics(ye, final_pred, Ae, Xe)

    avg_subset_size = float(np.mean([len(s) for s in subsets]))
    pct_humans_used = float(np.mean([len(s) > 0 for s in subsets])) * 100
    deferral_to_humans = pct_humans_used

    row = {
        "dataset": dataset_name,
        "setting": setting_label,
        "ai_kind": cfg["ai"],
        "policy_kind": cfg["policy"],
        "human_kind": cfg["humans"],
        "n_test": n_te,
        "budget": budget,
        "spent_cost": float(sum(costs)),
        "avg_subset_size": avg_subset_size,
        "pct_human_used": pct_humans_used,
        "ai_only_accuracy": ai_only_metrics["accuracy"],
        "ai_only_dp_diff": ai_only_metrics["demographic_parity_diff"],
        "ai_only_eo_diff": ai_only_metrics["equalized_odds_diff"],
        "ai_only_eopp_diff": ai_only_metrics["equal_opportunity_diff"],
        "ai_only_te_diff": ai_only_metrics["treatment_equality_diff"],
        "ai_only_subgroup_fairness": ai_only_metrics["subgroup_fairness"],
        "ai_only_individual_fairness": ai_only_metrics["individual_fairness"],
        **{f"final_{k}": v for k, v in metrics.items()},
        "wall_seconds": float(time.time() - t0),
    }
    return row, history


def run_all(datasets=None, quick=False, fair_constraint="equalized_odds"):
    datasets = datasets or DATASETS
    rows = []
    for ds in datasets:
        try:
            print(f"\n=== Loading {ds} ===")
            load_dataset(ds)
        except Exception as e:
            print(f"  skipping {ds}: {e}")
            continue
        for label, cfg in SETTINGS:
            try:
                print(f"  -> {ds:<11s} | {label}")
                row, _ = run_one(ds, label, cfg,
                                 fair_constraint=fair_constraint, quick=quick)
                rows.append(row)
                print(f"     acc={row['final_accuracy']:.3f}  "
                      f"DP={row['final_demographic_parity_diff']:.3f}  "
                      f"EO={row['final_equalized_odds_diff']:.3f}  "
                      f"EOpp={row['final_equal_opportunity_diff']:.3f}  "
                      f"sub={row['avg_subset_size']:.2f}  "
                      f"cost={row['spent_cost']:.1f}")
            except Exception as e:
                print(f"     FAILED: {e}")
    df = pd.DataFrame(rows)
    out = os.path.join(CSV_DIR, "results_long.csv")
    df.to_csv(out, index=False)
    print(f"\nWrote {out} ({len(df)} rows)")
    return df


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="Subsample large datasets for fast sanity run.")
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Subset of datasets to run (default: all).")
    p.add_argument("--fair-constraint", default="equalized_odds",
                   choices=("equalized_odds", "equal_opportunity", "demographic_parity"),
                   help="Constraint passed to ExponentiatedGradient for FairAI.")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip plot generation.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df = run_all(datasets=args.datasets, quick=args.quick,
                 fair_constraint=args.fair_constraint)
    if not args.no_plots:
        from plots import generate_all_plots
        generate_all_plots(df)
        print("Plots generated.")
    try:
        from analysis import generate_report
        generate_report(df)
        print("Analysis report written to output/analysis_report.md")
    except Exception as e:
        print(f"[analysis] skipped: {e}")
