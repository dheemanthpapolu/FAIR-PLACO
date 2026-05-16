import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from dependencies import PLOTS_DIR

sns.set_theme(style="whitegrid", context="talk")
mpl.rcParams["figure.dpi"] = 110

SETTING_SHORT = {
    "a_placo_basicAI_unbiased":          "a) PLACO+BasicAI+Unbias",
    "b_placo_basicAI_explicit":          "b) PLACO+BasicAI+Explicit",
    "c_placo_basicAI_automation":        "c) PLACO+BasicAI+Auto",
    "d_placo_fairAI_unbiased":           "d) PLACO+FairAI+Unbias",
    "e_placo_fairAI_explicit":           "e) PLACO+FairAI+Explicit",
    "f_placo_fairAI_automation":         "f) PLACO+FairAI+Auto",
    "g_FAIRPLACO_fairAI_unbiased":       "g) FAIR-PLACO+FairAI+Unbias",
    "h_FAIRPLACO_fairAI_explicit":       "h) FAIR-PLACO+FairAI+Explicit",
    "i_FAIRPLACO_fairAI_automation":     "i) FAIR-PLACO+FairAI+Auto",
    "j_FAIRPLACO_basicAI_unbiased":      "j) FAIR-PLACO+BasicAI+Unbias",
    "k_FAIRPLACO_basicAI_explicit":      "k) FAIR-PLACO+BasicAI+Explicit",
    "l_FAIRPLACO_basicAI_automation":    "l) FAIR-PLACO+BasicAI+Auto",
}
ORDER = list(SETTING_SHORT.keys())
SHORT_ORDER = [SETTING_SHORT[k] for k in ORDER]
GROUP_PALETTE = {
    "PLACO + BasicAI": "#9CA3AF",
    "PLACO + FairAI":  "#3B82F6",
    "FAIR-PLACO + FairAI": "#EF4444",
    "FAIR-PLACO + BasicAI": "#10B981",
}
def group_for(label):
    if label.startswith("a") or label.startswith("b") or label.startswith("c"):
        return "PLACO + BasicAI"
    if label.startswith("d") or label.startswith("e") or label.startswith("f"):
        return "PLACO + FairAI"
    if label.startswith("g") or label.startswith("h") or label.startswith("i"):
        return "FAIR-PLACO + FairAI"
    return "FAIR-PLACO + BasicAI"


def _save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"   wrote {path}")


def _bar_per_dataset(df, metric_col, title, fname, lower_better=True):
    df = df.copy()
    df["short"] = df["setting"].map(SETTING_SHORT)
    df["group"] = df["setting"].map(group_for)
    datasets = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(len(datasets), 1,
                             figsize=(12, 3.6 * len(datasets)),
                             sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds].copy()
        sub["short"] = pd.Categorical(sub["short"], categories=SHORT_ORDER, ordered=True)
        sub = sub.sort_values("short")
        colors = [GROUP_PALETTE[group_for(s)] for s in sub["setting"]]
        bars = ax.bar(range(len(sub)), sub[metric_col].values, color=colors,
                      edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub["short"], rotation=35, ha="right", fontsize=9)
        ax.set_title(f"{ds}", fontsize=12)
        ax.set_ylabel(metric_col)
    fig.suptitle(title + ("  (lower is better)" if lower_better else "  (higher is better)"),
                 fontsize=14, y=1.005)
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_PALETTE.values()]
    fig.legend(legend_handles, list(GROUP_PALETTE.keys()),
               loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    _save(fig, fname)


def plot_radar_main_per_dataset(df):
    metrics = [
        ("final_accuracy", "Accuracy", False),
        ("final_demographic_parity_diff", "DP gap", True),
        ("final_equalized_odds_diff", "EO gap", True),
        ("final_equal_opportunity_diff", "EOpp gap", True),
        ("final_treatment_equality_diff", "TE gap", True),
        ("final_subgroup_fairness", "Subgroup gap", True),
    ]
    targets = ["a_placo_basicAI_unbiased",
               "d_placo_fairAI_unbiased",
               "g_FAIRPLACO_fairAI_unbiased",
               "j_FAIRPLACO_basicAI_unbiased"]
    colors = ["#9CA3AF", "#3B82F6", "#EF4444", "#10B981"]
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += [angles[0]]
        for color, setting in zip(colors, targets):
            row = sub[sub["setting"] == setting]
            if row.empty:
                continue
            r = row.iloc[0]
            vals = []
            for col, _, lower_better in metrics:
                v = float(r[col])
                if lower_better:
                    vals.append(max(0.0, 1.0 - min(1.0, v)))
                else:
                    vals.append(min(1.0, v))
            vals += [vals[0]]
            ax.plot(angles, vals, color=color, linewidth=2,
                    label=SETTING_SHORT[setting])
            ax.fill(angles, vals, color=color, alpha=0.15)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m[1] for m in metrics], fontsize=10)
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels(["0.25", "0.5", "0.75"], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(f"Radar: {ds}\n(higher = better, fairness gaps inverted)",
                     fontsize=12)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
        _save(fig, f"p08_radar_{ds}.png")


def plot_pareto(df, fairness_col, fname, fairness_label):
    fig, ax = plt.subplots(figsize=(9, 6))
    df = df.copy()
    df["group"] = df["setting"].map(group_for)
    for group, sub in df.groupby("group"):
        ax.scatter(sub[fairness_col], sub["final_accuracy"],
                   color=GROUP_PALETTE[group], label=group, s=120,
                   edgecolor="black", linewidth=0.8, alpha=0.85)
    for _, r in df.iterrows():
        ax.annotate(r["dataset"][:3], (r[fairness_col], r["final_accuracy"]),
                    fontsize=7, alpha=0.6,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(f"{fairness_label} (lower = fairer)")
    ax.set_ylabel("Accuracy (higher = better)")
    ax.set_title(f"Pareto frontier: Accuracy vs {fairness_label}")
    ax.legend()
    _save(fig, fname)


def plot_setting_g_vs_baselines(df):
    metrics = [("final_demographic_parity_diff", "DP"),
               ("final_equalized_odds_diff",     "EO"),
               ("final_equal_opportunity_diff",  "EOpp"),
               ("final_subgroup_fairness",       "Subgroup")]
    rows = []
    for ds, sub in df.groupby("dataset"):
        try:
            base = sub[sub["setting"] == "a_placo_basicAI_unbiased"].iloc[0]
            mid  = sub[sub["setting"] == "d_placo_fairAI_unbiased"].iloc[0]
            target = sub[sub["setting"] == "g_FAIRPLACO_fairAI_unbiased"].iloc[0]
        except Exception:
            continue
        for col, name in metrics:
            base_v = float(base[col])
            mid_v  = float(mid[col])
            tar_v  = float(target[col])
            rows.append(dict(dataset=ds, metric=name,
                             vs="vs PLACO+BasicAI (a)",
                             pct_drop=100 * (base_v - tar_v) / max(base_v, 1e-9)))
            rows.append(dict(dataset=ds, metric=name,
                             vs="vs PLACO+FairAI (d)",
                             pct_drop=100 * (mid_v - tar_v) / max(mid_v, 1e-9)))
    if not rows:
        return
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=plot_df, x="metric", y="pct_drop", hue="dataset",
                ax=ax)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Relative change in fairness gap of setting (g) vs unbiased "
                 "baselines\n(positive = smaller gap than the baseline)")
    ax.set_ylabel("% change in gap (positive = smaller)")
    _save(fig, "p11_setting_g_vs_baselines.png")


def plot_cost_vs_accuracy(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    df = df.copy()
    df["group"] = df["setting"].map(group_for)
    for ds, sub in df.groupby("dataset"):
        for group, sub2 in sub.groupby("group"):
            ax.scatter(sub2["spent_cost"] / sub2["budget"],
                       sub2["final_accuracy"],
                       color=GROUP_PALETTE[group], s=120,
                       label=f"{ds} | {group}",
                       alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Fraction of budget spent")
    ax.set_ylabel("Combined accuracy")
    ax.set_title("Cost / accuracy: how much budget each setting consumed")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    _save(fig, "p12_cost_vs_accuracy.png")


def plot_human_usage(df):
    df = df.copy()
    df["short"] = df["setting"].map(SETTING_SHORT)
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(
        data=df, x="short", y="avg_subset_size", hue="dataset", ax=ax,
        order=SHORT_ORDER,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    ax.set_title("Average subset size selected per instance")
    _save(fig, "p13_human_usage_by_setting.png")


def plot_heatmap_per_dataset(df):
    metrics = ["final_accuracy", "final_demographic_parity_diff",
               "final_equalized_odds_diff", "final_equal_opportunity_diff",
               "final_treatment_equality_diff", "final_subgroup_fairness",
               "final_individual_fairness"]
    nice = ["Accuracy", "DP gap", "EO gap", "EOpp gap", "TE gap",
            "Subgroup gap", "Individual"]
    for ds, sub in df.groupby("dataset"):
        sub = sub.copy()
        sub["short"] = sub["setting"].map(SETTING_SHORT)
        sub = sub.set_index("short").reindex(SHORT_ORDER)[metrics]
        norm = sub.copy()
        for c in metrics:
            x = sub[c].values.astype(float)
            lo, hi = np.nanmin(x), np.nanmax(x)
            norm[c] = (x - lo) / (hi - lo + 1e-12)
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(norm, annot=sub.values, fmt=".3f",
                    xticklabels=nice, ax=ax, cmap="RdYlGn_r",
                    cbar_kws={"label": "normalised (0=best, 1=worst)"})
        ax.set_title(f"Metric heatmap (raw values annotated): {ds}")
        ax.set_ylabel("")
        _save(fig, f"p14_heatmap_{ds}.png")


def plot_3x3_summary(df):
    bias_order = ["unbiased", "explicit_biased", "automation_biased"]
    grp_order = [
        "PLACO + BasicAI",
        "PLACO + FairAI",
        "FAIR-PLACO + FairAI",
        "FAIR-PLACO + BasicAI",
    ]
    df = df.copy()
    df["group"] = df["setting"].map(group_for)
    fig, axes = plt.subplots(4, 3, figsize=(15, 14), sharey=True)
    for r, grp in enumerate(grp_order):
        for c, hk in enumerate(bias_order):
            ax = axes[r, c]
            sub = df[(df["group"] == grp) & (df["human_kind"] == hk)]
            datasets = sub["dataset"].tolist()
            accs = sub["final_accuracy"].values
            eos = sub["final_equalized_odds_diff"].values
            x = np.arange(len(datasets))
            ax.bar(x - 0.2, accs, width=0.4, color="#2563EB", label="Acc")
            ax.bar(x + 0.2, eos, width=0.4, color="#EF4444", label="EO gap")
            ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=8, rotation=20)
            ax.set_ylim(0, 1)
            if c == 0:
                ax.set_ylabel(grp, fontsize=10)
            if r == 0:
                ax.set_title(hk, fontsize=10)
            if r == 0 and c == 0:
                ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("4x3 summary: Accuracy vs Equalized-Odds gap "
                 "across (group × human-bias) combinations", y=1.01)
    _save(fig, "p15_grouped_settings_4x3_summary.png")


def generate_all_plots(df):
    if df.empty:
        print("No results to plot.")
        return
    _bar_per_dataset(df, "final_accuracy",
                     "Final combined accuracy by setting",
                     "p01_accuracy_by_setting_per_dataset.png",
                     lower_better=False)
    _bar_per_dataset(df, "final_demographic_parity_diff",
                     "Demographic Parity gap",
                     "p02_dp_diff_by_setting_per_dataset.png")
    _bar_per_dataset(df, "final_equalized_odds_diff",
                     "Equalized Odds gap",
                     "p03_eo_diff_by_setting_per_dataset.png")
    _bar_per_dataset(df, "final_equal_opportunity_diff",
                     "Equal Opportunity gap",
                     "p04_eopp_diff_by_setting_per_dataset.png")
    _bar_per_dataset(df, "final_treatment_equality_diff",
                     "Treatment Equality gap",
                     "p05_treatment_equality_by_setting_per_dataset.png")
    _bar_per_dataset(df, "final_subgroup_fairness",
                     "Subgroup Fairness gap",
                     "p06_subgroup_fairness_by_setting_per_dataset.png")
    _bar_per_dataset(df, "final_individual_fairness",
                     "Individual Fairness consistency",
                     "p07_individual_fairness_by_setting_per_dataset.png",
                     lower_better=False)
    plot_radar_main_per_dataset(df)
    plot_pareto(df, "final_equalized_odds_diff",
                "p09_pareto_acc_vs_eo.png", "Equalized Odds gap")
    plot_pareto(df, "final_demographic_parity_diff",
                "p10_pareto_acc_vs_dp.png", "Demographic Parity gap")
    plot_setting_g_vs_baselines(df)
    plot_cost_vs_accuracy(df)
    plot_human_usage(df)
    plot_heatmap_per_dataset(df)
    plot_3x3_summary(df)


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                  "output", "csv", "results_long.csv"))
    generate_all_plots(df)
