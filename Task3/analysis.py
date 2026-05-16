import os
import sys
import textwrap
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dependencies import CSV_DIR, OUTPUT_DIR


METRICS = {
    "final_accuracy": dict(
        label="Accuracy",
        lower_is_better=False,
        description=(
            "Fraction of test instances classified correctly. "
            "Higher is better. Range [0, 1]."
        ),
        interpretation=(
            "Measures overall predictive quality of the combined human-AI team."
        ),
    ),
    "final_demographic_parity_diff": dict(
        label="Demographic Parity (DP) gap",
        lower_is_better=True,
        description=(
            "|P(Ŷ=1|A=1) - P(Ŷ=1|A=0)|. "
            "Lower is fairer. Ideal = 0. "
            "Measures whether both groups receive positive predictions at the same rate."
        ),
        interpretation=(
            "Tracked by the 'dp' dual weight inside the FAIR-PLACO Lagrangian."
        ),
    ),
    "final_equalized_odds_diff": dict(
        label="Equalized Odds (EO) gap",
        lower_is_better=True,
        description=(
            "max(|TPR_1-TPR_0|, |FPR_1-FPR_0|). "
            "Lower is fairer. Ideal = 0. "
            "Both true positive *and* false positive rates should match across groups."
        ),
        interpretation=(
            "Targeted both at training time (FairAI's ExponentiatedGradient "
            "constraint) and at routing time (FAIR-PLACO's 'eo' dual weight)."
        ),
    ),
    "final_equal_opportunity_diff": dict(
        label="Equal Opportunity (EOpp) gap",
        lower_is_better=True,
        description=(
            "|TPR_1 - TPR_0|. "
            "Lower is fairer. Ideal = 0. "
            "The recall of the positive class should be equal across groups."
        ),
        interpretation=(
            "Tracked by the 'eopp' dual weight inside the FAIR-PLACO Lagrangian."
        ),
    ),
    "final_treatment_equality_diff": dict(
        label="Treatment Equality (TE) gap",
        lower_is_better=True,
        description=(
            "|log((FN_0+1)/(FP_0+1)) - log((FN_1+1)/(FP_1+1))|. "
            "Lower is fairer. Ideal = 0. "
            "Each group's ratio of false negatives to false positives should be equal. "
            "A high TE gap means the *type* of error differs systematically by group."
        ),
        interpretation=(
            "Tracked by the 'te' dual weight inside the FAIR-PLACO Lagrangian "
            "(in rate-normalised soft form, see value_function.py)."
        ),
    ),
    "final_subgroup_fairness": dict(
        label="Subgroup Fairness gap",
        lower_is_better=True,
        description=(
            "Worst-case TPR gap across (sensitive attribute × feature quantile) "
            "intersections. Lower is fairer. Ideal = 0."
        ),
        interpretation=(
            "Finer-grained than EO: even if the top-level EO gap is small, a "
            "subgroup at a specific feature quantile may be systematically under-served."
        ),
    ),
    "final_individual_fairness": dict(
        label="Individual Fairness (consistency)",
        lower_is_better=False,
        description=(
            "Average fraction of k-nearest neighbours that receive the same "
            "prediction. Higher is fairer (more consistent). Range [0, 1]."
        ),
        interpretation=(
            "Similar individuals should receive similar decisions."
        ),
    ),
}

SETTING_LABELS = {
    "a_placo_basicAI_unbiased":       "a) PLACO + BasicAI + Unbiased",
    "b_placo_basicAI_explicit":       "b) PLACO + BasicAI + Explicit-biased",
    "c_placo_basicAI_automation":     "c) PLACO + BasicAI + Auto-biased",
    "d_placo_fairAI_unbiased":        "d) PLACO + FairAI  + Unbiased",
    "e_placo_fairAI_explicit":        "e) PLACO + FairAI  + Explicit-biased",
    "f_placo_fairAI_automation":      "f) PLACO + FairAI  + Auto-biased",
    "g_FAIRPLACO_fairAI_unbiased":    "g) FAIR-PLACO + FairAI + Unbiased",
    "h_FAIRPLACO_fairAI_explicit":    "h) FAIR-PLACO + FairAI + Explicit-biased",
    "i_FAIRPLACO_fairAI_automation":  "i) FAIR-PLACO + FairAI + Auto-biased",
    "j_FAIRPLACO_basicAI_unbiased":   "j) FAIR-PLACO + BasicAI + Unbiased",
    "k_FAIRPLACO_basicAI_explicit":   "k) FAIR-PLACO + BasicAI + Explicit-biased",
    "l_FAIRPLACO_basicAI_automation": "l) FAIR-PLACO + BasicAI + Auto-biased",
}


def _rank_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, meta in METRICS.items():
        if col not in df.columns:
            continue
        ascending = meta["lower_is_better"]
        df[f"rank_{col}"] = df.groupby("dataset")[col].rank(
            ascending=ascending, method="min"
        )
    rank_cols = [f"rank_{c}" for c in METRICS if f"rank_{c}" in df.columns]
    if rank_cols:
        df["total_rank"] = df[rank_cols].mean(axis=1)
        df["wins_best_rank"] = df.groupby("dataset")["total_rank"].transform("min") == df["total_rank"]
    return df


def _pct_improvement(val, baseline, lower_is_better):
    if baseline == 0:
        return float("nan")
    if lower_is_better:
        return 100.0 * (baseline - val) / abs(baseline)
    else:
        return 100.0 * (val - baseline) / abs(baseline)


def generate_report(df: pd.DataFrame) -> str:
    df_r = _rank_df(df)
    out_csv = os.path.join(CSV_DIR, "results_with_ranks.csv")
    df_r.to_csv(out_csv, index=False)

    lines = []
    w = lines.append

    w("# FAIR-PLACO Analysis Report")
    w("")
    w("> Auto-generated by `analysis.py`. Covers the 12 settings × 6 datasets "
      "experiment grid defined in `experiments.py` "
      "(4 model groups × 3 human-bias regimes × 6 datasets = 72 cells).")
    w("")

    w("---")
    w("## §1  How each fairness metric is scored")
    w("")
    w("| Metric | Direction | What it measures |")
    w("|--------|-----------|-----------------|")
    for col, meta in METRICS.items():
        direction = "↓ lower = fairer" if meta["lower_is_better"] else "↑ higher = better"
        w(f"| **{meta['label']}** | {direction} | {meta['description']} |")
    w("")
    for col, meta in METRICS.items():
        w(f"### {meta['label']}")
        w(f"**Column in CSV:** `{col}`")
        w("")
        w(meta["description"])
        w("")
        w(f"**How the pipeline relates to this metric:** {meta['interpretation']}")
        w("")

    w("---")
    w("## §2  Per-dataset rank tables")
    w("")
    w("Rank 1 = best. Ties share the lower rank. "
      "Each row is one of the 12 settings.")
    w("")

    datasets = df["dataset"].unique()
    for ds in sorted(datasets):
        sub = df_r[df_r["dataset"] == ds].copy()
        sub = sub.set_index("setting")
        w(f"### {ds.upper()}")
        w("")

        header = ["Setting", "Acc↑", "DP↓", "EO↓", "EOpp↓", "TE↓",
                  "Subgrp↓", "IndivF↑", "AvgRank"]
        rows_t = [header]
        rows_t.append(["---"] * len(header))

        for s_key in SETTING_LABELS:
            if s_key not in sub.index:
                continue
            row_s = sub.loc[s_key]
            label = SETTING_LABELS[s_key]
            avg_rank = row_s.get("total_rank", float("nan"))

            def _fmt(col):
                v = row_s.get(col, float("nan"))
                r = row_s.get(f"rank_{col}", float("nan"))
                if pd.isna(v):
                    return "—"
                return f"{v:.3f} (#{int(r)})"

            rows_t.append([
                label,
                _fmt("final_accuracy"),
                _fmt("final_demographic_parity_diff"),
                _fmt("final_equalized_odds_diff"),
                _fmt("final_equal_opportunity_diff"),
                _fmt("final_treatment_equality_diff"),
                _fmt("final_subgroup_fairness"),
                _fmt("final_individual_fairness"),
                f"{avg_rank:.2f}" if not pd.isna(avg_rank) else "—",
            ])

        col_widths = [max(len(r[c]) for r in rows_t) for c in range(len(header))]
        for row_s in rows_t:
            cells = [row_s[c].ljust(col_widths[c]) for c in range(len(header))]
            w("| " + " | ".join(cells) + " |")
        w("")

    w("---")
    w("## §3  Aggregate win counts (rank 1 on each metric)")
    w("")
    w("How many (dataset, metric) combinations each setting achieves rank 1 on.")
    w("")

    win_rows = []
    for s_key in SETTING_LABELS:
        wins = {}
        for col, meta in METRICS.items():
            rank_col = f"rank_{col}"
            if rank_col not in df_r.columns:
                continue
            sub = df_r[(df_r["setting"] == s_key) & (df_r[rank_col] == 1.0)]
            wins[meta["label"]] = len(sub)
        wins["TOTAL wins"] = sum(wins.values())
        wins["Setting"] = SETTING_LABELS[s_key]
        win_rows.append(wins)

    win_df = pd.DataFrame(win_rows).set_index("Setting")
    w("| Setting | " + " | ".join(win_df.columns) + " |")
    w("| --- | " + " | ".join(["---"] * len(win_df.columns)) + " |")
    for idx, row_w in win_df.iterrows():
        w("| " + idx + " | " + " | ".join(str(v) for v in row_w) + " |")
    w("")

    w("---")
    w("## §4  Pairwise percentage improvement vs the two unbiased baselines")
    w("")
    w("For each setting and each dataset, we report the percentage improvement "
      "vs two reference points: ")
    w("- **a)** plain PLACO + BasicAI + Unbiased humans (the original PLACO baseline);")
    w("- **d)** plain PLACO + FairAI + Unbiased humans (model-level fairness only).")
    w("")
    w("Positive = the row's setting is better than the baseline on that metric.")
    w("")

    baselines = {
        "a": "a_placo_basicAI_unbiased",
        "d": "d_placo_fairAI_unbiased",
    }

    for bl_name, bl_key in baselines.items():
        w(f"### vs. baseline **{bl_name}** ({SETTING_LABELS[bl_key]})")
        w("")
        header = ["Dataset", "Setting"] + [meta["label"] for meta in METRICS.values()]
        w("| " + " | ".join(header) + " |")
        w("| " + " | ".join(["---"] * len(header)) + " |")

        for ds in sorted(datasets):
            bl_row = df_r[(df_r["dataset"] == ds) & (df_r["setting"] == bl_key)]
            if bl_row.empty:
                continue
            bl = bl_row.iloc[0]
            for s_key in SETTING_LABELS:
                if s_key == bl_key:
                    continue
                cmp_row = df_r[(df_r["dataset"] == ds) & (df_r["setting"] == s_key)]
                if cmp_row.empty:
                    continue
                cur = cmp_row.iloc[0]
                cells = [ds.upper(), SETTING_LABELS[s_key]]
                for col, meta in METRICS.items():
                    if col not in df.columns:
                        cells.append("—")
                        continue
                    pct = _pct_improvement(cur[col], bl[col],
                                           meta["lower_is_better"])
                    sign = "+" if pct >= 0 else ""
                    cells.append(f"{sign}{pct:.1f}%")
                w("| " + " | ".join(cells) + " |")
        w("")

    w("---")
    w("## §5  Mechanism notes — what each component does")
    w("")

    mechanism_text = textwrap.dedent("""\
    The 12-setting grid factors three independent levers; each row in the grid
    holds two of them fixed and varies the third.

    | Lever | Settings that turn it ON | Settings that turn it OFF |
    |-------|--------------------------|---------------------------|
    | Fairness-aware AI prior (FairAI) | d, e, f, g, h, i             | a, b, c, j, k, l |
    | Lagrangian routing (FAIR-PLACO)  | g, h, i, j, k, l             | a, b, c, d, e, f |
    | Adversarial humans               | b, c, e, f, h, i, k, l       | a, d, g, j (unbiased) |

    Concretely:

    **Lever 1 — FairAI prior.**
    `ExponentiatedGradient(HistGradientBoostingClassifier(...),
    constraints=EqualizedOdds())` from Fairlearn replaces the unconstrained
    HistGB used by plain PLACO. The reduction enforces a Hardt-style
    equalized-odds bound at training time, so the AI's probability column
    already has a smaller fairness gap before any human is consulted.

    **Lever 2 — Lagrangian routing.**
    The original PLACO greedy chooses humans to maximise

        V_PLACO(S) = Π_{h ∈ S} f(confusion_h, p_model, y_estimated)

    which is purely accuracy-driven. FAIR-PLACO replaces this with a
    log-space Lagrangian

        log V_fair(S) = log V_PLACO(S)
                      - K · Σ_m λ_m · max(0, Δψ_m(i, S))
                      - 0.1 K · Σ_m λ_m · ψ_m^current
                      - β · C(S)

    where Δψ_m is the *predicted increment* in fairness violation m if subset S
    handles instance i, and ψ_m^current is the running violation. K=200 puts
    the fairness term on the same log-scale as the per-human accuracy product.

    **Lever 3 — Primal-dual ascent.**
    Every batch_size=64 instances the dual weights are updated:

        λ_m ← clip(λ_m + lr · (Γ_m − ε), 0, λ_max)

    where Γ_m is the realised fairness violation on the last batch and ε=0.05
    is the allowed slack. If a violation grows, the penalty grows; if it
    shrinks below ε the penalty decays.

    **Treatment Equality (TE).**
    TE is included in the Lagrangian as a normalised soft-rate penalty
    (FNR_a / n_a and FPR_a / n_a), keeping it in the same O((1/n)^2) range
    as DP / EOpp / EO so a single K and λ_init are correctly calibrated.
    The evaluator reports the smoothed log-ratio
    |log((FN_0+1)/(FP_0+1)) − log((FN_1+1)/(FP_1+1))|, following Berk et al.
    (2018). Note that the Lagrangian and evaluator are not monotonically
    related, so TE results should be interpreted with care.

    **Three human-bias regimes (b/c, e/f, h/i, k/l vs a/d/g/j).**
    'Unbiased' humans flip the label with probability 1-acc.
    'Explicit-biased' humans hard-deny minority instances when the AI is
    uncertain (a deterministic group-prejudice pattern).
    'Automation-biased' humans copy the AI's hard label with prob 0.85
    regardless of their own competence.
    """)
    w(mechanism_text)

    w("---")
    w("## §6  Average metric values across all datasets (per setting)")
    w("")
    w("Each row is one of the 12 settings; values are arithmetic means across "
      "the six datasets (compas, german, adult, acs_income, cifar10h, "
      "imagenet). No row is highlighted: the reader is left to draw their own "
      "comparison.")
    w("")

    avg_rows = []
    for s_key in SETTING_LABELS:
        sub = df_r[df_r["setting"] == s_key]
        if sub.empty:
            continue
        row_avg = {"Setting": SETTING_LABELS[s_key]}
        for col, meta in METRICS.items():
            if col in sub.columns:
                row_avg[meta["label"]] = f"{sub[col].mean():.4f}"
        avg_rows.append(row_avg)

    avg_df = pd.DataFrame(avg_rows).set_index("Setting")
    w("| Setting | " + " | ".join(avg_df.columns) + " |")
    w("| --- | " + " | ".join(["---"] * len(avg_df.columns)) + " |")
    for idx, row_a in avg_df.iterrows():
        w("| " + idx + " | " + " | ".join(str(v) for v in row_a) + " |")
    w("")
    w("> Accuracy ↑ higher better. All other metrics ↓ lower better "
      "(except Individual Fairness, which is a consistency score, ↑ higher).")
    w("")

    w("---")
    w("## §7  How to run the full benchmark")
    w("")
    w("```bash")
    w("# Full 12-setting × 6-dataset run (~90–180 min)")
    w("~/.pyenv/versions/midast310/bin/python experiments.py")
    w("")
    w("# Fast sanity-check (subsamples large train/test sets, ~15–30 min)")
    w("~/.pyenv/versions/midast310/bin/python experiments.py --quick")
    w("")
    w("# Only specific datasets")
    w("~/.pyenv/versions/midast310/bin/python experiments.py --datasets compas cifar10h")
    w("")
    w("# PLACO benchmarks on 4 fairness datasets + original CIFAR-10H / ImageNet-16H")
    w("~/.pyenv/versions/midast310/bin/python run_placo_benchmarks.py")
    w("")
    w("# Re-generate this analysis report from an existing results_long.csv")
    w("~/.pyenv/versions/midast310/bin/python analysis.py")
    w("```")
    w("")
    w("Outputs written to:")
    w("```")
    w("output/csv/results_long.csv          # raw results (72 rows: 12 settings × 6 datasets)")
    w("output/csv/results_with_ranks.csv    # same + per-metric ranks + total_rank")
    w("output/csv/placo_benchmarks.csv      # PLACO benchmark on 6 datasets")
    w("output/analysis_report.md            # this report")
    w("output/plots/p01_*.png … p15_*.png   # 15+ analytical plots")
    w("```")
    w("")

    report = "\n".join(lines)
    out_md = os.path.join(OUTPUT_DIR, "analysis_report.md")
    with open(out_md, "w") as fh:
        fh.write(report)
    print(f"Wrote {out_md}")
    print(f"Wrote {out_csv}")
    return report


def main():
    csv_path = os.path.join(CSV_DIR, "results_long.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found. Run experiments.py first.")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    generate_report(df)


if __name__ == "__main__":
    main()
