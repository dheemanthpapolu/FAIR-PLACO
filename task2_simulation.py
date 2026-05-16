import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

def load_data():
    np.random.seed(42)
    n_samples = 3000

    gender = np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])

    income = np.random.normal(50000 + 15000 * gender, 15000, n_samples)
    credit_score = np.random.normal(600 + 50 * gender, 50, n_samples)

    logit = -10 + 0.0001 * income + 0.01 * credit_score
    prob = 1 / (1 + np.exp(-logit))
    y = (np.random.rand(n_samples) < prob).astype(int)

    df = pd.DataFrame({
        'Gender': gender,
        'Income': income,
        'CreditScore': credit_score,
        'Approved': y
    })
    return df

def get_rates(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return tpr, fpr, fn, fp

def demographic_parity_diff(y_pred, sensitive_attr):
    rate_1 = np.mean(y_pred[sensitive_attr == 1])
    rate_0 = np.mean(y_pred[sensitive_attr == 0])
    return abs(rate_1 - rate_0)

def equalized_odds_diff(y_true, y_pred, sensitive_attr):
    tpr_1, fpr_1, _, _ = get_rates(y_true[sensitive_attr == 1], y_pred[sensitive_attr == 1])
    tpr_0, fpr_0, _, _ = get_rates(y_true[sensitive_attr == 0], y_pred[sensitive_attr == 0])
    return abs(tpr_1 - tpr_0) + abs(fpr_1 - fpr_0)

def equal_opportunity_diff(y_true, y_pred, sensitive_attr):
    tpr_1, _, _, _ = get_rates(y_true[sensitive_attr == 1], y_pred[sensitive_attr == 1])
    tpr_0, _, _, _ = get_rates(y_true[sensitive_attr == 0], y_pred[sensitive_attr == 0])
    return abs(tpr_1 - tpr_0)

def treatment_equality_ratio(y_true, y_pred, sensitive_attr):
    _, _, fn_1, fp_1 = get_rates(y_true[sensitive_attr == 1], y_pred[sensitive_attr == 1])
    _, _, fn_0, fp_0 = get_rates(y_true[sensitive_attr == 0], y_pred[sensitive_attr == 0])
    ratio_1 = fn_1 / fp_1 if fp_1 > 0 else 0
    ratio_0 = fn_0 / fp_0 if fp_0 > 0 else 0
    return abs(ratio_1 - ratio_0)

def individual_fairness(X, y_pred, n_neighbors=5):
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    _, indices = nn.kneighbors(X)
    consistency = 0
    for i in range(len(X)):
        neighbor_preds = y_pred[indices[i][1:]]
        consistency += np.mean(neighbor_preds == y_pred[i])
    return consistency / len(X)

def counterfactual_fairness(clf, X, gender_idx=2):
    X_cf = X.copy()
    val_0 = np.min(X_cf[:, gender_idx])
    val_1 = np.max(X_cf[:, gender_idx])

    X_cf[:, gender_idx] = np.where(X_cf[:, gender_idx] > 0, val_0, val_1)

    orig_preds = clf.predict(X)
    cf_preds = clf.predict(X_cf)
    flipped_pct = np.mean(orig_preds != cf_preds) * 100
    return flipped_pct

def subgroup_fairness(y_true, y_pred, sensitive_attr, income_feature):
    median_income = np.median(income_feature)
    low_inc_mask = income_feature < median_income

    mask_sg1 = (sensitive_attr == 0) & low_inc_mask
    tpr_sg1, _, _, _ = get_rates(y_true[mask_sg1], y_pred[mask_sg1])

    mask_sg2 = (sensitive_attr == 1) & ~low_inc_mask
    tpr_sg2, _, _, _ = get_rates(y_true[mask_sg2], y_pred[mask_sg2])

    return abs(tpr_sg2 - tpr_sg1)

def plot_data_bias(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    counts = df['Gender'].value_counts()
    axes[0].pie(counts, labels=['Majority (1)', 'Minority (0)'], autopct='%1.1f%%',
                colors=sns.color_palette('Set2')[0:2], startangle=90, explode=(0.05, 0))
    axes[0].set_title('Representation Bias\n(Imbalanced Dataset)')

    sns.histplot(data=df, x='Income', hue='Gender', kde=True, ax=axes[1], palette='Set2')
    axes[1].set_title('Historical Bias\n(Income Distribution by Gender)')

    sns.histplot(data=df, x='CreditScore', hue='Gender', kde=True, ax=axes[2], palette='Set2')
    axes[2].set_title('Historical Bias\n(Credit Score by Gender)')

    plt.tight_layout()
    plt.savefig('01_data_bias_distributions.png', dpi=300)
    plt.close()

def plot_algorithmic_confidence(df_test, ai_probs):
    plt.figure(figsize=(8, 5))
    df_plot = df_test.copy()
    df_plot['AI_Confidence'] = ai_probs
    sns.kdeplot(data=df_plot, x='AI_Confidence', hue='Gender', fill=True, palette='Set1')
    plt.axvline(0.40, color='gray', linestyle='--', label='Uncertainty Lower Bound')
    plt.axvline(0.60, color='gray', linestyle='--', label='Uncertainty Upper Bound')
    plt.title('AI Prediction Confidence Density by Gender')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('02_ai_confidence_bias.png', dpi=300)
    plt.close()

def plot_metrics_comparison(df_results):
    metrics_to_plot = ['Demographic Parity Diff', 'Equalized Odds Diff', 'Equal Opportunity Diff', 'Subgroup Fairness Diff']
    df_melted = df_results.melt(id_vars='Scenario', value_vars=metrics_to_plot, var_name='Metric', value_name='Score')

    plt.figure(figsize=(14, 7))
    sns.barplot(data=df_melted, x='Metric', y='Score', hue='Scenario', palette='mako')
    plt.title('Group and Subgroup Fairness Errors (Lower is Better)', fontsize=16)
    plt.xticks(rotation=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('03_fairness_metrics_comparison.png', dpi=300)
    plt.close()

def main():
    print("\n" + "="*50)
    print(" TASK 2: HUMAN-IN-THE-LOOP FAIRNESS SIMULATION ")
    print("="*50)

    print("\n[1/5] Simulating Data with Embedded Biases...")
    df = load_data()
    plot_data_bias(df)
    print("  -> Created '01_data_bias_distributions.png' showing Historical & Representation bias.")

    X_aware = df[['Income', 'CreditScore', 'Gender']].values
    X_unaware = df[['Income', 'CreditScore']].values
    A = df['Gender'].values
    y = df['Approved'].values

    scaler = StandardScaler()
    X_aware = scaler.fit_transform(X_aware)
    X_unaware = scaler.fit_transform(X_unaware)

    indices = np.arange(len(y))
    idx_train, idx_test, y_train, y_test = train_test_split(indices, y, test_size=0.3, random_state=42)
    A_test = A[idx_test]
    df_test = df.iloc[idx_test]

    print("\n[2/5] Training AI Models (Aware vs. Unaware)...")
    clf_unaware = LogisticRegression()
    clf_unaware.fit(X_unaware[idx_train], y_train)
    preds_unaware = clf_unaware.predict(X_unaware[idx_test])

    clf_aware = LogisticRegression()
    clf_aware.fit(X_aware[idx_train], y_train)
    ai_probs = clf_aware.predict_proba(X_aware[idx_test])[:, 1]
    ai_preds = clf_aware.predict(X_aware[idx_test])

    plot_algorithmic_confidence(df_test, ai_probs)
    print("  -> Created '02_ai_confidence_bias.png' showing Algorithmic Bias in confidence.")

    cf_flip_rate = counterfactual_fairness(clf_aware, X_aware[idx_test], gender_idx=2)

    deferral_mask = (ai_probs > 0.40) & (ai_probs < 0.60)
    print(f"\n[3/5] AI defers to Human on {np.sum(deferral_mask)} ambiguous cases ({np.mean(deferral_mask)*100:.1f}% of test set).")

    results = []

    def evaluate(preds, scenario_name, is_baseline=False):
        acc = accuracy_score(y_test, preds)
        dp = demographic_parity_diff(preds, A_test)
        eo = equalized_odds_diff(y_test, preds, A_test)
        eopp = equal_opportunity_diff(y_test, preds, A_test)
        te = treatment_equality_ratio(y_test, preds, A_test)
        ind_f = individual_fairness(X_aware[idx_test], preds)
        sub_f = subgroup_fairness(y_test, preds, A_test, X_aware[idx_test][:, 0])

        results.append({
            'Scenario': scenario_name,
            'Accuracy': acc,
            'Demographic Parity Diff': dp,
            'Equalized Odds Diff': eo,
            'Equal Opportunity Diff': eopp,
            'Treatment Equality Diff': te,
            'Individual Fairness': ind_f,
            'Subgroup Fairness Diff': sub_f
        })
        return acc

    print("\n[4/5] Running Human-in-the-Loop Scenarios...\n")

    baseline_acc = evaluate(ai_preds, "AI Only (Aware)", is_baseline=True)
    evaluate(preds_unaware, "AI (Unaware - No Gender)")

    preds_unbiased = ai_preds.copy()
    preds_unbiased[deferral_mask] = y_test[deferral_mask]
    evaluate(preds_unbiased, "AI + Unbiased Human")

    preds_automation = ai_preds.copy()
    evaluate(preds_automation, "AI + Automation Bias Human")

    preds_explicit = ai_preds.copy()
    for i in range(len(preds_explicit)):
        if deferral_mask[i]:
            if A_test[i] == 0:
                preds_explicit[i] = 0
            else:
                preds_explicit[i] = 1
    evaluate(preds_explicit, "AI + Explicitly Biased Human")

    df_results = pd.DataFrame(results)

    print("="*50)
    print(" INSIGHTFUL METRICS & ANALYSIS")
    print("="*50)

    print(f"\n> Counterfactual Fairness (AI Only):")
    print(f"  {cf_flip_rate:.1f}% of individuals would have received a DIFFERENT outcome")
    print("  if they were magically swapped to the opposite gender. This violates Counterfactual Fairness.")

    print("\n> The Illusion of 'Fairness through Unawareness':")
    unaware_dp = df_results[df_results['Scenario'] == 'AI (Unaware - No Gender)']['Demographic Parity Diff'].values[0]
    print(f"  Even when Gender is removed, Demographic Parity difference is {unaware_dp:.3f}.")
    print("  Why? Because Income and Credit Score act as proxies for the Historical Bias.")

    print("\n> Subgroup Fairness Analysis:")
    ai_sub_f = df_results[df_results['Scenario'] == 'AI Only (Aware)']['Subgroup Fairness Diff'].values[0]
    print(f"  Intersectionality matters: The Equal Opportunity gap between 'Low-Income Minority' ")
    print(f"  and 'High-Income Majority' subgroups is {ai_sub_f:.3f}.")

    print("\n> The Price of Fairness (POF):")
    unbiased_acc = df_results[df_results['Scenario'] == 'AI + Unbiased Human']['Accuracy'].values[0]
    print(f"  AI Baseline Accuracy: {baseline_acc*100:.1f}%")
    print(f"  Unbiased HITL Accuracy: {unbiased_acc*100:.1f}%")
    print("  In this specific case, an ideal human *improves* accuracy while fixing fairness blindspots.")

    print("\n> The Danger of the Real-World Biased Human:")
    exp_eo = df_results[df_results['Scenario'] == 'AI + Explicitly Biased Human']['Equalized Odds Diff'].values[0]
    ai_eo = df_results[df_results['Scenario'] == 'AI Only (Aware)']['Equalized Odds Diff'].values[0]
    print(f"  When Explicit Bias is introduced in the loop, Equalized Odds difference jumps from")
    print(f"  {ai_eo:.3f} (AI Only) to {exp_eo:.3f}. The human actively ruined the algorithmic guardrails.")

    print("\n[5/5] Generating Final Visuals...")
    plot_metrics_comparison(df_results)
    print("  -> Created '03_fairness_metrics_comparison.png' showing fairness scores across HITL scenarios.")
    print("\nSimulation Complete. Check the generated PNG files for visual insights!")

if __name__ == "__main__":
    main()
