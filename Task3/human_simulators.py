import numpy as np

from dependencies import HUMAN_ACCURACIES, NUM_HUMANS, RNG_SEED


def _flip(rng, label, p_correct):
    return label if rng.random() < p_correct else 1 - label


def simulate_unbiased(y_true, A, ai_probs, accuracies=HUMAN_ACCURACIES, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    H = np.zeros((n, len(accuracies)), dtype=int)
    for j, acc in enumerate(accuracies):
        for i in range(n):
            H[i, j] = _flip(rng, int(y_true[i]), acc)
    return H


def simulate_explicit_biased(y_true, A, ai_probs, accuracies=HUMAN_ACCURACIES,
                              minority_penalty=0.30, seed=RNG_SEED + 1):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    num_humans = len(accuracies)
    H = np.zeros((n, num_humans), dtype=int)
    uncertain = (ai_probs > 0.40) & (ai_probs < 0.60)

    for j, acc in enumerate(accuracies):
        for i in range(n):
            if A[i] == 0:
                if j % 2 == 0 and uncertain[i]:
                    H[i, j] = 0
                else:
                    eff_acc = max(0.0, acc - minority_penalty)
                    H[i, j] = _flip(rng, int(y_true[i]), eff_acc)
            else:
                H[i, j] = _flip(rng, int(y_true[i]), acc)
    return H


def simulate_automation_biased(y_true, A, ai_probs, accuracies=HUMAN_ACCURACIES,
                                trust=0.85, seed=RNG_SEED + 2):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    num_humans = len(accuracies)
    H = np.zeros((n, num_humans), dtype=int)
    ai_label = (ai_probs >= 0.5).astype(int)

    for j, acc in enumerate(accuracies):
        for i in range(n):
            if rng.random() < trust:
                H[i, j] = ai_label[i]
            else:
                H[i, j] = _flip(rng, int(y_true[i]), acc)
    return H


HUMAN_SIM_REGISTRY = {
    "unbiased": simulate_unbiased,
    "explicit_biased": simulate_explicit_biased,
    "automation_biased": simulate_automation_biased,
}


def simulate_humans(kind, y_true, A, ai_probs, **kw):
    if kind not in HUMAN_SIM_REGISTRY:
        raise KeyError(kind)
    return HUMAN_SIM_REGISTRY[kind](y_true, A, ai_probs, **kw)
