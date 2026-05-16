import numpy as np

from dependencies import RNG_SEED


def make_costs(n_instances, num_humans, num_classes=2, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0001, num_classes, size=(n_instances, num_humans))


def subset_cost(subset, h_costs_row):
    if not len(subset):
        return 0.0
    return float(np.sum([h_costs_row[i] for i in subset]))
