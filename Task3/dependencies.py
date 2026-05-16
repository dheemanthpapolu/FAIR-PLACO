import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")

HUMAN_ACCURACIES = [0.48, 0.94, 0.53, 0.70, 0.43, 0.58, 0.30, 0.88, 0.62 , 0.57, 0.38, 0.49, 0.39, 0.55, 0.29]
NUM_HUMANS = len(HUMAN_ACCURACIES)

BUDGET_FRACTION = 0.30

LAMBDA_INIT = 1.0
LAMBDA_LR = 0.5
LAMBDA_MAX = 50.0
EPSILON_FAIRNESS = 0.05
BETA_COST = 0.0

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
CSV_DIR = os.path.join(OUTPUT_DIR, "csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)
