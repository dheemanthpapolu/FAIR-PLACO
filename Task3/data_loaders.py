import os
import io
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from dependencies import DATA_DIR, PROJECT_ROOT, RNG_SEED

CACHE = os.path.join(PROJECT_ROOT, "data_cache")
os.makedirs(CACHE, exist_ok=True)


def _cache_or_build(name, builder):
    path = os.path.join(CACHE, f"{name}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    obj = builder()
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return obj


def _split(X, y, A, test_size=0.3, seed=RNG_SEED):
    X_tr, X_te, y_tr, y_te, A_tr, A_te = train_test_split(
        X, y, A, test_size=test_size, random_state=seed, stratify=y
    )
    sc = StandardScaler().fit(X_tr)
    return sc.transform(X_tr), sc.transform(X_te), y_tr, y_te, A_tr, A_te


def load_compas():
    def _build():
        path = os.path.join(DATA_DIR, "compas-scores.csv")
        df = pd.read_csv(path)
        target_col = "two_year_recid" if "two_year_recid" in df.columns else "is_recid"
        keep = ["sex", "age", "race", "juv_fel_count", "juv_misd_count",
                "juv_other_count", "priors_count", "c_charge_degree", target_col]
        df = df[[c for c in keep if c in df.columns]].dropna()
        df = df[df[target_col] >= 0]
        A = (df["race"].astype(str).str.lower() == "caucasian").astype(int).values
        y = df[target_col].astype(int).values
        y = (1 - y).astype(int)
        df = df.drop(columns=["race", target_col])
        df["sex"] = (df["sex"].astype(str).str.lower() == "male").astype(int)
        df["c_charge_degree"] = (df["c_charge_degree"].astype(str).str.upper() == "F").astype(int)
        X = df.values.astype(float)
        return _split(X, y, A) + (list(df.columns),)
    return _cache_or_build("compas", _build)


def load_adult():
    def _build():
        cols = ["age", "workclass", "fnlwgt", "education", "education-num",
                "marital-status", "occupation", "relationship", "race", "sex",
                "capital-gain", "capital-loss", "hours-per-week",
                "native-country", "income"]
        train_path = os.path.join(DATA_DIR, "adult.data")
        test_path = os.path.join(DATA_DIR, "adult.test")
        df_tr = pd.read_csv(train_path, header=None, names=cols, sep=r",\s*",
                            engine="python", na_values="?")
        df_te = pd.read_csv(test_path, header=None, names=cols, sep=r",\s*",
                            engine="python", na_values="?", skiprows=1)
        df = pd.concat([df_tr, df_te], ignore_index=True).dropna()
        df["income"] = df["income"].astype(str).str.replace(".", "", regex=False).str.strip()
        y = (df["income"] == ">50K").astype(int).values
        A = (df["sex"].astype(str).str.strip().str.lower() == "male").astype(int).values
        df = df.drop(columns=["income", "sex", "fnlwgt", "native-country"])
        for c in df.select_dtypes(include="object").columns:
            df[c] = pd.Categorical(df[c]).codes
        X = df.values.astype(float)
        return _split(X, y, A) + (list(df.columns),)
    return _cache_or_build("adult", _build)


def load_german():
    def _build():
        path = os.path.join(DATA_DIR, "german.data")
        df = pd.read_csv(path, header=None, sep=r"\s+", engine="python")
        sex_col = df[8].astype(str)
        A = sex_col.isin(["A91", "A93", "A94"]).astype(int).values
        y = (df[20].astype(int) == 1).astype(int).values
        df = df.drop(columns=[20, 8])
        for c in df.columns:
            if df[c].dtype == "object":
                df[c] = pd.Categorical(df[c]).codes
        X = df.values.astype(float)
        return _split(X, y, A) + (list(df.columns),)
    return _cache_or_build("german", _build)


def load_acs_income(states=("CA",), n_max=15000):
    def _build():
        try:
            from folktables import ACSDataSource, ACSIncome
        except ImportError:
            raise RuntimeError("folktables not installed; pip install folktables")
        ds = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
        acs = ds.get_data(states=list(states), download=True)
        X, y, group = ACSIncome.df_to_pandas(acs)
        A = (group.values.flatten() == 1).astype(int)
        y = y.values.flatten().astype(int)
        X = X.values.astype(float)
        if n_max and len(y) > n_max:
            r = np.random.default_rng(RNG_SEED)
            idx = r.choice(len(y), size=n_max, replace=False)
            X, y, A = X[idx], y[idx], A[idx]
        feat = list(ACSIncome.features) if hasattr(ACSIncome, "features") else \
            [f"f{i}" for i in range(X.shape[1])]
        return _split(X, y, A) + (feat,)
    return _cache_or_build("acs_income", _build)


def _binarize_image(y_multi, threshold):
    return (y_multi >= threshold).astype(int)


def _synth_sensitive_from_confidence(model_probs):
    top = model_probs.max(axis=1)
    return (top >= np.median(top)).astype(int)


def load_cifar10h():
    def _build():
        path = os.path.join(DATA_DIR, "cifar10h",
                            "cnn_data.csv")
        data = np.genfromtxt(path, delimiter=",")
        y_multi = data[:, 0].astype(int)
        model_probs = data[:, 11:21].astype(float)
        y = _binarize_image(y_multi, threshold=5)
        A = _synth_sensitive_from_confidence(model_probs)
        X = model_probs
        feats = [f"prob_class_{i}" for i in range(10)]
        return _split(X, y, A) + (feats,)
    return _cache_or_build("cifar10h", _build)


def load_imagenet():
    def _build():
        path = os.path.join(DATA_DIR, "imagenet",
                            "imagenet_data.csv")
        data = np.genfromtxt(path, delimiter=",")
        y_multi = data[:, 164].astype(int)
        model_probs = data[:, 148:164].astype(float)
        y = _binarize_image(y_multi, threshold=8)
        A = _synth_sensitive_from_confidence(model_probs)
        X = model_probs
        feats = [f"prob_class_{i}" for i in range(16)]
        return _split(X, y, A) + (feats,)
    return _cache_or_build("imagenet", _build)


DATASET_REGISTRY = {
    "compas": load_compas,
    "adult": load_adult,
    "german": load_german,
    "acs_income": load_acs_income,
    "cifar10h": load_cifar10h,
    "imagenet": load_imagenet,
}


def load_dataset(name):
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset {name}; choose from {list(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name]()
