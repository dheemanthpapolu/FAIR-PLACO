import os
import sys
import importlib
import numpy as np
import pandas as pd

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)

from data_loaders import DATASET_REGISTRY, load_dataset
from ai_models import make_ai_model
from human_simulators import simulate_humans
from combiner import BinaryMAPCombiner
from cost_function import make_costs
from policy import placo_greedy_budget, estimate_run_data
from dependencies import HUMAN_ACCURACIES, NUM_HUMANS, BUDGET_FRACTION, CSV_DIR


def benchmark_fairness_dataset(name, seed=42):
    Xt, Xe, yt, ye, At, Ae, _ = load_dataset(name)
    ai = make_ai_model("basic", random_state=seed).fit(Xt, yt)
    mp_tr = ai.predict_proba(Xt); mp_te = ai.predict_proba(Xe)
    H_tr = simulate_humans("unbiased", yt, At, mp_tr[:, 1])
    H_te = simulate_humans("unbiased", ye, Ae, mp_te[:, 1])
    comb = BinaryMAPCombiner(num_humans=NUM_HUMANS).fit(H_tr, yt)
    n_te = len(ye)
    costs = make_costs(n_te, NUM_HUMANS, num_classes=2, seed=seed + 7)
    budget = NUM_HUMANS * 2.0 * BUDGET_FRACTION * n_te
    est_y, est_h = estimate_run_data(comb, mp_te, NUM_HUMANS, HUMAN_ACCURACIES)
    subs, csubs = placo_greedy_budget(comb, mp_te, H_te, NUM_HUMANS, costs,
                                      est_y, est_h, HUMAN_ACCURACIES, budget)
    final = comb.combine(mp_te, H_te, subs)
    ai_only = (ai.predict(Xe) == ye).mean()
    h_only = (H_te[:, 0] == ye).mean()
    placo = (final == ye).mean()
    return dict(dataset=name,
                source="fairness-binary",
                ai_only_acc=float(ai_only),
                best_human_acc=float(h_only),
                placo_acc=float(placo),
                avg_subset=float(np.mean([len(s) for s in subs])),
                budget_spent=float(sum(csubs)),
                budget_total=float(budget))


def try_image_benchmarks():
    rows = []
    placo_dir = os.path.abspath(os.path.join(THIS, "..", "PLACO"))
    if not os.path.isdir(placo_dir):
        print("  [skip] PLACO source dir not found.")
        return rows
    saved_path = list(sys.path)
    saved_modules = {}
    for m in ["combiner", "policy", "value_function", "cost_function",
              "estimation_methods", "dependencies", "experiment"]:
        if m in sys.modules:
            saved_modules[m] = sys.modules.pop(m)
    sys.path = [placo_dir] + [p for p in sys.path if p != THIS]
    cwd = os.getcwd()
    try:
        os.chdir(placo_dir)
        try:
            from sklearn.model_selection import train_test_split
            from combiner import MAPOracleCombiner
            from policy import PLACO_greedy
            from estimation_methods import posterior_estimation
            from value_function import value
            from dependencies import accuracies as acc_list, NUM_HUMANS as PNUM
        except Exception as e:
            print(f"  [skip] could not import original PLACO: {e}")
            return rows

        def _load(model_name, ds):
            data_path = os.path.join(placo_dir, f"dataset/{ds}/{model_name}.csv")
            data = np.genfromtxt(data_path, delimiter=",")
            if model_name == "cnn_data":
                yt = data[:, 0].astype(int)
                hc = data[:, 1:11]
                mp = data[:, 11:]
            else:
                yt = data[:, 164].astype(int)
                hc = data[:, 165:181]
                mp = data[:, 148:164]
            return hc, mp, yt

        def _sim_humans(hc, yt, accs, seed=0):
            rrng = np.random.default_rng(seed)
            out = []
            for i, dp in enumerate(hc):
                row = []
                for a in accs:
                    if rrng.random() < a:
                        row.append(int(yt[i]))
                    else:
                        prob = dp.copy(); prob[yt[i]] = 0
                        if prob.sum() == 0:
                            prob = np.ones_like(prob); prob[yt[i]] = 0
                        prob = prob / prob.sum()
                        row.append(int(rrng.choice(len(dp), p=prob)))
                out.append(row)
            return np.array(out)

        def _run_data(mp_te, comb):
            n_cls = mp_te.shape[1]
            est_h = []
            for mpv in mp_te:
                _, est = posterior_estimation(comb.confusion_matrix, mpv)
                est_h.append(est)
            est_y = []
            for i, mpv in enumerate(mp_te):
                vbest, ybest = -1.0, 0
                for j in range(n_cls):
                    v = value(np.ones(PNUM), comb.confusion_matrix, mpv, j, est_h[i])
                    if v > vbest:
                        vbest, ybest = v, j
                est_y.append(ybest)
            hcosts = [[np.random.uniform(0.0001, n_cls) for _ in range(PNUM)]
                      for _ in range(len(mp_te))]
            return est_y, hcosts, est_h
        for ds, mname in [("cifar10h", "cnn_data"), ("imagenet", "imagenet_data")]:
            try:
                print(f"-> {ds}")
                hc, mp, yt = _load(mname, ds)
            except Exception as e:
                print(f"  [skip] {ds}: {e}")
                continue
            if len(yt) > 1500:
                rs = np.random.default_rng(0)
                idx = rs.choice(len(yt), size=1500, replace=False)
                hc, mp, yt = hc[idx], mp[idx], yt[idx]
            yh = _sim_humans(hc, yt, acc_list)
            yh_tr, yh_te, mp_tr, mp_te, yt_tr, yt_te = train_test_split(
                yh, mp, yt, test_size=0.4, random_state=0)
            comb = MAPOracleCombiner()
            comb.fit(mp_tr, yh_tr, yt_tr)
            est_y, h_costs, est_h = _run_data(mp_te, comb)
            subs, csubs = PLACO_greedy(comb, yh_te, None, mp_te,
                                       PNUM, h_costs, est_y, est_h,
                                       num_classes=mp_te.shape[1])
            final = comb.combine(mp_te, yh_te, subs)
            ai_only = (np.argmax(mp_te, axis=1) == yt_te).mean()
            placo_acc = (final == yt_te).mean()
            rows.append(dict(
                dataset=ds,
                source="image-multi-class",
                ai_only_acc=float(ai_only),
                best_human_acc=float((yh_te[:, 0] == yt_te).mean()),
                placo_acc=float(placo_acc),
                avg_subset=float(np.mean([len(s) for s in subs])),
                budget_spent=float(np.mean([sum(c) for c in csubs])),
                budget_total=float(np.nan),
            ))
            print(f"   {rows[-1]}")
    finally:
        os.chdir(cwd)
        sys.path = saved_path
        for m, mod in saved_modules.items():
            sys.modules[m] = mod
    return rows


def main():
    rows = []
    for name in DATASET_REGISTRY:
        try:
            print(f"-> {name}")
            rows.append(benchmark_fairness_dataset(name))
            print(f"   {rows[-1]}")
        except Exception as e:
            print(f"   FAILED: {e}")
    rows += try_image_benchmarks()
    df = pd.DataFrame(rows)
    out = os.path.join(CSV_DIR, "placo_benchmarks.csv")
    df.to_csv(out, index=False)
    print(f"\nWrote {out}\n{df}")


if __name__ == "__main__":
    main()
