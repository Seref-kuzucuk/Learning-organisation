"""B. Leakage-free re-analysis on de-duplicated data.
Repeated stratified 5-fold outer CV (10 repeats). In each outer training fold:
Boruta feature selection -> inner 5-fold grid search for RF hyperparameters -> fit,
then evaluate on the untouched outer test fold. Compared against L2 logistic regression
and against reduced predictor sets."""
import sys, json, warnings
import numpy as np, pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score,
                             cohen_kappa_score, brier_score_loss, precision_score, recall_score)
warnings.filterwarnings("ignore")
np.int = int; np.float = float; np.bool = bool  # BorutaPy compatibility

SEED = 2026
d = pd.read_excel(sys.argv[1]).drop_duplicates().reset_index(drop=True)
y = d["y"].values
ALL = [f"x{i}" for i in range(1, 17)]
CORE = ["x12", "x13", "x14", "x15", "x16"]          # items closest to the outcome construct
NOMINAL = ["x4", "x11"]                              # sector, platform type
sets = {"all16": ALL, "no_x12_16": [c for c in ALL if c not in CORE], "only_x12_16": CORE}

def metrics(yt, p):
    yh = (p >= 0.5).astype(int)
    return dict(acc=accuracy_score(yt, yh), bal_acc=balanced_accuracy_score(yt, yh),
                auc=roc_auc_score(yt, p), f1=f1_score(yt, yh), precision=precision_score(yt, yh),
                recall=recall_score(yt, yh), kappa=cohen_kappa_score(yt, yh), brier=brier_score_loss(yt, p))

grid = {"max_features": [1, 2, "sqrt"], "min_samples_leaf": [1, 5]}
outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
rows, oof, sel_counts, best_params = [], {}, {c: 0 for c in ALL}, []

ONLY = sys.argv[2]
for name, cols in sets.items():
    if name != ONLY: continue
    X = d[cols]
    oof[name] = {"rf": np.zeros((3, len(d))), "lr": np.zeros((3, len(d)))}
    for k, (tr, te) in enumerate(outer.split(X, y)):
        rep = k // 5
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        # --- RF with Boruta inside the training fold (only for the full set) ---
        keep = cols
        if name == "all16":
            b = BorutaPy(RandomForestClassifier(n_jobs=-1, max_depth=5, random_state=SEED),
                         n_estimators=150, max_iter=50, random_state=SEED + k, verbose=0)
            b.fit(Xtr.values, y[tr])
            keep = [c for c, s, t in zip(cols, b.support_, b.support_weak_) if s or t] or cols
            for c in keep: sel_counts[c] += 1
        n = len(keep)
        g = {"max_features": [m for m in grid["max_features"] if m == "sqrt" or m <= n],
             "min_samples_leaf": grid["min_samples_leaf"]}
        gs = GridSearchCV(RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1), g,
                          cv=StratifiedKFold(5, shuffle=True, random_state=SEED + k), scoring="roc_auc")
        gs.fit(Xtr[keep], y[tr])
        if name == "all16": best_params.append(gs.best_params_)
        p_rf = gs.predict_proba(Xte[keep])[:, 1]
        # --- L2 logistic regression baseline (nominal vars one-hot) ---
        nom = [c for c in NOMINAL if c in cols]
        pre = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore"), nom)], remainder=StandardScaler())
        lr = make_pipeline(pre, LogisticRegressionCV(Cs=6, cv=5, scoring="roc_auc", max_iter=5000))
        lr.fit(Xtr, y[tr]); p_lr = lr.predict_proba(Xte)[:, 1]
        oof[name]["rf"][rep, te] = p_rf; oof[name]["lr"][rep, te] = p_lr
        rows.append({"set": name, "model": "RF", "fold": k, **metrics(y[te], p_rf)})
        rows.append({"set": name, "model": "LR", "fold": k, **metrics(y[te], p_lr)})
    print("done", name, flush=True)

r = pd.DataFrame(rows); r.to_csv(f"b_folds_{ONLY}.csv", index=False)
summ = r.groupby(["set", "model"]).agg(["mean", "std"]).drop(columns="fold").round(3)
print(summ.to_string())
summ.to_csv(f"b_summary_{ONLY}.csv")
pd.Series(sel_counts).div(15).sort_values(ascending=False).to_csv("b_boruta_selection_freq.csv")
print(pd.Series(sel_counts).div(15).sort_values(ascending=False))
print(pd.DataFrame(best_params).value_counts().head()) if best_params else None
np.savez(f"b_oof_{ONLY}.npz", y=y, **{f"{s}_{m}": oof[s][m] for s in oof for m in oof[s]})
