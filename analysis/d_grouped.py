"""D. Keep all 286 rows but put identical rows in the same fold (StratifiedGroupKFold),
compared with naive StratifiedKFold that lets copies straddle train/test."""
import sys, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score
warnings.filterwarnings("ignore")
d = pd.read_excel(sys.argv[1]); X, y = d.drop(columns="y"), d["y"].values
groups = d.astype(str).agg("|".join, axis=1).factorize()[0]
feats = {"all16": list(X.columns), "boruta6": ["x8", "x12", "x13", "x14", "x15", "x16"]}
rows = []
for fname, cols in feats.items():
    for scheme in ["naive", "grouped"]:
        for rep in range(5):
            cv = (StratifiedKFold(5, shuffle=True, random_state=rep) if scheme == "naive"
                  else StratifiedGroupKFold(5, shuffle=True, random_state=rep))
            for tr, te in cv.split(X, y, groups):
                rf = RandomForestClassifier(n_estimators=300, max_features=1, min_samples_leaf=3,
                                            random_state=rep, n_jobs=-1).fit(X.iloc[tr][cols], y[tr])
                p = rf.predict_proba(X.iloc[te][cols])[:, 1]; yh = (p >= .5).astype(int)
                rows.append(dict(features=fname, scheme=scheme, acc=accuracy_score(y[te], yh),
                                 auc=roc_auc_score(y[te], p), kappa=cohen_kappa_score(y[te], yh)))
r = pd.DataFrame(rows).groupby(["features", "scheme"]).agg(["mean", "std"]).round(3)
print(r.to_string()); r.to_csv("d_grouped_summary.csv")
