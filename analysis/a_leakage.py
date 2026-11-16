"""A. Duplicate leakage: how many test rows have an exact copy in the training set
under the manuscript's 80/20 split design (228/58), over many random splits."""
import sys
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

d = pd.read_excel(sys.argv[1])
X, y = d.drop(columns="y"), d["y"]
keys = X.astype(str).agg("|".join, axis=1)

res = []
for seed in range(200):
    tr, te = train_test_split(np.arange(len(d)), test_size=58, stratify=y, random_state=seed)
    leaked = keys.iloc[te].isin(set(keys.iloc[tr])).values
    rf = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1).fit(X.iloc[tr], y.iloc[tr])
    pred = rf.predict(X.iloc[te])
    yt = y.iloc[te].values
    res.append(dict(
        leaked_n=leaked.sum(),
        acc_all=accuracy_score(yt, pred),
        acc_leaked=accuracy_score(yt[leaked], pred[leaked]) if leaked.any() else np.nan,
        acc_clean=accuracy_score(yt[~leaked], pred[~leaked]),
    ))
r = pd.DataFrame(res)
print(r.describe().round(3))
r.to_csv("a_leakage.csv", index=False)
