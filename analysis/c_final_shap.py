"""C. Final deployable model on de-duplicated data: Boruta on all data, tuned RF (5-fold CV),
SHAP (TreeExplainer) for interpretation. Saves model for the web tool."""
import sys, warnings, pickle
import numpy as np, pandas as pd, shap
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
warnings.filterwarnings("ignore"); np.int = int; np.float = float; np.bool = bool
SEED = 2026
d = pd.read_excel(sys.argv[1]).drop_duplicates().reset_index(drop=True)
X, y = d.drop(columns="y"), d["y"].values
b = BorutaPy(RandomForestClassifier(n_jobs=-1, max_depth=5, random_state=SEED),
             n_estimators="auto", max_iter=200, random_state=SEED).fit(X.values, y)
status = pd.DataFrame({"var": X.columns, "confirmed": b.support_, "tentative": b.support_weak_, "rank": b.ranking_})
print(status.to_string()); status.to_csv("c_boruta_full.csv", index=False)
keep = list(X.columns[b.support_ | b.support_weak_])
gs = GridSearchCV(RandomForestClassifier(n_estimators=1000, random_state=SEED, oob_score=True, n_jobs=-1),
                  {"max_features": [1, 2, 3, "sqrt"], "min_samples_leaf": [1, 3, 5, 10]},
                  cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="roc_auc").fit(X[keep], y)
rf = gs.best_estimator_
print("best", gs.best_params_, "cvAUC", round(gs.best_score_, 3), "OOB acc", round(rf.oob_score_, 3))
sv = shap.TreeExplainer(rf).shap_values(X[keep])
sv = sv[:, :, 1] if sv.ndim == 3 else sv[1]
imp = pd.Series(np.abs(sv).mean(0), index=keep).sort_values(ascending=False)
print(imp.round(4)); imp.to_csv("c_shap_importance.csv")
labels = {"x1": "Q1 Gender", "x2": "Q2 Education", "x3": "Q3 Experience", "x4": "Q4 Sector", "x5": "Q5 Position",
          "x6": "Q6 Firm age", "x7": "Q7 Employees", "x8": "Q8 Big-data knowledge", "x9": "Q9 Big-data use",
          "x10": "Q10 Data-science staff", "x11": "Q11 Data platform", "x12": "Q12 Monitoring of analytics",
          "x13": "Q13 Sharing of results", "x14": "Q14 Advanced technologies", "x15": "Q15 Data security",
          "x16": "Q16 Data-based decisions"}
shap.summary_plot(sv, X[keep].rename(columns=labels), show=False, max_display=len(keep))
plt.tight_layout(); plt.savefig("fig_shap_summary.png", dpi=300); plt.close()
with open("final_rf.pkl", "wb") as f: pickle.dump({"model": rf, "features": keep, "labels": labels}, f)
