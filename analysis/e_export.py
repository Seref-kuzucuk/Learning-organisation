"""E. Precompute final RF probability and SHAP for all 5^6 answer combinations -> lookup.json"""
import pickle, json, itertools, numpy as np, pandas as pd, shap
m = pickle.load(open("final_rf.pkl", "rb")); rf, feats = m["model"], m["features"]
grid = np.array(list(itertools.product(range(1, 6), repeat=len(feats))))
X = pd.DataFrame(grid, columns=feats)
p = rf.predict_proba(X)[:, 1]
ex = shap.TreeExplainer(rf); sv = ex.shap_values(X)
sv = sv[:, :, 1] if sv.ndim == 3 else sv[1]
base = float(np.ravel(ex.expected_value)[-1])
assert np.allclose(base + sv.sum(1), p, atol=1e-6)
out = {"features": feats, "base": round(base, 4),
       "note": "index = sum((answer_k - 1) * 5^(n-1-k)) in feature order; p = probability of y=1; s = SHAP per feature",
       "p": [round(float(v), 3) for v in p],
       "s": [[round(float(v), 3) for v in row] for row in sv]}
json.dump(out, open("lookup.json", "w"), separators=(",", ":"))
print(feats, base, p.min(), p.max(), len(p))
