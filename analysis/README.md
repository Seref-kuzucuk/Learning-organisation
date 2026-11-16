# Analysis code — BD-LOC manuscript (revised)

Code for the leakage-free re-analysis reported in the revised manuscript
"Modeling Big Data–Driven Learning Organizations Using Machine Learning".
The survey data are not included; they are available from the corresponding author on request.

Requirements: Python 3, pandas, numpy, scikit-learn 1.9.1, boruta (BorutaPy), shap, matplotlib, openpyxl.
Random seed: 2026.

| Script | Purpose | Usage |
| --- | --- | --- |
| `a_leakage.py` | Simulates the original 228/58 split (200 splits) and measures how many test records have an exact copy in training | `python a_leakage.py DATA.xlsx` |
| `b_nested.py` | Nested CV on the 222 unique records: Boruta + RF tuning inside each outer training fold; L2 logistic regression baseline | `python b_nested.py DATA.xlsx all16` (also `no_x12_16`, `only_x12_16`) |
| `c_final_shap.py` | Final model on all unique records (Boruta, tuned RF, 1,000 trees) and SHAP summary plot; saves `final_rf.pkl` | `python c_final_shap.py DATA.xlsx` |
| `d_grouped.py` | All 286 records: naive vs. grouped CV (identical records kept in the same fold) | `python d_grouped.py DATA.xlsx` |
| `e_export.py` | Precomputes probability and SHAP values for all 15,625 answer combinations into `lookup.json` used by the web tool | `python e_export.py` (after `c_final_shap.py`) |

Data file: one sheet with columns `x1`–`x16` (predictors Q1–Q16) and `y` (1 = total scale score ≥ 88).
