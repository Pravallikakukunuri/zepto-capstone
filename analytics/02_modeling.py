"""
02_modeling.py — Part B: Predictive modeling, continuing from the same data.

This script NEVER calls sns.load_dataset(). It reads titanic.csv (the raw
snapshot produced once by 01_eda.py) and does its own preprocessing from
there, independent of Part A's EDA-stage cleaning.

Requires: 01_eda.py must have been run at least once already.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(OUT_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)


def savefig(name):
    path = os.path.join(CHART_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  saved chart -> charts/{name}")


print("=" * 78)
print("STEP 0: Reading titanic.csv (raw snapshot from 01_eda.py) — no network call here")
csv_path = os.path.join(OUT_DIR, "titanic.csv")
df = pd.read_csv(csv_path)
print(f"  Loaded shape: {df.shape}")

NUMERIC_FEATURES = ["pclass", "age", "sibsp", "parch", "fare"]
CATEGORICAL_FEATURES = ["sex", "embarked"]
TARGET = "survived"

df = df.dropna(subset=[TARGET])
X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET].astype(int)

print("\n" + "=" * 78)
print("STEP 1: Stratified train/test split")
print("-" * 78)
print(f"Class balance in full data: \n{y.value_counts(normalize=True).round(3)}")
print("Justification: survived is imbalanced (roughly 60/40, not 50/50). A plain "
      "random split risks train/test folds with noticeably different survival "
      "rates purely by chance, which would bias evaluation. Stratifying on "
      "`survived` forces both folds to preserve the same class ratio as the "
      "full dataset, making test-set metrics a fairer estimate of real "
      "performance.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Train class balance:\n{y_train.value_counts(normalize=True).round(3)}")
print(f"Test class balance:\n{y_test.value_counts(normalize=True).round(3)}")

print("\n" + "=" * 78)
print("STEP 2: Preprocessing pipeline (ColumnTransformer, fit on train only)")
print("-" * 78)
print("Chosen strategy: median-impute numeric columns, most-frequent-impute + "
      "one-hot-encode categorical columns, then StandardScaler the numeric "
      "columns. All wrapped inside a scikit-learn Pipeline, so .fit() only "
      "ever touches X_train, and .transform()/.predict() on X_test never "
      "refits anything -- structurally preventing leakage.")

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, NUMERIC_FEATURES),
    ("cat", categorical_transformer, CATEGORICAL_FEATURES),
])

print("\n" + "=" * 78)
print("STEP 3: Training Logistic Regression, Decision Tree, Random Forest")
print("-" * 78)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200),
}

fitted_pipelines = {}
for name, clf in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(X_train, y_train)
    fitted_pipelines[name] = pipe
    print(f"  trained: {name}")

dt_pipe = fitted_pipelines["Decision Tree"]
feature_names = dt_pipe.named_steps["preprocessor"].get_feature_names_out()
plt.figure(figsize=(22, 10))
plot_tree(
    dt_pipe.named_steps["classifier"],
    feature_names=feature_names,
    class_names=["Not Survived", "Survived"],
    filled=True, max_depth=3, fontsize=8,
)
plt.title("Decision Tree (top 3 levels shown for legibility; full tree used for prediction)")
savefig("decision_tree.png")

print("\n" + "=" * 78)
print("STEP 4: Evaluation — all three classifiers")
print("-" * 78)


def evaluate(name, pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_proba": y_proba,
    }


eval_results = [evaluate(name, pipe, X_test, y_test) for name, pipe in fitted_pipelines.items()]
comparison_df = pd.DataFrame(eval_results)[["model", "accuracy", "precision", "recall", "f1", "auc"]]
print("\nClassifier comparison table:")
print(comparison_df.round(3).to_string(index=False))

for r in eval_results:
    print(f"\nConfusion matrix — {r['model']}:\n{r['confusion_matrix']}")

plt.figure(figsize=(6, 6))
for r in eval_results:
    fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
    plt.plot(fpr, tpr, label=f"{r['model']} (AUC={r['auc']:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — all 3 classifiers")
plt.legend()
savefig("roc_curves.png")

print("\n" + "=" * 78)
print("STEP 5: Imbalance handling comparison (Random Forest)")
print("-" * 78)
print(f"Class balance in y_train:\n{y_train.value_counts(normalize=True).round(3)}")

baseline_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42, n_estimators=200)),
])
baseline_pipe.fit(X_train, y_train)

balanced_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42, n_estimators=200, class_weight="balanced")),
])
balanced_pipe.fit(X_train, y_train)

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    smote_pipe = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", RandomForestClassifier(random_state=42, n_estimators=200)),
    ])
    smote_pipe.fit(X_train, y_train)
    smote_available = True
except ImportError:
    print("  [imbalanced-learn not installed -- run: pip install imbalanced-learn]")
    smote_pipe = None
    smote_available = False

imbalance_variants = {"Baseline (no handling)": baseline_pipe, "class_weight=balanced": balanced_pipe}
if smote_available:
    imbalance_variants["SMOTE (train fold only)"] = smote_pipe

imbalance_rows = []
for name, pipe in imbalance_variants.items():
    y_pred = pipe.predict(X_test)
    imbalance_rows.append({
        "variant": name,
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    })
imbalance_df = pd.DataFrame(imbalance_rows)
print("\nImbalance-handling comparison:")
print(imbalance_df.round(3).to_string(index=False))

best_variant = imbalance_df.sort_values("f1", ascending=False).iloc[0]
print(f"\nConclusion: '{best_variant['variant']}' gave the best F1 "
      f"({best_variant['f1']:.3f}) among the three variants tested.")

print("\n" + "=" * 78)
print("STEP 6: GridSearchCV hyperparameter tuning (Random Forest)")
print("-" * 78)

rf_grid_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(oob_score=True, bootstrap=True, random_state=42)),
])
param_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [None, 5, 10],
    "classifier__max_features": ["sqrt", "log2"],
}
grid_search = GridSearchCV(rf_grid_pipe, param_grid, cv=5, scoring="f1", n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf_pipe = grid_search.best_estimator_
oob = best_rf_pipe.named_steps["classifier"].oob_score_
print(f"Best params: {grid_search.best_params_}")
print(f"OOB score of best Random Forest: {oob:.3f}")

print("\n" + "=" * 78)
print("STEP 7: Regression side-task — predicting fare")
print("-" * 78)

reg_numeric = ["pclass", "age", "sibsp", "parch", "survived"]
reg_categorical = ["sex", "embarked"]
df_reg = df.dropna(subset=["fare"])
X_reg = df_reg[reg_numeric + reg_categorical]
y_reg = df_reg["fare"]

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

reg_preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), reg_numeric),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), reg_categorical),
])
reg_pipe = Pipeline([("preprocessor", reg_preprocessor), ("regressor", LinearRegression())])
reg_pipe.fit(X_reg_train, y_reg_train)
y_reg_pred = reg_pipe.predict(X_reg_test)

mae = mean_absolute_error(y_reg_test, y_reg_pred)
rmse = mean_squared_error(y_reg_test, y_reg_pred) ** 0.5
r2 = r2_score(y_reg_test, y_reg_pred)
n_obs, n_pred = X_reg_test.shape[0], X_reg_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n_obs - 1) / (n_obs - n_pred - 1)

print(f"MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.3f}  Adjusted R2={adj_r2:.3f}")

residuals = y_reg_test - y_reg_pred
plt.figure(figsize=(6, 4))
plt.scatter(y_reg_pred, residuals, alpha=0.5)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted fare")
plt.ylabel("Residual")
plt.title("Residual plot — fare regression")
savefig("regression_residuals.png")

resid_corr = np.corrcoef(y_reg_pred, np.abs(residuals))[0, 1]
hetero = resid_corr > 0.3
print(f"Correlation between predicted fare and |residuals|: {resid_corr:.3f}")
print(f"Conclusion: the residual plot "
      f"{'DOES show heteroscedasticity (residual spread grows with predicted fare)' if hetero else 'does NOT show strong heteroscedasticity (spread looks roughly constant)'}.")

print("\n" + "=" * 78)
print("STEP 8: Final model comparison")
print("-" * 78)
print("\nClassification models:")
print(comparison_df.round(3).to_string(index=False))
print("\nRegression model (separate metric scale -- not directly comparable to the above):")
reg_table = pd.DataFrame([{"model": "Linear Regression (fare)", "MAE": mae, "RMSE": rmse,
                           "R2": r2, "Adjusted_R2": adj_r2}])
print(reg_table.round(3).to_string(index=False))

best_row = comparison_df.sort_values("f1", ascending=False).iloc[0]
print(f"""
Recommendation (auto-drafted -- review and tailor before submitting):
Deploy the {best_row['model']} model. It has the highest F1 score
({best_row['f1']:.3f}) among the three classifiers, balancing precision
({best_row['precision']:.3f}) and recall ({best_row['recall']:.3f}), with an
AUC of {best_row['auc']:.3f} indicating strong ranking ability between
survivors and non-survivors. The Random Forest's GridSearchCV-tuned variant
(OOB score {oob:.3f}) is a close comparison point if slightly more
accuracy/robustness is wanted at the cost of interpretability.
""")

print("=" * 78)
print("STEP 9: Saving the best fitted pipeline")
print("-" * 78)

pipeline_path = os.path.join(OUT_DIR, "best_pipeline.joblib")
joblib.dump(best_rf_pipe, pipeline_path)
print(f"Saved -> {pipeline_path}")

reloaded_pipe = joblib.load(pipeline_path)
sample_raw = X_test.iloc[[0]]
pred = reloaded_pipe.predict(sample_raw)
print(f"Reload check -- prediction on one raw test row: {pred} (actual: {y_test.iloc[0]})")

print("\n" + "=" * 78)
print("02_modeling.py complete.")