"""Student Performance Factors regression project.

Put StudentPerformanceFactors.csv beside this script or in ./data/.
Install: pip install pandas numpy scikit-learn matplotlib joblib
Run: python student_performance_regression_project.py
"""
from pathlib import Path
import joblib
from sklearn.base import clone
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=FutureWarning)
RANDOM_STATE, TARGET, STUDY_HOURS = 42, "Exam_Score", "Hours_Studied"
OUT = Path("outputs")


def find_dataset():
    likely = [Path("StudentPerformanceFactors.csv"), Path("Student_Performance_Factors.csv"),
              Path("data/StudentPerformanceFactors.csv"), Path("data/Student_Performance_Factors.csv")]
    for path in likely:
        if path.exists():
            return path
    for path in list(Path(".").glob("*.csv")) + list(Path("data").glob("*.csv")):
        try:
            cols = pd.read_csv(path, nrows=2).columns.str.strip()
            if TARGET in cols and STUDY_HOURS in cols:
                return path
        except Exception:
            pass
    raise FileNotFoundError(
        "Dataset CSV not found. Download StudentPerformanceFactors.csv from "
        "https://www.kaggle.com/datasets/lainguyn123/student-performance-factors and put it beside this script."
    )


def make_pipeline(X, estimator):
    cat = X.select_dtypes(include=["object", "category", "bool", "str"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("numeric", numeric, num), ("categorical", categorical, cat)])
    return Pipeline([("preprocess", prep), ("model", estimator)])


def main():
    OUT.mkdir(exist_ok=True)
    path = find_dataset()
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    df.columns = df.columns.str.strip()
    print("=" * 68, "\nSTUDENT ACADEMIC PERFORMANCE REGRESSION PROJECT\n", "=" * 68)
    print(f"Dataset: {path.resolve()} | Original shape: {df.shape}")
    if TARGET not in df or STUDY_HOURS not in df:
        raise ValueError(f"Required columns {TARGET!r} and {STUDY_HOURS!r} not found. Columns: {df.columns.tolist()}")

    initial_rows = len(df)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).drop_duplicates().copy()
    id_cols = [c for c in df if c.strip().lower() in {"student_id", "studentid", "id"}]
    df.drop(columns=id_cols, inplace=True, errors="ignore")
    y = df[TARGET].astype(float)
    X = df.drop(columns=[TARGET]).copy()
    X[STUDY_HOURS] = pd.to_numeric(X[STUDY_HOURS], errors="coerce")
    print(f"Rows retained: {len(df)} | rows removed for missing target/duplicates: {initial_rows-len(df)}")
    print(f"Predictor columns: {X.shape[1]} | missing predictor cells to impute: {X.isna().sum().sum()}")
    print(f"Target mean={y.mean():.3f}, SD={y.std():.3f}, range={y.min():.1f}–{y.max():.1f}")

    pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes],
                  "missing_values": [int(df[c].isna().sum()) for c in df.columns]}).to_csv(OUT/"dataset_audit.csv", index=False)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)
    Xtr_simple, Xte_simple = Xtr[[STUDY_HOURS]], Xte[[STUDY_HOURS]]
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    specs = {
        "Simple Linear Regression (study hours only)": (make_pipeline(Xtr_simple, LinearRegression()), {}, True),
        "Lasso Regression (all features)": (make_pipeline(Xtr, Lasso(max_iter=20000, random_state=RANDOM_STATE)),
                                             {"model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0]}, False),
        "Decision Tree Regression (all features)": (make_pipeline(Xtr, DecisionTreeRegressor(random_state=RANDOM_STATE)),
                                                       {"model__max_depth": [None, 5, 10, 20], "model__min_samples_leaf": [1, 5, 10]}, False),
        "Random Forest Regression (all features)": (make_pipeline(Xtr, RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
                                                       {"model__n_estimators": [200], "model__max_depth": [None, 12], "model__min_samples_leaf": [1, 5]}, False),
        "Gradient Boosting Regression (all features)": (make_pipeline(Xtr, GradientBoostingRegressor(random_state=RANDOM_STATE)),
                                                          {"model__n_estimators": [100, 200], "model__learning_rate": [0.03, 0.1], "model__max_depth": [2, 3]}, False),
    }
    rows, fitted = [], {}
    for name, (pipe, params, simple) in specs.items():
        print(f"\nTraining {name}...")
        a, b = (Xtr_simple, Xte_simple) if simple else (Xtr, Xte)
        if params:
            search = GridSearchCV(pipe, params, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1, refit=True)
            search.fit(a, ytr)
            model, cv_mae, best_params = search.best_estimator_, -search.best_score_, search.best_params_
        else:
            model = pipe.fit(a, ytr)
            best_params = {}
            cv_mae = np.nan
        pred = model.predict(b)
        mae = mean_absolute_error(yte, pred)
        rmse = np.sqrt(mean_squared_error(yte, pred))
        r2 = r2_score(yte, pred)
        print(f"Test MAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f}")
        rows.append({"Model": name, "CV_MAE": cv_mae, "Test_MAE": mae, "Test_RMSE": rmse,
                     "Test_R2": r2, "Best_parameters": str(best_params)})
        fitted[name] = (model, pred, b.copy())

    result = pd.DataFrame(rows).sort_values("Test_MAE")
    result.to_csv(OUT/"model_comparison.csv", index=False)
    print("\nMODEL COMPARISON (lower MAE/RMSE is better; higher R² is better)\n", result.to_string(index=False))
    # The best model is selected here by test MAE for a convenient demonstration.
    # For final research reporting, select using training/CV only, then evaluate test once.
    best_name = result.iloc[0]["Model"]
    best_model, best_pred, best_Xte = fitted[best_name]
    pd.DataFrame({"Actual_Exam_Score": yte, "Predicted_Exam_Score": best_pred,
                  "Error": yte.to_numpy()-best_pred}, index=yte.index).sort_index().to_csv(OUT/"best_model_test_predictions.csv", index_label="row_index")

    plt.figure(figsize=(8, 6))
    plt.scatter(yte, best_pred, alpha=0.55)
    lo, hi = min(yte.min(), np.min(best_pred)), max(yte.max(), np.max(best_pred))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Actual Exam Score"); plt.ylabel("Predicted Exam Score")
    plt.title(f"Actual vs Predicted — {best_name}"); plt.tight_layout()
    plt.savefig(OUT/"actual_vs_predicted.png", dpi=160); plt.close()

    chart = result.sort_values("Test_MAE", ascending=True)
    plt.figure(figsize=(10, 6)); plt.barh(chart["Model"], chart["Test_MAE"])
    plt.gca().invert_yaxis(); plt.xlabel("Test MAE (lower is better)")
    plt.title("Model Comparison by Test MAE"); plt.tight_layout()
    plt.savefig(OUT/"model_comparison_mae.png", dpi=160); plt.close()

    # Model-agnostic importance at the original feature level. This measures
    # how much test MAE worsens when one input feature is shuffled; it does not
    # establish causation.
    print("\nCalculating permutation feature importance for the selected model...")
    perm = permutation_importance(
        best_model, best_Xte, yte,
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance = pd.DataFrame({
        "Feature": best_Xte.columns,
        "Importance_MAE_Increase": perm.importances_mean,
        "Importance_Std": perm.importances_std,
    }).sort_values("Importance_MAE_Increase", ascending=False)
    importance.to_csv(OUT / "feature_importance.csv", index=False)

    top = importance.head(15).sort_values("Importance_MAE_Increase", ascending=True)
    plt.figure(figsize=(10, 7))
    plt.barh(top["Feature"], top["Importance_MAE_Increase"], xerr=top["Importance_Std"])
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Increase in MAE when feature is shuffled (higher = more important)")
    plt.title(f"Top 15 Feature Importances — {best_name}")
    plt.tight_layout()
    plt.savefig(OUT / "feature_importance.png", dpi=180)
    plt.close()

    numeric = df.select_dtypes(include=[np.number])
    if TARGET in numeric:
        numeric.corr(numeric_only=True)[TARGET].drop(TARGET).sort_values(key=lambda s: s.abs(), ascending=False).to_csv(OUT/"numeric_feature_correlations.csv", header=["correlation_with_exam_score"])
    # Export a production-ready model for the website.
    # Select using training-set cross-validation, then refit on all labelled rows.
    cv_candidates = result.dropna(subset=["CV_MAE"])
    if cv_candidates.empty:
        raise RuntimeError("No cross-validated model is available for website export.")

    deployment_name = cv_candidates.sort_values("CV_MAE").iloc[0]["Model"]
    deployment_pipeline = clone(fitted[deployment_name][0])
    deployment_pipeline.fit(X, y)

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool", "str"]
    ).columns.tolist()
    numeric_features = [c for c in X.columns if c not in categorical_features]

    category_options = {
        column: sorted(X[column].dropna().astype(str).unique().tolist())
        for column in categorical_features
    }
    numeric_ranges = {}
    for column in numeric_features:
        values = pd.to_numeric(X[column], errors="coerce").dropna()
        if not values.empty:
            numeric_ranges[column] = {
                "min": float(values.min()),
                "max": float(values.max()),
                "median": float(values.median())
            }

    model_bundle = {
        "model": deployment_pipeline,
        "features": X.columns.tolist(),
        "target": TARGET,
        "model_name": deployment_name,
        "category_options": category_options,
        "numeric_ranges": numeric_ranges,
        "training_rows": int(len(X)),
        "random_state": RANDOM_STATE,
    }
    model_path = OUT / "student_performance_model.joblib"
    joblib.dump(model_bundle, model_path)

    import json
    schema = {
        "target": TARGET,
        "features": X.columns.tolist(),
        "categorical_features": category_options,
        "numeric_features": numeric_ranges,
        "model_name": deployment_name
    }
    with open(OUT / "prediction_input_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    print(f"\nBest model by held-out test MAE: {best_name}")
    print(f"Outputs saved to: {OUT.resolve()}")
    print("Reminder: correlation does not prove causation. Simple Linear Regression uses Hours_Studied only; other models use all predictors.")

if __name__ == "__main__":
    main()
