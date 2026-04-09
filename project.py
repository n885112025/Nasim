"""
Student Performance ML Project
===============================
Dataset : UCI Student Performance (Math course)
Objective: Predict whether a student passes (G3 >= 10) — classification —
           and predict the exact final grade G3 — regression.
           Compare multiple machine learning algorithms on both tasks.
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for environments without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "student-mat.csv")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the UCI Student Performance dataset (semicolon-separated)."""
    df = pd.read_csv(path, sep=";")
    return df


# ---------------------------------------------------------------------------
# 2. Exploratory Data Analysis
# ---------------------------------------------------------------------------

def exploratory_data_analysis(df: pd.DataFrame) -> dict:
    """
    Perform EDA and return a summary dict.
    Also saves plots to the plots/ directory.
    """
    summary = {}

    # Basic statistics
    summary["shape"] = df.shape
    summary["missing_values"] = int(df.isnull().sum().sum())
    summary["G3_mean"] = float(df["G3"].mean())
    summary["G3_std"] = float(df["G3"].std())
    summary["G3_min"] = float(df["G3"].min())
    summary["G3_max"] = float(df["G3"].max())
    summary["pass_rate"] = float((df["G3"] >= 10).mean())

    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"Shape            : {summary['shape']}")
    print(f"Missing values   : {summary['missing_values']}")
    print(f"G3 mean ± std    : {summary['G3_mean']:.2f} ± {summary['G3_std']:.2f}")
    print(f"G3 range         : [{summary['G3_min']}, {summary['G3_max']}]")
    print(f"Pass rate (G3≥10): {summary['pass_rate']:.1%}")
    print()
    print(df.describe())

    # --- Plot 1: G3 distribution ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["G3"], bins=range(0, 21), color="steelblue", edgecolor="white")
    ax.axvline(10, color="red", linestyle="--", label="Pass threshold (10)")
    ax.set_xlabel("Final Grade (G3)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Final Grade (G3)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "g3_distribution.png"), dpi=100)
    plt.close()

    # --- Plot 2: Correlation heatmap (numeric features) ---
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=100)
    plt.close()

    # --- Plot 3: G3 by sex ---
    fig, ax = plt.subplots(figsize=(6, 4))
    df.boxplot(column="G3", by="sex", ax=ax)
    ax.set_title("Final Grade (G3) by Sex")
    ax.set_xlabel("Sex")
    ax.set_ylabel("G3")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "g3_by_sex.png"), dpi=100)
    plt.close()

    # --- Plot 4: G3 vs study time scatter ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["studytime"], df["G3"], alpha=0.6, color="teal")
    ax.set_xlabel("Study Time (1=<2h, 2=2-5h, 3=5-10h, 4=>10h)")
    ax.set_ylabel("Final Grade (G3)")
    ax.set_title("G3 vs Study Time")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "g3_vs_studytime.png"), dpi=100)
    plt.close()

    print(f"\nPlots saved to: {PLOTS_DIR}/")
    return summary


# ---------------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame):
    """
    Encode categoricals, create pass/fail label, split features/targets.
    Returns X, y_class (pass/fail), y_reg (G3 score).
    """
    df = df.copy()

    # Binary target: pass = 1 if G3 >= 10, else 0
    df["pass"] = (df["G3"] >= 10).astype(int)

    # Encode binary yes/no columns
    yes_no_cols = [
        "schoolsup", "famsup", "paid", "activities",
        "nursery", "higher", "internet", "romantic",
    ]
    for col in yes_no_cols:
        df[col] = df[col].map({"yes": 1, "no": 0})

    # Encode remaining categoricals with LabelEncoder
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    feature_cols = [c for c in df.columns if c not in ("G3", "pass")]
    X = df[feature_cols]
    y_class = df["pass"]
    y_reg = df["G3"]

    return X, y_class, y_reg


# ---------------------------------------------------------------------------
# 4. Classification — Predict Pass / Fail
# ---------------------------------------------------------------------------

def run_classification(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Train and evaluate multiple classifiers. Returns accuracy dict.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel="rbf", random_state=42),
    }

    results = {}
    print("=" * 60)
    print("CLASSIFICATION: Predict Pass / Fail (G3 >= 10)")
    print("=" * 60)

    for name, clf in classifiers.items():
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(clf, scaler.transform(X), y, cv=5, scoring="accuracy")
        results[name] = {
            "test_accuracy": acc,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
        }
        print(f"\n{name}")
        print(f"  Test Accuracy : {acc:.4f}")
        print(f"  CV Accuracy   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))

    # --- Plot: Classifier comparison ---
    names = list(results.keys())
    accs = [results[n]["test_accuracy"] for n in names]
    cv_means = [results[n]["cv_mean"] for n in names]
    cv_stds = [results[n]["cv_std"] for n in names]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x - 0.2, accs, 0.35, label="Test Accuracy", color="steelblue")
    ax.bar(x + 0.2, cv_means, 0.35, yerr=cv_stds, label="CV Accuracy (5-fold)",
           color="coral", capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Classifier Comparison — Pass/Fail Prediction")
    ax.legend()
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "classifier_comparison.png"), dpi=100)
    plt.close()

    # --- Confusion matrix for best classifier ---
    best_name = max(results, key=lambda n: results[n]["cv_mean"])
    best_clf = classifiers[best_name]
    best_clf.fit(X_train_s, y_train)
    y_pred_best = best_clf.predict(X_test_s)
    cm = confusion_matrix(y_test, y_pred_best)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=100)
    plt.close()

    print(f"\nBest classifier (CV): {best_name}")
    return results


# ---------------------------------------------------------------------------
# 5. Regression — Predict Final Grade G3
# ---------------------------------------------------------------------------

def run_regression(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Train and evaluate multiple regressors. Returns RMSE dict.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    regressors = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    }

    results = {}
    print("=" * 60)
    print("REGRESSION: Predict Final Grade G3")
    print("=" * 60)

    for name, reg in regressors.items():
        reg.fit(X_train_s, y_train)
        y_pred = reg.predict(X_test_s)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        results[name] = {"rmse": rmse, "mae": mae, "r2": r2}
        print(f"\n{name}")
        print(f"  RMSE : {rmse:.4f}")
        print(f"  MAE  : {mae:.4f}")
        print(f"  R²   : {r2:.4f}")

    # --- Plot: Regressor comparison ---
    names = list(results.keys())
    rmses = [results[n]["rmse"] for n in names]
    r2s = [results[n]["r2"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(names, rmses, color="steelblue")
    axes[0].set_title("RMSE by Regressor")
    axes[0].set_ylabel("RMSE (lower is better)")
    axes[0].set_xlabel("Regressor")
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=15, ha="right")
    axes[1].bar(names, r2s, color="coral")
    axes[1].set_title("R² Score by Regressor")
    axes[1].set_ylabel("R² (higher is better)")
    axes[1].set_xlabel("Regressor")
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=15, ha="right")
    plt.suptitle("Regressor Comparison — G3 Prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "regressor_comparison.png"), dpi=100)
    plt.close()

    # --- Predicted vs actual for best regressor ---
    best_name = min(results, key=lambda n: results[n]["rmse"])
    best_reg = regressors[best_name]
    best_reg.fit(X_train_s, y_train)
    y_pred_best = best_reg.predict(X_test_s)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, y_pred_best, alpha=0.7, color="teal")
    lims = [min(y_test.min(), y_pred_best.min()) - 1,
            max(y_test.max(), y_pred_best.max()) + 1]
    ax.plot(lims, lims, "r--", label="Perfect prediction")
    ax.set_xlabel("Actual G3")
    ax.set_ylabel("Predicted G3")
    ax.set_title(f"Actual vs Predicted G3 — {best_name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "actual_vs_predicted.png"), dpi=100)
    plt.close()

    print(f"\nBest regressor (RMSE): {best_name}")
    return results


# ---------------------------------------------------------------------------
# 6. Feature Importance
# ---------------------------------------------------------------------------

def feature_importance(X: pd.DataFrame, y_class: pd.Series) -> None:
    """Plot feature importances from a Random Forest classifier."""
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    rf.fit(X_s, y_class)

    importances = pd.Series(rf.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot(kind="bar", color="teal", ax=ax)
    ax.set_title("Top 15 Feature Importances (Random Forest — Pass/Fail)")
    ax.set_ylabel("Importance")
    ax.set_xlabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=100)
    plt.close()
    print("\nTop features for pass/fail prediction:")
    print(importances.to_string())


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main():
    df = load_data()
    eda_summary = exploratory_data_analysis(df)
    X, y_class, y_reg = preprocess(df)
    clf_results = run_classification(X, y_class)
    reg_results = run_regression(X, y_reg)
    feature_importance(X, y_class)

    print("\n" + "=" * 60)
    print("PROJECT SUMMARY")
    print("=" * 60)
    print(f"Dataset shape   : {eda_summary['shape']}")
    print(f"Pass rate       : {eda_summary['pass_rate']:.1%}")
    print(f"G3 mean         : {eda_summary['G3_mean']:.2f}")

    best_clf = max(clf_results, key=lambda n: clf_results[n]["cv_mean"])
    best_reg = min(reg_results, key=lambda n: reg_results[n]["rmse"])
    print(f"Best classifier : {best_clf}  "
          f"(CV acc={clf_results[best_clf]['cv_mean']:.4f})")
    print(f"Best regressor  : {best_reg}  "
          f"(RMSE={reg_results[best_reg]['rmse']:.4f}, "
          f"R²={reg_results[best_reg]['r2']:.4f})")
    print(f"\nAll plots saved in: {PLOTS_DIR}/")

    return eda_summary, clf_results, reg_results


if __name__ == "__main__":
    main()
