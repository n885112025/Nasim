"""
Tests for the Student Performance ML project.
"""

import pytest
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_project():
    import importlib, sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import project as p
    return p


# ---------------------------------------------------------------------------
# 1. Data loading tests
# ---------------------------------------------------------------------------

class TestDataLoading:
    def test_load_returns_dataframe(self):
        p = get_project()
        df = p.load_data()
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns_present(self):
        p = get_project()
        df = p.load_data()
        required = {"G1", "G2", "G3", "age", "studytime", "failures", "absences", "sex", "school"}
        assert required.issubset(set(df.columns))

    def test_row_count(self):
        p = get_project()
        df = p.load_data()
        assert len(df) >= 30, "Dataset must have at least 30 rows"

    def test_no_all_null_columns(self):
        p = get_project()
        df = p.load_data()
        for col in df.columns:
            assert df[col].notna().any(), f"Column {col} is entirely null"

    def test_g3_range(self):
        p = get_project()
        df = p.load_data()
        assert df["G3"].between(0, 20).all(), "G3 values must be in [0, 20]"


# ---------------------------------------------------------------------------
# 2. EDA tests
# ---------------------------------------------------------------------------

class TestEDA:
    def test_eda_returns_dict(self):
        p = get_project()
        df = p.load_data()
        result = p.exploratory_data_analysis(df)
        assert isinstance(result, dict)

    def test_eda_keys(self):
        p = get_project()
        df = p.load_data()
        result = p.exploratory_data_analysis(df)
        for key in ("shape", "missing_values", "G3_mean", "G3_std", "pass_rate"):
            assert key in result, f"Missing EDA key: {key}"

    def test_g3_mean_in_valid_range(self):
        p = get_project()
        df = p.load_data()
        result = p.exploratory_data_analysis(df)
        assert 0 <= result["G3_mean"] <= 20

    def test_pass_rate_in_valid_range(self):
        p = get_project()
        df = p.load_data()
        result = p.exploratory_data_analysis(df)
        assert 0.0 <= result["pass_rate"] <= 1.0

    def test_missing_values_non_negative(self):
        p = get_project()
        df = p.load_data()
        result = p.exploratory_data_analysis(df)
        assert result["missing_values"] >= 0


# ---------------------------------------------------------------------------
# 3. Preprocessing tests
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_preprocess_returns_three_objects(self):
        p = get_project()
        df = p.load_data()
        X, y_class, y_reg = p.preprocess(df)
        assert X is not None and y_class is not None and y_reg is not None

    def test_no_missing_after_preprocess(self):
        p = get_project()
        df = p.load_data()
        X, y_class, y_reg = p.preprocess(df)
        assert X.isnull().sum().sum() == 0

    def test_binary_labels(self):
        p = get_project()
        df = p.load_data()
        _, y_class, _ = p.preprocess(df)
        assert set(y_class.unique()).issubset({0, 1})

    def test_g3_not_in_features(self):
        p = get_project()
        df = p.load_data()
        X, _, _ = p.preprocess(df)
        assert "G3" not in X.columns

    def test_pass_col_not_in_features(self):
        p = get_project()
        df = p.load_data()
        X, _, _ = p.preprocess(df)
        assert "pass" not in X.columns


# ---------------------------------------------------------------------------
# 4. Classification tests
# ---------------------------------------------------------------------------

class TestClassification:
    @pytest.fixture(scope="class")
    def clf_results(self):
        p = get_project()
        df = p.load_data()
        X, y_class, _ = p.preprocess(df)
        return p.run_classification(X, y_class)

    def test_returns_dict(self, clf_results):
        assert isinstance(clf_results, dict)

    def test_all_classifiers_present(self, clf_results):
        expected = {"Logistic Regression", "Decision Tree", "Random Forest", "SVM"}
        assert expected.issubset(set(clf_results.keys()))

    def test_accuracy_above_threshold(self, clf_results):
        for name, metrics in clf_results.items():
            assert metrics["test_accuracy"] >= 0.5, (
                f"{name} test accuracy {metrics['test_accuracy']:.4f} is below 0.5"
            )

    def test_cv_accuracy_above_threshold(self, clf_results):
        for name, metrics in clf_results.items():
            assert metrics["cv_mean"] >= 0.5, (
                f"{name} CV accuracy {metrics['cv_mean']:.4f} is below 0.5"
            )

    def test_best_classifier_cv_above_60(self, clf_results):
        best = max(clf_results, key=lambda n: clf_results[n]["cv_mean"])
        assert clf_results[best]["cv_mean"] >= 0.60, (
            f"Best classifier CV accuracy {clf_results[best]['cv_mean']:.4f} is below 0.60"
        )


# ---------------------------------------------------------------------------
# 5. Regression tests
# ---------------------------------------------------------------------------

class TestRegression:
    @pytest.fixture(scope="class")
    def reg_results(self):
        p = get_project()
        df = p.load_data()
        X, _, y_reg = p.preprocess(df)
        return p.run_regression(X, y_reg)

    def test_returns_dict(self, reg_results):
        assert isinstance(reg_results, dict)

    def test_all_regressors_present(self, reg_results):
        expected = {"Linear Regression", "Ridge Regression", "Decision Tree", "Random Forest"}
        assert expected.issubset(set(reg_results.keys()))

    def test_rmse_non_negative(self, reg_results):
        for name, metrics in reg_results.items():
            assert metrics["rmse"] >= 0, f"{name} RMSE is negative"

    def test_rmse_below_10(self, reg_results):
        for name, metrics in reg_results.items():
            assert metrics["rmse"] <= 10, (
                f"{name} RMSE {metrics['rmse']:.4f} seems too large (> 10)"
            )

    def test_best_regressor_rmse_below_5(self, reg_results):
        best = min(reg_results, key=lambda n: reg_results[n]["rmse"])
        assert reg_results[best]["rmse"] <= 5.0, (
            f"Best regressor RMSE {reg_results[best]['rmse']:.4f} is above 5.0"
        )
