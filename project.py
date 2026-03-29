from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ProjectResult:
    """Container for the main project outputs."""

    accuracy: float
    class_distribution: Dict[str, int]
    feature_stats: Dict[str, Dict[str, float]]
    report: str
    model: Pipeline


def load_dataset() -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Load the Iris dataset as the project data source."""
    data = load_iris()
    return data.data, data.target, list(data.feature_names), list(data.target_names)


def exploratory_data_analysis(
    features: np.ndarray, labels: np.ndarray, feature_names: List[str], target_names: List[str]
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, int]]:
    """Compute simple EDA summaries for the dataset."""
    feature_stats: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(feature_names):
        column = features[:, idx]
        feature_stats[name] = {
            "mean": float(np.mean(column)),
            "std": float(np.std(column, ddof=1)),
            "min": float(np.min(column)),
            "max": float(np.max(column)),
        }

    class_distribution: Dict[str, int] = {name: 0 for name in target_names}
    for label in labels:
        class_distribution[target_names[int(label)]] += 1

    return feature_stats, class_distribution


def train_and_evaluate(
    features: np.ndarray, labels: np.ndarray, random_state: int = 42
) -> Tuple[Pipeline, float, str]:
    """Train a simple classifier and return the model, accuracy, and report."""
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=random_state, stratify=labels
    )

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, random_state=random_state)),
        ]
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=["setosa", "versicolor", "virginica"])
    return pipeline, accuracy, report


def run_project(random_state: int = 42) -> ProjectResult:
    """Execute the end-to-end project flow."""
    features, labels, feature_names, target_names = load_dataset()
    feature_stats, class_distribution = exploratory_data_analysis(features, labels, feature_names, target_names)
    model, accuracy, report = train_and_evaluate(features, labels, random_state=random_state)

    return ProjectResult(
        accuracy=accuracy,
        class_distribution=class_distribution,
        feature_stats=feature_stats,
        report=report,
        model=model,
    )


def format_summary(result: ProjectResult) -> str:
    """Create a human-readable project summary."""
    lines = [
        "HCAI - Final Project Summary",
        "",
        "Objective: Multi-class classification of iris flowers using sepal/petal measurements.",
        f"Test Accuracy: {result.accuracy:.3f}",
        "",
        "Class distribution:",
    ]
    for label, count in result.class_distribution.items():
        lines.append(f"  - {label}: {count}")

    lines.append("")
    lines.append("Feature statistics (mean ± std):")
    for feature, stats in result.feature_stats.items():
        lines.append(f"  - {feature}: {stats['mean']:.2f} ± {stats['std']:.2f} (min {stats['min']:.2f}, max {stats['max']:.2f})")

    lines.append("")
    lines.append("Classification report:")
    lines.append(result.report)
    return "\n".join(lines)


if __name__ == "__main__":
    project_result = run_project()
    print(format_summary(project_result))
