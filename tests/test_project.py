import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from project import exploratory_data_analysis, load_dataset, run_project, train_and_evaluate


def test_run_project_returns_high_accuracy():
    result = run_project()
    assert result.accuracy > 0.85
    assert "precision" in result.report
    assert len(result.class_distribution) == 3


def test_exploratory_data_analysis_produces_stats():
    features, labels, feature_names, target_names = load_dataset()
    stats, distribution = exploratory_data_analysis(features, labels, feature_names, target_names)

    assert set(stats.keys()) == set(feature_names)
    for feature in feature_names:
        assert {"mean", "std", "min", "max"}.issubset(stats[feature].keys())
    assert sum(distribution.values()) == len(labels)


def test_train_and_evaluate_reproducible():
    features, labels, _, _ = load_dataset()
    _, accuracy_first, _ = train_and_evaluate(features, labels, random_state=1)
    _, accuracy_second, _ = train_and_evaluate(features, labels, random_state=1)
    assert accuracy_first == accuracy_second
