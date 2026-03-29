# HCAI Final Project – Iris Classification

This repository contains a minimal end-to-end project that satisfies the requirements in the attached final project brief:

- **Objective:** classify iris flowers into three species based on sepal and petal measurements.
- **Dataset:** the public Iris dataset (150 samples, 4 numeric features).
- **Exploratory data analysis:** summary statistics and class distribution are computed automatically.
- **Machine learning task:** multi-class classification.
- **Methods:** standardization + logistic regression.
- **Results:** test accuracy > 0.85 with full classification report; reproducible via a fixed random seed.

## Project structure

- `project.py` – main pipeline (data loading, EDA, model training, summary formatting).
- `tests/test_project.py` – focused tests ensuring the pipeline meets the expected accuracy and outputs EDA summaries.
- `requirements.txt` – Python dependencies.

## How to run

```bash
pip install -r requirements.txt
python project.py
```

The script prints the class distribution, feature statistics, and classification report.

## How to test

```bash
pip install -r requirements.txt
pytest
```
