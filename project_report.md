# Final Project Deliverable (Text Version)

## Objective
Classify iris flowers into three species (setosa, versicolor, virginica) using sepal and petal measurements.

## Dataset
- Source: scikit-learn Iris dataset (public, 150 rows, 4 numeric features).
- Features: sepal length, sepal width, petal length, petal width.
- Split: 80% training / 20% testing with stratification for balanced evaluation.

## Exploratory Data Analysis
- Class distribution: 50 samples per class (balanced).
- Feature summaries (mean ± std):
  - Sepal length: 5.84 ± 0.83
  - Sepal width: 3.05 ± 0.43
  - Petal length: 3.76 ± 1.76
  - Petal width: 1.20 ± 0.76
- Observations: petal measurements vary more across classes, making them strong discriminators.

## Machine Learning Task
Multi-class classification (3 classes).

## Methods
- Preprocessing: feature standardization (zero mean, unit variance).
- Model: logistic regression (multinomial), max_iter=200, random_state=42.

## Results and Interpretation
- Test accuracy: ~0.97 on the held-out set (varies slightly with random seed but exceeds 0.85 in tests).
- Classification report (example run):

```
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      0.90      0.95        10
   virginica       0.91      1.00      0.95        10

    accuracy                           0.97        30
   macro avg       0.97      0.97      0.97        30
weighted avg       0.97      0.97      0.97        30
```

- Limitations: small dataset; performance may vary with different seeds or without stratified splits. Further work could compare other algorithms (SVM, Random Forest) or add cross-validation.

## Reproducibility
- Fixed random seed (42) in the default run.
- `pytest` suite checks accuracy threshold (>0.85) and verifies EDA outputs.
