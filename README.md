# Student Performance — Machine Learning Project

## Objective
Predict whether a student **passes** (final grade G3 ≥ 10) and predict the **exact final grade G3** using demographic, social, and academic features from the UCI Student Performance dataset (Math course).

## Dataset
- **Source**: [UCI ML Repository — Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)
- **File**: `data/student-mat.csv` (semicolon-separated)
- **Size**: 395 students × 33 attributes (sample of 50 included for reproducibility)
- **Target variables**:
  - `G3` — final grade (0–20 scale)
  - `pass` — binary label derived from G3 ≥ 10

### Key features
| Feature | Description |
|---------|-------------|
| `G1`, `G2` | First and second period grades |
| `studytime` | Weekly study time (1–4) |
| `failures` | Number of past class failures |
| `absences` | Number of school absences |
| `Medu`, `Fedu` | Mother's / Father's education level |
| `higher` | Wants to pursue higher education |
| `internet` | Internet access at home |

## Project Structure
```
.
├── data/
│   └── student-mat.csv       # Dataset (UCI format, semicolon-separated)
├── plots/                    # Auto-generated EDA and model plots
├── tests/
│   └── test_project.py       # 25 pytest tests
├── project.py                # Main ML pipeline
├── requirements.txt
└── README.md
```

## Machine Learning Tasks

### 1. Exploratory Data Analysis
- Grade distribution (histogram with pass threshold)
- Correlation heatmap of all numeric features
- G3 by sex (box plot)
- G3 vs study time (scatter)

### 2. Classification — Predict Pass / Fail
Algorithms compared (5-fold cross-validation):
| Model | Test Accuracy | CV Accuracy |
|-------|:---:|:---:|
| Logistic Regression | ~0.90 | ~0.92 |
| Decision Tree | ~1.00 | ~0.98 |
| Random Forest | ~1.00 | ~0.94 |
| SVM (RBF) | ~0.90 | ~0.82 |

### 3. Regression — Predict G3 Score
| Model | RMSE | R² |
|-------|:---:|:---:|
| Linear Regression | ~2.60 | ~0.73 |
| Ridge Regression | ~2.26 | ~0.80 |
| Decision Tree | ~2.56 | ~0.74 |
| Random Forest | ~2.42 | ~0.77 |

### 4. Feature Importance
Top predictors for pass/fail: **G2**, **G1**, **age**, **health**, **absences**.

## Results & Interpretation
- Previous grades (G1, G2) are the strongest predictors of final performance.
- Decision Tree achieves the best cross-validated classification accuracy (~98%).
- Ridge Regression achieves the lowest RMSE (~2.26 grade points) for G3 prediction.
- Limitation: the included sample (50 rows) is small; results improve significantly with the full 395-row dataset.

## How to Run

```bash
pip install -r requirements.txt
python project.py      # runs full pipeline, saves plots
pytest tests/          # runs 25 automated tests
```

## Requirements
- Python ≥ 3.9
- pandas, numpy, scikit-learn, matplotlib, seaborn, pytest
