# Titanic Survival Prediction — Supervised Learning Mini Project

## 1. Problem Statement

The objective of this mini project is to build a supervised machine learning model that predicts whether a passenger survived the Titanic disaster based on passenger-related characteristics such as passenger class, sex, age, number of siblings/spouses aboard, number of parents/children aboard, fare and port of embarkation.

This project demonstrates statistical analysis and machine learning concepts using Python.

## 2. Dataset Description

**Dataset:** Titanic passenger dataset  
**Source:** Seaborn public Titanic dataset  
**Target variable:** `survived`

The dataset contains passenger-level observations with demographic and travel-related variables.

Important variables used:

| Variable | Description |
|---|---|
| `pclass` | Passenger class |
| `sex` | Passenger sex |
| `age` | Passenger age |
| `sibsp` | Number of siblings/spouses aboard |
| `parch` | Number of parents/children aboard |
| `fare` | Passenger fare |
| `embarked` | Port of embarkation |
| `survived` | Target: 0 = did not survive, 1 = survived |

## 3. Data Preprocessing

The project performs:

- Missing-value detection
- Median imputation for numerical variables
- Most-frequent imputation for categorical variables
- One-hot encoding of categorical variables
- Standardization of numerical variables
- Train-test split using stratification
- IQR-based outlier identification

Extreme values are not automatically removed because they may represent genuine observations.

## 4. Exploratory Data Analysis

The project generates:

- Passenger class distribution
- Survival distribution
- Survival by sex
- Survival by passenger class
- Age distribution
- Fare boxplot
- Correlation heatmap
- ROC curves
- Confusion matrices

## 5. Statistical Analysis

Two statistical analyses are included:

1. **Chi-square test** between sex and survival.
2. **Welch's t-test** comparing fare distributions of survivors and non-survivors.

The script also calculates descriptive statistics and correlations.

## 6. Machine Learning Models

Two supervised classification algorithms are implemented:

### Logistic Regression
A linear classification model used as a baseline.

### Random Forest
An ensemble learning algorithm that combines multiple decision trees.

## 7. Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Classification report

The script saves a model comparison table to:

`outputs/model_comparison.csv`

## 8. How to Run

### Step 1 — Clone/download the repository

```bash
git clone <your-github-repository-url>
cd titanic-supervised-learning-mini-project
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Run the project

```bash
python main.py
```

The dataset is loaded through Seaborn. Internet access may be required the first time the dataset is downloaded.

## 9. Project Structure

```text
titanic-supervised-learning-mini-project/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── outputs/
    ├── descriptive_statistics.csv
    ├── model_comparison.csv
    ├── 01_passenger_class_distribution.png
    ├── 02_survival_distribution.png
    ├── 03_survival_by_sex.png
    ├── 04_survival_by_class.png
    ├── 05_age_distribution.png
    ├── 06_fare_boxplot.png
    ├── 07_correlation_heatmap.png
    ├── 08_roc_curves.png
    ├── logistic_regression_confusion_matrix.png
    └── random_forest_confusion_matrix.png
```

## 10. Conclusion

The project demonstrates how secondary data can be analyzed using statistical techniques and supervised machine learning. Passenger characteristics can be used to build predictive classification models. The final model performance should be discussed using the metrics generated when the code is executed.

### Limitations

- The dataset represents a historical event and may not generalize to modern populations.
- Some variables contain missing information.
- Model performance depends on the train-test split and selected features.
- Correlation does not by itself establish causation.
- The project focuses on prediction rather than causal inference.

## Academic Note

This is an educational mini-project. The generated metric values should be taken from the actual execution of `main.py` rather than copied without verification.
