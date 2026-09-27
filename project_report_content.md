# Titanic Survival Prediction — Mini Project Report

## 1. Problem Statement

To predict whether a Titanic passenger survived using supervised machine learning based on passenger class, sex, age, family-related variables, fare and port of embarkation.

## 2. Dataset Description

The project uses the publicly available Titanic passenger dataset provided through Seaborn.

The target variable is `survived`:
- 0 = Did not survive
- 1 = Survived

The selected predictor variables are:
`pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, and `embarked`.

Run `main.py` to obtain the exact dataset shape, descriptive statistics and missing-value counts.

## 3. Data Preprocessing

The data is checked for missing values. Numerical missing values are replaced with the median, while categorical missing values are replaced with the most frequent category.

Categorical variables are converted into numerical form using one-hot encoding. Numerical variables are standardized for Logistic Regression.

The data is divided into training and testing subsets using an 80:20 split with stratification.

Potential outliers are identified using the Interquartile Range (IQR) method.

## 4. Exploratory Data Analysis

The project investigates:
- Distribution of passengers by class
- Overall survival distribution
- Survival according to sex
- Survival according to passenger class
- Age distribution
- Fare outliers
- Correlation among numerical variables

The generated graphs are stored in the `outputs` directory.

## 5. Statistical Analysis

A chi-square test is performed to study the relationship between sex and survival.

Welch's t-test is used to compare fare distributions between survivors and non-survivors.

Descriptive statistics and correlation analysis are also performed.

## 6. Model Building

Two supervised classification algorithms are applied:

1. Logistic Regression
2. Random Forest Classifier

The models use the same preprocessing pipeline.

## 7. Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

The actual results should be copied from `outputs/model_comparison.csv` after running the program.

## 8. Interpretation

The analysis helps identify passenger characteristics that contain predictive information about survival.

The model metrics indicate how well the algorithms classify passengers into survived and not-survived categories.

The confusion matrix provides the number of correct and incorrect classifications.

## 9. Conclusion

The project demonstrates a complete supervised learning workflow starting from secondary data collection and preprocessing through EDA, statistical analysis, model training and evaluation.

The results should be interpreted based on the metrics produced during execution.

## 10. Limitations

1. The Titanic dataset represents a historical event.
2. Missing data may reduce the reliability of some analyses.
3. The selected features do not contain every factor that could influence survival.
4. Model performance can vary with different train-test splits.
5. Predictive association does not prove causal relationships.
