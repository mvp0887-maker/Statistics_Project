"""
Titanic Survival Prediction - Supervised Learning Mini Project

Project requirements covered:
1. Problem Statement
2. Dataset Description
3. Data Preprocessing
4. Exploratory Data Analysis
5. Model Building
6. Model Evaluation
7. Interpretation
8. Conclusion

Dataset: Titanic passenger dataset
Source: Seaborn's public Titanic dataset
Target variable: survived
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("TITANIC SURVIVAL PREDICTION - MINI PROJECT")
print("=" * 70)

print("\n[1] Loading public Titanic dataset...")

# Seaborn provides this publicly available dataset.
# If internet access is unavailable, download titanic.csv manually
# and replace this line with: pd.read_csv("data/titanic.csv")
df = sns.load_dataset("titanic")

print(f"Dataset shape: {df.shape}")
print("\nFirst 5 observations:")
print(df.head())


# ---------------------------------------------------------
# 2. DATASET DESCRIPTION
# ---------------------------------------------------------
print("\n[2] DATASET DESCRIPTION")
print("-" * 70)
print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDescriptive statistics:")
print(df.describe(include="all").T)

df.describe(include="all").T.to_csv(
    os.path.join(OUTPUT_DIR, "descriptive_statistics.csv")
)


# ---------------------------------------------------------
# 3. DATA PREPROCESSING
# ---------------------------------------------------------
print("\n[3] DATA PREPROCESSING")
print("-" * 70)

# Select useful variables.
# We intentionally avoid 'alive', 'class', 'adult_male', 'alone',
# and 'who' to reduce target leakage/redundancy.
features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked",
]

target = "survived"

model_df = df[features + [target]].copy()

# Remove rows where target is missing (normally none in this dataset).
model_df = model_df.dropna(subset=[target])

X = model_df[features]
y = model_df[target].astype(int)

numeric_features = ["pclass", "age", "sibsp", "parch", "fare"]
categorical_features = ["sex", "embarked"]

# Numeric preprocessing:
# - Median imputation handles missing numeric values.
# - StandardScaler standardizes numerical features for Logistic Regression.
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

# Categorical preprocessing:
# - Most-frequent imputation handles missing categories.
# - OneHotEncoder converts categories to numerical columns.
categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"Training observations: {len(X_train)}")
print(f"Testing observations : {len(X_test)}")

print("\nTarget distribution:")
print(y.value_counts())
print("\nTarget percentage:")
print((y.value_counts(normalize=True) * 100).round(2))


# ---------------------------------------------------------
# 4. OUTLIER ANALYSIS
# ---------------------------------------------------------
print("\n[4] OUTLIER ANALYSIS")
print("-" * 70)

for col in ["age", "fare", "sibsp", "parch"]:
    q1 = model_df[col].quantile(0.25)
    q3 = model_df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = model_df[(model_df[col] < lower) | (model_df[col] > upper)]

    print(
        f"{col:>6}: {len(outliers):>3} potential outliers "
        f"(IQR lower={lower:.2f}, upper={upper:.2f})"
    )

# We do not delete outliers automatically because extreme fares/ages
# can be genuine observations. The model pipeline retains them.


# ---------------------------------------------------------
# 5. STATISTICAL ANALYSIS
# ---------------------------------------------------------
print("\n[5] STATISTICAL ANALYSIS")
print("-" * 70)

# Survival rate by sex
sex_survival = model_df.groupby("sex")["survived"].mean().mul(100).round(2)
print("\nSurvival percentage by sex:")
print(sex_survival)

# Survival rate by passenger class
class_survival = model_df.groupby("pclass")["survived"].mean().mul(100).round(2)
print("\nSurvival percentage by passenger class:")
print(class_survival)

# Correlation among numeric variables
print("\nNumeric correlation matrix:")
print(model_df[numeric_features + [target]].corr().round(2))

# Chi-square test: sex vs survival
contingency = pd.crosstab(model_df["sex"], model_df["survived"])
chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

print("\nChi-square test: Sex vs Survival")
print(f"Chi-square statistic = {chi2:.4f}")
print(f"p-value              = {p_value:.6f}")

# T-test: fare for survivors vs non-survivors
survived_fare = model_df.loc[model_df["survived"] == 1, "fare"].dropna()
not_survived_fare = model_df.loc[model_df["survived"] == 0, "fare"].dropna()

t_stat, fare_p = stats.ttest_ind(
    survived_fare,
    not_survived_fare,
    equal_var=False
)

print("\nWelch's t-test: Fare vs Survival")
print(f"t-statistic = {t_stat:.4f}")
print(f"p-value     = {fare_p:.6f}")


# ---------------------------------------------------------
# 6. EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------
print("\n[6] CREATING EDA VISUALIZATIONS")
print("-" * 70)

# Class distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=model_df, x="pclass")
plt.title("Passenger Count by Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_passenger_class_distribution.png"), dpi=150)
plt.close()

# Survival distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=model_df, x="survived")
plt.title("Survival Distribution")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_survival_distribution.png"), dpi=150)
plt.close()

# Survival by sex
plt.figure(figsize=(7, 5))
sns.countplot(data=model_df, x="sex", hue="survived")
plt.title("Survival by Sex")
plt.xlabel("Sex")
plt.ylabel("Number of Passengers")
plt.legend(title="Survived")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_survival_by_sex.png"), dpi=150)
plt.close()

# Survival by class
plt.figure(figsize=(7, 5))
sns.countplot(data=model_df, x="pclass", hue="survived")
plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.legend(title="Survived")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_survival_by_class.png"), dpi=150)
plt.close()

# Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=model_df, x="age", kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_age_distribution.png"), dpi=150)
plt.close()

# Fare boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(data=model_df, x="fare")
plt.title("Fare Boxplot - Outlier Analysis")
plt.xlabel("Fare")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "06_fare_boxplot.png"), dpi=150)
plt.close()

# Correlation heatmap
plt.figure(figsize=(8, 6))
corr = model_df[numeric_features + [target]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "07_correlation_heatmap.png"), dpi=150)
plt.close()

print("EDA graphs saved in the 'outputs' folder.")


# ---------------------------------------------------------
# 7. MODEL BUILDING
# ---------------------------------------------------------
print("\n[7] MODEL BUILDING")
print("-" * 70)

# Model 1: Logistic Regression
logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(max_iter=1000, random_state=42),
        ),
    ]
)

# Model 2: Random Forest
random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                max_depth=6,
            ),
        ),
    ]
)

models = {
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest_model,
}


# ---------------------------------------------------------
# 8. MODEL EVALUATION
# ---------------------------------------------------------
print("\n[8] MODEL EVALUATION")
print("-" * 70)

results = []

for model_name, model in models.items():
    print(f"\n--- {model_name} ---")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Survived", "Survived"],
        yticklabels=["Not Survived", "Survived"],
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    plt.savefig(
        os.path.join(OUTPUT_DIR, f"{safe_name}_confusion_matrix.png"),
        dpi=150,
    )
    plt.close()

    results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1,
            "ROC_AUC": auc,
        }
    )

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

print("\nModel comparison:")
print(results_df.round(4).to_string(index=False))


# ---------------------------------------------------------
# 9. ROC CURVES
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))

for model_name, model in models.items():
    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
        name=model_name,
    )

plt.title("ROC Curves")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "08_roc_curves.png"), dpi=150)
plt.close()


# ---------------------------------------------------------
# 10. FINAL INTERPRETATION
# ---------------------------------------------------------
print("\n[9] INTERPRETATION")
print("-" * 70)

best_row = results_df.sort_values("F1_Score", ascending=False).iloc[0]

print(
    f"The model with the highest F1-score in this run is "
    f"{best_row['Model']} ({best_row['F1_Score']:.4f})."
)
print(
    "The analysis also shows that passenger characteristics such as "
    "sex, passenger class, age and fare contain useful information "
    "for predicting survival."
)
print(
    "Missing values were handled using imputation, categorical "
    "variables were converted using one-hot encoding, and numerical "
    "variables were standardized for Logistic Regression."
)
print(
    "Potential outliers were identified using the IQR method. "
    "They were not automatically deleted because extreme values "
    "may represent genuine passengers."
)

print("\nProject execution completed successfully.")
print(f"Check the '{OUTPUT_DIR}' folder for graphs and result files.")
