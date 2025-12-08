# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Customer Churn Prediction
# 
# This notebook demonstrates a complete workflow for predicting customer churn using a dataset of Netflix customers.
# We will cover:
# 1. Data Loading and Exploration
# 2. Data Cleaning and Preprocessing
# 3. Feature Engineering
# 4. Model Training and Hyperparameter Tuning
# 5. Model Stacking and Final Evaluation
# 6. Feature Importance Analysis

# %% id="y2o_mHqPg9sT"
# Basic libraries
import numpy as np
import pandas as pd

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Handling imbalance

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Metrics / Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# %% colab={"base_uri": "https://localhost:8080/"} id="9ANX8rjbus2w" outputId="ae038d24-2c20-475b-baea-ffa5ead7c8f1"
# !pip install ace_tools

# %% [markdown]
# ## 1. Imports and Configuration
# Import necessary libraries for data manipulation, visualization, and machine learning.

# %% id="J1eHZg_6vFhs"
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
from time import time

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier



# optional imports (will continue gracefully if absent)
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns


# %% [markdown] id="9zITXP3Jvfuq"
# ## **Config**

# %% [markdown] id="9zITXP3Jvfuq"
# ## **Config**
# Define paths and constants used throughout the notebook.

# %% id="EMd5TturvPgu"
CSV_PATH = "netflix_customer_churn.csv"
OUTPUT_DIR = "/content/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_MODEL_PATH = "model.pkl"
OUTPUT_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "model_comparison_summary.csv")
OUTPUT_TEXT_SUMMARY = os.path.join(OUTPUT_DIR, "best_model_summary.txt")

RANDOM_STATE = 42
N_JOBS = -1
N_SPLITS = 5
TOP_K_FEATURES = 15

# %% [markdown] id="r8tBJgWiwAda"
# # **Load** **Data**

# %% [markdown] id="r8tBJgWiwAda"
# # **Load Data**
# Load the dataset and perform an initial inspection of the data structure and statistics.

# %% colab={"base_uri": "https://localhost:8080/"} id="e9y39c7ov--s" outputId="0d3ae0e1-e64f-4313-ee4e-351fa887bdb1"
df = pd.read_csv(CSV_PATH)
print("Loaded:", CSV_PATH, "shape:", df.shape)

# %% id="1iP7ymoYhYaN"
df.info()

# %% id="z8ohw6nFhfBl"
df["churned"].value_counts()[1]

# %% id="txNU3WCchiFx"
df.describe()

# %% id="HuLyYJhjhkn7"
df["gender"].value_counts()

# %% id="dQISfNPyhlQs"
df["watch_hours"].max()


# %% id="Uf6UHLrdhpaZ"
df["customer_id"].duplicated().sum()


# %% [markdown]
# ## Data Cleaning
# Rename columns for consistency and drop unnecessary columns that won't be used for prediction.

# %% id="JHfiK5HJhsfK"
df = df.rename(columns={"watch_hours": "month_watch_hours"})


# %% id="H04NWEV4hx3k"
df = df.drop(columns=[
    "region",
    "device",
    "payment_method",
    "avg_watch_time_per_day",
    "favorite_genre",
    "number_of_profiles",
    "customer_id"
])
df.info()


# %% id="dgAzLUs4h87J"
df.isnull().sum()


# %% id="vXv1sRYUh_CR"
(df== "").sum()


# %% id="lyFP7izsiBn0"
df.duplicated().sum()


# %% id="m1L8Em7WiE3J"
df= df.drop_duplicates()


# %% id="13VlTzeHiH-3"
df.duplicated().sum()

# %% [markdown]
# ## Exploratory Data Analysis (EDA)
# Visualize the distribution of key variables like Gender and Churn status.

# %% id="ycc2ICpriKgC"
df["churned"].value_counts()


# %% id="1WEic9JniO4E"
df["gender"].value_counts().plot(kind="bar")

plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# %% id="rIxcYd1oiMcx"
df["churned"].value_counts().plot(kind="bar")
plt.title("Churn Distribution")
plt.xlabel("Churn (0=No, 1=Yes)")
plt.ylabel("Count")
plt.show()


# %% [markdown] id="u6Gx5D-jwMLR"
# # **Column mapping & selection (as before)**

# %% [markdown] id="u6Gx5D-jwMLR"
# # **Column mapping & selection**
# Select the relevant columns for modeling and handle any missing values.

# %% id="Fxur0lqRwHO6"
monthly_col = "watch_hours" if "watch_hours" in df.columns else ("month_watch_hours" if "month_watch_hours" in df.columns else None)
txn_col = "num_transactions" if "num_transactions" in df.columns else ("num_transactions_scaled" if "num_transactions_scaled" in df.columns else None)
target_col = "churned" if "churned" in df.columns else ("churn" if "churn" in df.columns else None)

if not (monthly_col and txn_col and target_col):
    raise RuntimeError("Required columns missing in CSV. Found: " + ", ".join(df.columns.tolist()))

use_cols = ["age", "gender", monthly_col, txn_col, "complaints", "subscription_type", "last_login_days", target_col]
df_sel = df[use_cols].copy()
df_sel = df_sel.rename(columns={monthly_col: "monthly_usage_hours", txn_col: "num_transactions", target_col: "churn"})

# drop NA rows in these selected cols
df_sel = df_sel.dropna().reset_index(drop=True)


# %% [markdown] id="K_najz5cwSdH"
# # Feature engineering (important additions)

# %% [markdown] id="K_najz5cwSdH"
# # Feature Engineering
# Create new features to capture more complex patterns in the data, such as usage intensity and complaint ratios.
# We also apply log transformations to skewed variables and bin age into categories.

# %% id="6NOwa1wmwLCc"
# usage intensity (hours per transaction) — handles division by zero
df_sel["usage_intensity"] = df_sel["monthly_usage_hours"] / (df_sel["num_transactions"] + 1e-6)

# complaint ratio
df_sel["complaint_ratio"] = df_sel["complaints"] / (df_sel["num_transactions"] + 1e-6)

# log transforms to reduce skew
df_sel["log_monthly_usage"] = np.log1p(df_sel["monthly_usage_hours"])
df_sel["log_num_transactions"] = np.log1p(df_sel["num_transactions"])


# %% id="KfBaM4_ZweLN"
# age bucket
def age_bucket(a):
    if a <= 30:
        return "Young"
    elif a <= 50:
        return "Adult"
    else:
        return "Senior"
df_sel["age_bucket"] = df_sel["age"].apply(age_bucket)

# %% id="O1FeQ2WswiUg"
# Clip winsorize numeric extremes (1st-99th)
for col in ["monthly_usage_hours", "num_transactions", "usage_intensity", "complaint_ratio", "log_monthly_usage", "log_num_transactions"]:
    low, high = df_sel[col].quantile(0.01), df_sel[col].quantile(0.99)
    df_sel[col] = df_sel[col].clip(low, high)

# %% colab={"base_uri": "https://localhost:8080/"} id="BVApq8ihwlT-" outputId="3e4bc47c-6b25-41fe-e027-26e7e288748c"

# Ensure types
df_sel["age"] = df_sel["age"].astype(int)
df_sel["num_transactions"] = df_sel["num_transactions"].astype(int)
df_sel["complaints"] = df_sel["complaints"].astype(int)
df_sel["churn"] = df_sel["churn"].astype(int)
df_sel["gender"] = df_sel["gender"].astype(str)
df_sel["subscription_type"] = df_sel["subscription_type"].astype(str)

print("After FE, columns:", df_sel.columns.tolist())
print("Churn distribution:\n", df_sel["churn"].value_counts(normalize=True))


# %% [markdown] id="ThgtnvHlwrIK"
# # Prepare X,y and split
#

# %% [markdown] id="ThgtnvHlwrIK"
# # Prepare X, y and Split
# Separate the target variable from the features and split the data into training and testing sets.

# %% colab={"base_uri": "https://localhost:8080/"} id="b7tGQ-GzwpFp" outputId="51c86bb6-cebe-4e38-b90a-5ed6497a5c0c"
target = "churn"
X = df_sel.drop(columns=[target])
y = df_sel[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print("Train/test shapes:", X_train.shape, X_test.shape)


# %% [markdown] id="S0wgEi7mw2LZ"
# # **Preprocessing definition**

# %% [markdown] id="S0wgEi7mw2LZ"
# # **Preprocessing Pipeline**
# Define a preprocessing pipeline to handle numeric scaling and categorical encoding automatically.

# %% id="dSwsh99qwzl0"
numeric_features = ["age", "monthly_usage_hours", "num_transactions", "complaints", "usage_intensity", "complaint_ratio", "log_monthly_usage", "log_num_transactions", "last_login_days"]
categorical_features = ["gender", "subscription_type", "age_bucket"]

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])


# %% [markdown] id="HoI54tepw9Tk"
# OneHotEncoder with sparse=False for easier feature-name mapping
#

# %% id="Awyr1pkMw7LW"
categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
], remainder="drop")



# %% id="E8q6DDp-xDCU"
# get feature names helper
def get_feature_names(preprocessor):
    num_feats = numeric_features
    cat_transformer = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_transformer.get_feature_names_out(categorical_features))
    return num_feats + cat_names



# %% [markdown] id="_51i6txZxK0N"
# # Model candidates (with sensible defaults)
#

# %% [markdown] id="_51i6txZxK0N"
# # Model Selection
# Define a list of candidate models to evaluate. We include Logistic Regression, Random Forest, Gradient Boosting, and XGBoost.

# %% id="Fckwk3jyxI21"
candidates = {
    "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=N_JOBS),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=300, random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=N_JOBS)
}

# %% colab={"base_uri": "https://localhost:8080/"} id="a0m4Bkn8xQ0v" outputId="bc8e3925-dba4-4e8a-bf1a-dd26e4c0b479"
if CATBOOST_AVAILABLE:
    candidates["CatBoost"] = CatBoostClassifier(verbose=0, random_state=RANDOM_STATE)
if LGB_AVAILABLE:
    candidates["LightGBM"] = lgb.LGBMClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS)

print("Candidate estimators:", list(candidates.keys()))

# %% [markdown] id="aUWC7IBNxagj"
# Evaluate each candidate with Stratified K-Fold CV (ROC-AUC)
#
# We'll fit pipeline: preprocessor -> standard scaling -> estimator using sklearn Pipeline is not straightforward with CV scoring.
#
# So we'll use a simple approach: prefit preprocessing, then perform standard scaling on training folds inside cross-validation manually.
#
# Simpler: use pipeline with standard scaling and cross_val_score on pipeline (sklearn supports cv).
#

# %% [markdown] id="aUWC7IBNxagj"
# ## Cross-Validation
# Evaluate each model using Stratified K-Fold Cross-Validation to ensure robust performance estimates.

# %% colab={"base_uri": "https://localhost:8080/"} id="_BJFlamixUMX" outputId="f8c90570-e905-4a20-8857-73cf7affdb7d"

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

model_scores = []

for name, estimator in candidates.items():
    print(f"\nEvaluating {name} ...")
    # pipeline: preprocessor -> SMOTE -> estimator
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clf", estimator)
    ])
    # cross_val_score with ROC-AUC
    try:
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=N_JOBS)
        mean_auc = float(np.mean(scores))
        std_auc = float(np.std(scores))
        print(f"{name} CV ROC-AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    except Exception as e:
        print("CV failed for", name, "->", e)
        mean_auc, std_auc = np.nan, np.nan
    model_scores.append({"model": name, "mean_roc_auc": mean_auc, "std_roc_auc": std_auc, "estimator": estimator})

# Convert to DataFrame
scores_df = pd.DataFrame(model_scores).sort_values("mean_roc_auc", ascending=False).reset_index(drop=True)
print("\nModel CV ranking:")
print(scores_df[["model","mean_roc_auc","std_roc_auc"]])


# %% [markdown] id="aAd7Uld1xp7L"
# **# Quick hyperparameter tuning for top 3 models (RandomizedSearchCV)**
#

# %% [markdown] id="aAd7Uld1xp7L"
# # Hyperparameter Tuning
# Select the top performing models and tune their hyperparameters using RandomizedSearchCV to optimize performance.

# %% colab={"base_uri": "https://localhost:8080/"} id="Zs2x70BVxm_l" outputId="cf0df502-9351-4159-f910-0c0031171fc2"
top_models = scores_df["model"].tolist()[:3]
print("\nTop models for tuning:", top_models)

tuned_estimators = {}
random_search_results = []

for name in top_models:
    base = candidates[name]
    print(f"\nTuning {name} ...")
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clf", base)
    ])
    param_distributions = None
    n_iter = 20
    if name == "RandomForest":
        param_distributions = {
            "clf__n_estimators": [200, 300, 500],
            "clf__max_depth": [6, 10, 16, None],
            "clf__min_samples_split": [2, 5, 10],
            "clf__max_features": ["sqrt", "log2"]
        }
    elif name == "XGBoost":
        param_distributions = {
            "clf__n_estimators": [200, 400, 700],
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__max_depth": [3, 5, 7],
            "clf__subsample": [0.7, 0.8, 1.0],
            "clf__colsample_bytree": [0.6, 0.8, 1.0]
        }
    elif name == "GradientBoosting":
        param_distributions = {
            "clf__n_estimators": [200, 300, 500],
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__max_depth": [3, 5, 7]
        }
    else:
        param_distributions = {"clf__C": [0.01, 0.1, 1, 10]}  # fallback for Logistic

    # run RandomizedSearchCV
    rsearch = RandomizedSearchCV(pipe, param_distributions, n_iter=n_iter, scoring="roc_auc",
                                 cv=cv, random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=0)
    try:
        t0 = time()
        rsearch.fit(X_train, y_train)
        t1 = time()
        print(f"Tuning {name} done in {t1-t0:.1f}s. Best ROC-AUC: {rsearch.best_score_:.4f}")
        best_pipe = rsearch.best_estimator_
        tuned_estimators[name] = best_pipe
        random_search_results.append({"model": name, "best_score": rsearch.best_score_, "best_params": rsearch.best_params_})
    except Exception as e:
        print("Tuning failed for", name, ":", e)

# %% [markdown] id="qqC2AOIzyFBX"
# # Build a stacking ensemble from the tuned estimators (if available)
#

# %% [markdown] id="qqC2AOIzyFBX"
# # Stacking Ensemble
# Combine the tuned models into a Stacking Classifier to leverage the strengths of multiple models.

# %% id="_gXK-UVMx6SI"
estimators_for_stack = []
for name, pipe in tuned_estimators.items():
    # use the clf step inside the pipeline as estimator
    estimators_for_stack.append((name, pipe.named_steps["clf"]))

# fallback: if no tuned estimators, use top raw candidates
if not estimators_for_stack:
    for nm in top_models:
        estimators_for_stack.append((nm, candidates[nm]))

# define final meta-estimator
meta_clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

# stacking requires base estimators that accept fit(X,y) numpy arrays
# We'll build a final pipeline: preprocessor -> SMOTE -> StackingClassifier
base_estimators = [(n, e) for n, e in estimators_for_stack]
stack_clf = StackingClassifier(estimators=base_estimators, final_estimator=meta_clf, n_jobs=N_JOBS, passthrough=False)

final_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("clf", stack_clf)
])


# %% [markdown] id="cl5m-vzwyUBX"
# # Evaluate final stacked pipeline with CV
#

# %% [markdown] id="cl5m-vzwyUBX"
# # Final Model Evaluation
# Evaluate the final Stacking Ensemble on the held-out test set to get an unbiased estimate of its performance.

# %% colab={"base_uri": "https://localhost:8080/", "height": 488} id="Ed_T5aGWyWUS" outputId="59681baa-db8b-4f8a-ec6a-b651de118a96"
print("\nEvaluating final stacking pipeline with cross-validation...")
stack_scores = cross_val_score(final_pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=N_JOBS)
print("Stacking CV ROC-AUC:", np.mean(stack_scores), np.std(stack_scores))

# Fit stack on whole training set
print("Fitting final pipeline on full training set...")
final_pipe.fit(X_train, y_train)


# %% [markdown] id="vB3C2m3qzMp8"
# # Final Evaluation

# %% colab={"base_uri": "https://localhost:8080/"} id="8AvxOkGCybSM" outputId="0a388470-42b2-4233-d451-503cdca73ae6"
y_pred = final_pipe.predict(X_test)
y_proba = final_pipe.predict_proba(X_test)[:, 1] if hasattr(final_pipe, "predict_proba") else None

test_metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "recall": recall_score(y_test, y_pred, zero_division=0),
    "f1": f1_score(y_test, y_pred, zero_division=0),
    "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
    "confusion_matrix": confusion_matrix(y_test, y_pred),
    "classification_report": classification_report(y_test, y_pred, zero_division=0)
}

print("\n==============================")
print("Final model: Stacking_ensemble")
print("==============================")
for k, v in test_metrics.items():
    if k not in ["confusion_matrix", "classification_report"]:
        print(f"{k}: {v:.4f}")
print("\nConfusion matrix:\n", test_metrics["confusion_matrix"])
print("\nClassification report:\n", test_metrics["classification_report"])
print("==============================\n")


# %% [markdown] id="B0KZ2weG2Q_T"
# # Feature Importance Extraction
#

# %% [markdown] id="B0KZ2weG2Q_T"
# # Feature Importance
# Analyze which features are driving the model's predictions using a surrogate Random Forest model.

# %% colab={"base_uri": "https://localhost:8080/", "height": 826} id="BsDXg3X82Stu" outputId="77107b7c-e2ff-45eb-d539-904b84942151"

# get feature names now that preprocessor is fitted
feature_names = get_feature_names(final_pipe.named_steps["preprocessor"])
print("\nFinal feature names (count):", len(feature_names))

fi = None  # so we can safely refer to it later
try:
    # transform X_train to preprocessed array
    X_train_trans = final_pipe.named_steps["preprocessor"].transform(X_train)

    # surrogate RF to approximate feature importances
    rf_for_importance = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    )
    rf_for_importance.fit(X_train_trans, y_train)

    importances = rf_for_importance.feature_importances_
    fi = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi = fi.sort_values("importance", ascending=False).head(TOP_K_FEATURES)

    print("\nTop feature importances (approx, via surrogate RF):")
    print(fi.to_string(index=False))

    # plot
    plt.figure(figsize=(8, max(3, 0.3 * len(fi))))
    sns.barplot(x="importance", y="feature", data=fi)
    plt.title("Surrogate RF Feature importances (top features)")
    plt.tight_layout()
    plt.show()

except Exception as e:
    print("Could not compute surrogate feature importances:", e)




# %% [markdown] id="cZVL_93g2ghp"
# # Save final pipeline & summaries
#

# %% [markdown] id="cZVL_93g2ghp"
# # Save Model
# Save the final trained pipeline and performance summaries to disk for later use.

# %% colab={"base_uri": "https://localhost:8080/"} id="PePj02Ph2fMB" outputId="a94c095c-98e8-4f41-87df-c8bdf020deaf"

# Save model pipeline
# Inject feature importances into the pipeline object if possible
try:
    stacking_clf = final_pipe.named_steps['clf']
    # Try to find a tree-based model among base estimators
    for estimator in stacking_clf.estimators_:
        if hasattr(estimator, 'feature_importances_'):
            final_pipe.feature_importances_ = estimator.feature_importances_
            print("Injected feature_importances_ from base estimator into final_pipe.")
            break
except Exception as e:
    print(f"Could not inject feature importances: {e}")

joblib.dump(final_pipe, OUTPUT_MODEL_PATH)
print("Saved final pipeline to:", OUTPUT_MODEL_PATH)

# Save metrics + CV summary to CSV
summary_rows = [{
    "model": "Stacking_ensemble",
    "accuracy": test_metrics["accuracy"],
    "precision": test_metrics["precision"],
    "recall": test_metrics["recall"],
    "f1": test_metrics["f1"],
    "roc_auc": test_metrics["roc_auc"]
}]

for r in model_scores:
    summary_rows.append({
        "model": r["model"],
        "mean_roc_auc": r["mean_roc_auc"],
        "std_roc_auc": r["std_roc_auc"]
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)
print("Saved model comparison to:", OUTPUT_SUMMARY_CSV)

# Save text summary
with open(OUTPUT_TEXT_SUMMARY, "w") as f:
    f.write("Final selected model: Stacking_ensemble\n\n")
    f.write("Test metrics:\n")
    for k, v in test_metrics.items():
        f.write(f"{k}:\n{v}\n\n")

    f.write("Top features (approx):\n")
    if fi is not None:
        f.write(fi.to_string(index=False))
    else:
        f.write("n/a\n")

print("Saved text summary to:", OUTPUT_TEXT_SUMMARY)

print("\nDone. Model pipeline saved at (local path):", OUTPUT_MODEL_PATH)
print("You can use joblib.load(OUTPUT_MODEL_PATH) to load it.")


# %% [markdown]
# ## Prediction on New Data
# Demonstrate how to load the saved model and make predictions on new, unseen customer data.

# %% id="0oeJwb1NLnNY"
import joblib
import numpy as np
import pandas as pd

# load pipeline
final_pipe = joblib.load("model.pkl")

def add_features(df):
    df = df.copy()
    df["usage_intensity"] = df["monthly_usage_hours"] / (df["num_transactions"] + 1e-6)
    df["complaint_ratio"] = df["complaints"] / (df["num_transactions"] + 1e-6)
    df["log_monthly_usage"] = np.log1p(df["monthly_usage_hours"])
    df["log_num_transactions"] = np.log1p(df["num_transactions"])

    def age_bucket(a):
        if a <= 30:
            return "Young"
        elif a <= 50:
            return "Adult"
        else:
            return "Senior"
    df["age_bucket"] = df["age"].apply(age_bucket)
    return df



# %% colab={"base_uri": "https://localhost:8080/"} id="tNI5EMpHEj5C" outputId="a7e13366-71ec-417b-891a-5ed54a77e8e1"
new_customers_raw = pd.DataFrame([
    # 1. Very loyal, heavy user, zero complaints → expected low churn
    {
        "age": 28,
        "gender": "Male",
        "monthly_usage_hours": 120,   # watches a lot
        "num_transactions": 12,       # many payments/renewals
        "complaints": 0,
        "subscription_type": "Premium"
    },
    # 2. Low usage, many complaints, basic plan → expected high churn
    {
        "age": 35,
        "gender": "Female",
        "monthly_usage_hours": 5,
        "num_transactions": 2,
        "complaints": 3,
        "subscription_type": "Basic"
    },
    # 3. Medium usage but very high complaint ratio → likely to churn
    {
        "age": 22,
        "gender": "Male",
        "monthly_usage_hours": 40,
        "num_transactions": 3,
        "complaints": 4,  # more complaints than transactions
        "subscription_type": "Standard"
    },
    # 4. Older user, moderate usage, no complaints, long history → probably stable
    {
        "age": 55,
        "gender": "Female",
        "monthly_usage_hours": 60,
        "num_transactions": 20,
        "complaints": 0,
        "subscription_type": "Standard"
    },
    # 5. New user: very low usage, low transactions, no complaints yet → ambiguous / moderate churn risk
    {
        "age": 30,
        "gender": "Other",
        "monthly_usage_hours": 8,
        "num_transactions": 1,
        "complaints": 0,
        "subscription_type": "Basic"
    },
])

# Add engineered features
new_customers = add_features(new_customers_raw)

# Predict
probs = final_pipe.predict_proba(new_customers)[:, 1]  # probability of churn
preds = final_pipe.predict(new_customers)

for i, (p, prob) in enumerate(zip(preds, probs), start=1):
    print(f"Customer {i}: predicted_churn={p}, churn_probability={prob:.3f}")


# %% id="2RQe8B-pEk3l"
