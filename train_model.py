import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Config
CSV_PATH = "netflix_customer_churn.csv"
OUTPUT_MODEL_PATH = "model.pkl"
RANDOM_STATE = 42
N_JOBS = -1

def train():
    print("Loading data...")
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return

    df = pd.read_csv(CSV_PATH)
    
    # --- Feature Engineering (Matching Notebook) ---
    
    # Rename columns if necessary (based on notebook logic)
    if "watch_hours" in df.columns:
        df = df.rename(columns={"watch_hours": "month_watch_hours"})
    
    # Drop columns
    cols_to_drop = [
        "last_login_days", "region", "device", "payment_method",
        "avg_watch_time_per_day", "favorite_genre", "number_of_profiles",
        "customer_id", "num_transactions"
    ]
    # Note: notebook drops "num_transactions" but then uses it in "use_cols"?
    # Let's check the notebook logic again carefully.
    # Line 268 drops "num_transactions".
    # Line 397 defines txn_col.
    # Line 403 uses txn_col.
    # Line 405 renames txn_col to "num_transactions".
    
    # Wait, if "num_transactions" is dropped in line 277, how is it used in line 403?
    # Ah, the notebook has:
    # df = df.drop(columns=[... "num_transactions"])
    # df2.info()  <-- This cell (id H04NWEV4hx3k) seems to have a bug or refers to a previous state.
    # BUT later in line 397: txn_col = "num_transactions" if "num_transactions" in df.columns ...
    # If it was dropped, it wouldn't be there.
    # However, the notebook output at line 183 says shape (5000, 17).
    # Let's look at the columns in the CSV.
    
    # I will inspect the CSV columns first to be sure.
    # But assuming the notebook logic *intended* to keep it for feature engineering:
    # The notebook cell 266 *drops* it.
    # But cell 394 *selects* it.
    # This implies cell 266 might not have been run or the user restarted the kernel and skipped it, OR the logic is contradictory.
    # However, "num_transactions" is crucial for "usage_intensity" and "complaint_ratio".
    # So I MUST NOT drop "num_transactions" if I want to use it.
    
    # Let's look at the "Column mapping & selection" cell (394).
    # It creates `df_sel` from `df`.
    # If `df` had `num_transactions` dropped, `df_sel` creation would fail if it tries to select it.
    # So I will assume the drop cell was either experimental or I should only drop UNUSED columns.
    
    # Let's proceed with a safer approach: Select what we need, ignore the drop cell if it conflicts.
    
    monthly_col = "watch_hours" if "watch_hours" in df.columns else ("month_watch_hours" if "month_watch_hours" in df.columns else None)
    txn_col = "num_transactions" if "num_transactions" in df.columns else ("num_transactions_scaled" if "num_transactions_scaled" in df.columns else None)
    target_col = "churned" if "churned" in df.columns else ("churn" if "churn" in df.columns else None)
    
    if not (monthly_col and txn_col and target_col):
        raise RuntimeError(f"Required columns missing. Found: {df.columns.tolist()}")

    use_cols = ["age", "gender", monthly_col, txn_col, "complaints", "subscription_type", target_col]
    df_sel = df[use_cols].copy()
    df_sel = df_sel.rename(columns={monthly_col: "monthly_usage_hours", txn_col: "num_transactions", target_col: "churn"})
    
    # Drop NA
    df_sel = df_sel.dropna().reset_index(drop=True)
    
    # Feature Engineering
    # usage intensity
    df_sel["usage_intensity"] = df_sel["monthly_usage_hours"] / (df_sel["num_transactions"] + 1e-6)
    
    # complaint ratio
    df_sel["complaint_ratio"] = df_sel["complaints"] / (df_sel["num_transactions"] + 1e-6)
    
    # log transforms
    df_sel["log_monthly_usage"] = np.log1p(df_sel["monthly_usage_hours"])
    df_sel["log_num_transactions"] = np.log1p(df_sel["num_transactions"])
    
    # age bucket
    def age_bucket(a):
        if a <= 30:
            return "Young"
        elif a <= 50:
            return "Adult"
        else:
            return "Senior"
    df_sel["age_bucket"] = df_sel["age"].apply(age_bucket)
    
    # Winsorize
    for col in ["monthly_usage_hours", "num_transactions", "usage_intensity", "complaint_ratio", "log_monthly_usage", "log_num_transactions"]:
        low, high = df_sel[col].quantile(0.01), df_sel[col].quantile(0.99)
        df_sel[col] = df_sel[col].clip(low, high)
        
    # Ensure types
    df_sel["age"] = df_sel["age"].astype(int)
    df_sel["num_transactions"] = df_sel["num_transactions"].astype(int)
    df_sel["complaints"] = df_sel["complaints"].astype(int)
    df_sel["churn"] = df_sel["churn"].astype(int)
    df_sel["gender"] = df_sel["gender"].astype(str)
    df_sel["subscription_type"] = df_sel["subscription_type"].astype(str)
    
    print("Data prepared. Shape:", df_sel.shape)
    print("Churn distribution:", df_sel["churn"].value_counts(normalize=True).to_dict())
    
    # Split
    target = "churn"
    X = df_sel.drop(columns=[target])
    y = df_sel[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    
    # Preprocessing
    numeric_features = ["age", "monthly_usage_hours", "num_transactions", "complaints", "usage_intensity", "complaint_ratio", "log_monthly_usage", "log_num_transactions"]
    categorical_features = ["gender", "subscription_type", "age_bucket"]
    
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ], remainder="drop")
    
    # Model Definition (Stacking without SMOTE)
    estimators = [
        ("GradientBoosting", GradientBoostingClassifier(n_estimators=200, learning_rate=0.01, random_state=RANDOM_STATE)),
        ("LogisticRegression", LogisticRegression(max_iter=2000, C=0.1, random_state=RANDOM_STATE)),
        ("RandomForest", RandomForestClassifier(n_estimators=200, max_depth=6, max_features='log2', min_samples_split=5, n_jobs=N_JOBS, random_state=RANDOM_STATE))
    ]
    
    meta_clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    
    stack_clf = StackingClassifier(estimators=estimators, final_estimator=meta_clf, n_jobs=N_JOBS, passthrough=False)
    
    # Final Pipeline
    final_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clf", stack_clf)
    ])
    
    print("Training model...")
    final_pipe.fit(X_train, y_train)
    
    print("Evaluating...")
    y_pred = final_pipe.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, final_pipe.predict_proba(X_test)[:, 1]))
    print(classification_report(y_test, y_pred))
    
    print(f"Saving model to {OUTPUT_MODEL_PATH}...")
    joblib.dump(final_pipe, OUTPUT_MODEL_PATH)
    print("Done.")

if __name__ == "__main__":
    train()
