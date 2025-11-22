"""
Script to retrain the churn prediction model with the current environment
This ensures compatibility with the current Python and scikit-learn versions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

def retrain_model():
    """Retrain the model using the existing dataset"""
    
    print("Loading dataset...")
    try:
        df = pd.read_csv('netflix_customer_churn.csv')
    except FileNotFoundError:
        print("Error: netflix_customer_churn.csv not found!")
        return
    
    print(f"Dataset loaded: {len(df)} rows")
    
    # Check for required columns
    required_features = ['Age', 'Gender', 'MonthlyUsageHours', 'NumTransactions', 
                        'SubscriptionType', 'Complaints']
    
    # Find the churn column (might have different names)
    churn_col = None
    for col in df.columns:
        if 'churn' in col.lower():
            churn_col = col
            break
    
    if churn_col is None:
        print("Error: No 'Churn' column found in dataset!")
        return
    
    print(f"Using '{churn_col}' as target variable")
    
    # Prepare features
    X = df[required_features].copy()
    y = df[churn_col]
    
    # Encode categorical variables
    print("Encoding categorical variables...")
    gender_mapping = {'Male': 0, 'Female': 1, 'Other': 2}
    subscription_mapping = {'Basic': 0, 'Standard': 1, 'Premium': 2, 'Gold': 3}
    
    X['Gender'] = X['Gender'].map(gender_mapping)
    X['SubscriptionType'] = X['SubscriptionType'].map(subscription_mapping)
    
    # Handle any missing values
    X = X.fillna(X.median())
    
    # Split the data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train the model
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\nModel Performance:")
    print(f"Training Accuracy: {train_score:.4f}")
    print(f"Testing Accuracy: {test_score:.4f}")
    
    # Save the model using both pickle and joblib
    print("\nSaving model...")
    
    # Save with pickle
    try:
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        print("✓ Model saved as 'model.pkl' (pickle format)")
    except Exception as e:
        print(f"✗ Failed to save with pickle: {e}")
    
    # Save with joblib (recommended for scikit-learn)
    try:
        joblib.dump(model, 'model_joblib.pkl')
        print("✓ Model saved as 'model_joblib.pkl' (joblib format)")
    except Exception as e:
        print(f"✗ Failed to save with joblib: {e}")
    
    # Display feature importance
    print("\nFeature Importance:")
    feature_importance = pd.DataFrame({
        'Feature': required_features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(feature_importance.to_string(index=False))
    
    print("\n✓ Model retraining complete!")
    print("\nEnvironment Info:")
    print(f"- Python: {__import__('sys').version.split()[0]}")
    print(f"- scikit-learn: {__import__('sklearn').__version__}")
    print(f"- pandas: {pd.__version__}")
    print(f"- numpy: {np.__version__}")

if __name__ == "__main__":
    print("=" * 60)
    print("Customer Churn Model Retraining Script")
    print("=" * 60)
    print()
    retrain_model()
