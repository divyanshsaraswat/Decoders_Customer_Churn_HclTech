"""
Sample Model and Dataset Generator for Customer Churn Prediction
This script creates a sample model.pkl and dataset CSV for testing the Streamlit app
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Set random seed for reproducibility
np.random.seed(42)

def generate_sample_dataset(n_samples=1000):
    """Generate a realistic sample customer churn dataset"""
    
    # Generate features
    data = {
        'Age': np.random.randint(18, 71, n_samples),
        'Gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.48, 0.48, 0.04]),
        'MonthlyUsageHours': np.random.randint(0, 201, n_samples),
        'NumTransactions': np.random.randint(1, 51, n_samples),
        'SubscriptionType': np.random.choice(['Basic', 'Standard', 'Premium', 'Gold'], n_samples, p=[0.3, 0.35, 0.25, 0.1]),
        'Complaints': np.random.randint(0, 11, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic churn labels based on features
    # Higher complaints, lower usage, and basic subscription increase churn probability
    churn_probability = (
        (df['Complaints'] / 10) * 0.4 +  # Complaints contribute 40%
        (1 - df['MonthlyUsageHours'] / 200) * 0.3 +  # Low usage contributes 30%
        (df['SubscriptionType'].map({'Basic': 0.3, 'Standard': 0.2, 'Premium': 0.1, 'Gold': 0.05})) +  # Subscription type
        np.random.random(n_samples) * 0.2  # Random noise
    )
    
    # Convert probability to binary churn (1 = churned, 0 = not churned)
    df['Churn'] = (churn_probability > 0.5).astype(int)
    
    return df

def train_and_save_model(df, model_path='model.pkl'):
    """Train a Random Forest model and save it"""
    
    # Prepare features
    X = df[['Age', 'Gender', 'MonthlyUsageHours', 'NumTransactions', 'SubscriptionType', 'Complaints']].copy()
    y = df['Churn']
    
    # Encode categorical variables
    gender_mapping = {'Male': 0, 'Female': 1, 'Other': 2}
    subscription_mapping = {'Basic': 0, 'Standard': 1, 'Premium': 2, 'Gold': 3}
    
    X['Gender'] = X['Gender'].map(gender_mapping)
    X['SubscriptionType'] = X['SubscriptionType'].map(subscription_mapping)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Model Training Complete!")
    print(f"Training Accuracy: {train_score:.3f}")
    print(f"Testing Accuracy: {test_score:.3f}")
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\nModel saved to: {model_path}")
    
    return model

def main():
    """Main function to generate dataset and train model"""
    
    print("=" * 60)
    print("Customer Churn Prediction - Sample Data Generator")
    print("=" * 60)
    
    # Generate dataset
    print("\n1. Generating sample dataset...")
    df = generate_sample_dataset(n_samples=1000)
    
    # Save dataset
    dataset_path = 'netflix_customer_churn_with_new_columns.csv'
    df.to_csv(dataset_path, index=False)
    print(f"   ✓ Dataset saved to: {dataset_path}")
    print(f"   ✓ Total samples: {len(df)}")
    print(f"   ✓ Churned customers: {df['Churn'].sum()} ({df['Churn'].mean()*100:.1f}%)")
    
    # Display sample
    print("\n2. Sample data (first 5 rows):")
    print(df.head())
    
    # Train and save model
    print("\n3. Training Random Forest model...")
    model = train_and_save_model(df, model_path='model.pkl')
    
    # Feature importance
    print("\n4. Feature Importance:")
    feature_names = ['Age', 'Gender', 'MonthlyUsageHours', 'NumTransactions', 'SubscriptionType', 'Complaints']
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(importance_df.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("✓ Setup Complete!")
    print("=" * 60)
    print("\nYou can now run the Streamlit app:")
    print("  streamlit run app.py")
    print("\nFiles created:")
    print(f"  - {dataset_path}")
    print("  - model.pkl")
    print("=" * 60)

if __name__ == "__main__":
    main()
