# Customer Churn Prediction - Quick Start Guide

## Option 1: Use Sample Data (Recommended for Testing)

If you don't have a model or dataset yet, generate sample data:

```bash
# Install dependencies
pip install -r requirements.txt

# Generate sample model and dataset
python generate_sample_data.py

# Run the app
streamlit run app.py
```

This will create:
- `model.pkl` - A trained Random Forest model
- `netflix_customer_churn_with_new_columns.csv` - Sample dataset with 1000 customers

## Option 2: Use Your Own Data

1. **Prepare your model:**
   - Save your trained model as `model.pkl` in this directory
   - Model should accept features: Age, Gender, MonthlyUsageHours, NumTransactions, SubscriptionType, Complaints
   - Gender should be encoded: Male=0, Female=1, Other=2
   - SubscriptionType should be encoded: Basic=0, Standard=1, Premium=2, Gold=3

2. **Prepare your dataset (optional):**
   - Save as CSV with columns: Age, Gender, MonthlyUsageHours, NumTransactions, SubscriptionType, Complaints, Churn
   - Or use the file uploader in the app sidebar

3. **Run the app:**
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

## Features Overview

### 🎯 Churn Prediction Page
- Enter customer details
- Get instant churn prediction
- View probability gauge
- See risk assessment

### 📊 EDA Dashboard
- 7 interactive visualizations
- Churn patterns analysis
- Feature correlations
- Model insights

### ℹ️ About Dataset
- Dataset statistics
- Data quality report
- Column information
- Sample data viewer

## Troubleshooting

**No model found?**
- Run `python generate_sample_data.py` to create one
- Or upload via sidebar

**No dataset?**
- Use the sidebar file uploader
- Or run the sample data generator

**Dependencies error?**
- Run `pip install -r requirements.txt`

## Next Steps

1. ✅ Install dependencies
2. ✅ Generate or provide model and data
3. ✅ Run the app
4. 🎉 Start predicting churn!

For detailed documentation, see README.md
