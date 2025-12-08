# 🎯 Customer Churn Prediction System

A comprehensive machine learning solution for predicting customer churn with an interactive Streamlit web application. This project includes end-to-end model training, evaluation, and deployment with rich visualizations and insights.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Dataset Information](#dataset-information)
- [Troubleshooting](#troubleshooting)

---

## 🌟 Overview

This project provides a complete customer churn prediction solution built with:
- **Machine Learning**: Stacking ensemble model (Logistic Regression, Random Forest, Gradient Boosting, XGBoost).
- **Interactive Web App**: Streamlit-based dashboard for predictions and data exploration.
- **Production-Ready**: Comprehensive error handling, model versioning, and deployment support.

### Key Highlights
- ✅ **High Performance**: Stacking ensemble combining multiple algorithms for superior accuracy.
- ✅ **Feature Engineering**: Includes critical features like `LastLoginDays` and `UsageIntensity`.
- ✅ **Interactive UI**: Real-time predictions with visual risk assessment and Dark/Light mode support.
- ✅ **Comprehensive EDA**: Interactive visualizations for data insights.

---

## 📁 Project Structure

```
Churning/
│
├── app.py                             # Streamlit web application
├── Customer_Churn_Prediction_.ipynb   # Model training notebook
├── Customer_Churn_Prediction_.py      # Model training script (Jupytext)
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
│
├── netflix_customer_churn.csv         # Training dataset
├── model.pkl                          # Trained model (pickle format)
```

---

## 🎨 Features

### 1. **Single Customer Prediction** 🔮
- Interactive input form for customer attributes:
    - **Age**, **Gender**, **Subscription Type**
    - **Monthly Usage**, **Transactions**, **Complaints**
    - **Days Since Last Login** (New!)
- Real-time churn prediction with probability.
- Visual risk gauge (0-100%).
- Actionable recommendations based on risk level.
- **Model Details View**: Inspect the underlying model architecture and expected features.

### 2. **Batch Prediction** 📁
- Upload CSV files for bulk predictions.
- **New**: Supports `LastLoginDays` column for accurate batch processing.
- Predict churn for thousands of customers at once.
- Download results with risk levels.
- Visual analysis of prediction distribution.

### 3. **Exploratory Data Analysis Dashboard** 📊
- **Churn Distribution**: Overall churn rate and class balance.
- **Age Analysis**: Age patterns by churn status.
- **Complaints Impact**: Correlation between complaints and churn.
- **Usage Patterns**: Monthly usage vs transactions scatter plot.
- **Subscription Analysis**: Churn rates by subscription tier.
- **Correlation Heatmap**: Feature correlation matrix.
- **Feature Importance**: Top predictive features from the model.

### 4. **Theme Toggle** 🌓
- Switch between **Light** and **Dark** modes in the sidebar.
- Dynamic CSS injection ensures consistent styling for all widgets and text.

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download the Repository

```bash
git clone <repository-url>
cd Churning
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Running the Streamlit Application

```bash
streamlit run app.py
```

The application will open automatically in your default browser at `http://localhost:8501`.

---

## 🧠 Model Details

### Architecture
The model is a **Stacking Classifier** that combines the predictions of multiple base estimators:
1.  **Logistic Regression**
2.  **Random Forest Classifier**
3.  **Gradient Boosting Classifier**
4.  **XGBoost Classifier**

The final prediction is made by a meta-learner (Logistic Regression) based on the outputs of these base models.

### Key Features
The model uses the following features for prediction:
- **Age**: Customer's age.
- **Gender**: Customer's gender.
- **Monthly Usage Hours**: Average hours watched per month.
- **Number of Transactions**: Total transactions.
- **Subscription Type**: Basic, Standard, Premium, or Gold.
- **Complaints**: Number of complaints filed.
- **Days Since Last Login**: Recency of user activity (Critical feature).
- **Engineered Features**: `Usage Intensity`, `Complaint Ratio`, `Log Transformed Usage`.

---

## 📊 Dataset Information

### Source
Netflix Customer Churn Dataset

### Features Description

| Feature | Type | Description |
|---------|------|-------------|
| **Age** | Numeric | Customer age (18-70) |
| **Gender** | Categorical | Male, Female, Other |
| **MonthlyUsageHours** | Numeric | Average monthly usage (0-200) |
| **NumTransactions** | Numeric | Transactions per month (1-50) |
| **SubscriptionType** | Categorical | Basic, Standard, Premium, Gold |
| **Complaints** | Numeric | Number of complaints (0-10) |
| **LastLoginDays** | Numeric | Days since last login (0-365) |
| **Churn** | Binary | Target variable (0=No, 1=Yes) |

---

## 🔧 Troubleshooting

### Model Won't Load
**Error**: `STACK_GLOBAL requires str` or pickle errors.
**Solution**: Retrain the model using the provided script:
```bash
python Customer_Churn_Prediction_.py
```

### Streamlit Command Not Found
**Error**: `'streamlit' is not recognized...`
**Solution**: Ensure your virtual environment is activated:
```bash
venv\Scripts\activate
```

### Prediction Errors
**Error**: Missing columns in Batch Prediction.
**Solution**: Ensure your CSV includes the `LastLoginDays` column. Download the updated example template from the "Batch Prediction" page.

---

**Built with ❤️ using Python, Streamlit, and Machine Learning**
