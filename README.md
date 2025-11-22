# 🎯 Customer Churn Prediction System

A comprehensive machine learning solution for predicting customer churn with an interactive Streamlit web application. This project includes end-to-end model training, evaluation, and deployment with rich visualizations and insights.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Streamlit Application](#streamlit-application)
- [Dataset Information](#dataset-information)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---
## ✒️ Design Architecture
![Design Architecture Image](https://github.com/user-attachments/assets/c3b1eebb-5dfd-4e3f-b315-03f326e1967e)

---

## 🖥️ Deployed Showcase
<img width="1913" height="910" alt="1" src="https://github.com/user-attachments/assets/b30a9757-cc06-4f8e-abd2-49f381f4e37b" />
<img width="1902" height="887" alt="2" src="https://github.com/user-attachments/assets/c84178b8-ea6e-4085-ac5f-9f7f61ec7b57" />
<img width="1885" height="892" alt="3" src="https://github.com/user-attachments/assets/1724fc96-00cd-4d63-afb8-d5701c484370" />
<img width="1908" height="875" alt="4" src="https://github.com/user-attachments/assets/b271aad2-a123-45f4-bc80-be6936e9066e" />
<img width="1918" height="892" alt="5" src="https://github.com/user-attachments/assets/e1cb9980-99ce-43db-91c5-3938c4ddd531" />

---

## 🌟 Overview

This project provides a complete customer churn prediction solution built with:
- **Machine Learning**: Stacking ensemble model achieving 76.8% accuracy and 0.849 ROC-AUC
- **Interactive Web App**: Streamlit-based dashboard for predictions and data exploration
- **Production-Ready**: Comprehensive error handling, model versioning, and deployment support

### Key Highlights
- ✅ **High Performance**: 76.8% accuracy with balanced precision and recall
- ✅ **Advanced ML**: Stacking ensemble combining multiple algorithms
- ✅ **Feature Engineering**: 15+ engineered features for better predictions
- ✅ **Interactive UI**: Real-time predictions with visual risk assessment
- ✅ **Comprehensive EDA**: 7+ interactive visualizations for data insights

---

## 📁 Project Structure

```
Churning/
│
├── Customer_Churn_Prediction_.ipynb  # Complete model training notebook
├── app.py                             # Streamlit web application
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
│
├── netflix_customer_churn.csv         # Training dataset
├── model.pkl                          # Trained model (pickle format)
│
└── output/
    ├── best_model_pipeline.pkl        # Best model from training
    ├── best_model_summary.txt         # Model performance summary
    └── model_comparison_summary.csv   # Comparison of all models
```

---

## 🏆 Model Performance

### Best Model: Stacking Ensemble

The final model is a **Stacking Ensemble** that combines multiple base learners for superior performance.

#### Performance Metrics (Test Set)

| Metric | Score |
|--------|-------|
| **Accuracy** | 76.8% |
| **Precision** | 77.6% |
| **Recall** | 75.7% |
| **F1-Score** | 76.7% |
| **ROC-AUC** | 84.9% |

#### Confusion Matrix

```
                Predicted
                No    Yes
Actual  No     387    110
        Yes    122    381
```

- **True Negatives**: 387 (correctly identified non-churners)
- **False Positives**: 110 (incorrectly flagged as churners)
- **False Negatives**: 122 (missed churners)
- **True Positives**: 381 (correctly identified churners)

#### Model Comparison

| Model | Mean ROC-AUC | Std Dev |
|-------|-------------|---------|
| **Stacking Ensemble** | **0.849** | - |
| Gradient Boosting | 0.824 | 0.006 |
| Logistic Regression | 0.819 | 0.010 |
| Random Forest | 0.818 | 0.011 |
| XGBoost | 0.796 | 0.010 |

---

## 🎨 Features

### 1. **Single Customer Prediction** 🔮
- Interactive input form for customer attributes
- Real-time churn prediction with probability
- Visual risk gauge (0-100%)
- Actionable recommendations based on risk level
- Customer profile summary

### 2. **Batch Prediction** 📁
- Upload CSV files for bulk predictions
- Predict churn for thousands of customers at once
- Download results with risk levels
- Visual analysis of prediction distribution
- Identify high-risk customers automatically

### 3. **Exploratory Data Analysis Dashboard** 📊
Seven comprehensive visualizations:
1. **Churn Distribution**: Overall churn rate and class balance
2. **Age Analysis**: Age patterns by churn status
3. **Complaints Impact**: Box plots showing complaint correlation
4. **Usage Patterns**: Monthly usage vs transactions scatter plot
5. **Subscription Analysis**: Churn rates by subscription tier
6. **Correlation Heatmap**: Feature correlation matrix
7. **Feature Importance**: Top predictive features from the model

### 4. **About Dataset** ℹ️
- Dataset statistics and overview
- Column information and data types
- Statistical summary
- Data quality report
- Sample data viewer

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

### Required Packages
- `streamlit==1.29.0` - Web application framework
- `pandas==2.1.3` - Data manipulation
- `numpy==1.26.2` - Numerical computing
- `plotly==5.18.0` - Interactive visualizations
- `scikit-learn==1.6.1` - Machine learning
- `imbalanced-learn==0.12.0` - Handling imbalanced datasets
- `joblib` - Model serialization

---

## 💻 Usage

### Running the Streamlit Application

```bash
streamlit run app.py
```

The application will open automatically in your default browser at `http://localhost:8501`

### Quick Start Guide

See [QUICKSTART.md](QUICKSTART.md) for a step-by-step walkthrough.

---

## 🧠 Model Training

### Training Process

The model was trained using the Jupyter notebook `Customer_Churn_Prediction_.ipynb` with the following pipeline:

#### 1. **Data Preprocessing**
- Handled missing values
- Encoded categorical variables (Gender, Subscription Type)
- Feature scaling and normalization

#### 2. **Feature Engineering**
Created 15+ engineered features:
- `log_monthly_usage`: Log transformation of usage hours
- `log_num_transactions`: Log transformation of transactions
- `usage_intensity`: Usage per transaction ratio
- `complaint_ratio`: Complaints per transaction
- `age_bucket`: Age categorization (Young/Adult/Senior)
- One-hot encoding for categorical variables

#### 3. **Model Selection**
Trained and compared 5 different models:
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- **Stacking Ensemble** (Final choice)

#### 4. **Hyperparameter Tuning**
- Cross-validation with 5 folds
- Grid search for optimal parameters
- Stratified sampling to handle class imbalance

#### 5. **Model Evaluation**
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC curve analysis
- Confusion matrix
- Feature importance analysis

### Top Predictive Features

| Feature | Importance |
|---------|-----------|
| Monthly Usage Hours | 22.1% |
| Log Monthly Usage | 22.0% |
| Usage Intensity | 13.5% |
| Age | 7.8% |
| Complaint Ratio | 7.7% |
| Log Num Transactions | 6.2% |
| Num Transactions | 6.1% |
| Complaints | 4.6% |

### Retraining the Model

If you need to retrain the model with your current environment:

```bash
python retrain_model.py
```

This will:
- Load the dataset
- Preprocess features
- Train a Random Forest model
- Save the model as `model.pkl`
- Display performance metrics

**Note**: The retrain script uses a simplified Random Forest model. For the full Stacking Ensemble, use the Jupyter notebook.

---

## 🖥️ Streamlit Application

### Page 1: Single Prediction

**Input Features:**
- **Age**: 18-70 years (slider)
- **Gender**: Male/Female/Other (dropdown)
- **Monthly Usage Hours**: 0-200 hours (slider)
- **Number of Transactions**: 1-50 (slider)
- **Subscription Type**: Basic/Standard/Premium/Gold (dropdown)
- **Number of Complaints**: 0-10 (slider)

**Output:**
- Churn prediction (Yes/No)
- Churn probability (0-100%)
- Risk level gauge
- Actionable recommendations
- Customer profile summary

### Page 2: Batch Prediction

**Features:**
- Upload CSV file with customer data
- Automatic validation of required columns
- Bulk prediction processing
- Results table with:
  - Original customer data
  - Churn prediction
  - Churn probability
  - Risk level (Low/Medium/High)
- Download predictions as CSV
- Visual analysis:
  - Prediction distribution pie chart
  - Probability histogram
  - Risk level bar chart
  - High-risk customer table

**CSV Format:**
```csv
Age,Gender,MonthlyUsageHours,NumTransactions,SubscriptionType,Complaints
35,Male,50,10,Basic,0
42,Female,120,25,Premium,2
28,Other,30,5,Standard,5
```

### Page 3: EDA Dashboard

**Visualizations:**
1. **Churn Distribution**: Bar chart showing churned vs retained customers
2. **Age Distribution**: Overlapping histograms by churn status
3. **Complaints Analysis**: Box plots comparing churned vs retained
4. **Usage Patterns**: Scatter plot of usage hours vs transactions
5. **Subscription Analysis**: Churn rates by subscription tier
6. **Correlation Heatmap**: Feature correlation matrix
7. **Feature Importance**: Bar chart of top predictive features

### Page 4: About Dataset

- Total customer count
- Churn rate statistics
- Column information and data types
- Statistical summary (mean, std, min, max, quartiles)
- Missing value report
- Sample data preview

---

## 📊 Dataset Information

### Source
Netflix Customer Churn Dataset

### Size
- **Total Customers**: 5,000+ records
- **Features**: 6 input features + 1 target variable
- **File**: `netflix_customer_churn.csv`

### Features Description

| Feature | Type | Description | Range/Values |
|---------|------|-------------|--------------|
| **Age** | Numeric | Customer age | 18-70 years |
| **Gender** | Categorical | Customer gender | Male, Female, Other |
| **MonthlyUsageHours** | Numeric | Average monthly usage | 0-200 hours |
| **NumTransactions** | Numeric | Transactions per month | 1-50 |
| **SubscriptionType** | Categorical | Subscription tier | Basic, Standard, Premium, Gold |
| **Complaints** | Numeric | Number of complaints | 0-10 |
| **Churn** | Binary | Target variable | 0 (No), 1 (Yes) |

### Data Characteristics
- **Class Balance**: Relatively balanced (check EDA dashboard for exact ratio)
- **Missing Values**: Handled during preprocessing
- **Outliers**: Minimal, within expected ranges
- **Correlations**: See correlation heatmap in EDA dashboard

---

## 🔧 Troubleshooting

### Model Won't Load

**Error**: `STACK_GLOBAL requires str` or pickle errors

**Cause**: Version mismatch between model training and current environment

**Solution**:
```bash
python retrain_model.py
```

This retrains the model with your current Python and scikit-learn versions.

**Alternative**: Check if `output/best_model_pipeline.pkl` exists and copy it to `model.pkl`:
```bash
# Windows
copy output\best_model_pipeline.pkl model.pkl

# macOS/Linux
cp output/best_model_pipeline.pkl model.pkl
```

### Dataset Not Found

**Error**: Dataset not loaded or file not found

**Solution**:
1. Ensure `netflix_customer_churn.csv` is in the root directory
2. Or upload the dataset via the sidebar in the Streamlit app
3. Check file name spelling (case-sensitive on Linux/macOS)

### Prediction Errors

**Error**: Predictions fail or return errors

**Possible Causes**:
1. **Feature mismatch**: Ensure input features match training data
2. **Encoding issues**: Check categorical variable encoding
3. **Missing values**: Verify all inputs are provided

**Solution**:
- Review the preprocessing in `app.py` (lines 137-156)
- Ensure categorical mappings match training:
  - Gender: Male=0, Female=1, Other=2
  - Subscription: Basic=0, Standard=1, Premium=2, Gold=3

### Streamlit App Crashes

**Error**: App crashes or shows errors

**Solution**:
1. Check Python version: `python --version` (should be 3.8+)
2. Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`
3. Clear Streamlit cache: Delete `.streamlit` folder
4. Check console for detailed error messages

### Version Compatibility

**Current Environment Requirements**:
- Python: 3.8 - 3.11 (3.11 recommended)
- scikit-learn: 1.6.1 or compatible
- If using different versions, retrain the model

**Check Versions**:
```bash
python --version
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
```

---

## 🛠️ Advanced Configuration

### Customizing the Model

To modify model parameters, edit `retrain_model.py`:

```python
model = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Maximum tree depth
    min_samples_split=5,   # Minimum samples to split
    min_samples_leaf=2,    # Minimum samples per leaf
    random_state=42,       # Reproducibility
    n_jobs=-1             # Use all CPU cores
)
```

### Customizing the UI

Edit CSS in `app.py` (lines 34-65):

```python
st.markdown("""
    <style>
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        # Change colors here
    }
    </style>
""", unsafe_allow_html=True)
```

### Adding New Features

1. **Update Dataset**: Add new columns to CSV
2. **Modify Preprocessing**: Update `preprocess_input()` in `app.py`
3. **Retrain Model**: Run `retrain_model.py` or the Jupyter notebook
4. **Update UI**: Add input widgets in the prediction page

---

## 📈 Performance Optimization

### For Large Datasets
- Use batch prediction instead of single predictions
- Enable caching with `@st.cache_data` and `@st.cache_resource`
- Consider using `joblib` for faster model loading

### For Faster Predictions
- The app uses caching for model and dataset loading
- Predictions are computed on-demand
- Visualizations are cached when possible

---

## 🚧 Challenges Faced and Future Improvements

### Challenges Encountered

#### 1. **Model Deployment and Serialization Issues** 🔴

**Problem**: The trained model (`model.pkl`) fails to load in the deployed Streamlit environment due to library version mismatches.

**Root Cause**:
- The model was trained using specific versions of `scikit-learn` and `imbalanced-learn`
- Deployment environment may have different versions installed
- Pickle serialization is not version-agnostic and breaks with version mismatches
- Error: `STACK_GLOBAL requires str` - pickle compatibility issue between Python versions

**Impact**:
- Model pickel's predictions has been verified by random test-cases in the notebook attached itself.
- Model cannot be loaded in production environment
- Application fails to make predictions
- Requires manual intervention to fix version conflicts

**Current Workaround**:
- Pin exact library versions in `requirements.txt`
- Retrain model in deployment environment
- Use `joblib` instead of `pickle` for better compatibility


#### 2. **Library Version Compatibility**

**Issue**: `scikit-learn` and `imbalanced-learn` version conflicts across different environments

**Specific Problems**:
- Training environment: `scikit-learn==1.6.1`, `imbalanced-learn==0.12.0`
- Deployment environment: May have `scikit-learn==1.7.1` or other versions
- Breaking changes in scikit-learn API between versions
- SMOTE and other imbalanced-learn functions behave differently

**Solutions Attempted**:
- ✅ Strict version pinning in `requirements.txt`
- ✅ Virtual environment isolation
- ⚠️ Model retraining (time-consuming)
- ❌ Backward compatibility (not always possible)

---

### 🔮 Future Improvements

#### 1. **Model Deployment and Versioning** 🎯

**Priority**: HIGH

- [ ] **Implement MLflow** for model versioning and tracking
  - Track experiments, parameters, and metrics
  - Version models with metadata
  - Easy model rollback and comparison

- [ ] **Use ONNX format** for model serialization
  - Framework-agnostic model format
  - Better cross-platform compatibility
  - Faster inference

- [ ] **Docker containerization**
  - Package entire environment (Python + libraries + model)
  - Ensure consistency across development and production
  - Easier deployment and scaling

- [ ] **Model registry**
  - Centralized model storage
  - Version control for models
  - Automated deployment pipelines

---


## 📄 License

This project is open source and available for educational and commercial use.

---

## 🙏 Acknowledgments

- **Dataset**: Netflix Customer Churn Dataset
- **Framework**: Streamlit for the web application
- **ML Libraries**: scikit-learn, imbalanced-learn, XGBoost
- **Visualization**: Plotly for interactive charts

---

## 📧 Support

For questions or issues:

1. **Check this README** for common solutions
2. **Review the code comments** in `app.py` and notebooks
3. **Consult documentation**:
   - [Streamlit Docs](https://docs.streamlit.io)
   - [scikit-learn Docs](https://scikit-learn.org/stable/)
   - [Plotly Docs](https://plotly.com/python/)

---

## 🎯 Quick Commands Reference

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py





# Check Python version
python --version

# Check package versions
pip list | grep -E "streamlit|scikit-learn|pandas"
```

---

**Built with ❤️ using Python, Streamlit, and Machine Learning**

*Last Updated: November 2025*
