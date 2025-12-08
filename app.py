"""
Customer Churn Prediction Streamlit Application
A comprehensive ML-powered app for predicting customer churn with interactive visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Try to import joblib for more robust model loading
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Configuration
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Single Prediction", "📁 Batch Prediction", "📊 EDA Dashboard", "ℹ️ About Dataset"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Settings")
dark_mode = st.sidebar.toggle("Dark Mode", value=True)

# Define Theme Colors
if dark_mode:
    # Dark Mode Colors
    primary_color = "#1f77b4"
    background_color = "#0e1117"
    secondary_background_color = "#262730"
    text_color = "#fafafa"
    card_shadow = "rgba(0, 0, 0, 0.3)"
else:
    # Light Mode Colors
    primary_color = "#1f77b4"
    background_color = "#ffffff"
    secondary_background_color = "#f0f2f6"
    text_color = "#31333F"
    card_shadow = "rgba(0, 0, 0, 0.1)"

# Custom CSS for better styling
st.markdown(f"""
    <style>
    /* Force Theme Colors */
    :root {{
        --primary-color: {primary_color};
        --background-color: {background_color};
        --secondary-background-color: {secondary_background_color};
        --text-color: {text_color};
    }}
    
    /* App Background */
    .stApp {{
        background-color: {background_color};
        color: {text_color};
    }}
    
    /* Sidebar Background */
    section[data-testid="stSidebar"] {{
        background-color: {secondary_background_color};
        color: {text_color};
    }}
    
    /* Main container styling */
    .main {{
        padding: 2rem;
    }}
    
    /* Card-like container for sections */
    .st-card {{
        background-color: {secondary_background_color};
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px {card_shadow};
        margin-bottom: 20px;
        color: {text_color};
    }}
    
    /* Metric styling */
    div[data-testid="stMetricValue"] {{
        font-size: 28px;
        font-weight: bold;
        color: {primary_color};
    }}
    
    /* Custom prediction box styling */
    .prediction-card {{
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 15px;
        box-shadow: 0 4px 15px {card_shadow};
        transition: transform 0.3s ease;
    }}
    .prediction-card:hover {{
        transform: translateY(-5px);
    }}
    
    .success-card {{
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        color: #155724;
        border: 1px solid #c3e6cb;
    }}
    
    .warning-card {{
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 99%, #fecfef 100%);
        color: #721c24;
        border: 1px solid #f5c6cb;
    }}
    
    .prediction-title {{
        font-size: 24px;
        font-weight: 800;
        margin-bottom: 10px;
    }}
    
    .prediction-subtitle {{
        font-size: 16px;
        opacity: 0.9;
    }}
    
    /* Interpretation box styling */
    .interpretation-box {{
        background-color: {secondary_background_color};
        border-left: 5px solid {primary_color};
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
        color: {text_color};
    }}
    
    /* Headers */
    h1, h2, h3 {{
        font-weight: 700;
        color: {text_color} !important;
    }}
    
    /* Divider */
    hr {{
        margin: 30px 0;
        border-color: {secondary_background_color};
    }}
    
    /* Plotly Chart Backgrounds */
    .js-plotly-plot .plotly .main-svg {{
        background: rgba(0,0,0,0) !important;
    }}
    
    /* Global Text Color Override */
    p, label, span, li, .stMarkdown {{
        color: {text_color} !important;
    }}
    
    /* Widget Labels */
    .stTextInput label, .stSelectbox label, .stSlider label, .stNumberInput label {{
        color: {text_color} !important;
    }}
    
    /* Input Widget Backgrounds & Text */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div,
    .stNumberInput > div > div > input {{
        background-color: {secondary_background_color} !important;
        color: {text_color} !important;
    }}
    
    /* Selectbox Dropdown Options */
    div[data-baseweb="popover"] {{
        background-color: {secondary_background_color} !important;
    }}
    div[data-baseweb="menu"] {{
        background-color: {secondary_background_color} !important;
        color: {text_color} !important;
    }}
    
    /* Expander and other containers */
    .streamlit-expanderHeader {{
        background-color: {secondary_background_color} !important;
        color: {text_color} !important;
    }}
    
    /* Alert Boxes (Info, Success, Warning, Error) */
    /* We need to ensure text is readable. 
       In Light Mode (Dark Text), standard alerts (light bg) work.
       In Dark Mode (White Text), standard alerts (dark bg) work.
       Since we are forcing text color globally, we need to be careful with alerts.
       We'll let alerts inherit the global text color which should match the theme.
    */
    div[data-testid="stAlert"] {{
        background-color: {secondary_background_color};
        color: {text_color};
    }}
    div[data-testid="stAlert"] p {{
        color: {text_color} !important;
    }}
    
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}

# Helper Functions
@st.cache_resource
def load_model(model_path):
    """Load the pre-trained ML model"""
    errors = []
    
    # Try loading with pickle first
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model, None
    except FileNotFoundError:
        return None, "Model file not found. Please ensure 'model.pkl' is in the root directory."
    except Exception as e:
        errors.append(f"Pickle loading failed: {type(e).__name__}: {str(e)}")
    
    # Try loading with joblib if available
    if JOBLIB_AVAILABLE:
        try:
            model = joblib.load(model_path)
            return model, None
        except Exception as e:
            errors.append(f"Joblib loading failed: {type(e).__name__}: {str(e)}")
    
    # If both methods failed, provide comprehensive error message
    error_msg = f"""
    **Model Loading Failed**
    
    Attempted methods:
    {chr(10).join(f"- {err}" for err in errors)}
    
    **Common causes:**
    1. The model was trained with a different version of scikit-learn
    2. The Python version used to create the model differs from the current one
    3. The pickle file is corrupted or incompatible
    
    **Solutions to try:**
    1. Retrain the model with the current environment
    2. Check if the model file is compatible with your scikit-learn version
    3. Install joblib: `pip install joblib`
    
    **Current environment:**
    - Python: {__import__('sys').version.split()[0]}
    - scikit-learn: {__import__('sklearn').__version__}
    - joblib available: {JOBLIB_AVAILABLE}
    """
    return None, error_msg

@st.cache_data
def load_dataset(file_path=None, uploaded_file=None):
    """Load the dataset from file path or uploaded file"""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        elif file_path and Path(file_path).exists():
            df = pd.read_csv(file_path)
        else:
            return None, "Dataset not found. Please upload a CSV file."
        return df, None
    except Exception as e:
        return None, f"Error loading dataset: {str(e)}"

def preprocess_input(age, gender, monthly_usage, num_transactions, subscription_type, complaints, last_login):
    """Preprocess user input to match model training format"""
    # Create input dataframe with initial columns
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'monthly_usage_hours': [monthly_usage],
        'num_transactions': [num_transactions],
        'subscription_type': [subscription_type],
        'complaints': [complaints],
        'last_login_days': [last_login]
    })
    
    # Feature Engineering (must match training logic)
    # 1. Usage Intensity
    input_data["usage_intensity"] = input_data["monthly_usage_hours"] / (input_data["num_transactions"] + 1e-6)
    
    # 2. Complaint Ratio
    input_data["complaint_ratio"] = input_data["complaints"] / (input_data["num_transactions"] + 1e-6)
    
    # 3. Log Transforms
    input_data["log_monthly_usage"] = np.log1p(input_data["monthly_usage_hours"])
    input_data["log_num_transactions"] = np.log1p(input_data["num_transactions"])
    
    # 4. Age Bucket
    def get_age_bucket(a):
        if a <= 30:
            return "Young"
        elif a <= 50:
            return "Adult"
        else:
            return "Senior"
            
    input_data["age_bucket"] = input_data["age"].apply(get_age_bucket)
    
    # Note: Label encoding is NOT needed here because the pipeline handles it via OneHotEncoder
    # The pipeline expects raw strings for categorical variables: 'gender', 'subscription_type', 'age_bucket'
    
    return input_data

def get_feature_importance(model, feature_names):
    """Extract feature importance from the model pipeline"""
    try:
        # Try to get transformed feature names from the pipeline first
        # This is crucial because the model's importances correspond to transformed features
        if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
            preprocessor = model.named_steps['preprocessor']
            try:
                if hasattr(preprocessor, 'get_feature_names_out'):
                    # This works for scikit-learn >= 1.0
                    feature_names = preprocessor.get_feature_names_out()
            except:
                pass # Fallback to provided feature_names

        # Case 0: Check for injected feature importances on the main object (from our retraining)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            if len(importances) == len(feature_names):
                return pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=True)

        # Check if model is a pipeline
        if hasattr(model, 'named_steps'):
            classifier = model.named_steps['classifier']
            
            # Case 1: Stacking Classifier
            if hasattr(classifier, 'estimators_'):
                # Try to find a tree-based model among base estimators
                for estimator in classifier.estimators_:
                    if hasattr(estimator, 'feature_importances_'):
                        importances = estimator.feature_importances_
                        # Match lengths if possible
                        if len(importances) == len(feature_names):
                            return pd.DataFrame({
                                'Feature': feature_names,
                                'Importance': importances
                            }).sort_values('Importance', ascending=True)
            
            # Case 2: Single Tree-based Model
            elif hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                if len(importances) == len(feature_names):
                    return pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=True)
                    
            # Case 3: Linear Model (use coefficients)
            elif hasattr(classifier, 'coef_'):
                importances = np.abs(classifier.coef_[0])
                if len(importances) == len(feature_names):
                    return pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=True)
                    
        return None
    except Exception as e:
        # st.error(f"Error extracting feature importance: {e}")
        return None



# Load model automatically
if st.session_state.model is None:
    model, error = load_model('model.pkl')
    if model:
        st.session_state.model = model
    elif error:
        st.sidebar.error(f"⚠️ {error}")

# Load dataset automatically
if st.session_state.dataset is None:
    # Try default paths
    default_paths = [
        'netflix_customer_churn.csv',
        'netflix_customer_churn_with_new_columns.csv',
        'data/netflix_customer_churn.csv',
    ]
    for path in default_paths:
        df, error = load_dataset(file_path=path)
        if df is not None:
            st.session_state.dataset = df
            break


# PAGE 1: SINGLE PREDICTION
if page == "🏠 Single Prediction":
    st.title("🎯 Single Customer Churn Prediction")
    st.markdown("### Predict whether a customer will churn based on their profile")
    
    if st.session_state.model is None:
        st.error("⚠️ **Model not loaded!** Please upload a model file in the sidebar or ensure 'model.pkl' exists in the root directory.")
    else:
        st.success("✅ Model is ready for predictions!")
        
        # Model Details Expander
        with st.expander("ℹ️ View Model Details"):
            model = st.session_state.model
            st.markdown("### Model Architecture")
            
            # Check if it's a pipeline
            if hasattr(model, 'named_steps'):
                st.write("**Type:** Scikit-learn Pipeline")
                st.write("**Steps:**")
                for step_name, step_obj in model.named_steps.items():
                    st.code(f"{step_name}: {type(step_obj).__name__}")
                    
                # If classifier is StackingClassifier, show base estimators
                if 'classifier' in model.named_steps:
                    clf = model.named_steps['classifier']
                    if hasattr(clf, 'estimators_'):
                        st.markdown("#### Stacking Ensemble Details")
                        st.write(f"**Final Estimator:** {type(clf.final_estimator_).__name__}")
                        st.write("**Base Estimators:**")
                        for estimator in clf.estimators_:
                            st.code(f"{type(estimator).__name__}")
            else:
                st.write(f"**Type:** {type(model).__name__}")
                
            st.markdown("### Expected Features")
            st.code("""
- Age (Numeric)
- Gender (Categorical)
- MonthlyUsageHours (Numeric)
- NumTransactions (Numeric)
- SubscriptionType (Categorical)
- Complaints (Numeric)
- LastLoginDays (Numeric)
            """)
    
    st.markdown("---")
    
    # Input Section
    st.subheader("📝 Enter Customer Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider(
            "Age",
            min_value=18,
            max_value=70,
            value=35,
            help="Customer's age (18-70 years)"
        )
        
        gender = st.selectbox(
            "Gender",
            options=['Male', 'Female', 'Other'],
            help="Customer's gender"
        )
        
        last_login = st.slider(
            "Days Since Last Login",
            min_value=0,
            max_value=365,
            value=10,
            help="Number of days since the customer last logged in"
        )
    
    with col2:
        monthly_usage = st.slider(
            "Monthly Usage Hours",
            min_value=0,
            max_value=200,
            value=50,
            help="Average monthly usage hours (0-200)"
        )
        
        num_transactions = st.slider(
            "Number of Transactions",
            min_value=1,
            max_value=50,
            value=10,
            help="Number of transactions per month (1-50)"
        )
    
    with col3:
        subscription_type = st.selectbox(
            "Subscription Type",
            options=['Basic', 'Standard', 'Premium', 'Gold'],
            help="Customer's subscription tier"
        )
        
        complaints = st.slider(
            "Number of Complaints",
            min_value=0,
            max_value=10,
            value=0,
            help="Number of complaints filed (0-10)"
        )
    
    st.markdown("---")
    
    # Prediction Section
    if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
        if st.session_state.model is None:
            st.error("❌ Cannot make prediction: Model not loaded!")
        else:
            try:
                # Preprocess input
                input_data = preprocess_input(
                    age, gender, monthly_usage, 
                    num_transactions, subscription_type, complaints, last_login
                )
                
                # Make prediction
                prediction = st.session_state.model.predict(input_data)[0]
                
                # Get probability if available
                try:
                    probability = st.session_state.model.predict_proba(input_data)[0]
                    churn_probability = probability[1] * 100  # Probability of churn
                except:
                    churn_probability = None
                
                # Display Results
                st.markdown("### 🎯 Prediction Results")
                
                # Create a container for results with card styling
                with st.container():
                    col1, col2, col3 = st.columns([1, 1, 1.2], gap="large")
                    
                    with col1:
                        st.markdown("#### Status")
                        if prediction == 1:
                            st.markdown("""
                                <div class="prediction-card warning-card">
                                    <div class="prediction-title">⚠️ Will Churn</div>
                                    <div class="prediction-subtitle">High Risk Customer</div>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                                <div class="prediction-card success-card">
                                    <div class="prediction-title">✅ Will Not Churn</div>
                                    <div class="prediction-subtitle">Low Risk Customer</div>
                                </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("#### Probability")
                        if churn_probability is not None:
                            st.metric(
                                "Churn Probability",
                                f"{churn_probability:.1f}%",
                                delta=f"{churn_probability - 50:.1f}%",
                                delta_color="inverse"
                            )
                            
                            # Probability gauge
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=churn_probability,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                gauge={
                                    'axis': {'range': [None, 100], 'tickwidth': 1},
                                    'bar': {'color': "darkred" if churn_probability > 50 else "darkgreen"},
                                    'bgcolor': "white",
                                    'borderwidth': 2,
                                    'bordercolor': "gray",
                                    'steps': [
                                        {'range': [0, 33], 'color': "#d4fc79"},
                                        {'range': [33, 66], 'color': "#ffe259"},
                                        {'range': [66, 100], 'color': "#ff9a9e"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 50
                                    }
                                }
                            ))
                            fig.update_layout(height=160, margin=dict(l=10, r=10, t=10, b=10))
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col3:
                        st.markdown("#### Interpretation")
                        if prediction == 1:
                            if churn_probability is not None and churn_probability > 75:
                                st.markdown("""
                                    <div class="interpretation-box">
                                        <strong>🚨 High Risk Alert!</strong>
                                        <ul>
                                            <li>Churn probability is very high</li>
                                            <li>Immediate retention action recommended</li>
                                            <li>Consider personalized offers or support</li>
                                        </ul>
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                    <div class="interpretation-box">
                                        <strong>⚠️ Moderate Risk</strong>
                                        <ul>
                                            <li>Customer shows signs of potential churn</li>
                                            <li>Monitor engagement closely</li>
                                            <li>Proactive outreach suggested</li>
                                        </ul>
                                    </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                                <div class="interpretation-box">
                                    <strong>🌟 Low Risk Customer</strong>
                                    <ul>
                                        <li>Customer is likely to stay</li>
                                        <li>Continue providing quality service</li>
                                        <li>Opportunity for upselling</li>
                                    </ul>
                                </div>
                            """, unsafe_allow_html=True)
                
                # Feature Summary
                st.markdown("---")
                st.subheader("📋 Customer Profile Summary")
                
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                
                with summary_col1:
                    st.metric("Age", f"{age} years")
                    st.metric("Gender", gender)
                
                with summary_col2:
                    st.metric("Monthly Usage", f"{monthly_usage} hours")
                    st.metric("Transactions", num_transactions)
                
                with summary_col3:
                    st.metric("Subscription", subscription_type)
                    st.metric("Complaints", complaints)
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
                st.info("Please ensure the model is compatible with the input features.")

# PAGE 2: BATCH PREDICTION
elif page == "📁 Batch Prediction":
    st.title("📁 Batch Customer Churn Prediction")
    st.markdown("### Upload a CSV file to predict churn for multiple customers")
    
    if st.session_state.model is None:
        st.error("⚠️ **Model not loaded!** Please upload a model file in the sidebar or ensure 'test-model.pkl' exists in the root directory.")
    else:
        st.success("✅ Model is ready for batch predictions!")
    
    st.markdown("---")
    
    # File upload section
    st.subheader("📤 Upload Data for Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        prediction_file = st.file_uploader(
            "Upload CSV file with customer data",
            type=['csv'],
            help="CSV should contain: Age, Gender, MonthlyUsageHours, NumTransactions, SubscriptionType, Complaints"
        )
    
    with col2:
        st.info("""
            **Required Columns:**
            - Age
            - Gender
            - MonthlyUsageHours
            - NumTransactions
            - SubscriptionType
            - Complaints
        """)
    
    if prediction_file is not None:
        try:
            # Load the uploaded file
            input_df = pd.read_csv(prediction_file)
            
            st.markdown("---")
            st.subheader("📋 Uploaded Data Preview")
            st.dataframe(input_df.head(10), use_container_width=True)
            
            # Check for required columns
            required_cols = ['Age', 'Gender', 'MonthlyUsageHours', 'NumTransactions', 'SubscriptionType', 'Complaints', 'LastLoginDays']
            missing_cols = [col for col in required_cols if col not in input_df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            else:
                st.success(f"✅ All required columns present! ({len(input_df)} rows loaded)")
                
                # Predict button
                if st.button("🔮 Predict Churn for All Customers", type="primary", use_container_width=True):
                    if st.session_state.model is None:
                        st.error("❌ Cannot make predictions: Model not loaded!")
                    else:
                        try:
                            with st.spinner("Making predictions..."):
                                # Preprocess the data
                                processed_df = input_df.copy()
                                
                                # Rename columns to match training data if needed
                                col_mapping = {
                                    'Age': 'age',
                                    'Gender': 'gender',
                                    'MonthlyUsageHours': 'monthly_usage_hours',
                                    'NumTransactions': 'num_transactions',
                                    'SubscriptionType': 'subscription_type',
                                    'Complaints': 'complaints',
                                    'LastLoginDays': 'last_login_days'
                                }
                                processed_df = processed_df.rename(columns=col_mapping)
                                
                                # Feature Engineering
                                processed_df["usage_intensity"] = processed_df["monthly_usage_hours"] / (processed_df["num_transactions"] + 1e-6)
                                processed_df["complaint_ratio"] = processed_df["complaints"] / (processed_df["num_transactions"] + 1e-6)
                                processed_df["log_monthly_usage"] = np.log1p(processed_df["monthly_usage_hours"])
                                processed_df["log_num_transactions"] = np.log1p(processed_df["num_transactions"])
                                
                                def get_age_bucket(a):
                                    if a <= 30:
                                        return "Young"
                                    elif a <= 50:
                                        return "Adult"
                                    else:
                                        return "Senior"
                                processed_df["age_bucket"] = processed_df["age"].apply(get_age_bucket)
                                
                                # Make predictions
                                # The pipeline expects specific columns
                                model_cols = ["age", "gender", "monthly_usage_hours", "num_transactions", "complaints", "subscription_type", "usage_intensity", "complaint_ratio", "log_monthly_usage", "log_num_transactions", "age_bucket", "last_login_days"]
                                predictions = st.session_state.model.predict(processed_df[model_cols])
                                
                                # Get probabilities if available
                                try:
                                    probabilities = st.session_state.model.predict_proba(processed_df[model_cols])
                                    churn_probabilities = probabilities[:, 1] * 100
                                except:
                                    churn_probabilities = None
                                
                                # Add predictions to the original dataframe
                                results_df = input_df.copy()
                                results_df['Churn_Prediction'] = predictions
                                results_df['Churn_Prediction_Label'] = results_df['Churn_Prediction'].map({0: 'Will Not Churn', 1: 'Will Churn'})
                                
                                if churn_probabilities is not None:
                                    results_df['Churn_Probability_%'] = churn_probabilities.round(2)
                                    results_df['Risk_Level'] = pd.cut(
                                        churn_probabilities,
                                        bins=[0, 33, 66, 100],
                                        labels=['Low', 'Medium', 'High']
                                    )
                                
                                # Display results
                                st.markdown("---")
                                st.subheader("🎯 Prediction Results")
                                
                                # Summary metrics
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Total Customers", len(results_df))
                                
                                with col2:
                                    churned_count = (predictions == 1).sum()
                                    st.metric("Predicted to Churn", churned_count)
                                
                                with col3:
                                    churn_rate = (churned_count / len(results_df)) * 100
                                    st.metric("Predicted Churn Rate", f"{churn_rate:.1f}%")
                                
                                with col4:
                                    retention_count = (predictions == 0).sum()
                                    st.metric("Predicted to Retain", retention_count)
                                
                                # Results table
                                st.markdown("---")
                                st.subheader("📊 Detailed Predictions")
                                st.dataframe(results_df, use_container_width=True)
                                
                                # Download button
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Predictions as CSV",
                                    data=csv,
                                    file_name="churn_predictions.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                                
                                # Visualizations
                                st.markdown("---")
                                st.subheader("📈 Prediction Analysis")
                                
                                # Prediction distribution
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    fig1 = go.Figure(data=[
                                        go.Pie(
                                            labels=['Will Not Churn', 'Will Churn'],
                                            values=[(predictions == 0).sum(), (predictions == 1).sum()],
                                            marker=dict(colors=['#38ef7d', '#ee0979']),
                                            hole=.4
                                        )
                                    ])
                                    fig1.update_layout(
                                        title="Churn Prediction Distribution",
                                        height=400
                                    )
                                    st.plotly_chart(fig1, use_container_width=True)
                                
                                with col2:
                                    if churn_probabilities is not None:
                                        fig2 = go.Figure(data=[
                                            go.Histogram(
                                                x=churn_probabilities,
                                                nbinsx=20,
                                                marker_color='#667eea',
                                                opacity=0.7
                                            )
                                        ])
                                        fig2.update_layout(
                                            title="Churn Probability Distribution",
                                            xaxis_title="Churn Probability (%)",
                                            yaxis_title="Number of Customers",
                                            height=400
                                        )
                                        st.plotly_chart(fig2, use_container_width=True)
                                
                                # Risk level distribution
                                if churn_probabilities is not None:
                                    st.markdown("---")
                                    st.subheader("⚠️ Risk Level Distribution")
                                    
                                    risk_counts = results_df['Risk_Level'].value_counts()
                                    
                                    fig3 = go.Figure(data=[
                                        go.Bar(
                                            x=risk_counts.index,
                                            y=risk_counts.values,
                                            marker_color=['#38ef7d', '#ffd700', '#ee0979'],
                                            text=risk_counts.values,
                                            textposition='auto'
                                        )
                                    ])
                                    fig3.update_layout(
                                        title="Customer Risk Level Distribution",
                                        xaxis_title="Risk Level",
                                        yaxis_title="Number of Customers",
                                        height=400
                                    )
                                    st.plotly_chart(fig3, use_container_width=True)
                                    
                                    # High-risk customers
                                    high_risk = results_df[results_df['Risk_Level'] == 'High']
                                    if len(high_risk) > 0:
                                        st.markdown("---")
                                        st.subheader("🚨 High-Risk Customers (Immediate Action Required)")
                                        st.dataframe(high_risk, use_container_width=True)
                                        st.warning(f"⚠️ {len(high_risk)} customers are at high risk of churning. Consider immediate retention strategies.")
                                
                        except Exception as e:
                            st.error(f"❌ Error making predictions: {str(e)}")
                            st.info("Please ensure the CSV file has the correct format and column names.")
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.info("Please ensure the file is a valid CSV.")
    else:
        st.info("👆 Upload a CSV file to get started with batch predictions")
        
        # Show example format
        st.markdown("---")
        st.subheader("📝 Example CSV Format")
        
        example_data = pd.DataFrame({
            'Age': [35, 42, 28],
            'Gender': ['Male', 'Female', 'Other'],
            'MonthlyUsageHours': [50, 120, 30],
            'NumTransactions': [10, 25, 5],
            'SubscriptionType': ['Basic', 'Premium', 'Standard'],
            'Complaints': [0, 2, 5],
            'LastLoginDays': [10, 2, 20]
        })
        
        st.dataframe(example_data, use_container_width=True)
        
        # Download example CSV
        example_csv = example_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Example CSV Template",
            data=example_csv,
            file_name="example_churn_data.csv",
            mime="text/csv",
            use_container_width=True
        )

# PAGE 3: EDA DASHBOARD
elif page == "📊 EDA Dashboard":
    st.title("📊 Exploratory Data Analysis Dashboard")
    st.markdown("### Comprehensive visualization of customer churn patterns")
    
    if st.session_state.dataset is None:
        st.warning("⚠️ **No dataset loaded!** Please upload a dataset in the sidebar to view visualizations.")
        st.info("The dataset should contain columns: Age, Gender, MonthlyUsageHours, NumTransactions, SubscriptionType, Complaints, LastLoginDays, and Churn")
    else:

        df = st.session_state.dataset.copy()
        
        # Standardize column names to lowercase
        df.columns = df.columns.str.lower()
        
        # Check for required columns (now lowercase)
        required_cols = ['age', 'gender', 'monthly_usage_hours', 'num_transactions', 
                        'subscription_type', 'complaints', 'last_login_days']
        
        # Check if Churn column exists
        churn_col = None
        if 'churn' in df.columns:
            churn_col = 'churn'
        elif 'churned' in df.columns:
            churn_col = 'churned'
            df.rename(columns={'churned': 'churn'}, inplace=True)
            churn_col = 'churn'
            
        if churn_col is None:
            st.error("❌ Dataset must contain a 'Churn' column!")
        else:
            # Ensure proper capitalization for display if needed, but use lowercase for logic
            # We will use the lowercase names for all plotting code below
            
            churned = df[churn_col].sum()
            churn_rate = (churned / len(df)) * 100
            # Rename to standard 'Churn' if needed
            if churn_col != 'Churn':
                df['Churn'] = df[churn_col]
            
            st.success(f"✅ Dataset loaded successfully! ({len(df)} customers)")
            
            # Dataset Overview
            st.markdown("---")
            st.markdown('<div class="st-card">', unsafe_allow_html=True)
            st.subheader("📈 Dataset Overview")
            
            overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
            
            with overview_col1:
                st.metric("Total Customers", f"{len(df):,}")
            
            with overview_col2:
                churned = df['Churn'].sum()
                st.metric("Churned Customers", f"{churned:,}")
            
            with overview_col3:
                churn_rate = (churned / len(df)) * 100
                st.metric("Churn Rate", f"{churn_rate:.1f}%")
            
            with overview_col4:
                retention_rate = 100 - churn_rate
                st.metric("Retention Rate", f"{retention_rate:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # VISUALIZATION 1: Churn Distribution
            st.markdown('<div class="st-card">', unsafe_allow_html=True)
            st.subheader("1️⃣ Churn Distribution")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                churn_counts = df['churn'].value_counts()
                fig1 = go.Figure(data=[
                    go.Bar(
                        x=['Not Churned', 'Churned'],
                        y=[churn_counts.get(0, 0), churn_counts.get(1, 0)],
                        marker_color=['#38ef7d', '#ee0979'],
                        text=[churn_counts.get(0, 0), churn_counts.get(1, 0)],
                        textposition='auto',
                    )
                ])
                fig1.update_layout(
                    title="Customer Churn Distribution",
                    xaxis_title="Churn Status",
                    yaxis_title="Number of Customers",
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                st.markdown("### 📊 Insights")
                st.info(f"""
                    - **Total**: {len(df):,} customers
                    - **Churned**: {churned:,} ({churn_rate:.1f}%)
                    - **Retained**: {len(df) - churned:,} ({retention_rate:.1f}%)
                    - **Class Balance**: {'Imbalanced' if abs(churn_rate - 50) > 20 else 'Balanced'}
                """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # VISUALIZATION 2: Age Distribution by Churn
            st.markdown('<div class="st-card">', unsafe_allow_html=True)
            st.subheader("2️⃣ Age Distribution by Churn Status")
            
            if 'age' in df.columns:
                fig2 = go.Figure()
                
                for churn_status, color, name in [(0, '#38ef7d', 'Not Churned'), (1, '#ee0979', 'Churned')]:
                    data = df[df['churn'] == churn_status]['age']
                    fig2.add_trace(go.Histogram(
                        x=data,
                        name=name,
                        marker_color=color,
                        opacity=0.7,
                        nbinsx=20
                    ))
                
                fig2.update_layout(
                    title="Age Distribution: Churned vs Non-Churned Customers",
                    xaxis_title="Age",
                    yaxis_title="Count",
                    barmode='overlay',
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig2, use_container_width=True)
                
                # Age statistics
                col1, col2 = st.columns(2)
                with col1:
                    avg_age_churned = df[df['churn'] == 1]['age'].mean()
                    st.metric("Avg Age (Churned)", f"{avg_age_churned:.1f} years")
                with col2:
                    avg_age_retained = df[df['churn'] == 0]['age'].mean()
                    st.metric("Avg Age (Retained)", f"{avg_age_retained:.1f} years")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # VISUALIZATION 2.5: Gender Distribution (New)
            if 'gender' in df.columns:
                st.markdown('<div class="st-card">', unsafe_allow_html=True)
                st.subheader("2️⃣.5️⃣ Gender Distribution by Churn")
                
                gender_churn = df.groupby(['gender', 'churn']).size().unstack(fill_value=0)
                
                fig_gender = go.Figure()
                fig_gender.add_trace(go.Bar(
                    name='Not Churned',
                    x=gender_churn.index,
                    y=gender_churn[0] if 0 in gender_churn.columns else [0] * len(gender_churn),
                    marker_color='#38ef7d'
                ))
                fig_gender.add_trace(go.Bar(
                    name='Churned',
                    x=gender_churn.index,
                    y=gender_churn[1] if 1 in gender_churn.columns else [0] * len(gender_churn),
                    marker_color='#ee0979'
                ))
                
                fig_gender.update_layout(
                    title="Churn by Gender",
                    xaxis_title="Gender",
                    yaxis_title="Number of Customers",
                    barmode='group',
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_gender, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # VISUALIZATION 3: Complaints vs Churn
            st.markdown('<div class="st-card">', unsafe_allow_html=True)
            st.subheader("3️⃣ Complaints Analysis")
            
            if 'complaints' in df.columns:
                fig3 = go.Figure()
                
                for churn_status, color, name in [(0, '#38ef7d', 'Not Churned'), (1, '#ee0979', 'Churned')]:
                    data = df[df['churn'] == churn_status]['complaints']
                    fig3.add_trace(go.Box(
                        y=data,
                        name=name,
                        marker_color=color,
                        boxmean='sd'
                    ))
                
                fig3.update_layout(
                    title="Complaints Distribution by Churn Status",
                    yaxis_title="Number of Complaints",
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig3, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    avg_complaints_churned = df[df['churn'] == 1]['complaints'].mean()
                    st.metric("Avg Complaints (Churned)", f"{avg_complaints_churned:.2f}")
                with col2:
                    avg_complaints_retained = df[df['churn'] == 0]['complaints'].mean()
                    st.metric("Avg Complaints (Retained)", f"{avg_complaints_retained:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # VISUALIZATION 4: Monthly Usage Hours vs Churn
            st.markdown('<div class="st-card">', unsafe_allow_html=True)
            st.subheader("4️⃣ Monthly Usage Hours vs Churn")
            
            if 'monthly_usage_hours' in df.columns:
                fig4 = px.scatter(
                    df,
                    x='monthly_usage_hours',
                    y='num_transactions' if 'num_transactions' in df.columns else 'age',
                    color='churn',
                    color_discrete_map={0: '#38ef7d', 1: '#ee0979'},
                    labels={'churn': 'Churn Status'},
                    title="Monthly Usage Hours vs Transactions (colored by Churn)",
                    height=400,
                    opacity=0.6
                )
                fig4.update_traces(marker=dict(size=8))
                fig4.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig4, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    avg_usage_churned = df[df['churn'] == 1]['monthly_usage_hours'].mean()
                    st.metric("Avg Usage (Churned)", f"{avg_usage_churned:.1f} hrs")
                with col2:
                    avg_usage_retained = df[df['churn'] == 0]['monthly_usage_hours'].mean()
                    st.metric("Avg Usage (Retained)", f"{avg_usage_retained:.1f} hrs")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # VISUALIZATION 4.5: Transactions Distribution (New)
            if 'num_transactions' in df.columns:
                st.markdown('<div class="st-card">', unsafe_allow_html=True)
                st.subheader("4️⃣.5️⃣ Transactions Distribution")
                
                fig_trans = go.Figure()
                for churn_status, color, name in [(0, '#38ef7d', 'Not Churned'), (1, '#ee0979', 'Churned')]:
                    data = df[df['churn'] == churn_status]['num_transactions']
                    fig_trans.add_trace(go.Box(
                        y=data,
                        name=name,
                        marker_color=color,
                        boxmean='sd'
                    ))
                
                fig_trans.update_layout(
                    title="Number of Transactions by Churn Status",
                    yaxis_title="Number of Transactions",
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_trans, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # VISUALIZATION 5: Subscription Type vs Churn
            st.markdown('<div class="st-card">', unsafe_allow_html=True)
            st.subheader("5️⃣ Subscription Type Analysis")
            
            if 'subscription_type' in df.columns:
                subscription_churn = df.groupby(['subscription_type', 'churn']).size().unstack(fill_value=0)
                
                fig5 = go.Figure()
                fig5.add_trace(go.Bar(
                    name='Not Churned',
                    x=subscription_churn.index,
                    y=subscription_churn[0] if 0 in subscription_churn.columns else [0] * len(subscription_churn),
                    marker_color='#38ef7d'
                ))
                fig5.add_trace(go.Bar(
                    name='Churned',
                    x=subscription_churn.index,
                    y=subscription_churn[1] if 1 in subscription_churn.columns else [0] * len(subscription_churn),
                    marker_color='#ee0979'
                ))
                
                fig5.update_layout(
                    title="Churn by Subscription Type",
                    xaxis_title="Subscription Type",
                    yaxis_title="Number of Customers",
                    barmode='group',
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig5, use_container_width=True)
                
                # Churn rate by subscription
                st.markdown("### Churn Rate by Subscription Type")
                subscription_stats = df.groupby('subscription_type').agg({
                    'churn': ['sum', 'count', 'mean']
                }).round(3)
                subscription_stats.columns = ['Churned', 'Total', 'Churn Rate']
                subscription_stats['Churn Rate'] = (subscription_stats['Churn Rate'] * 100).round(1)
                st.dataframe(subscription_stats, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # VISUALIZATION 6: Correlation Heatmap
            st.markdown('<div class="st-card">', unsafe_allow_html=True)
            st.subheader("6️⃣ Feature Correlation Heatmap")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                
                fig6 = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_matrix.values.round(2),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    colorbar=dict(title="Correlation")
                ))
                
                fig6.update_layout(
                    title="Feature Correlation Matrix",
                    height=500,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig6, use_container_width=True)
                
                # Top correlations with Churn
                if 'churn' in corr_matrix.columns:
                    st.markdown("### 🔍 Top Correlations with Churn")
                    churn_corr = corr_matrix['churn'].drop('churn').abs().sort_values(ascending=False)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(churn_corr.head(5), use_container_width=True)
                    with col2:
                        st.info(f"""
                            **Strongest predictor**: {churn_corr.index[0]}
                            
                            **Correlation**: {churn_corr.iloc[0]:.3f}
                            
                            Higher values indicate stronger relationship with churn.
                        """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # VISUALIZATION 7: Feature Importance
            st.markdown('<div class="st-card">', unsafe_allow_html=True)
            st.subheader("7️⃣ Feature Importance")
            
            if st.session_state.model is not None:
                feature_names = ['Age', 'Gender', 'MonthlyUsageHours', 
                               'NumTransactions', 'SubscriptionType', 'Complaints']
                importance_df = get_feature_importance(st.session_state.model, feature_names)
                
                if importance_df is not None:
                    fig7 = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Feature Importance from ML Model",
                        color='Importance',
                        color_continuous_scale='Viridis',
                        height=400
                    )
                    fig7.update_layout(
                        showlegend=False,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig7, use_container_width=True)
                    
                    st.info(f"""
                        **Most Important Feature**: {importance_df.iloc[-1]['Feature']}
                        
                        This feature has the strongest influence on the model's churn predictions.
                    """)
                else:
                    st.warning("⚠️ Feature importance not available for this model type.")
            else:
                st.warning("⚠️ Load a model to view feature importance analysis.")
            st.markdown('</div>', unsafe_allow_html=True)

# PAGE 3: ABOUT DATASET
elif page == "ℹ️ About Dataset":
    st.title("ℹ️ About the Dataset")
    st.markdown("### Customer Churn Dataset Overview")
    
    if st.session_state.dataset is None:
        st.info("📁 Upload a dataset in the sidebar to view detailed information.")
        
        # Show expected schema
        st.markdown("---")
        st.subheader("📋 Expected Dataset Schema")
        
        schema_data = {
            'Feature': ['Age', 'Gender', 'MonthlyUsageHours', 'NumTransactions', 
                       'SubscriptionType', 'Complaints', 'Churn'],
            'Type': ['Numeric', 'Categorical', 'Numeric', 'Numeric', 
                    'Categorical', 'Numeric', 'Binary'],
            'Range/Values': ['18-70', 'Male/Female/Other', '0-200', '1-50', 
                           'Basic/Standard/Premium/Gold', '0-10', '0/1'],
            'Description': [
                'Customer age in years',
                'Customer gender',
                'Average monthly usage hours',
                'Number of monthly transactions',
                'Subscription tier level',
                'Number of complaints filed',
                'Target variable (1=Churned, 0=Retained)'
            ]
        }
        
        schema_df = pd.DataFrame(schema_data)
        st.dataframe(schema_df, use_container_width=True, hide_index=True)
        
    else:
        df = st.session_state.dataset
        
        st.success(f"✅ Dataset loaded successfully!")
        
        # Basic Statistics
        st.markdown("---")
        st.subheader("📊 Dataset Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        
        with col2:
            st.metric("Total Columns", len(df.columns))
        
        with col3:
            missing_values = df.isnull().sum().sum()
            st.metric("Missing Values", missing_values)
        
        with col4:
            duplicates = df.duplicated().sum()
            st.metric("Duplicate Rows", duplicates)
        
        # Column Information
        st.markdown("---")
        st.subheader("📋 Column Information")
        
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Unique Values': df.nunique()
        })
        st.dataframe(col_info, use_container_width=True, hide_index=True)
        
        # Statistical Summary
        st.markdown("---")
        st.subheader("📈 Statistical Summary")
        
        st.dataframe(df.describe(), use_container_width=True)
        
        # Sample Data
        st.markdown("---")
        st.subheader("👀 Sample Data")
        
        sample_size = st.slider("Number of rows to display", 5, 50, 10)
        st.dataframe(df.head(sample_size), use_container_width=True)
        
        # Data Quality
        st.markdown("---")
        st.subheader("✅ Data Quality Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Completeness")
            completeness = (1 - df.isnull().sum() / len(df)) * 100
            completeness_df = pd.DataFrame({
                'Column': completeness.index,
                'Completeness %': completeness.values.round(2)
            }).sort_values('Completeness %')
            
            fig = px.bar(
                completeness_df,
                x='Completeness %',
                y='Column',
                orientation='h',
                color='Completeness %',
                color_continuous_scale='RdYlGn',
                range_color=[0, 100]
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Data Types Distribution")
            dtype_counts = df.dtypes.value_counts()
            
            fig = go.Figure(data=[go.Pie(
                labels=dtype_counts.index.astype(str),
                values=dtype_counts.values,
                hole=.3
            )])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🎯 Customer Churn Prediction Application</p>
        <p>Built with Streamlit • Powered by Machine Learning</p>
    </div>
""", unsafe_allow_html=True)
