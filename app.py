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

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 20px 0;
    }
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .warning-box {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
    }
    h1 {
        color: #1f77b4;
    }
    h2 {
        color: #2c3e50;
    }
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

def preprocess_input(age, gender, monthly_usage, num_transactions, subscription_type, complaints):
    """Preprocess user input to match model training format"""
    # Create input dataframe
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'MonthlyUsageHours': [monthly_usage],
        'NumTransactions': [num_transactions],
        'SubscriptionType': [subscription_type],
        'Complaints': [complaints]
    })
    
    # Label encoding for categorical variables
    gender_mapping = {'Male': 0, 'Female': 1, 'Other': 2}
    subscription_mapping = {'Basic': 0, 'Standard': 1, 'Premium': 2, 'Gold': 3}
    
    input_data['Gender'] = input_data['Gender'].map(gender_mapping)
    input_data['SubscriptionType'] = input_data['SubscriptionType'].map(subscription_mapping)
    
    return input_data

def get_feature_importance(model, feature_names):
    """Extract feature importance from the model if available"""
    try:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            return pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=True)
        else:
            return None
    except:
        return None

# Sidebar Navigation
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Single Prediction", "📁 Batch Prediction", "📊 EDA Dashboard", "ℹ️ About Dataset"]
)

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
                    num_transactions, subscription_type, complaints
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
                
                col1, col2, col3 = st.columns([2, 2, 3])
                
                with col1:
                    if prediction == 1:
                        st.markdown("""
                            <div class="prediction-box warning-box">
                                <h2>⚠️ Will Churn</h2>
                                <p>High risk customer</p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div class="prediction-box success-box">
                                <h2>✅ Will Not Churn</h2>
                                <p>Low risk customer</p>
                            </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    if churn_probability is not None:
                        st.metric(
                            "Churn Probability",
                            f"{churn_probability:.1f}%",
                            delta=f"{churn_probability - 50:.1f}% from baseline"
                        )
                        
                        # Probability gauge
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=churn_probability,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Risk Level"},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkred" if churn_probability > 50 else "darkgreen"},
                                'steps': [
                                    {'range': [0, 33], 'color': "lightgreen"},
                                    {'range': [33, 66], 'color': "yellow"},
                                    {'range': [66, 100], 'color': "lightcoral"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50
                                }
                            }
                        ))
                        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig, use_container_width=True)
                
                with col3:
                    st.markdown("### 💡 Interpretation")
                    if prediction == 1:
                        if churn_probability is not None and churn_probability > 75:
                            st.warning("""
                                **High Risk Alert!**
                                - Churn probability is very high
                                - Immediate retention action recommended
                                - Consider personalized offers or support
                            """)
                        else:
                            st.info("""
                                **Moderate Risk**
                                - Customer shows signs of potential churn
                                - Monitor engagement closely
                                - Proactive outreach suggested
                            """)
                    else:
                        st.success("""
                            **Low Risk Customer**
                            - Customer is likely to stay
                            - Continue providing quality service
                            - Opportunity for upselling
                        """)
                
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
            required_cols = ['Age', 'Gender', 'MonthlyUsageHours', 'NumTransactions', 'SubscriptionType', 'Complaints']
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
                                
                                # Label encoding for categorical variables
                                gender_mapping = {'Male': 0, 'Female': 1, 'Other': 2}
                                subscription_mapping = {'Basic': 0, 'Standard': 1, 'Premium': 2, 'Gold': 3}
                                
                                processed_df['Gender'] = processed_df['Gender'].map(gender_mapping)
                                processed_df['SubscriptionType'] = processed_df['SubscriptionType'].map(subscription_mapping)
                                
                                # Make predictions
                                predictions = st.session_state.model.predict(processed_df[required_cols])
                                
                                # Get probabilities if available
                                try:
                                    probabilities = st.session_state.model.predict_proba(processed_df[required_cols])
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
            'Complaints': [0, 2, 5]
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
        st.info("The dataset should contain columns: Age, Gender, MonthlyUsageHours, NumTransactions, SubscriptionType, Complaints, and Churn")
    else:
        df = st.session_state.dataset.copy()
        
        # Check for required columns
        required_cols = ['Age', 'Gender', 'MonthlyUsageHours', 'NumTransactions', 
                        'SubscriptionType', 'Complaints']
        
        # Check if Churn column exists (might be named differently)
        churn_col = None
        for col in df.columns:
            if 'churn' in col.lower():
                churn_col = col
                break
        
        if churn_col is None:
            st.error("❌ Dataset must contain a 'Churn' column!")
        else:
            # Rename to standard 'Churn' if needed
            if churn_col != 'Churn':
                df['Churn'] = df[churn_col]
            
            st.success(f"✅ Dataset loaded successfully! ({len(df)} customers)")
            
            # Dataset Overview
            st.markdown("---")
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
            
            st.markdown("---")
            
            # VISUALIZATION 1: Churn Distribution
            st.subheader("1️⃣ Churn Distribution")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                churn_counts = df['Churn'].value_counts()
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
                    height=400
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
            
            st.markdown("---")
            
            # VISUALIZATION 2: Age Distribution by Churn
            st.subheader("2️⃣ Age Distribution by Churn Status")
            
            if 'Age' in df.columns:
                fig2 = go.Figure()
                
                for churn_status, color, name in [(0, '#38ef7d', 'Not Churned'), (1, '#ee0979', 'Churned')]:
                    data = df[df['Churn'] == churn_status]['Age']
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
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)
                
                # Age statistics
                col1, col2 = st.columns(2)
                with col1:
                    avg_age_churned = df[df['Churn'] == 1]['Age'].mean()
                    st.metric("Avg Age (Churned)", f"{avg_age_churned:.1f} years")
                with col2:
                    avg_age_retained = df[df['Churn'] == 0]['Age'].mean()
                    st.metric("Avg Age (Retained)", f"{avg_age_retained:.1f} years")
            
            st.markdown("---")
            
            # VISUALIZATION 3: Complaints vs Churn
            st.subheader("3️⃣ Complaints Analysis")
            
            if 'Complaints' in df.columns:
                fig3 = go.Figure()
                
                for churn_status, color, name in [(0, '#38ef7d', 'Not Churned'), (1, '#ee0979', 'Churned')]:
                    data = df[df['Churn'] == churn_status]['Complaints']
                    fig3.add_trace(go.Box(
                        y=data,
                        name=name,
                        marker_color=color,
                        boxmean='sd'
                    ))
                
                fig3.update_layout(
                    title="Complaints Distribution by Churn Status",
                    yaxis_title="Number of Complaints",
                    height=400
                )
                st.plotly_chart(fig3, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    avg_complaints_churned = df[df['Churn'] == 1]['Complaints'].mean()
                    st.metric("Avg Complaints (Churned)", f"{avg_complaints_churned:.2f}")
                with col2:
                    avg_complaints_retained = df[df['Churn'] == 0]['Complaints'].mean()
                    st.metric("Avg Complaints (Retained)", f"{avg_complaints_retained:.2f}")
            
            st.markdown("---")
            
            # VISUALIZATION 4: Monthly Usage Hours vs Churn
            st.subheader("4️⃣ Monthly Usage Hours vs Churn")
            
            if 'MonthlyUsageHours' in df.columns:
                fig4 = px.scatter(
                    df,
                    x='MonthlyUsageHours',
                    y='NumTransactions' if 'NumTransactions' in df.columns else 'Age',
                    color='Churn',
                    color_discrete_map={0: '#38ef7d', 1: '#ee0979'},
                    labels={'Churn': 'Churn Status'},
                    title="Monthly Usage Hours vs Transactions (colored by Churn)",
                    height=400,
                    opacity=0.6
                )
                fig4.update_traces(marker=dict(size=8))
                st.plotly_chart(fig4, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    avg_usage_churned = df[df['Churn'] == 1]['MonthlyUsageHours'].mean()
                    st.metric("Avg Usage (Churned)", f"{avg_usage_churned:.1f} hrs")
                with col2:
                    avg_usage_retained = df[df['Churn'] == 0]['MonthlyUsageHours'].mean()
                    st.metric("Avg Usage (Retained)", f"{avg_usage_retained:.1f} hrs")
            
            st.markdown("---")
            
            # VISUALIZATION 5: Subscription Type vs Churn
            st.subheader("5️⃣ Subscription Type Analysis")
            
            if 'SubscriptionType' in df.columns:
                subscription_churn = df.groupby(['SubscriptionType', 'Churn']).size().unstack(fill_value=0)
                
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
                    height=400
                )
                st.plotly_chart(fig5, use_container_width=True)
                
                # Churn rate by subscription
                st.markdown("### Churn Rate by Subscription Type")
                subscription_stats = df.groupby('SubscriptionType').agg({
                    'Churn': ['sum', 'count', 'mean']
                }).round(3)
                subscription_stats.columns = ['Churned', 'Total', 'Churn Rate']
                subscription_stats['Churn Rate'] = (subscription_stats['Churn Rate'] * 100).round(1)
                st.dataframe(subscription_stats, use_container_width=True)
            
            st.markdown("---")
            
            # VISUALIZATION 6: Correlation Heatmap
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
                    height=500
                )
                st.plotly_chart(fig6, use_container_width=True)
                
                # Top correlations with Churn
                if 'Churn' in corr_matrix.columns:
                    st.markdown("### 🔍 Top Correlations with Churn")
                    churn_corr = corr_matrix['Churn'].drop('Churn').abs().sort_values(ascending=False)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(churn_corr.head(5), use_container_width=True)
                    with col2:
                        st.info(f"""
                            **Strongest predictor**: {churn_corr.index[0]}
                            
                            **Correlation**: {churn_corr.iloc[0]:.3f}
                            
                            Higher values indicate stronger relationship with churn.
                        """)
            
            st.markdown("---")
            
            # VISUALIZATION 7: Feature Importance
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
                    fig7.update_layout(showlegend=False)
                    st.plotly_chart(fig7, use_container_width=True)
                    
                    st.info(f"""
                        **Most Important Feature**: {importance_df.iloc[-1]['Feature']}
                        
                        This feature has the strongest influence on the model's churn predictions.
                    """)
                else:
                    st.warning("⚠️ Feature importance not available for this model type.")
            else:
                st.warning("⚠️ Load a model to view feature importance analysis.")

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
