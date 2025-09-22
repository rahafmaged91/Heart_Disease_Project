# ===================================
# STREAMLIT WEB UI FOR HEART DISEASE PREDICTION
# ===================================

import streamlit as st
import pandas as pd
import joblib
import json
import os
import plotly.express as px

# --- CUSTOM CLASS DEFINITION ---
# The definition of the custom class used in the pipeline must be available
# in the script that loads the pipeline.
class FeatureSelector:
    def __init__(self, feature_names):
        self.feature_names = feature_names
    
    def fit(self, X, y=None):
        # Ensure feature names exist in the data
        available_features = [f for f in self.feature_names if f in X.columns]
        if len(available_features) != len(self.feature_names):
            print(f"[WARNING] Some features not found. Using {len(available_features)} out of {len(self.feature_names)}")
        self.feature_names = available_features
        return self
    
    def transform(self, X):
        return X[self.feature_names]
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- INTELLIGENT PATH HANDLING ---
def get_project_root():
    """Dynamically find the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Assumes app.py is in 'ui' folder, so go up one level
    return os.path.abspath(os.path.join(current_dir, '..'))

PROJECT_ROOT = get_project_root()
# UPDATED PATH to point to the complete pipeline created by your export script
PIPELINE_PATH = os.path.join(PROJECT_ROOT, 'models', 'deployment', 'heart_disease_pipeline_latest.joblib')
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'models', 'model_config.json')
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'heart_disease_final.csv')

# --- LOADING FUNCTIONS (with caching) ---
@st.cache_resource
def load_pipeline(pipeline_path):
    """Loads the trained machine learning pipeline."""
    try:
        with open(pipeline_path, 'rb') as f:
            pipeline = joblib.load(f)
        return pipeline
    except FileNotFoundError:
        st.error(f"Error: Pipeline file not found at {pipeline_path}. Please run the model export script (07_model_export.py).")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the pipeline: {e}")
        return None

@st.cache_data
def load_config(config_path):
    """Loads the model configuration (required features)."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        st.error(f"Error: Model config file not found at {config_path}. This file is crucial for predictions.")
        return None

@st.cache_data
def load_data(data_path):
    """Loads the dataset for exploration."""
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        st.warning(f"Warning: Data file not found at {data_path}. Data exploration will be unavailable.")
        return None

# --- LOAD ARTIFACTS ---
pipeline = load_pipeline(PIPELINE_PATH)
# The scaler is now part of the pipeline, so we don't load it separately
config = load_config(CONFIG_PATH)
df = load_data(DATA_PATH)

# --- UI LAYOUT ---
st.title("❤️ Heart Disease Prediction Dashboard")
st.markdown("An interactive dashboard to predict heart disease risk using a machine learning model.")

# --- SIDEBAR FOR USER INPUT ---
st.sidebar.header("Patient Health Data")
st.sidebar.markdown("Enter the patient's information below to get a prediction.")

if config:
    # All original features should be collected, the pipeline will select the right ones.
    input_data = {}

    # Create sliders and select boxes for user input
    with st.sidebar.form("prediction_form"):
        st.write("**Patient Demographics**")
        input_data['age'] = st.slider("Age", 20, 90, 55)
        input_data['sex'] = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")

        st.write("**Symptoms & Vitals**")
        input_data['cp'] = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3], format_func=lambda x: {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"}[x])
        input_data['trestbps'] = st.slider("Resting Blood Pressure (trestbps)", 90, 200, 120)
        input_data['chol'] = st.slider("Serum Cholestoral (chol) in mg/dl", 100, 600, 200)
        input_data['fbs'] = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[1, 0], format_func=lambda x: "True" if x == 1 else "False")

        st.write("**Test Results**")
        input_data['restecg'] = st.selectbox("Resting Electrocardiographic Results (restecg)", options=[0, 1, 2], format_func=lambda x: {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Probable or Definite Left Ventricular Hypertrophy"}[x])
        input_data['thalach'] = st.slider("Maximum Heart Rate Achieved (thalach)", 70, 220, 150)
        input_data['exang'] = st.selectbox("Exercise Induced Angina (exang)", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        input_data['oldpeak'] = st.slider("ST depression induced by exercise relative to rest (oldpeak)", 0.0, 6.2, 1.0, 0.1)
        input_data['slope'] = st.selectbox("Slope of the peak exercise ST segment (slope)", options=[0, 1, 2], format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x])
        input_data['ca'] = st.selectbox("Number of major vessels colored by flourosopy (ca)", options=[0, 1, 2, 3, 4])
        input_data['thal'] = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3], format_func=lambda x: {0: "Normal", 1: "Fixed Defect", 2: "Reversable Defect", 3: "Unknown"}[x])

        submit_button = st.form_submit_button(label="🩺 Predict Heart Disease Risk")

else:
    st.sidebar.error("Cannot create input form because model configuration is missing.")
    submit_button = False

# --- PREDICTION LOGIC ---
if pipeline and config and submit_button:
    # 1. Create a DataFrame from user input
    # We can pass the raw dictionary keys because the pipeline expects the original feature names
    input_df = pd.DataFrame([input_data])
    
    try:
        # 2. Make Prediction using the entire pipeline
        # The pipeline automatically handles feature selection, scaling, and prediction
        prediction = pipeline.predict(input_df)
        prediction_proba = pipeline.predict_proba(input_df)
        
        # --- DISPLAY PREDICTION RESULT ---
        st.header("Prediction Result")
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction[0] == 1:
                st.error("High Risk of Heart Disease Detected")
                st.markdown("""
                    **Interpretation:** The model predicts a significant probability of heart disease based on the provided data.
                    
                    **Disclaimer:** This is a prediction from a machine learning model and **not a medical diagnosis**. Please consult a healthcare professional for an accurate assessment.
                """)
            else:
                st.success("Low Risk of Heart Disease Detected")
                st.markdown("""
                    **Interpretation:** The model predicts a low probability of heart disease. Maintaining a healthy lifestyle is always recommended.
                    
                    **Disclaimer:** This tool provides a risk assessment, not a diagnosis. Always consult a qualified doctor for medical advice.
                """)
        
        with col2:
            prob_df = pd.DataFrame({
                'Risk Level': ['No Disease', 'Heart Disease'],
                'Probability': prediction_proba[0]
            })
            fig = px.pie(prob_df, names='Risk Level', values='Probability', 
                         title='Prediction Probability', hole=0.4,
                         color_discrete_map={'No Disease':'green', 'Heart Disease':'red'})
            fig.update_traces(textinfo='percent+label', pull=[0, 0.1])
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Please ensure all input values are reasonable.")


# --- DATA EXPLORATION SECTION ---
st.header("🔬 Data Exploration")
st.markdown("Explore the dataset used to train the model.")

if df is not None:
    # Show a sample of the data
    if st.checkbox("Show Raw Data Sample"):
        st.write(df.head())

    # Create interactive plots
    st.subheader("Interactive Data Visualizations")
    col1, col2 = st.columns(2)

    with col1:
        x_axis = st.selectbox("Select X-axis for Scatter Plot", df.columns, index=list(df.columns).index('age'))
        y_axis = st.selectbox("Select Y-axis for Scatter Plot", df.columns, index=list(df.columns).index('thalach'))
        
        fig = px.scatter(df, x=x_axis, y=y_axis, color="num",
                         title=f"{y_axis.title()} vs. {x_axis.title()}",
                         labels={"num": "Heart Disease"},
                         color_discrete_map={0: "green", 1: "red"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        hist_axis = st.selectbox("Select Feature for Distribution Plot", df.columns, index=list(df.columns).index('chol'))
        
        fig2 = px.histogram(df, x=hist_axis, color="num",
                            title=f"Distribution of {hist_axis.title()}",
                            labels={"num": "Heart Disease"},
                            color_discrete_map={0: "green", 1: "red"},
                            marginal="box")
        st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("Dataset for exploration is not available.")

