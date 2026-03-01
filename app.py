import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime

st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .prediction-result {
        background-color: #d4f1d4;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 8px;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/model.pkl")
        with open("models/metrics.json") as f:
            metrics = json.load(f)
        return model, metrics
    except FileNotFoundError:
        st.error("❌ Model files not found. Please train the model first using: python3 src/train.py")
        st.stop()

@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("data/ames.csv")
    except FileNotFoundError:
        return None

model, metrics = load_model()
df = load_dataset()

st.title("🏡 Real Estate Price Predictor")
st.markdown("Predict property prices using Machine Learning (Random Forest + Ridge Ensemble)")

tab1, tab2, tab3 = st.tabs(["💰 Price Prediction", "📊 Model Info", "ℹ️ About"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Property Details")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            gr_liv_area = st.number_input(
                "Living Area (sq ft)",
                min_value=500,
                max_value=6000,
                value=1500,
                step=100
            )
            total_bsmt_sf = st.number_input(
                "Basement Area (sq ft)",
                min_value=0,
                max_value=3500,
                value=1000,
                step=100
            )
            garage_area = st.number_input(
                "Garage Area (sq ft)",
                min_value=0,
                max_value=1500,
                value=500,
                step=50
            )
        
        with col_b:
            overall_qual = st.slider(
                "Overall Quality (1-10)",
                min_value=1,
                max_value=10,
                value=7
            )
            overall_cond = st.slider(
                "Overall Condition (1-10)",
                min_value=1,
                max_value=10,
                value=7
            )
            year_built = st.number_input(
                "Year Built",
                min_value=1800,
                max_value=2026,
                value=2000,
                step=1
            )
        
        with col_c:
            bedrooms = st.number_input(
                "Bedrooms",
                min_value=0,
                max_value=10,
                value=3,
                step=1
            )
            bathrooms = st.number_input(
                "Bathrooms",
                min_value=0,
                max_value=10,
                value=2,
                step=1
            )
            garage_cars = st.number_input(
                "Garage Cars",
                min_value=0,
                max_value=4,
                value=2,
                step=1
            )
        
        col_d, col_e = st.columns(2)
        with col_d:
            neighborhood = st.selectbox(
                "Neighborhood",
                ["CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel", "Somerst", "NWAmes", "OldTown", "BrkSide", "Sawyer", "NridgHt", "NAmes", "Blmngtn", "BrDale", "IDOTRR", "MeadowV"]
            )
        
        with col_e:
            bldg_type = st.selectbox(
                "Building Type",
                ["1Fam", "2FmCon", "Duplex", "TwnhsE", "TwnhsI"]
            )
        
        house_style = st.selectbox(
            "House Style",
            ["2Story", "1Story", "1.5Fin", "1.5Unf", "SFoyer", "SLvl"]
        )
        
        lot_area = st.number_input(
            "Lot Area (sq ft)",
            min_value=1000,
            max_value=50000,
            value=10000,
            step=500
        )
        
        first_flr_sf = st.number_input(
            "1st Floor Area (sq ft)",
            min_value=400,
            max_value=4000,
            value=1200,
            step=100
        )
    
    with col2:
        st.subheader("Model Performance")
        st.metric("Accuracy", f"{metrics['accuracy']:.2f}%")
        st.metric("Precision", f"{metrics['precision']:.2f}%")
        st.metric("R² Score", f"{metrics['test_r2']:.4f}")
        st.metric("Test MAE", f"${metrics['test_mae']:,.0f}")
        st.metric("Test RMSE", f"${metrics['test_rmse']:,.0f}")
    
    st.divider()
    
    if st.button("🔮 Predict Price", use_container_width=True, type="primary"):
        house_age = 2026 - year_built
        quality_area = overall_qual * gr_liv_area
        quality_cond_score = overall_qual * overall_cond
        total_floor = first_flr_sf + total_bsmt_sf
        
        input_df = pd.DataFrame({
            'Gr Liv Area': [gr_liv_area],
            'Total Bsmt SF': [total_bsmt_sf],
            '1st Flr SF': [first_flr_sf],
            'Garage Area': [garage_area],
            'Lot Area': [lot_area],
            'Overall Qual': [overall_qual],
            'Overall Cond': [overall_cond],
            'Year Built': [year_built],
            'House_Age': [house_age],
            'Bedroom AbvGr': [bedrooms],
            'Full Bath': [bathrooms],
            'Half Bath': [0],
            'Kitchen AbvGr': [1],
            'TotRms AbvGrd': [bedrooms + bathrooms + 3],
            'Garage Cars': [garage_cars],
            'Quality_Area': [quality_area],
            'Quality_Condition_Score': [quality_cond_score],
            'Total_Floor_Area': [total_floor],
            'Neighborhood': [neighborhood],
            'Bldg Type': [bldg_type],
            'House Style': [house_style]
        })
        
        try:
            predicted_price = model.predict(input_df)[0]
            
            st.markdown("<div class='prediction-result'>", unsafe_allow_html=True)
            st.metric("Predicted Price", f"${predicted_price:,.0f}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.info(f"""
                **Property Summary:**
                - Living Area: {gr_liv_area:,} sq ft
                - Bedrooms: {bedrooms}
                - Bathrooms: {bathrooms}
                - Year Built: {year_built}
                - Quality: {overall_qual}/10
                """)
            
            with col_info2:
                st.success(f"""
                **Model Info:**
                - Accuracy: {metrics['accuracy']:.2f}%
                - R² Score: {metrics['test_r2']:.4f}
                - Train Samples: {metrics['train_size']:,}
                - Features Used: {metrics['num_features']}
                """)
        
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")

with tab2:
    st.subheader("Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test MAE", f"${metrics['test_mae']:,.0f}")
    col2.metric("Test RMSE", f"${metrics['test_rmse']:,.0f}")
    col3.metric("Precision", f"{metrics['precision']:.2f}%")
    col4.metric("CV Mean", f"{metrics['cv_mean']:.4f}")
    
    st.subheader("Training Details")
    st.info(f"""
    **Dataset:** {metrics['train_size']:,} training samples
    **Features:** {metrics['num_features']} selected features
    **Architecture:** Random Forest (500 estimators) + Ridge (α=0.1)
    **Ensemble:** VotingRegressor with weights [0.9, 0.1]
    """)

with tab3:
    st.subheader("About This Model")
    st.markdown("""
    **Intelligent Property Price Prediction** estimates residential property values using machine learning.
    
    **Model Details:**
    - Algorithm: Random Forest + Ridge Regression Ensemble
    - Framework: Scikit-Learn Pipeline with ColumnTransformer
    - Training Data: 2,296 properties (80% of 2,870)
    - Test Data: 574 properties (20% of 2,870)
    - Features: 21 selected (18 numeric + 3 categorical)
    
    **Performance:**
    - Accuracy: 91.27%
    - R² Score: 0.8999
    - MAE: $14,965
    - RMSE: $21,790
    
    **Architecture:**
    - Random Forest: 500 estimators, max_depth=30
    - Ridge Regression: alpha=0.1
    - Weighted voting: 90% RF, 10% Ridge
    
    **Preprocessing:**
    - KNN Imputation (k=5 neighbors)
    - StandardScaler (numeric features)
    - OneHotEncoder (categorical features)
    
    **GitHub:** https://github.com/CosmicGalactus/real-estate-ml
    """)
