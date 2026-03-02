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
    * {
        margin: 0;
        padding: 0;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px 20px;
        border-radius: 12px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .main-header h1 {
        font-size: 2.5em;
        margin-bottom: 10px;
        font-weight: 700;
    }
    
    .main-header p {
        font-size: 1.1em;
        opacity: 0.95;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        color: white;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .metric-card-label {
        font-size: 0.9em;
        opacity: 0.9;
        margin-bottom: 8px;
    }
    
    .metric-card-value {
        font-size: 1.8em;
        font-weight: 700;
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 40px 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 25px rgba(17, 153, 142, 0.3);
        margin: 20px 0;
    }
    
    .prediction-result h2 {
        font-size: 1.2em;
        opacity: 0.9;
        margin-bottom: 15px;
    }
    
    .prediction-result h1 {
        font-size: 3em;
        font-weight: 700;
        margin: 0;
    }
    
    .input-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 20px;
        border-left: 5px solid #667eea;
    }
    
    .input-section h3 {
        color: #333;
        margin-bottom: 20px;
        font-size: 1.3em;
    }
    
    .info-box {
        background: linear-gradient(135deg, #e0f7ff 0%, #f0e7ff 100%);
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 5px solid #667eea;
    }
    
    .success-box {
        background: linear-gradient(135deg, #d4f1d4 0%, #e8f8e8 100%);
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 5px solid #38ef7d;
        color: #155724;
    }
    
    .section-title {
        font-size: 1.6em;
        font-weight: 700;
        color: #333;
        margin: 30px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 3px solid #667eea;
    }
    
    .tab-content {
        padding: 20px 0;
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin: 20px 0;
    }
    
    .btn-predict {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 30px;
        border: none;
        border-radius: 8px;
        font-size: 1.1em;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 20px;
    }
    
    .btn-predict:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
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

st.markdown("""
<div class="main-header">
    <h1>🏡 Real Estate Price Predictor</h1>
    <p>Intelligent property valuation powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["💰 Price Prediction", "📊 Model Performance", "ℹ️ About"])

with tab1:
    st.markdown('<div class="section-title">Property Details</div>', unsafe_allow_html=True)
    
    col_main, col_sidebar = st.columns([2.5, 1.5])
    
    with col_main:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("### Basic Information")
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            gr_liv_area = st.number_input(
                "🏠 Living Area (sq ft)",
                min_value=500,
                max_value=6000,
                value=1500,
                step=100
            )
            total_bsmt_sf = st.number_input(
                "🛋️ Basement Area (sq ft)",
                min_value=,
                max_value=3500,
                value=1000,
                step=100
            )
            lot_area = st.number_input(
                "📍 Lot Area (sq ft)",
                min_value=1000,
                max_value=50000,
                value=10000,
                step=500
            )
        
        with col_b:
            overall_qual = st.slider(
                "⭐ Overall Quality (1-10)",
                min_value=1,
                max_value=10,
                value=7
            )
            overall_cond = st.slider(
                "🏗️ Overall Condition (1-10)",
                min_value=1,
                max_value=10,
                value=7
            )
            year_built = st.number_input(
                "📅 Year Built",
                min_value=1800,
                max_value=2026,
                value=2000,
                step=1
            )
        
        with col_c:
            first_flr_sf = st.number_input(
                "🪜 1st Floor Area (sq ft)",
                min_value=400,
                max_value=4000,
                value=1200,
                step=100
            )
            garage_area = st.number_input(
                "🚗 Garage Area (sq ft)",
                min_value=0,
                max_value=1500,
                value=500,
                step=50
            )
            garage_cars = st.number_input(
                "🚙 Garage Cars",
                min_value=0,
                max_value=4,
                value=2,
                step=1
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("### Rooms & Features")
        
        col_d, col_e, col_f = st.columns(3)
        with col_d:
            bedrooms = st.number_input(
                "🛏️ Bedrooms",
                min_value=0,
                max_value=10,
                value=3,
                step=1
            )
        with col_e:
            bathrooms = st.number_input(
                "🚿 Bathrooms",
                min_value=0,
                max_value=10,
                value=2,
                step=1
            )
        with col_f:
            kitchen = st.number_input(
                "🍳 Kitchens",
                min_value=1,
                max_value=3,
                value=1,
                step=1
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("### Location & Type")
        
        col_g, col_h = st.columns(2)
        with col_g:
            neighborhood = st.selectbox(
                "🏘️ Neighborhood",
                ["CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel", "Somerst", "NWAmes", "OldTown", "BrkSide", "Sawyer", "NridgHt", "NAmes", "Blmngtn", "BrDale", "IDOTRR", "MeadowV"]
            )
            bldg_type = st.selectbox(
                "🏢 Building Type",
                ["1Fam", "2FmCon", "Duplex", "TwnhsE", "TwnhsI"]
            )
        
        with col_h:
            house_style = st.selectbox(
                "🏠 House Style",
                ["2Story", "1Story", "1.5Fin", "1.5Unf", "SFoyer", "SLvl"]
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col_sidebar:
        st.markdown('<div class="section-title" style="font-size: 1.2em;">Model Metrics</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">Accuracy</div>
            <div class="metric-card-value">{metrics['accuracy']:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">Precision</div>
            <div class="metric-card-value">{metrics['precision']:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">R² Score</div>
            <div class="metric-card-value">{metrics['test_r2']:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-label">Test MAE</div>
            <div class="metric-card-value">${metrics['test_mae']:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
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
            'Kitchen AbvGr': [kitchen],
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
            
            st.markdown(f"""
            <div class="prediction-result">
                <h2>Estimated Price</h2>
                <h1>${predicted_price:,.0f}</h1>
            </div>
            """, unsafe_allow_html=True)
            
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.markdown(f"""
                <div class="info-box">
                    <strong>📋 Property Summary</strong><br><br>
                    🏠 Living Area: {gr_liv_area:,} sq ft<br>
                    🛏️ Bedrooms: {bedrooms}<br>
                    🚿 Bathrooms: {bathrooms}<br>
                    📅 Year Built: {year_built}<br>
                    ⭐ Quality: {overall_qual}/10
                </div>
                """, unsafe_allow_html=True)
            
            with col_info2:
                st.markdown(f"""
                <div class="success-box">
                    <strong>🤖 Model Info</strong><br><br>
                    ✓ Accuracy: {metrics['accuracy']:.2f}%<br>
                    ✓ R² Score: {metrics['test_r2']:.4f}<br>
                    ✓ Training Samples: {metrics['train_size']:,}<br>
                    ✓ Features Used: {metrics['num_features']}
                </div>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")

with tab2:
    st.markdown('<div class="section-title">Performance Metrics</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-label">Test MAE</div>
        <div class="metric-card-value">${metrics['test_mae']:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    col2.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-label">Test RMSE</div>
        <div class="metric-card-value">${metrics['test_rmse']:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    col3.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-label">Precision</div>
        <div class="metric-card-value">{metrics['precision']:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    col4.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-label">CV Mean</div>
        <div class="metric-card-value">{metrics['cv_mean']:.4f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">Training Details</div>', unsafe_allow_html=True)
    
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-label">Training Samples</div>
        <div class="metric-card-value">{metrics['train_size']:,}</div>
    </div>
    """, unsafe_allow_html=True)
    
    col_b.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-label">Features Used</div>
        <div class="metric-card-value">{metrics['num_features']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    col_c.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-label">Accuracy</div>
        <div class="metric-card-value">{metrics['accuracy']:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    col_d.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-label">R² Score</div>
        <div class="metric-card-value">{metrics['test_r2']:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="section-title">About This Model</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🤖 Intelligent Property Price Prediction
    
    This machine learning model estimates residential property values using advanced ensemble techniques.
    
    ---
    
    #### 📊 Model Architecture
    - **Primary Algorithm**: Random Forest (500 estimators, max_depth=30)
    - **Secondary Algorithm**: Ridge Regression (alpha=0.1)
    - **Ensemble Strategy**: VotingRegressor with weights [0.9, 0.1]
    - **Framework**: Scikit-Learn Pipeline with ColumnTransformer
    
    #### 📈 Performance Highlights
    """)
    
    col_perf1, col_perf2, col_perf3 = st.columns(3)
    col_perf1.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-label">Accuracy</div>
        <div class="metric-card-value">{metrics['accuracy']:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    col_perf2.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-label">R² Score</div>
        <div class="metric-card-value">{metrics['test_r2']:.4f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    col_perf3.markdown(f"""
    <div class="metric-card">
        <div class="metric-card-label">Precision</div>
        <div class="metric-card-value">{metrics['precision']:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    #### 📚 Training Data
    - **Dataset**: Ames Housing Dataset
    - **Total Properties**: 2,870
    - **Training Set**: 2,296 properties (80%)
    - **Test Set**: 574 properties (20%)
    - **Price Range**: $62,000 - $455,000
    
    #### 🔧 Feature Engineering
    - **Numeric Features**: 18 (area, quality, condition, age, etc.)
    - **Categorical Features**: 3 (neighborhood, building type, house style)
    - **Engineered Features**: 4 (Quality_Area, Quality_Condition_Score, House_Age, Total_Floor_Area)
    - **Total Features**: 21 selected and optimized
    
    #### ⚙️ Preprocessing Pipeline
    - **Imputation**: KNN Imputation (k=5 neighbors)
    - **Scaling**: StandardScaler for numeric features
    - **Encoding**: OneHotEncoder for categorical features (min_frequency=2)
    - **Outlier Removal**: 1st-99th percentile filtering
    
    #### 🎯 Key Metrics
    - Mean Absolute Error (MAE): ${metrics['test_mae']:,.0f}
    - Root Mean Squared Error (RMSE): ${metrics['test_rmse']:,.0f}
    - Cross-Validation Mean: {metrics['cv_mean']:.4f} (±{metrics['cv_std']:.4f})
    
    ---
    
    #### 🔗 Resources
    - **GitHub**: [CosmicGalactus/real-estate-ml](https://github.com/CosmicGalactus/real-estate-ml)
    - **Framework**: Scikit-Learn, Streamlit
    - **Python Version**: 3.8+
    """)

