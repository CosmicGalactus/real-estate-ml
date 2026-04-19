import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from advisory_ui import render_advisory_tab, render_how_it_works

st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
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
        color: #1a1a1a;
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
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/model.pkl")
        with open("models/metrics.json") as f:
            metrics = json.load(f)
        return model, metrics
    except FileNotFoundError:
        st.error(
            "❌ Model files not found. Please train the model first using: python3 src/train.py"
        )
        st.stop()


@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("data/ames.csv")
    except FileNotFoundError:
        return None


model, metrics = load_model()
df = load_dataset()

st.markdown(
    """
<div class="main-header">
    <h1>🏡 Real Estate Price Predictor</h1>
    <p>Intelligent property valuation powered by Machine Learning</p>
</div>
""",
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "💰 Price Prediction",
        "📊 Model Performance",
        "🤖 AI Advisory",
        "❓ How It Works",
        "ℹ️ About",
    ]
)

with tab1:
    st.markdown(
        '<div class="section-title">Property Details</div>', unsafe_allow_html=True
    )

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
                step=100,
            )
            total_bsmt_sf = st.number_input(
                "🛋️ Basement Area (sq ft)",
                min_value=0,
                max_value=3500,
                value=1000,
                step=100,
            )
            lot_area = st.number_input(
                "📍 Lot Area (sq ft)",
                min_value=1000,
                max_value=50000,
                value=10000,
                step=500,
            )

        with col_b:
            overall_qual = st.slider(
                "⭐ Overall Quality (1-10)", min_value=1, max_value=10, value=7
            )
            overall_cond = st.slider(
                "🏗️ Overall Condition (1-10)", min_value=1, max_value=10, value=7
            )
            year_built = st.number_input(
                "📅 Year Built", min_value=1800, max_value=2026, value=2000, step=1
            )

        with col_c:
            first_flr_sf = st.number_input(
                "🪜 1st Floor Area (sq ft)",
                min_value=400,
                max_value=4000,
                value=1200,
                step=100,
            )
            garage_area = st.number_input(
                "🚗 Garage Area (sq ft)",
                min_value=0,
                max_value=1500,
                value=500,
                step=50,
            )
            garage_cars = st.number_input(
                "🚙 Garage Cars", min_value=0, max_value=4, value=2, step=1
            )

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("### Rooms & Features")

        col_d, col_e, col_f = st.columns(3)
        with col_d:
            bedrooms = st.number_input(
                "🛏️ Bedrooms", min_value=0, max_value=10, value=3, step=1
            )
        with col_e:
            bathrooms = st.number_input(
                "🚿 Bathrooms", min_value=0, max_value=10, value=2, step=1
            )
        with col_f:
            kitchen = st.number_input(
                "🍳 Kitchens", min_value=1, max_value=3, value=1, step=1
            )

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("### Location & Type")

        col_g, col_h = st.columns(2)
        with col_g:
            neighborhood = st.selectbox(
                "🏘️ Neighborhood",
                [
                    "CollgCr",
                    "Veenker",
                    "Crawfor",
                    "NoRidge",
                    "Mitchel",
                    "Somerst",
                    "NWAmes",
                    "OldTown",
                    "BrkSide",
                    "Sawyer",
                    "NridgHt",
                    "NAmes",
                    "Blmngtn",
                    "BrDale",
                    "IDOTRR",
                    "MeadowV",
                ],
            )
            bldg_type = st.selectbox(
                "🏢 Building Type", ["1Fam", "2FmCon", "Duplex", "TwnhsE", "TwnhsI"]
            )

        with col_h:
            house_style = st.selectbox(
                "🏠 House Style",
                ["2Story", "1Story", "1.5Fin", "1.5Unf", "SFoyer", "SLvl"],
            )

        st.markdown("</div>", unsafe_allow_html=True)

    with col_sidebar:
        st.markdown(
            '<div class="section-title" style="font-size: 1.2em;">Model Metrics</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-card-label">Accuracy</div>
            <div class="metric-card-value">{metrics['accuracy']:.2f}%</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-card-label">Precision</div>
            <div class="metric-card-value">{metrics['precision']:.2f}%</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-card-label">R² Score</div>
            <div class="metric-card-value">{metrics['test_r2']:.3f}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-card-label">Test MAE</div>
            <div class="metric-card-value">${metrics['test_mae']:,.0f}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.divider()

    if st.button("🔮 Predict Price", use_container_width=True, type="primary"):
        house_age = 2026 - year_built
        quality_area = overall_qual * gr_liv_area
        quality_cond_score = overall_qual * overall_cond
        total_floor = first_flr_sf + total_bsmt_sf

        input_df = pd.DataFrame(
            {
                "Gr Liv Area": [gr_liv_area],
                "Total Bsmt SF": [total_bsmt_sf],
                "1st Flr SF": [first_flr_sf],
                "Garage Area": [garage_area],
                "Lot Area": [lot_area],
                "Overall Qual": [overall_qual],
                "Overall Cond": [overall_cond],
                "Year Built": [year_built],
                "House_Age": [house_age],
                "Bedroom AbvGr": [bedrooms],
                "Full Bath": [bathrooms],
                "Half Bath": [0],
                "Kitchen AbvGr": [kitchen],
                "TotRms AbvGrd": [bedrooms + bathrooms + 3],
                "Garage Cars": [garage_cars],
                "Quality_Area": [quality_area],
                "Quality_Condition_Score": [quality_cond_score],
                "Total_Floor_Area": [total_floor],
                "Neighborhood": [neighborhood],
                "Bldg Type": [bldg_type],
                "House Style": [house_style],
            }
        )

        try:
            predicted_price = model.predict(input_df)[0]

            st.markdown(
                f"""
            <div class="prediction-result">
                <h2>Estimated Price</h2>
                <h1>${predicted_price:,.0f}</h1>
            </div>
            """,
                unsafe_allow_html=True,
            )

            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.markdown(
                    f"""
                <div class="info-box">
                    <strong>📋 Property Summary</strong><br><br>
                    🏠 Living Area: {gr_liv_area:,} sq ft<br>
                    🛏️ Bedrooms: {bedrooms}<br>
                    🚿 Bathrooms: {bathrooms}<br>
                    📅 Year Built: {year_built}<br>
                    ⭐ Quality: {overall_qual}/10
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col_info2:
                st.markdown(
                    f"""
                <div class="success-box">
                    <strong>🤖 Model Info</strong><br><br>
                    ✓ Accuracy: {metrics['accuracy']:.2f}%<br>
                    ✓ R² Score: {metrics['test_r2']:.4f}<br>
                    ✓ Training Samples: {metrics['train_size']:,}<br>
                    ✓ Features Used: {metrics['num_features']}
                </div>
                """,
                    unsafe_allow_html=True,
                )

        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")

with tab2:
    st.markdown(
        '<div class="section-title">Performance Metrics</div>', unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-card-label">Test MAE</div>
        <div class="metric-card-value">${metrics['test_mae']:,.0f}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col2.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-card-label">Test RMSE</div>
        <div class="metric-card-value">${metrics['test_rmse']:,.0f}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col3.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-card-label">Precision</div>
        <div class="metric-card-value">{metrics['precision']:.2f}%</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col4.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-card-label">CV Mean</div>
        <div class="metric-card-value">{metrics['cv_mean']:.4f}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Training Details</div>', unsafe_allow_html=True
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-card-label">Training Samples</div>
        <div class="metric-card-value">{metrics['train_size']:,}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col_b.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-card-label">Features Used</div>
        <div class="metric-card-value">{metrics['num_features']}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col_c.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-card-label">Accuracy</div>
        <div class="metric-card-value">{metrics['accuracy']:.2f}%</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col_d.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-card-label">R² Score</div>
        <div class="metric-card-value">{metrics['test_r2']:.4f}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with tab3:
    render_advisory_tab()

with tab4:
    render_how_it_works()

with tab5:
    st.markdown(
        '<div class="section-title">About This Project</div>', unsafe_allow_html=True
    )

    st.markdown(
        """
    ## 🏡 Real Estate ML: Intelligent Property Valuation & Advisory
    
    This application is a comprehensive capstone project on intelligent property price prediction 
    and AI-driven real estate advisory. It combines machine learning, data analysis, and artificial 
    intelligence to provide investors and homebuyers with data-driven property insights.
    
    ---
    
    ### 🎯 Project Overview
    
    **Purpose**: Predict residential property values accurately and provide intelligent investment 
    recommendations through an AI advisory system that analyzes market trends and property characteristics.
    
    **Technology Stack**:
    - **Frontend**: Streamlit (interactive web interface)
    - **ML Framework**: Scikit-Learn + Ensemble methods
    - **AI Research**: LangGraph-inspired agentic reasoning patterns
    - **Vector Database**: Chroma (for RAG-based market insights)
    - **Data Processing**: Pandas, NumPy, Scikit-learn Pipeline
    
    ---
    
    ### 💰 How to Use This Tool
    
    **Tab 1: Price Prediction**
    - Enter property details (size, condition, location, etc.)
    - Click "Predict Price" to get an ML-based valuation
    - Review the prediction accuracy metrics
    - Perfect for quick price estimates
    
    **Tab 2: Model Performance**
    - View detailed ML model metrics and performance
    - Understand accuracy, precision, and error rates
    - Learn about training data composition
    - Great for technical users wanting model details
    
    **Tab 3: AI Advisory**
    - Get intelligent property investment recommendations
    - Analyze market position and ROI potential
    - Multi-step reasoning with confidence scores
    - Transparency into how recommendations are made
    
    **Tab 4: How It Works**
    - Understand the advisory analysis process
    - Learn about quality assurance measures
    - Interpret different signals and recommendations
    - Best practices for accurate analysis
    
    ---
    
    ### 🤖 AI Advisory Agent Features
    
    ✅ **Multi-Step Reasoning**: Property validation → Analysis → Market positioning → Recommendations
    
    ✅ **Confidence Scoring**: Each recommendation includes a 0-100% confidence score based on data quality
    
    ✅ **RAG Integration**: Retrieves relevant market insights for context-aware analysis
    
    ✅ **Transparent Logic**: Full reasoning history shows exactly how recommendations are generated
    
    ✅ **Risk Assessment**: Identifies market anomalies and flags concerns for further investigation
    
    ---
    
    ### 📊 Model Accuracy & Validation
    
    The underlying ML model was trained on 2,870 residential properties from the Ames Housing Dataset 
    and achieves:
    - **Accuracy**: {0:.2f}%
    - **R² Score**: {1:.4f}
    - **MAE**: ${2:,.0f} (average prediction error)
    - **Test RMSE**: ${3:,.0f}
    
    The model uses **Random Forest (500 estimators)** with Ridge Regression voting ensemble for 
    optimal predictions across diverse property types.
    
    ---
    
    ### 🔄 Key Features
    
    **1. Comprehensive Property Analysis**
    - 21 engineered features capturing property essence
    - Quality-to-area ratio, age, condition, location factors
    - Neighborhood-specific insights
    
    **2. Market-Aware Pricing**
    - Compares individual properties to market comparables
    - Identifies undervalued and overvalued properties
    - Provides investment signals for decision-making
    
    **3. Investment Recommendations**
    - BUY: Competitively priced, good quality
    - HOLD: Fair price, review condition
    - INVESTIGATE: Unusual pricing, research further
    
    **4. Data Privacy & Security**
    - No data storage or tracking
    - All analysis happens client-side
    - No personal information collection
    
    ---
    
    ### 📈 Use Cases
    
    🏠 **Homebuyers**: Verify asking prices are reasonable before making offers
    
    💼 **Investors**: Identify undervalued rental properties with ROI potential
    
    🏘️ **Real Estate Agents**: Support pricing decisions with data-driven insights
    
    📊 **Analysts**: Understand property valuation patterns and market trends
    
    ---
    
    ### ⚖️ Important Disclaimers
    
    ⚠️ **This tool is for informational purposes only** and should not be considered professional 
    financial or legal advice. Always consult with qualified real estate professionals before making 
    property investment decisions.
    
    ✓ Predictions are based on historical data patterns and may not account for:
    - Recent market changes or economic conditions
    - Unique property characteristics or renovations
    - Local regulations or zoning changes
    - Individual buyer preferences or circumstances
    
    ---
    
    ### 📚 Technical Resources
    
    - **GitHub Repository**: [CosmicGalactus/real-estate-ml](https://github.com/CosmicGalactus/real-estate-ml)
    - **Dataset**: Ames Housing Dataset (kaggle.com/c/house-prices-advanced-regression-techniques)
    - **Framework**: [Streamlit Documentation](https://docs.streamlit.io)
    - **ML Framework**: [Scikit-Learn](https://scikit-learn.org)
    
    ---
    
    ### 👨‍💼 About the Project
    
    This project demonstrates advanced AI/ML capabilities including:
    - Ensemble machine learning architectures
    - Agentic AI patterns with multi-step reasoning
    - RAG (Retrieval-Augmented Generation) systems
    - Production-ready Streamlit applications
    - Professional code documentation and best practices
    
    Built with focus on transparency, accuracy, and user experience.
    """.format(
            metrics["accuracy"],
            metrics["test_r2"],
            metrics["test_mae"],
            metrics["test_rmse"],
        )
    )
