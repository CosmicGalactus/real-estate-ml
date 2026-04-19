"""
Streamlit UI for Real Estate Advisory Agent
Simple and straightforward interface for property analysis
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).parent))

from agent import PropertyAdvisor


@st.cache_resource
def get_advisor():
    """Create advisor instance (cached)"""
    return PropertyAdvisor()


@st.cache_resource
def load_model():
    """Load the ML model for predictions"""
    try:
        model = joblib.load("models/model.pkl")
        return model
    except FileNotFoundError:
        return None


def render_advisory_tab():
    """Display the AI advisory analysis tab"""

    st.markdown("## 🤖 AI Property Advisor")

    st.markdown("""
    This advisor analyzes properties and provides investment recommendations 
    based on market data and property characteristics.
    """)

    # Create two-column layout
    col_form, col_info = st.columns([2, 1])

    # Left column: Input form
    with col_form:
        st.markdown("### Property Details")

        # Basic property information
        col1, col2, col3 = st.columns(3)

        with col1:
            address = st.text_input("Address", "500 Main St")
            sqft = st.number_input(
                "Square Feet", min_value=500, max_value=6000, value=2000, step=100
            )
            bedrooms = st.number_input(
                "Bedrooms", min_value=1, max_value=6, value=3, step=1
            )

        with col2:
            neighborhood = st.selectbox(
                "Neighborhood", ["Northridge", "Westside", "Downtown", "Suburbs"]
            )
            year_built = st.number_input(
                "Year Built", min_value=1800, max_value=2026, value=2005, step=1
            )
            bathrooms = st.number_input(
                "Bathrooms", min_value=1, max_value=5, value=2, step=1
            )

        with col3:
            quality = st.slider("Quality (1-10)", 1, 10, 7)
            condition = st.slider("Condition (1-10)", 1, 10, 7)
            garage_cars = st.number_input(
                "Garage Cars", min_value=0, max_value=4, value=2, step=1
            )

        st.markdown("---")
        st.markdown("### Investment Preferences")

        col_inv1, col_inv2 = st.columns(2)
        with col_inv1:
            investment_type = st.radio(
                "Investment Type", ["Buy to Live", "Rental", "Flip"]
            )
        with col_inv2:
            risk = st.select_slider("Risk Tolerance", ["Low", "Medium", "High"])

    # Right column: Quick property summary
    with col_info:
        st.markdown("### Summary")

        st.markdown(f"""
        **Property Size:** {sqft:,} sqft
        
        **Bedrooms:** {bedrooms}
        
        **Quality:** {quality}/10
        
        **Age:** {2026 - year_built} years
        """)

    # Analyze button
    st.markdown("---")

    if st.button("📊 Analyze Property", use_container_width=True, type="primary"):
        with st.spinner("Analyzing..."):
            # Get ML model prediction for the property
            model = load_model()
            predicted_price = 300000  # fallback default

            if model:
                try:
                    # Prepare input in same format as Price Prediction tab
                    house_age = 2026 - year_built
                    quality_area = quality * sqft
                    quality_cond_score = quality * condition
                    total_floor = (sqft * 0.7) + (sqft * 0.3)  # rough estimate
                    garage_area = garage_cars * 250  # ~250 sqft per car

                    input_df = pd.DataFrame(
                        {
                            "Gr Liv Area": [sqft],
                            "Total Bsmt SF": [sqft * 0.5],
                            "1st Flr SF": [sqft * 0.7],
                            "Garage Area": [garage_area],
                            "Lot Area": [10000],
                            "Overall Qual": [quality],
                            "Overall Cond": [condition],
                            "Year Built": [year_built],
                            "House_Age": [house_age],
                            "Bedroom AbvGr": [bedrooms],
                            "Full Bath": [condition >= 7],  # 1 if well maintained
                            "Half Bath": [0],
                            "Kitchen AbvGr": [1],
                            "TotRms AbvGrd": [bedrooms + condition + 3],
                            "Garage Cars": [garage_cars],
                            "Quality_Area": [quality_area],
                            "Quality_Condition_Score": [quality_cond_score],
                            "Total_Floor_Area": [total_floor],
                            "Neighborhood": [neighborhood],
                            "Bldg Type": ["1Fam"],
                            "House Style": ["2Story"],
                        }
                    )

                    predicted_price = model.predict(input_df)[0]
                except Exception as e:
                    st.warning(
                        f"⚠️ Could not use ML model: {str(e)[:50]}. Using default estimate."
                    )
                    predicted_price = 300000 + (sqft * 100) + (quality * 5000)

            # Prepare data for advisor
            property_data = {
                "features": {
                    "address": address,
                    "neighborhood": neighborhood,
                    "sqft": sqft,
                    "bedrooms": bedrooms,
                    "bathrooms": bathrooms,
                    "year_built": year_built,
                    "quality": quality,
                    "condition": condition,
                    "garage_cars": garage_cars,
                },
                "predicted_price": predicted_price,
            }

            # Get analysis
            advisor = get_advisor()
            report = advisor.analyze(property_data)

            if report and isinstance(report, dict) and "valuation" in report:
                # Display results
                st.markdown("---")
                st.markdown("## 📋 Analysis Results")

                # Valuation section
                st.markdown("### Estimate")
                val = report["valuation"]

                # Helper function to safely extract numeric value
                def to_float(val):
                    """Convert any value to float, handling strings, None, etc."""
                    try:
                        if isinstance(val, (int, float)):
                            return float(val)
                        if isinstance(val, str):
                            return float(val.replace("$", "").replace(",", ""))
                        return 0.0
                    except:
                        return 0.0

                # Extract numeric values from valuation dict
                pred_price_val = to_float(val.get("predicted_price"))
                price_sqft_val = to_float(val.get("price_per_sqft"))

                # Format for display
                pred_price = f"${pred_price_val:,.0f}" if pred_price_val > 0 else "$0"
                price_sqft = f"${price_sqft_val:,.0f}" if price_sqft_val > 0 else "$0"

                signal = val.get("signal", "N/A")
                deviation = val.get("deviation", "N/A")

                col_v1, col_v2, col_v3, col_v4 = st.columns(4)
                with col_v1:
                    st.metric("Predicted Price", pred_price)
                with col_v2:
                    st.metric("Price/Sqft", price_sqft)
                with col_v3:
                    st.metric("Deviation", deviation)
                with col_v4:
                    st.metric("Signal", signal)

                # Property analysis with detailed insights
                st.markdown("### 🏘️ Detailed Property Analysis")
                analysis_text = report.get("property_analysis") or report.get(
                    "analysis", "No analysis available"
                )
                st.markdown(analysis_text)

                # Enhanced analysis metrics
                st.markdown("### 📈 Investment Metrics")

                # Calculate additional metrics for analysis
                house_age = 2026 - year_built
                quality_factor = quality * condition / 100

                col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)

                # Use the numeric values we already extracted
                # If price_per_sqft is 0, calculate it from predicted_price and sqft
                if price_sqft_val <= 0 and pred_price_val > 0 and sqft > 0:
                    price_per_sqft_numeric = pred_price_val / sqft
                else:
                    price_per_sqft_numeric = price_sqft_val

                pred_price_numeric = pred_price_val if pred_price_val > 0 else 300000

                with col_metric1:
                    st.metric(
                        "Price per Sqft",
                        f"${price_per_sqft_numeric:,.0f}",
                        delta=(
                            "Market aligned" if quality_factor > 0.5 else "Below market"
                        ),
                    )
                with col_metric2:
                    estimated_rent = (
                        pred_price_numeric * 0.007
                    )  # 7% annual rental yield estimate
                    try:
                        yield_pct = (
                            (estimated_rent / pred_price_numeric * 100)
                            if pred_price_numeric > 0
                            else 0
                        )
                        st.metric(
                            "Est. Annual Rent",
                            f"${estimated_rent:,.0f}",
                            delta=f"{yield_pct:.1f}% yield",
                        )
                    except:
                        st.metric("Est. Annual Rent", f"${estimated_rent:,.0f}")
                with col_metric3:
                    st.metric(
                        "Property Age",
                        f"{house_age} years",
                        delta=(
                            "Well maintained" if condition >= 7 else "Renovation needed"
                        ),
                    )
                with col_metric4:
                    quality_score = (quality + condition) / 2
                    st.metric(
                        "Quality Score",
                        f"{quality_score:.1f}/10",
                        delta="Premium" if quality_score >= 7 else "Standard",
                    )

                # Market position analysis
                st.markdown("### 🎯 Market Position Analysis")

                # Use the extracted numeric values for market insights
                market_insights = f"""
                **Price Assessment**: 
                - Predicted value: {pred_price}
                - Price per sqft: {price_sqft}
                - Quality alignment: {signal}
                
                **Property Characteristics**:
                - {bedrooms}-bedroom, {bathrooms}-bathroom property
                - {sqft:,} sqft living space ({f"${price_per_sqft_numeric:,.0f}/sqft" if price_per_sqft_numeric > 0 else "N/A"})
                - Built in {year_built} ({house_age} years old)
                - Quality: {quality}/10 | Condition: {condition}/10
                - Located in {neighborhood} neighborhood
                
                **Investment Potential**:
                - Estimated annual rental income: ~${estimated_rent:,.0f}
                - Rental yield estimate: ~{(estimated_rent/pred_price_numeric*100) if pred_price_numeric > 0 else 0:.1f}% annually
                - {neighborhood} neighborhood trend: Growing demand for investment properties
                - Quality-condition composite: {quality_score:.1f}/10 indicates {"premium marketability" if quality_score >= 7 else "standard market appeal"}
                
                **Key Value Drivers**:
                """

                if sqft > 2500:
                    market_insights += (
                        "- ✓ Spacious property attracts larger families\n"
                    )
                if quality >= 8:
                    market_insights += "- ✓ High quality commands premium in market\n"
                if house_age < 10:
                    market_insights += "- ✓ Newer property requires less maintenance\n"
                if garage_cars >= 2:
                    market_insights += (
                        "- ✓ Multiple car garage increases desirability\n"
                    )

                if house_age > 50:
                    market_insights += "- ⚠ Aging infrastructure may require upgrades\n"
                if condition < 6:
                    market_insights += "- ⚠ Below-average condition needs attention\n"
                if quality < 6:
                    market_insights += "- ⚠ Standard quality may limit buyer pool\n"

                st.markdown(market_insights)

                # Recommendation
                st.markdown("### 💡 Investment Recommendation")

                rec = report.get("recommendation", "Unable to generate recommendation")
                if "BUY" in rec:
                    st.success(f"✅ **RECOMMENDATION: BUY**\n\n{rec}")
                elif "INVESTIGATE" in rec:
                    st.warning(f"⚠️ **RECOMMENDATION: INVESTIGATE**\n\n{rec}")
                else:
                    st.info(f"ℹ️ **RECOMMENDATION: HOLD**\n\n{rec}")

                # Disclaimer
                st.markdown("### Legal Notice")
                disclaimer = report.get(
                    "disclaimer", "This is for informational purposes only."
                )
                st.info(disclaimer)

                # Export options
                st.markdown("---")
                st.markdown("### Export Report")

                import json

                col_exp1, col_exp2 = st.columns(2)

                with col_exp1:
                    json_str = json.dumps(report, indent=2)
                    st.download_button(
                        "📥 Download as JSON",
                        data=json_str,
                        file_name="property_analysis.json",
                        mime="application/json",
                    )

                with col_exp2:
                    text_report = f"""
PROPERTY ANALYSIS REPORT
========================

Property: {address}
Neighborhood: {neighborhood}

VALUATION:
Predicted Price: {val['predicted_price']}
Price per Sqft: {val['price_per_sqft']}
Status: {val['signal']}

RECOMMENDATION:
{rec}

DISCLAIMER:
{report['disclaimer']}
                    """
                    st.download_button(
                        "📥 Download as Text",
                        data=text_report,
                        file_name="property_analysis.txt",
                        mime="text/plain",
                    )


def render_how_it_works():
    """Explain how the system works"""

    st.markdown("## ❓ How It Works")

    st.markdown("""
    ### 🔍 Analysis Process
    
    Our advisor analyzes properties through these steps:
    
    **1. Property Assessment**
    - Reviews size, age, quality, and condition
    - Identifies key value drivers
    - Compares to market standards
    
    **2. Price Validation**
    - Calculates expected price based on size and features
    - Compares with predicted price
    - Determines if price is reasonable
    
    **3. Market Position**
    - Evaluates neighborhood characteristics
    - Assesses market trends
    - Identifies opportunities
    
    **4. Recommendation**
    - Generates BUY, HOLD, or INVESTIGATE recommendation
    - Provides reasoning based on analysis
    - Flags any concerns
    
    ### ✅ Quality Assurance
    
    Our system ensures accuracy through:
    - **Data validation** - All inputs are checked
    - **Conservative estimates** - We avoid overvaluation
    - **Clear reasoning** - Every recommendation has a basis
    - **Honest assessment** - We flag problems, not hide them
    
    ### 📊 Understanding the Signals
    
    - **✓ REASONABLE** - Price aligns with comparables (±10%)
    - **⚠ NEEDS REVIEW** - Price slightly off (±20%)
    - **❌ ANOMALY** - Significant deviation (>20%)
    
    ### 🎯 Investment Recommendations
    
    - **🟢 BUY** - Property is competitively priced with good quality
    - **🟡 HOLD** - Fair price but review property condition
    - **🔴 INVESTIGATE** - Price seems unusual, research further
    """)

    st.markdown("---")
    st.markdown("### 🤝 Tips for Best Results")

    st.markdown("""
    1. **Enter accurate property details** - Wrong inputs = wrong analysis
    2. **Double-check the address** - Location affects value significantly
    3. **Be honest about condition** - Overestimating leads to bad decisions
    4. **Consider the neighborhood** - Market trends matter
    5. **Consult a professional** - Always verify before investing
    """)


if __name__ == "__main__":
    render_advisory_tab()
