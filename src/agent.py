"""Real Estate Advisory Agent with Agentic AI Reasoning Patterns

This module implements an intelligent PropertyAdvisor agent that uses multi-step reasoning loops
inspired by LangGraph workflows. The agent performs structured analysis of properties through
multiple reasoning steps before generating investment recommendations.

Architecture:
    - State Management: Tracks analysis progress through reasoning steps
    - Multi-Step Reasoning: Validates prices, analyzes features, retrieves market insights
    - Tool-Like Pattern: Separate methods act as reasoning tools
    - Decision Logic: Rule-based recommendations based on analysis convergence

Example:
    >>> advisor = PropertyAdvisor()
    >>> property_data = {
    ...     "features": {"sqft": 2000, "bedrooms": 3, "quality": 7},
    ...     "predicted_price": 350000
    ... }
    >>> report = advisor.analyze(property_data)
    >>> print(report["recommendation"])
"""

from typing import Dict, Any, List, Optional
import json
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime


class AnalysisState(Enum):
    """States in the agentic reasoning workflow."""

    INIT = "initialization"
    PRICE_VALIDATION = "price_validation"
    PROPERTY_ANALYSIS = "property_analysis"
    MARKET_ASSESSMENT = "market_assessment"
    RECOMMENDATION = "recommendation_generation"
    COMPLETE = "analysis_complete"


@dataclass
class AgentState:
    """Represents the current state of the analysis workflow.

    Attributes:
        step: Current stage in the reasoning loop
        features: Property features being analyzed
        predicted_price: ML model's price prediction
        validations: Results from price validation step
        analysis: Property characteristics analysis
        market: Market positioning assessment
        confidence_score: Confidence level (0-1) in recommendation
        reasoning_history: Log of reasoning steps taken
    """

    step: AnalysisState
    features: Dict[str, Any]
    predicted_price: float
    validations: Optional[Dict[str, Any]] = None
    analysis: Optional[str] = None
    market: Optional[Dict[str, Any]] = None
    confidence_score: float = 0.0
    reasoning_history: List[str] = None

    def __post_init__(self):
        if self.reasoning_history is None:
            self.reasoning_history = []

    def log_step(self, message: str) -> None:
        """Log a reasoning step for transparency.

        Args:
            message: Description of the reasoning step performed
        """
        timestamp = datetime.now().isoformat()
        self.reasoning_history.append(f"[{timestamp}] {self.step.value}: {message}")


class PropertyAdvisor:
    """Intelligent real estate advisor with agentic reasoning patterns.

    This advisor analyzes properties through structured reasoning loops inspired by LangGraph.
    It implements a step-by-step analysis workflow similar to ReAct patterns:
    1. Validate ML predictions against market comparables
    2. Analyze property characteristics comprehensively
    3. Assess market positioning and opportunities
    4. Generate recommendations with confidence scores
    5. Return detailed report with reasoning transparency

    The agent uses rule-based reasoning for interpretability and includes provenance
    tracking of how recommendations were derived.

    Attributes:
        analysis_results: Cache of recent analyses
        max_confidence: Maximum confidence threshold for recommendations
    """

    def __init__(self, use_RAG: bool = False):
        """Initialize the PropertyAdvisor agent.

        Args:
            use_RAG: Whether to use RAG system for market insights (if available).
                     If True, will attempt to import and use RealEstateRAG.
        """
        self.analysis_results = {}
        self.max_confidence = 1.0
        self.use_RAG = use_RAG
        self.rag = None

        # Try to initialize RAG if requested
        if use_RAG:
            try:
                from rag_system import RealEstateRAG, initialize_sample_market_data

                self.rag = RealEstateRAG()
                initialize_sample_market_data(self.rag)
            except ImportError:
                self.use_RAG = False

    def analyze(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate the multi-step agentic reasoning workflow.

        This method implements a ReAct-style reasoning loop where each step refines
        the analysis and builds confidence in the final recommendation. The workflow is:
        1. Initialize state - Prepare features and establish baseline
        2. Validate price - Check ML prediction against market comparables
        3. Analyze property - Assess structural and quality characteristics
        4. Assess market - Evaluate positioning and opportunities
        5. Generate recommendation - Synthesize findings into actionable advice

        Args:
            property_data: Dict containing:
                - features: Property characteristics dict
                - predicted_price: ML model's price prediction (float)

        Returns:
            Dict with comprehensive analysis report including:
                - property: Property details
                - valuation: Price analysis results
                - analysis: Characteristics assessment
                - market_position: Market insights
                - recommendation: Investment advice with confidence
                - reasoning_steps: Log of analysis steps for transparency

        Raises:
            ValueError: If required fields are missing from property_data
        """
        # Extract and validate input
        features = property_data.get("features", {})
        predicted_price = property_data.get("predicted_price", 0)

        if not features or predicted_price <= 0:
            raise ValueError(
                "Invalid property_data: requires 'features' and 'predicted_price'"
            )

        # Initialize agent state with reasoning tracking
        state = AgentState(
            step=AnalysisState.INIT, features=features, predicted_price=predicted_price
        )
        state.log_step(
            f"Analysis started for property in {features.get('neighborhood', 'Unknown')}"
        )

        # ===== REASONING STEP 1: PRICE VALIDATION =====
        state.step = AnalysisState.PRICE_VALIDATION
        state.validations = self._validate_price(features, predicted_price)
        state.log_step(f"Price validation complete: {state.validations['signal']}")

        # ===== REASONING STEP 2: PROPERTY ANALYSIS =====
        state.step = AnalysisState.PROPERTY_ANALYSIS
        state.analysis = self._analyze_property(features)
        state.log_step(
            f"Property characteristics analyzed ({len(state.analysis.split(chr(10)))} insights)"
        )

        # ===== REASONING STEP 3: MARKET ASSESSMENT =====
        state.step = AnalysisState.MARKET_ASSESSMENT
        state.market = self._assess_market_position(features)
        state.log_step(
            f"Market position assessed: {state.market.get('market_outlook', 'Unknown')}"
        )

        # Optionally retrieve RAG insights if available
        rag_insights = None
        if self.use_RAG and self.rag:
            query = f"{features.get('neighborhood', '')} property market {state.market.get('quality_rating', '')} quality"
            rag_insights = self.rag.retrieve_market_insights(query, top_k=2)
            state.log_step(f"Retrieved {len(rag_insights)} market insights from RAG")

        # ===== REASONING STEP 4: RECOMMENDATION GENERATION =====
        state.step = AnalysisState.RECOMMENDATION
        recommendation, confidence = self._generate_recommendation_with_confidence(
            state.validations, state.market, rag_insights
        )
        state.confidence_score = confidence
        state.log_step(f"Recommendation generated with {confidence:.1%} confidence")

        # ===== STEP 5: SYNTHESIZE FINDINGS INTO REPORT =====
        state.step = AnalysisState.COMPLETE
        report = self._build_comprehensive_report(state, recommendation, rag_insights)

        # End analysis
        state.log_step("Analysis workflow completed successfully")
        report["reasoning_steps"] = state.reasoning_history

        return report

    def _validate_price(self, features: Dict, predicted_price: float) -> Dict:
        """Validate ML prediction against market comparables (REASONING TOOL #1).

        This is the first reasoning step. It checks if the predicted price is within
        expected market ranges using comparable property analysis. The validation uses:
        - Base price-per-sqft benchmarks
        - Bedroom premiums/discounts
        - Neighborhood context (if available)

        The signal helps determine whether the prediction is reasonable, needs further
        review, or appears anomalous. This directly influences confidence in the
        final recommendation.

        Args:
            features: Property characteristics dict with at minimum 'sqft' and 'bedrooms'
            predicted_price: ML model's predicted price in dollars

        Returns:
            Dict with validation results:
                - price_per_sqft: Normalized price metric
                - expected_price: Market comparable range midpoint
                - predicted_price: The input prediction (for reference)
                - deviation_percent: % difference from expected (±10% is "reasonable")
                - signal: Quality indicator (✓ REASONABLE, ⚠ NEEDS REVIEW, ❌ ANOMALY)
        """
        # Extract key features for valuation logic
        sqft = features.get("sqft", 1)
        bedrooms = features.get("bedrooms", 1)
        neighborhood = features.get("neighborhood", "Unknown")

        # Calculate normalized price metric
        price_per_sqft = predicted_price / sqft if sqft > 0 else 0

        # Market comparable pricing logic:
        # - Base $150/sqft for 3BR homes (Ames market baseline)
        # - Each additional BR adds 5% premium, each fewer BR discounts 5%
        # This simple model captures the main value drivers
        base_price_per_sqft = 150  # $150/sqft baseline for mid-market
        bedroom_adjustment = 1.0 + (bedrooms - 3) * 0.05  # 5% per BR difference
        expected_price = base_price_per_sqft * sqft * bedroom_adjustment

        # Calculate deviation from market comparables
        if expected_price > 0:
            deviation = ((predicted_price - expected_price) / expected_price) * 100
        else:
            deviation = 0

        # Determine confidence signal based on deviation magnitude:
        # < ±10%: ML model likely correct (high confidence)
        # ±10-20%: May indicate unique features or market shifts (medium confidence)
        # > ±20%: Potential data quality issue or market anomaly (low confidence)
        if abs(deviation) <= 10:
            signal = "✓ REASONABLE"  # Prediction aligns with market comparables
        elif abs(deviation) <= 20:
            signal = "⚠ NEEDS REVIEW"  # Outside typical range, investigate further
        else:
            signal = "❌ ANOMALY"  # Significant deviation, handle with caution

        return {
            "price_per_sqft": round(price_per_sqft, 2),
            "expected_price": round(expected_price, 0),
            "predicted_price": round(predicted_price, 0),
            "deviation_percent": round(deviation, 1),
            "signal": signal,
        }

    def _analyze_property(self, features: Dict) -> str:
        """Analyze property characteristics (REASONING TOOL #2).

        This is the second reasoning step. It performs qualitative assessment of
        key property characteristics that affect investment potential:
        - Size (sqft) - impacts rental/resale potential
        - Age - affects maintenance costs and modernization value
        - Bedroom count - market segmentation (family vs. investment)
        - Quality score - construction and finishes assessment

        The analysis produces a human-readable bullet-point summary that feeds into
        the recommendation decision logic.

        Args:
            features: Property characteristics dict

        Returns:
            str: Multi-line analysis with bullet points describing key characteristics
        """
        analysis_points = []

        # Analyze size
        sqft = features.get("sqft", 0)
        if sqft > 2500:
            analysis_points.append("• Large property - good for families")
        elif sqft < 1500:
            analysis_points.append("• Compact property - good for first-time buyers")
        else:
            analysis_points.append("• Medium-sized property - versatile")

        # Analyze age
        year_built = features.get("year_built", 2000)
        age = 2026 - year_built
        if age > 50:
            analysis_points.append("• Older property - may need updates")
        elif age < 10:
            analysis_points.append("• Recently built - modern construction")
        else:
            analysis_points.append("• Well-established property")

        # Analyze bedrooms
        beds = features.get("bedrooms", 0)
        if beds >= 4:
            analysis_points.append("• Good family home with multiple bedrooms")
        elif beds <= 2:
            analysis_points.append("• Good for couples or small families")

        # Quality assessment
        quality = features.get("quality", 5)
        if quality >= 8:
            analysis_points.append("• High quality construction")
        elif quality <= 4:
            analysis_points.append("• Average quality - may require maintenance")

        return "\n".join(analysis_points)

    def _assess_market_position(self, features: Dict) -> Dict:
        """Assess property's market positioning (REASONING TOOL #3).

        This is the third reasoning step. It evaluates how the property positions
        relative to market conditions and opportunities:
        - Neighborhood desirability and market trends
        - Quality tier (affects both demand and appreciation potential)
        - Investment opportunity classification

        Args:
            features: Property characteristics dict

        Returns:
            Dict with market assessment:
                - neighborhood: Location name
                - quality_rating: 1-10 quality score
                - market_outlook: Market condition assessment
                - investment_class: Property classification for investors
        """
        neighborhood = features.get("neighborhood", "Unknown")
        quality = features.get("quality", 5)
        sqft = features.get("sqft", 0)
        beds = features.get("bedrooms", 3)

        # Market outlook determination:
        # High quality (7+) in all aspects = Strong market conditions
        # Mixed quality = Stable but limited growth
        # Lower quality = Cautious (potential fixer-upper/value play)
        market_outlook = (
            "Strong" if quality >= 8 else ("Stable" if quality >= 6 else "Cautious")
        )

        # Investment classification:
        # - Primary residence: 2-4 BR, medium quality, stable neighborhoods
        # - Investment/Rental: Higher yields expected
        # - Fix-and-Flip: Lower quality, discount pricing
        if quality >= 7 and 3 <= beds <= 4 and sqft > 1500:
            investment_class = "Primary Residence / Quality Home"
        elif quality <= 5 and sqft > 1800:
            investment_class = "Fixer-Upper / Value Play"
        else:
            investment_class = "Investment/Rental Property"

        market_status = {
            "neighborhood": neighborhood,
            "quality_rating": quality,
            "market_outlook": market_outlook,
            "investment_class": investment_class,
        }

        return market_status

    def _generate_recommendation_with_confidence(
        self,
        price_validation: Dict,
        market_position: Dict,
        rag_insights: Optional[List[Dict]] = None,
    ) -> tuple:
        """Generate recommendation with confidence score (REASONING TOOL #4).

        This is the final reasoning step. It synthesizes all prior analysis into
        an investment recommendation with an associated confidence score (0-100%).

        Args:
            price_validation: Results from _validate_price
            market_position: Results from _assess_market_position
            rag_insights: Optional RAG-retrieved market insights

        Returns:
            Tuple of (recommendation_string, confidence_score 0-1)
        """
        signal = price_validation.get("signal", "")
        quality = market_position.get("quality_rating", 5)
        deviation = abs(price_validation.get("deviation_percent", 0))

        # Initialize base confidence from price signal
        if "REASONABLE" in signal:
            base_confidence = 0.75
        elif "NEEDS REVIEW" in signal:
            base_confidence = 0.50
        else:
            base_confidence = 0.20

        # Adjust based on quality
        quality_factor = min(quality / 10.0, 1.0)
        combined_confidence = base_confidence * (0.7 + 0.3 * quality_factor)

        # Boost if RAG found insights
        if rag_insights and len(rag_insights) > 0:
            insight_boost = min(len(rag_insights) * 0.05, 0.15)
            combined_confidence = min(combined_confidence + insight_boost, 1.0)

        # Generate recommendation
        if "REASONABLE" in signal and quality >= 6:
            recommendation = (
                "🟢 BUY - Price is competitive and property quality is good"
            )
            final_confidence = min(combined_confidence + 0.10, 1.0)
        elif "REASONABLE" in signal:
            recommendation = "🟡 HOLD - Price is fair but consider property condition"
            final_confidence = combined_confidence
        elif "NEEDS REVIEW" in signal:
            recommendation = (
                "🟡 INVESTIGATE FURTHER - Price deviation requires expert review"
            )
            final_confidence = combined_confidence
        else:
            recommendation = (
                "🔴 CAUTION - Price seems unusual, review with professional"
            )
            final_confidence = combined_confidence * 0.8

        return recommendation, final_confidence

    def _build_comprehensive_report(
        self,
        state: AgentState,
        recommendation: str,
        rag_insights: Optional[List[Dict]] = None,
    ) -> Dict:
        """Build comprehensive analysis report with full transparency.

        Assembles all analysis components into a report that includes complete
        reasoning history and all intermediate analysis results.

        Args:
            state: AgentState containing all analysis results
            recommendation: Final investment recommendation string
            rag_insights: Optional market insights from RAG retrieval

        Returns:
            Dict: Comprehensive report with all analysis details
        """
        features = state.features
        validation = state.validations
        market = state.market

        report = {
            "status": "success",
            "analysis_confidence": f"{state.confidence_score:.1%}",
            "property": {
                "address": features.get("address", "Unknown"),
                "size_sqft": features.get("sqft", 0),
                "bedrooms": features.get("bedrooms", 0),
                "bathrooms": features.get("bathrooms", 0),
                "year_built": features.get("year_built", 0),
                "neighborhood": features.get("neighborhood", "Unknown"),
                "quality_rating": f"{features.get('quality', 5)}/10",
            },
            "valuation": {
                "predicted_price": validation["predicted_price"],
                "predicted_price_formatted": f"${validation['predicted_price']:,}",
                "price_per_sqft": validation["price_per_sqft"],
                "price_per_sqft_formatted": f"${validation['price_per_sqft']}",
                "expected_range": f"${validation['expected_price'] * 0.95:,.0f} - ${validation['expected_price'] * 1.05:,.0f}",
                "signal": validation["signal"],
                "deviation": f"{((validation['predicted_price'] - validation['expected_price']) / validation['expected_price'] * 100):.1f}%",
            },
            "property_analysis": state.analysis,
            "market_position": market,
            "recommendation": recommendation,
            "confidence_score": state.confidence_score,
            "disclaimer": "This is an automated analysis for informational purposes only. Consult with a licensed real estate professional before making investment decisions.",
        }

        if rag_insights:
            report["market_insights"] = [
                {"text": insight["text"], "relevance": insight.get("relevance", "N/A")}
                for insight in rag_insights[:3]
            ]

        return report


class RealEstateAdvisoryAgent:
    """Wrapper for backward compatibility with Streamlit UI.

    Provides drop-in compatibility with existing code that expects
    the old RealEstateAdvisoryAgent interface.
    """

    def __init__(self, rag_system=None):
        """Initialize with optional RAG system.

        Args:
            rag_system: Optional pre-initialized RAG system (unused in new version)
        """
        self.advisor = PropertyAdvisor(use_RAG=True)

    def analyze_property(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze property using new agentic workflow.

        Args:
            property_data: Property data dict with 'features' and 'predicted_price'

        Returns:
            Dict with success status, advisory report, and any errors
        """
        try:
            report = self.advisor.analyze(property_data)
            return {"success": True, "advisory_report": report, "errors": []}
        except Exception as e:
            return {"success": False, "advisory_report": None, "errors": [str(e)]}


if __name__ == "__main__":
    """Test and demonstrate the agentic AI reasoning workflow."""
    print("\n" + "=" * 80)
    print("Real Estate Advisory Agent - Agentic AI Reasoning Demo")
    print("=" * 80 + "\n")

    advisor = PropertyAdvisor(use_RAG=True)

    test_property = {
        "features": {
            "address": "500 Main St, Northridge",
            "sqft": 2000,
            "bedrooms": 3,
            "bathrooms": 2,
            "year_built": 2005,
            "quality": 7,
            "condition": 7,
            "garage_cars": 2,
            "neighborhood": "Northridge",
        },
        "predicted_price": 350000,
    }

    print("Analyzing property...\n")
    result = advisor.analyze(test_property)

    print(f"Recommendation: {result.get('recommendation', 'N/A')}")
    print(f"Confidence: {result.get('analysis_confidence', 'N/A')}\n")
    print("Full Report:")
    print(json.dumps(result, indent=2))
