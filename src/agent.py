"""
Real Estate Advisory Agent - Simplified Version
Analyzes properties and generates investment recommendations
"""

from typing import Dict, Any, List
import json


class PropertyAdvisor:
    """
    Simple advisor that analyzes properties and generates recommendations.
    No complex ML needed - uses straightforward business logic.
    """
    
    def __init__(self):
        """Initialize advisor with no dependencies"""
        self.analysis_results = {}
    
    def analyze(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main analysis function - takes property data and returns recommendation
        
        Args:
            property_data: Dict with property details and predicted price
            
        Returns:
            Dict with analysis and recommendation
        """
        
        # Extract input data
        features = property_data.get("features", {})
        predicted_price = property_data.get("predicted_price", 0)
        
        # Step 1: Validate the predicted price
        price_validation = self._validate_price(features, predicted_price)
        
        # Step 2: Analyze the property
        property_analysis = self._analyze_property(features)
        
        # Step 3: Get market position
        market_position = self._assess_market_position(features)
        
        # Step 4: Generate recommendation based on all factors
        recommendation = self._generate_recommendation(price_validation, market_position)
        
        # Step 5: Build final report
        report = self._build_report(
            features, 
            predicted_price, 
            price_validation, 
            property_analysis,
            market_position,
            recommendation
        )
        
        return report
    
    def _validate_price(self, features: Dict, predicted_price: float) -> Dict:
        """
        Check if predicted price makes sense
        Uses simple market comparables logic
        """
        sqft = features.get("sqft", 1)
        bedrooms = features.get("bedrooms", 1)
        neighborhood = features.get("neighborhood", "Unknown")
        
        # Simple price per sqft calculation
        price_per_sqft = predicted_price / sqft if sqft > 0 else 0
        
        # Expected price based on market averages (these are reasonable for mid-market)
        base_price_per_sqft = 150  # $150/sqft baseline
        expected_price = base_price_per_sqft * sqft * (1 + (bedrooms - 3) * 0.05)
        
        # Check deviation
        if expected_price > 0:
            deviation = ((predicted_price - expected_price) / expected_price) * 100
        else:
            deviation = 0
        
        # Determine signal
        if abs(deviation) <= 10:
            signal = "✓ REASONABLE"
        elif abs(deviation) <= 20:
            signal = "⚠ NEEDS REVIEW"
        else:
            signal = "❌ ANOMALY"
        
        return {
            "price_per_sqft": round(price_per_sqft, 2),
            "expected_price": round(expected_price, 0),
            "predicted_price": round(predicted_price, 0),
            "deviation_percent": round(deviation, 1),
            "signal": signal
        }
    
    def _analyze_property(self, features: Dict) -> str:
        """
        Simple property analysis based on key features
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
        """
        Simple market assessment
        """
        neighborhood = features.get("neighborhood", "Unknown")
        quality = features.get("quality", 5)
        
        # Simple market assessment
        market_status = {
            "neighborhood": neighborhood,
            "quality_rating": quality,
            "market_outlook": "Stable" if quality >= 6 else "Cautious"
        }
        
        return market_status
    
    def _generate_recommendation(self, price_validation: Dict, market_position: Dict) -> str:
        """
        Generate investment recommendation based on analysis
        Simple rule-based approach
        """
        signal = price_validation.get("signal", "")
        quality = market_position.get("quality_rating", 5)
        
        # Decision logic
        if "REASONABLE" in signal and quality >= 6:
            recommendation = "🟢 BUY - Price is competitive and property quality is good"
        elif "REASONABLE" in signal:
            recommendation = "🟡 HOLD - Price is fair but consider property condition"
        elif "NEEDS REVIEW" in signal:
            recommendation = "🟡 INVESTIGATE FURTHER - Price deviation requires analysis"
        else:
            recommendation = "🔴 CAUTION - Price seems unusual, review carefully"
        
        return recommendation
    
    def _build_report(self, features: Dict, price: float, validation: Dict, 
                     analysis: str, market: Dict, rec: str) -> Dict:
        """
        Build the final advisory report
        """
        report = {
            "status": "success",
            "property": {
                "address": features.get("address", "Unknown"),
                "size_sqft": features.get("sqft", 0),
                "bedrooms": features.get("bedrooms", 0),
                "bathrooms": features.get("bathrooms", 0),
                "year_built": features.get("year_built", 0),
                "neighborhood": features.get("neighborhood", "Unknown")
            },
            "valuation": {
                "predicted_price": f"${price:,.0f}",
                "price_per_sqft": f"${validation['price_per_sqft']}",
                "expected_range": f"${validation['expected_price'] * 0.95:,.0f} - ${validation['expected_price'] * 1.05:,.0f}",
                "deviation": f"{validation['deviation_percent']}%",
                "signal": validation["signal"]
            },
            "analysis": analysis,
            "market_position": market,
            "recommendation": rec,
            "disclaimer": "This is an automated analysis for informational purposes only. Consult with a licensed real estate professional before making investment decisions."
        }
        
        return report


# For backward compatibility with existing code
class RealEstateAdvisoryAgent:
    """Wrapper for backward compatibility"""
    
    def __init__(self, rag_system=None):
        self.advisor = PropertyAdvisor()
    
    def analyze_property(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze property and return results"""
        try:
            report = self.advisor.analyze(property_data)
            return {
                "success": True,
                "advisory_report": report,
                "errors": []
            }
        except Exception as e:
            return {
                "success": False,
                "advisory_report": None,
                "errors": [str(e)]
            }


if __name__ == "__main__":
    # Quick test
    advisor = PropertyAdvisor()
    
    test_property = {
        "features": {
            "address": "500 Main St, Northridge",
            "sqft": 2000,
            "bedrooms": 3,
            "bathrooms": 2,
            "year_built": 2005,
            "quality": 7,
            "neighborhood": "Northridge"
        },
        "predicted_price": 350000
    }
    
    result = advisor.analyze(test_property)
    print(json.dumps(result, indent=2))

