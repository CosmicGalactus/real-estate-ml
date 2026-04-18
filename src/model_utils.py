"""
Utilities for ML model integration and feature engineering
Simple wrapper around the trained model
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path


class ModelPredictor:
    """Wrapper for the trained ML model"""
    
    def __init__(self):
        """Load the trained model"""
        model_path = Path(__file__).parent.parent / "models" / "model.pkl"
        
        if not model_path.exists():
            self.model = None
            self.scaler = None
        else:
            self.model = joblib.load(model_path)
            
            # Try to load scaler if it exists
            scaler_path = Path(__file__).parent.parent / "models" / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            else:
                self.scaler = None
    
    def predict(self, features):
        """Predict price for a property"""
        if self.model is None:
            return self._estimate_price(features)
        
        try:
            # Convert to DataFrame for consistency
            df = pd.DataFrame([features])
            
            # Scale if scaler available
            if self.scaler is not None:
                df_scaled = self.scaler.transform(df)
                prediction = self.model.predict(df_scaled)[0]
            else:
                prediction = self.model.predict(df)[0]
            
            return float(prediction)
        except:
            # Fallback to simple estimation
            return self._estimate_price(features)
    
    def _estimate_price(self, features):
        """Simple price estimation (fallback)"""
        base_price = 300000
        
        # Adjust for size
        sqft = features.get("sqft", 2000)
        price = base_price * (sqft / 2000)
        
        # Adjust for quality and condition
        quality = features.get("quality", 7)
        condition = features.get("condition", 7)
        price = price * (quality / 7) * (condition / 7)
        
        # Adjust for bedrooms
        bedrooms = features.get("bedrooms", 3)
        price = price * (bedrooms / 3)
        
        # Adjust for age
        year_built = features.get("year_built", 2005)
        age = 2026 - year_built
        age_factor = 1.0 - (age * 0.005)  # 0.5% depreciation per year
        price = price * max(age_factor, 0.5)  # Don't depreciate below 50%
        
        return price


def prepare_property_features(property_data):
    """
    Prepare property features for model prediction
    
    Args:
        property_data: dict with property information
    
    Returns:
        dict: features suitable for model input
    """
    features = {
        "sqft": property_data.get("sqft", 2000),
        "bedrooms": property_data.get("bedrooms", 3),
        "bathrooms": property_data.get("bathrooms", 2),
        "year_built": property_data.get("year_built", 2005),
        "quality": property_data.get("quality", 7),
        "condition": property_data.get("condition", 7),
        "garage_cars": property_data.get("garage_cars", 2),
    }
    
    # Add derived features
    features["age"] = 2026 - features["year_built"]
    features["rooms"] = features["bedrooms"] + features["bathrooms"]
    features["price_per_sqft"] = 150  # Placeholder
    
    return features


def get_property_options_from_dataset():
    """Get sample properties for UI dropdown"""
    csv_path = Path(__file__).parent.parent / "data" / "ames.csv"
    
    try:
        df = pd.read_csv(csv_path)
        
        # Get unique neighborhoods and other options
        neighborhoods = df["Neighborhood"].unique().tolist() if "Neighborhood" in df else []
        
        return {
            "neighborhoods": neighborhoods[:10],
            "qualities": list(range(1, 11)),
            "conditions": list(range(1, 11)),
        }
    except:
        return {
            "neighborhoods": ["Northridge", "Westside", "Downtown", "Suburbs"],
            "qualities": list(range(1, 11)),
            "conditions": list(range(1, 11)),
        }


def format_price(price):
    """Format price as currency string"""
    return f"${price:,.2f}"


def get_price_per_sqft(price, sqft):
    """Calculate price per square foot"""
    if sqft > 0:
        return price / sqft
    return 0
