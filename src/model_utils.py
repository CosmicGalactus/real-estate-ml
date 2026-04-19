"""Utilities for ML Model Integration and Feature Engineering

This module provides a wrapper around the trained price prediction model and utilities
for feature preparation. It handles:
- Loading and caching the trained model and scaler
- Feature preparation and normalization
- Price estimation (with fallback logic if model unavailable)
- Feature importance tracking

The ModelPredictor class is designed to be robust - it includes fallback heuristics
if the trained model is unavailable, ensuring the system continues to function even
in degraded mode.

Example:
    >>> predictor = ModelPredictor()
    >>> features = prepare_property_features(property_data)
    >>> price = predictor.predict(features)
    >>> print(f\"Predicted price: ${price:,.0f}\")
\"\"\"

import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional


class ModelPredictor:
    """Wrapper for the trained ML model.
    
    Handles loading and predicting with the trained Random Forest model.
    Includes fallback heuristic estimation if model is unavailable.
    """
    
    def __init__(self):
        """Load the trained model from disk.
        
        Attempts to load model.pkl and scaler.pkl from models/ directory.
        If files don't exist, sets to None - the predict() method will
        then fall back to heuristic estimation.
        """
        model_path = Path(__file__).parent.parent / "models" / "model.pkl"
        
        if not model_path.exists():
            self.model = None
            self.scaler = None
        else:
            self.model = joblib.load(model_path)
            
            # Load scaler if it exists (for feature preprocessing)
            scaler_path = Path(__file__).parent.parent / "models" / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            else:
                self.scaler = None
    
    def predict(self, features):
        """Predict price for a property.
        
        Uses the trained model if available, otherwise falls back to
        heuristic estimation to ensure robustness.
        
        Args:
            features: Dict with property characteristics (sqft, bedrooms, etc.)
            
        Returns:
            float: Predicted property price in dollars
        """
        if self.model is None:
            return self._estimate_price(features)
        
        try:
            # Convert to DataFrame for consistency with training
            df = pd.DataFrame([features])
            
            # Scale features if scaler available
            if self.scaler is not None:
                df_scaled = self.scaler.transform(df)
                prediction = self.model.predict(df_scaled)[0]
            else:
                prediction = self.model.predict(df)[0]
            
            return float(prediction)
        except Exception as e:
            # Fallback to heuristic if model prediction fails
            print(f"Warning: Model prediction failed, using heuristic: {e}")
            return self._estimate_price(features)
    
    def _estimate_price(self, features):
        """Heuristic price estimation (fallback method).
        
        Implements rule-of-thumb pricing based on market fundamentals.
        Used when trained model is unavailable for robustness.
        
        Args:
            features: Property characteristics dict
            
        Returns:
            float: Estimated price based on market heuristics
        """
        # Base price for reference property
        base_price = 300000
        
        # Adjust for size (reference: 2000 sqft)
        sqft = features.get("sqft", 2000)
        price = base_price * (sqft / 2000)
        
        # Adjust for quality and condition (both 1-10 scale)
        quality = features.get("quality", 7)
        condition = features.get("condition", 7)
        price = price * (quality / 7) * (condition / 7)
        
        # Adjust for bedrooms (reference: 3 bedrooms)
        bedrooms = features.get("bedrooms", 3)
        price = price * (bedrooms / 3)
        
        # Adjust for age (depreciate ~0.5% per year, min 50%)
        year_built = features.get("year_built", 2005)
        age = 2026 - year_built
        age_factor = 1.0 - (age * 0.005)
        price = price * max(age_factor, 0.5)
        
        return price


def prepare_property_features(property_data):
    """Prepare and normalize property features for model input.
    
    Extracts relevant features from raw property data and formats them
    for the ML model. This function serves as a bridge between the UI
    input and the model's expected feature format.
    
    Args:
        property_data: Raw property information dict containing any of:
            - sqft: Living area in square feet
            - bedrooms: Number of bedrooms
            - bathrooms: Number of bathrooms
            - year_built: Year of construction
            - quality: Quality rating (1-10 scale)
            - condition: Condition rating (1-10 scale)
            - garage_cars: Garage spaces
            
    Returns:
        dict: Normalized features dict suitable for model input.
              Includes default values for missing fields.
              
    Example:
        >>> raw_data = {"sqft": 2000, "bedrooms": 3}
        >>> features = prepare_property_features(raw_data)
        >>> price = predictor.predict(features)
    \"\"\"
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
