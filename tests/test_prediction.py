"""
Unit tests for inference and prediction pipeline.
"""

import os
import pytest
import pandas as pd
from src.prediction import (
    load_model_artifact,
    predict_single_order,
    predict_bulk_orders,
    determine_risk_tier
)


def test_determine_risk_tier():
    cat_high, color_high, _ = determine_risk_tier(0.65)
    assert cat_high == "High Risk"
    assert color_high == "#DC2626"

    cat_med, color_med, _ = determine_risk_tier(0.35)
    assert cat_med == "Medium Risk"
    assert color_med == "#D97706"

    cat_low, color_low, _ = determine_risk_tier(0.10)
    assert cat_low == "Low Risk"
    assert color_low == "#16A34A"


def test_single_order_prediction():
    sample_order = {
        'ProductCategory': 'Clothing',
        'ProductSubCategory': 'Tops',
        'UnitPrice': 120.0,
        'Quantity': 2,
        'DiscountPercent': 15,
        'PaymentMethod': 'Debit Card',
        'DeviceType': 'Mobile',
        'MembershipTier': 'Silver',
        'ShippingDays': 5,
        'State': 'CA',
        'City': 'Sacramento',
        'OrderSeason': 'Spring',
        'OrderDayOfWeek': 'Monday'
    }
    
    artifact_path = "models/best_model.joblib"
    assert os.path.exists(artifact_path), "Model artifact must exist for testing"
    
    result = predict_single_order(sample_order, artifact_path=artifact_path)
    assert 'prediction' in result
    assert result['prediction'] in ['RETURN', 'NOT RETURN']
    assert 0.0 <= result['probability'] <= 1.0
    assert result['risk_category'] in ['Low Risk', 'Medium Risk', 'High Risk']
    assert 'financials' in result
    assert result['financials']['total_amount'] == 240.0


def test_bulk_order_prediction():
    df_bulk = pd.DataFrame([
        {
            'OrderID': 'ORD-T1',
            'ProductCategory': 'Clothing',
            'ProductSubCategory': 'Footwear',
            'UnitPrice': 80.0,
            'Quantity': 1,
            'DiscountPercent': 10,
            'PaymentMethod': 'Credit Card',
            'DeviceType': 'Mobile',
            'MembershipTier': 'Bronze',
            'ShippingDays': 4,
            'State': 'NY',
            'City': 'Albany',
            'OrderSeason': 'Winter',
            'OrderDayOfWeek': 'Tuesday'
        },
        {
            'OrderID': 'ORD-T2',
            'ProductCategory': 'Grocery',
            'ProductSubCategory': 'Bakery',
            'UnitPrice': 15.0,
            'Quantity': 3,
            'DiscountPercent': 0,
            'PaymentMethod': 'PayPal',
            'DeviceType': 'Desktop',
            'MembershipTier': 'Gold',
            'ShippingDays': 2,
            'State': 'TX',
            'City': 'Houston',
            'OrderSeason': 'Summer',
            'OrderDayOfWeek': 'Friday'
        }
    ])
    
    artifact_path = "models/best_model.joblib"
    res_df, summary = predict_bulk_orders(df_bulk, artifact_path=artifact_path)
    
    assert len(res_df) == 2
    assert 'Predicted_Return' in res_df.columns
    assert 'Return_Probability' in res_df.columns
    assert 'Risk_Category' in res_df.columns
    assert summary['total_orders'] == 2
    assert summary['predicted_returns'] + summary['predicted_non_returns'] == 2
