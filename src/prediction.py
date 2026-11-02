"""
Inference and Prediction Module for E-Commerce Return Intelligence.

Provides single-order scoring, batch bulk processing, risk tiering,
business recommendations, and downloadable prediction report generation.
"""

import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import joblib

from src.data_preprocessing import clean_inference_data
from src.feature_engineering import extract_features


def load_model_artifact(artifact_path: str = "models/best_model.joblib") -> Dict[str, Any]:
    """
    Load deployable Scikit-Learn artifact containing model, preprocessor, and threshold.
    """
    if not os.path.exists(artifact_path):
        # Fallback to absolute or relative path resolution
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        alt_path = os.path.join(base_dir, artifact_path)
        if os.path.exists(alt_path):
            artifact_path = alt_path
        else:
            raise FileNotFoundError(f"Model artifact not found at {artifact_path}")
            
    artifact = joblib.load(artifact_path)
    return artifact


def determine_risk_tier(prob: float, low_bound: float = 0.25, high_bound: float = 0.50) -> Tuple[str, str, str]:
    """
    Map return probability to enterprise operational risk tier.
    
    Returns:
        (risk_category, hex_color, business_recommendation)
    """
    if prob >= high_bound:
        category = "High Risk"
        color = "#DC2626"  # Red
        action = (
            "Flag for pre-dispatch inspection: Verify sizing with customer, bundle return instructions, "
            "and alert warehouse logistics for potential RMA processing."
        )
    elif prob >= low_bound:
        category = "Medium Risk"
        color = "#D97706"  # Amber/Orange
        action = (
            "Deploy proactive retention: Send detailed usage and sizing guides via email/SMS, "
            "track delivery transit milestones, and offer live support assistance."
        )
    else:
        category = "Low Risk"
        color = "#16A34A"  # Green
        action = (
            "Standard automated fulfillment: Low probability of return. Proceed with standard logistics."
        )
        
    return category, color, action


def predict_single_order(
    order_data: Dict[str, Any],
    model_artifact: Optional[Dict[str, Any]] = None,
    artifact_path: str = "models/best_model.joblib",
    custom_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Predict return probability and class for an individual order.
    """
    if model_artifact is None:
        model_artifact = load_model_artifact(artifact_path)

    model = model_artifact['model']
    preprocessor = model_artifact['preprocessor']
    default_thresh = model_artifact.get('threshold', 0.50)
    threshold = custom_threshold if custom_threshold is not None else default_thresh

    # Normalize inputs and calculate financial fields if missing
    unit_price = float(order_data.get('UnitPrice', 50.0))
    quantity = int(order_data.get('Quantity', 1))
    disc_pct = float(order_data.get('DiscountPercent', 0.0))
    
    total_amount = float(order_data.get('TotalAmount', unit_price * quantity))
    discount_amount = float(order_data.get('DiscountAmount', total_amount * (disc_pct / 100.0)))
    final_amount = float(order_data.get('FinalAmount', total_amount - discount_amount))

    input_dict = {
        'ProductCategory': str(order_data.get('ProductCategory', 'Clothing')),
        'ProductSubCategory': str(order_data.get('ProductSubCategory', 'Footwear')),
        'UnitPrice': unit_price,
        'Quantity': quantity,
        'TotalAmount': total_amount,
        'DiscountPercent': disc_pct,
        'DiscountAmount': discount_amount,
        'FinalAmount': final_amount,
        'PaymentMethod': str(order_data.get('PaymentMethod', 'Credit Card')),
        'DeviceType': str(order_data.get('DeviceType', 'Mobile')),
        'MembershipTier': str(order_data.get('MembershipTier', 'Silver')),
        'ShippingDays': int(order_data.get('ShippingDays', 4)),
        'State': str(order_data.get('State', 'CA')),
        'City': str(order_data.get('City', 'Los Angeles')),
        'OrderSeason': str(order_data.get('OrderSeason', 'Summer')),
        'OrderDayOfWeek': str(order_data.get('OrderDayOfWeek', 'Monday')),
        'OrderDate': str(order_data.get('OrderDate', '2024-06-15'))
    }

    df_input = pd.DataFrame([input_dict])
    
    # Preprocess & Feature Engineering
    df_clean, _ = clean_inference_data(df_input)
    df_features = extract_features(df_clean)
    
    X_vec = preprocessor.transform(df_features)
    prob = float(model.predict_proba(X_vec)[0, 1])
    is_return = int(prob >= threshold)
    
    risk_cat, risk_color, business_action = determine_risk_tier(prob)

    return {
        'prediction': "RETURN" if is_return == 1 else "NOT RETURN",
        'is_returned': is_return,
        'probability': round(prob, 4),
        'probability_pct': f"{prob * 100:.1f}%",
        'risk_category': risk_cat,
        'risk_color': risk_color,
        'recommended_action': business_action,
        'threshold_used': round(float(threshold), 3),
        'model_name': model_artifact.get('model_name', 'Random Forest Classifier'),
        'financials': {
            'unit_price': unit_price,
            'quantity': quantity,
            'total_amount': total_amount,
            'discount_amount': discount_amount,
            'final_amount': final_amount
        },
        'disclaimer': (
            "Notice: Predictions are statistical risk probabilities generated by the trained machine learning pipeline. "
            "They should guide supply-chain verification and customer service workflows rather than serve as absolute guarantees."
        )
    }


def predict_bulk_orders(
    df: pd.DataFrame,
    model_artifact: Optional[Dict[str, Any]] = None,
    artifact_path: str = "models/best_model.joblib",
    custom_threshold: Optional[float] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Process batch order records from an uploaded CSV dataframe.
    """
    if model_artifact is None:
        model_artifact = load_model_artifact(artifact_path)

    model = model_artifact['model']
    preprocessor = model_artifact['preprocessor']
    default_thresh = model_artifact.get('threshold', 0.50)
    threshold = custom_threshold if custom_threshold is not None else default_thresh

    df_work = df.copy()

    # Defensive financial derivations if missing in upload
    if 'UnitPrice' in df_work.columns and 'Quantity' in df_work.columns and 'TotalAmount' not in df_work.columns:
        df_work['TotalAmount'] = df_work['UnitPrice'] * df_work['Quantity']
        
    if 'TotalAmount' in df_work.columns and 'DiscountPercent' in df_work.columns and 'DiscountAmount' not in df_work.columns:
        df_work['DiscountAmount'] = df_work['TotalAmount'] * (df_work['DiscountPercent'] / 100.0)
        
    if 'TotalAmount' in df_work.columns and 'DiscountAmount' in df_work.columns and 'FinalAmount' not in df_work.columns:
        df_work['FinalAmount'] = df_work['TotalAmount'] - df_work['DiscountAmount']

    # Clean inference data and extract original IDs
    df_clean, id_df = clean_inference_data(df_work)
    df_features = extract_features(df_clean)

    X_vec = preprocessor.transform(df_features)
    probs = model.predict_proba(X_vec)[:, 1]
    preds = (probs >= threshold).astype(int)

    # Attach prediction outputs
    results_df = df.copy()
    results_df['Return_Probability'] = np.round(probs, 4)
    results_df['Return_Probability_Pct'] = (probs * 100).round(1).astype(str) + '%'
    results_df['Predicted_Return'] = np.where(preds == 1, 'RETURN', 'NOT RETURN')

    # Assign Risk Categories
    risk_categories = []
    actions = []
    for p in probs:
        rc, _, act = determine_risk_tier(p)
        risk_categories.append(rc)
        actions.append(act)

    results_df['Risk_Category'] = risk_categories
    results_df['Recommended_Action'] = actions

    # Compute batch summary statistics
    total = len(results_df)
    ret_count = int(np.sum(preds))
    ret_rate = round((ret_count / total * 100), 2) if total > 0 else 0.0

    high_risk_cnt = int(np.sum(results_df['Risk_Category'] == 'High Risk'))
    med_risk_cnt = int(np.sum(results_df['Risk_Category'] == 'Medium Risk'))
    low_risk_cnt = int(np.sum(results_df['Risk_Category'] == 'Low Risk'))

    potential_loss = 0.0
    if 'FinalAmount' in results_df.columns:
        potential_loss = float(results_df.loc[preds == 1, 'FinalAmount'].sum())

    summary = {
        'total_orders': total,
        'predicted_returns': ret_count,
        'predicted_non_returns': total - ret_count,
        'predicted_return_rate': ret_rate,
        'high_risk_count': high_risk_cnt,
        'medium_risk_count': med_risk_cnt,
        'low_risk_count': low_risk_cnt,
        'potential_returned_revenue': round(potential_loss, 2),
        'threshold_used': round(float(threshold), 3)
    }

    return results_df, summary
