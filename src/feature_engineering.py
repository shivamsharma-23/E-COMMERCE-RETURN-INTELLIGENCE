"""
Feature Engineering and Transformation Module for E-Commerce Return Intelligence.

Creates domain-specific derived features, extracts temporal signals from OrderDate,
and constructs reusable Scikit-Learn transformers for preprocessing.
"""

from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Core numerical features prior to feature engineering
BASE_NUMERIC_COLS: List[str] = [
    'UnitPrice', 'Quantity', 'TotalAmount',
    'DiscountPercent', 'DiscountAmount', 'FinalAmount',
    'ShippingDays'
]

# Engineered numerical features
ENGINEERED_NUMERIC_COLS: List[str] = [
    'DiscountRatio', 'EffectiveUnitPrice', 'ShippingPerDollar',
    'IsHighDiscount', 'IsBulkOrder', 'OrderMonth', 'IsWeekend'
]

ALL_NUMERIC_FEATURES: List[str] = BASE_NUMERIC_COLS + ENGINEERED_NUMERIC_COLS

# Categorical features for one-hot encoding
CATEGORICAL_FEATURES: List[str] = [
    'ProductCategory', 'ProductSubCategory', 'PaymentMethod',
    'DeviceType', 'MembershipTier', 'State', 'City',
    'OrderSeason', 'OrderDayOfWeek'
]


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate domain-specific and temporal engineered features:
    1. Parse OrderDate to derive month and day components.
    2. Fallback to OrderDayOfWeek if OrderDate is missing or pre-parsed.
    3. DiscountRatio: Ratio of discount to total transaction amount.
    4. EffectiveUnitPrice: Net final unit price paid per item.
    5. ShippingPerDollar: Shipping time relative to spend magnitude.
    6. IsHighDiscount: Binary flag for aggressive discounting (>= 20%).
    7. IsBulkOrder: Binary flag indicating order volume (Quantity >= 3).
    8. IsWeekend: Binary indicator for weekend shopping behavior.
    """
    df_feat = df.copy()
    
    # 1. Temporal feature extraction
    if 'OrderDate' in df_feat.columns:
        parsed_dates = pd.to_datetime(df_feat['OrderDate'], errors='coerce')
        # Fill missing months with mode/median if unparseable
        df_feat['OrderMonth'] = parsed_dates.dt.month.fillna(6).astype(int)
        
        # Check weekend from date if possible, else from OrderDayOfWeek
        date_is_weekend = parsed_dates.dt.dayofweek.isin([5, 6]).astype(int)
        if 'OrderDayOfWeek' in df_feat.columns:
            dow_is_weekend = df_feat['OrderDayOfWeek'].astype(str).str.strip().isin(['Saturday', 'Sunday']).astype(int)
            df_feat['IsWeekend'] = np.where(parsed_dates.notnull(), date_is_weekend, dow_is_weekend)
        else:
            df_feat['IsWeekend'] = date_is_weekend
    else:
        df_feat['OrderMonth'] = 6  # Sensible default (June/mid-year)
        if 'OrderDayOfWeek' in df_feat.columns:
            df_feat['IsWeekend'] = df_feat['OrderDayOfWeek'].astype(str).str.strip().isin(['Saturday', 'Sunday']).astype(int)
        else:
            df_feat['IsWeekend'] = 0

    # 2. Financial & behavioral ratios
    # Ensure numeric columns exist and are numeric
    for col in BASE_NUMERIC_COLS:
        if col not in df_feat.columns:
            df_feat[col] = 0.0
        else:
            df_feat[col] = pd.to_numeric(df_feat[col], errors='coerce').fillna(0.0)

    # Avoid division by zero with defensive epsilon
    eps = 1e-5
    df_feat['DiscountRatio'] = df_feat['DiscountAmount'] / (df_feat['TotalAmount'] + eps)
    df_feat['EffectiveUnitPrice'] = df_feat['FinalAmount'] / (df_feat['Quantity'] + eps)
    df_feat['ShippingPerDollar'] = df_feat['ShippingDays'] / (df_feat['FinalAmount'] + eps)
    
    # 3. Behavioral flags
    df_feat['IsHighDiscount'] = (df_feat['DiscountPercent'] >= 20).astype(int)
    df_feat['IsBulkOrder'] = (df_feat['Quantity'] >= 3).astype(int)

    # 4. Fill categorical columns with 'Unknown' if missing
    for cat_col in CATEGORICAL_FEATURES:
        if cat_col not in df_feat.columns:
            df_feat[cat_col] = 'Unknown'
        else:
            df_feat[cat_col] = df_feat[cat_col].fillna('Unknown').astype(str)

    return df_feat


def build_preprocessor() -> ColumnTransformer:
    """
    Build Scikit-Learn ColumnTransformer:
    - StandardScaler for numerical features
    - OneHotEncoder with handle_unknown='ignore' for categorical features
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, ALL_NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Retrieve output feature names from fitted ColumnTransformer."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback manual reconstruction
        cat_names = []
        try:
            ohe = preprocessor.named_transformers_['cat']
            cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
        except Exception:
            pass
        return ALL_NUMERIC_FEATURES + cat_names
