"""
Data Preprocessing Module for E-Commerce Return Intelligence.

Handles data loading, deduplication, identifier removal, data leakage prevention,
imputation, and target binarization.
"""

from typing import Tuple, List, Optional
import pandas as pd
import numpy as np


# Identifying columns to remove from ML modeling
ID_COLUMNS: List[str] = ['OrderID', 'CustomerID', 'FirstName', 'LastName']

# Target leakage columns to exclude
LEAKAGE_COLUMNS: List[str] = ['ReviewScore']

# Core categorical features expected in raw data
CATEGORICAL_COLUMNS: List[str] = [
    'ProductCategory', 'ProductSubCategory', 'PaymentMethod',
    'DeviceType', 'MembershipTier', 'State', 'City',
    'OrderSeason', 'OrderDayOfWeek'
]

# Core numeric features expected in raw data
NUMERICAL_COLUMNS: List[str] = [
    'UnitPrice', 'Quantity', 'TotalAmount',
    'DiscountPercent', 'DiscountAmount', 'FinalAmount',
    'ShippingDays'
]

TARGET_COLUMN: str = 'Returned'


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw dataset from CSV file."""
    df = pd.read_csv(filepath)
    return df


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Remove duplicate rows from the dataset.
    
    Returns:
        Deduplicated DataFrame and the count of dropped duplicate records.
    """
    initial_count = len(df)
    df_clean = df.drop_duplicates().copy()
    duplicates_removed = initial_count - len(df_clean)
    return df_clean, duplicates_removed


def binarize_target(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> pd.DataFrame:
    """
    Convert target column ('Yes'/'No') to binary (1/0).
    """
    df_copy = df.copy()
    if target_col in df_copy.columns:
        if df_copy[target_col].dtype == object or isinstance(df_copy[target_col].iloc[0], str):
            df_copy[target_col] = df_copy[target_col].astype(str).str.strip().str.capitalize()
            df_copy[target_col] = df_copy[target_col].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    return df_copy


def handle_missing_values(
    df: pd.DataFrame,
    numerical_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Defensive missing value handling:
    - Imputes numeric columns with their median (or 0 if empty)
    - Imputes categorical columns with 'Missing' or mode
    """
    df_clean = df.copy()
    num_cols = numerical_cols or [c for c in NUMERICAL_COLUMNS if c in df_clean.columns]
    cat_cols = categorical_cols or [c for c in CATEGORICAL_COLUMNS if c in df_clean.columns]

    for col in num_cols:
        if col in df_clean.columns and df_clean[col].isnull().any():
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val if not pd.isna(median_val) else 0)

    for col in cat_cols:
        if col in df_clean.columns and df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna('Missing').astype(str)

    return df_clean


def prepare_training_data(filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Full data preparation for model training:
    1. Loads dataset
    2. Removes duplicates
    3. Excludes customer IDs/names and leakage columns (ReviewScore)
    4. Binarizes target variable
    5. Imputes defensive missing values
    
    Returns:
        (clean_modeling_df, raw_dedup_df, duplicates_count)
    """
    raw_df = load_raw_data(filepath)
    dedup_df, duplicates_count = remove_duplicates(raw_df)
    
    # Store clean raw copy with binarized target for EDA/visualizations
    dedup_df_bin = binarize_target(dedup_df)
    
    # Prepare modeling dataframe: drop IDs and data leakage columns
    cols_to_drop = [c for c in ID_COLUMNS + LEAKAGE_COLUMNS if c in dedup_df_bin.columns]
    modeling_df = dedup_df_bin.drop(columns=cols_to_drop)
    
    modeling_df = handle_missing_values(modeling_df)
    
    return modeling_df, dedup_df_bin, duplicates_count


def clean_inference_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Cleans uploaded or single-order inference data:
    - Retains original IDs in an auxiliary dataframe for user reporting if present
    - Drops leakage and ID columns from model input
    - Handles missing values defensibly
    """
    df_clean = df.copy()
    
    # Extract identifiers for reporting if available
    id_df = None
    present_ids = [c for c in ['OrderID', 'CustomerID'] if c in df_clean.columns]
    if present_ids:
        id_df = df_clean[present_ids].copy()
    
    # Drop identifiers and leakage columns if present
    cols_to_drop = [c for c in ID_COLUMNS + LEAKAGE_COLUMNS if c in df_clean.columns]
    if TARGET_COLUMN in df_clean.columns:
        cols_to_drop.append(TARGET_COLUMN)
        
    df_clean = df_clean.drop(columns=cols_to_drop)
    df_clean = handle_missing_values(df_clean)
    
    return df_clean, id_df
