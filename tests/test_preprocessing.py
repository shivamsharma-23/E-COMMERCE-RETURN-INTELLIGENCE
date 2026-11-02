"""
Unit tests for data preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np

from src.data_preprocessing import (
    remove_duplicates,
    binarize_target,
    handle_missing_values,
    prepare_training_data,
    clean_inference_data,
    ID_COLUMNS,
    LEAKAGE_COLUMNS
)


def test_remove_duplicates():
    df = pd.DataFrame({
        'OrderID': ['ORD-1', 'ORD-2', 'ORD-1'],
        'Val': [10, 20, 10]
    })
    clean_df, removed = remove_duplicates(df)
    assert len(clean_df) == 2
    assert removed == 1


def test_binarize_target():
    df = pd.DataFrame({
        'Returned': ['Yes', 'No', 'yes', 'NO']
    })
    res = binarize_target(df)
    assert res['Returned'].tolist() == [1, 0, 1, 0]


def test_handle_missing_values():
    df = pd.DataFrame({
        'UnitPrice': [10.0, np.nan, 30.0],
        'ProductCategory': ['Clothing', np.nan, 'Electronics']
    })
    res = handle_missing_values(df, numerical_cols=['UnitPrice'], categorical_cols=['ProductCategory'])
    assert not res['UnitPrice'].isnull().any()
    assert res['UnitPrice'].iloc[1] == 20.0  # Median
    assert not res['ProductCategory'].isnull().any()
    assert res['ProductCategory'].iloc[1] == 'Missing'


def test_clean_inference_data():
    df = pd.DataFrame({
        'OrderID': ['ORD-101'],
        'FirstName': ['Alice'],
        'ReviewScore': [5],
        'UnitPrice': [99.0],
        'Quantity': [2]
    })
    clean_df, id_df = clean_inference_data(df)
    assert 'OrderID' not in clean_df.columns
    assert 'FirstName' not in clean_df.columns
    assert 'ReviewScore' not in clean_df.columns
    assert 'UnitPrice' in clean_df.columns
    assert id_df is not None
    assert 'OrderID' in id_df.columns
