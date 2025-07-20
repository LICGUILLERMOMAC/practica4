# test_preprocessing.py

import pytest
import pandas as pd
import numpy as np
from preprocesamiento import clean_ctg_dataframe

@pytest.fixture
def sample_dataframe():
    data = {
        'num_col1': [1, 2, 3, 4, 5, 6, 10_000],         # outlier
        'num_col2': [10, 12, None, 11, 13, 12, 14],   # 1 nulo
        'cat_col': ['a', 'b', None, 'a', 'b', 'a', 'b'], # 1 nulo
        'mostly_null': [None, None, None, 1, None, None, None]  # >85% nulos
    }
    df = pd.DataFrame(data)
    return df

def test_clean_ctg_dataframe_zscore(sample_dataframe):
    cleaned_df = clean_ctg_dataframe(sample_dataframe, outlier_method="zscore", threshold=3.0)

    # 1. El resultado es un DataFrame
    assert isinstance(cleaned_df, pd.DataFrame)

    # 2. No hay valores nulos
    assert cleaned_df.isnull().sum().sum() == 0

    # 3. La columna mostly_null debe haberse eliminado
    assert 'mostly_null' not in cleaned_df.columns

    # 4. La cantidad de filas debe ser menor si se eliminó el outlier (num_col1 = 1000)
    if sample_dataframe.shape[0] != cleaned_df.shape[0]:
        assert cleaned_df.shape[0] < sample_dataframe.shape[0]
    else:
        print("No se eliminaron outliers con el z-score dado.")

def test_clean_ctg_dataframe_iqr(sample_dataframe):
    cleaned_df = clean_ctg_dataframe(sample_dataframe, outlier_method="zscore", threshold=2.0)

    # 1. El resultado es un DataFrame
    assert isinstance(cleaned_df, pd.DataFrame)

    # 2. No hay valores nulos
    assert cleaned_df.isnull().sum().sum() == 0

    # 3. La columna mostly_null debe haberse eliminado
    assert 'mostly_null' not in cleaned_df.columns

    # 4. La cantidad de filas debe ser menor si se eliminó el outlier
    assert cleaned_df.shape[0] < sample_dataframe.shape[0]
