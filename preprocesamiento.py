# Importamos librerias necesarias
# preprocessing.py

import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple

def clean_ctg_dataframe(df: pd.DataFrame, outlier_method: str = "zscore", threshold: float = 3.0) -> pd.DataFrame:
    """
    Realiza limpieza completa del DataFrame: 
    - Elimina columnas con >20% de nulos
    - Imputa valores nulos
    - Trata outliers con z-score o IQR
    
    Args:
        df (pd.DataFrame): DataFrame original
        outlier_method (str): Método para outliers ('zscore' o 'iqr')
        threshold (float): Umbral para z-score o IQR

    Returns:
        pd.DataFrame: DataFrame limpio
    """
    df = df.copy()

    # 1. Eliminar columnas con > 20% nulos
    null_threshold = 0.3
    df = df.loc[:, df.isnull().mean() <= null_threshold]

    # 2. Imputación de valores nulos
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            mean_value = df[col].mean()
            df[col] = df[col].fillna(mean_value)
        else:
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(mode_value)

    # 3. Tratar outliers
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    if outlier_method == "zscore":
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(df[col]))
            df = df[(z_scores < threshold) | pd.isnull(z_scores)]
    elif outlier_method == "iqr":
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    df.reset_index(drop=True, inplace=True)
    return df