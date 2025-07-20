from typing import Literal
import pandas as pd
import numpy as np

def resumen_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un resumen de un DataFrame con:
    - Conteo de nulos
    - Porcentaje de completitud
    - Tipo de dato
    - Estadísticos de dispersión
    - Clasificación automática: continua, discreta o categórica

    Parámetros:
        df (pd.DataFrame): DataFrame de entrada

    Retorna:
        pd.DataFrame: Resumen de las columnas
    """
    resumen = []

    for col in df.columns:
        serie = df[col]
        nulos = serie.isnull().sum()
        completitud = 100 * (1 - nulos / len(serie))
        tipo_dato = serie.dtype

        # Estadísticos básicos
        try:
            minimo = serie.min()
            maximo = serie.max()
            media = serie.mean()
            std = serie.std()
        except:
            minimo = maximo = media = std = None

        # Clasificación automática
        unicos = serie.nunique(dropna=True)
        if pd.api.types.is_numeric_dtype(serie):
            if unicos > 10:
                clasificacion = "Continua"
            else:
                clasificacion = "Discreta"
        else:
            clasificacion = "Categórica"

        resumen.append({
            "columna": col,
            "nulos": nulos,
            "completitud (%)": round(completitud, 2),
            "tipo_dato": str(tipo_dato),
            "min": minimo,
            "max": maximo,
            "media": media,
            "std": std,
            "valores_unicos": unicos,
            "clasificacion": clasificacion
        })

    return pd.DataFrame(resumen)
