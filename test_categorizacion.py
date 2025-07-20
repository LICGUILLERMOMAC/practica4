import pandas as pd
import pytest
from categorizacion import resumen_dataframe

@pytest.fixture
def df_ejemplo():
    data = {
        "edad": [25, 30, 22, 40, None],
        "sexo": ["F", "M", "F", "F", "M"],
        "puntos": [10, 10, 10, 15, None],
        "grupo": ["A", "B", "A", "A", "B"]
    }
    return pd.DataFrame(data)

def test_resumen_es_dataframe(df_ejemplo):
    resumen = resumen_dataframe(df_ejemplo)
    assert isinstance(resumen, pd.DataFrame)

def test_columnas_esperadas(df_ejemplo):
    resumen = resumen_dataframe(df_ejemplo)
    columnas_esperadas = {
        "columna", "nulos", "completitud (%)", "tipo_dato", 
        "min", "max", "media", "std", "valores_unicos", "clasificacion"
    }
    assert set(resumen.columns) == columnas_esperadas

def test_filas_correspondientes_a_columnas(df_ejemplo):
    resumen = resumen_dataframe(df_ejemplo)
    assert resumen.shape[0] == df_ejemplo.shape[1]

def test_clasificacion_correcta(df_ejemplo):
    resumen = resumen_dataframe(df_ejemplo)
    clasificaciones_validas = {"Continua", "Discreta", "Categórica"}
    assert all(clasif in clasificaciones_validas for clasif in resumen["clasificacion"])
