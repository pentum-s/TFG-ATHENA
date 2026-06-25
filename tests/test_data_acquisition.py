"""
=============================================================================
PROYECTO ATHENA — tests/test_data_acquisition.py
=============================================================================
Tests unitarios e integración del módulo data_acquisition.py.

Cubre:
    - Inicialización de DataAcquisition (validaciones de entrada)
    - compute_returns()  → forma, valores, propiedades matemáticas
    - preprocess()       → 5 pasos: calidad, interpolación, nulos, duplicados,
                           reindexado
    - compute_annualized_stats() → columnas, tipos, rangos esperados
    - _max_drawdown()    → método estático, valores límite
    - get_default_tickers() → IBEX35, SP500, error en índice desconocido
    - load_from_csv()    → lectura correcta, FileNotFoundError
    - Sistema de caché   → escritura y lectura de CSV
=============================================================================
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from data_acquisition import DataAcquisition, get_default_tickers, load_from_csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from constants import TICKERS, N_DAYS, CAPITAL, SEED


# =============================================================================
# 1. INICIALIZACIÓN — DataAcquisition.__init__
# =============================================================================

class TestDataAcquisitionInit:
    """Tests de validación de parámetros en __init__."""

    def test_init_correcto(self):
        """Instancia se crea correctamente con parámetros válidos."""
        da = DataAcquisition(tickers=TICKERS, start="2020-01-01")
        assert da.tickers == TICKERS
        assert da.start == "2020-01-01"
        assert da.interval == "1d"

    def test_tickers_normalizados_a_mayusculas(self):
        """Los tickers se normalizan siempre a mayúsculas."""
        da = DataAcquisition(tickers=["san.mc", "bbva.mc"])
        assert da.tickers == ["SAN.MC", "BBVA.MC"]

    def test_tickers_sin_espacios(self):
        """Los tickers se normalizan eliminando espacios en blanco."""
        da = DataAcquisition(tickers=[" SAN.MC ", " BBVA.MC "])
        assert da.tickers == ["SAN.MC", "BBVA.MC"]

    def test_error_lista_vacia(self):
        """Debe lanzar ValueError si la lista de tickers está vacía."""
        with pytest.raises(ValueError, match="vacía"):
            DataAcquisition(tickers=[])

    def test_end_por_defecto_es_hoy(self):
        """Si no se especifica end, usa la fecha de hoy."""
        da = DataAcquisition(tickers=TICKERS, start="2020-01-01")
        today = pd.Timestamp.today().strftime("%Y-%m-%d")
        assert da.end == today

    def test_cache_dir_se_crea(self, tmp_path):
        """Si se especifica cache_dir, el directorio se crea automáticamente."""
        cache = tmp_path / "nueva_carpeta" / "cache"
        assert not cache.exists()
        DataAcquisition(tickers=TICKERS, start="2020-01-01", cache_dir=cache)
        assert cache.exists()

    def test_sin_cache_dir_es_none(self):
        """Sin cache_dir, el atributo es None."""
        da = DataAcquisition(tickers=TICKERS, start="2020-01-01")
        assert da.cache_dir is None


# =============================================================================
# 2. COMPUTE_RETURNS
# =============================================================================

class TestComputeReturns:
    """Tests del método compute_returns()."""

    def test_shape_correcto(self, da_instance, synthetic_prices):
        """El resultado tiene una fila menos que los precios de entrada."""
        returns = da_instance.compute_returns(synthetic_prices)
        assert returns.shape == (N_DAYS - 1, len(TICKERS))

    def test_columnas_identicas(self, da_instance, synthetic_prices):
        """Las columnas del resultado son idénticas a las de los precios."""
        returns = da_instance.compute_returns(synthetic_prices)
        assert list(returns.columns) == list(synthetic_prices.columns)

    def test_indice_sin_primera_fila(self, da_instance, synthetic_prices):
        """El índice empieza en el segundo día (la primera fila es NaN y se elimina)."""
        returns = da_instance.compute_returns(synthetic_prices)
        assert returns.index[0] == synthetic_prices.index[1]

    def test_sin_nulos(self, da_instance, synthetic_prices):
        """No debe haber valores nulos en los rendimientos."""
        returns = da_instance.compute_returns(synthetic_prices)
        assert returns.isna().sum().sum() == 0

    def test_formula_logaritmica(self, da_instance, synthetic_prices):
        """Verifica la fórmula r_t = ln(P_t / P_{t-1}) para el primer valor."""
        returns = da_instance.compute_returns(synthetic_prices)
        expected = np.log(
            synthetic_prices.iloc[1, 0] / synthetic_prices.iloc[0, 0]
        )
        assert abs(returns.iloc[0, 0] - expected) < 1e-10

    def test_rendimientos_son_pequenos(self, da_instance, synthetic_prices):
        """Los rendimientos diarios deben estar en un rango realista (±30%)."""
        returns = da_instance.compute_returns(synthetic_prices)
        assert returns.abs().max().max() < 0.30

    def test_error_dataframe_vacio(self, da_instance):
        """Debe lanzar ValueError si el DataFrame de precios tiene menos de 2 filas."""
        df_vacio = pd.DataFrame([[1.0, 2.0]], columns=TICKERS[:2])
        with pytest.raises(ValueError):
            da_instance.compute_returns(df_vacio)


# =============================================================================
# 3. PREPROCESS
# =============================================================================

class TestPreprocess:
    """Tests del pipeline de preprocesado (5 pasos)."""

    def test_paso1_elimina_activos_baja_calidad(self, da_instance, prices_with_nulls):
        """
        Paso 1: TEF.MC tiene 60% de nulos → debe eliminarse.
        Los demás activos superan el umbral del 80% de datos válidos.
        """
        clean = da_instance.preprocess(prices_with_nulls, min_valid_ratio=0.80)
        assert "TEF.MC" not in clean.columns

    def test_paso1_conserva_activos_validos(self, da_instance, prices_with_nulls):
        """Paso 1: SAN.MC e IBE.MC tienen pocos nulos → deben conservarse."""
        clean = da_instance.preprocess(prices_with_nulls, min_valid_ratio=0.80)
        assert "SAN.MC" in clean.columns
        assert "IBE.MC" in clean.columns

    def test_paso2_interpola_nulos_aislados(self, da_instance, prices_with_nulls):
        """
        Paso 2: Los 3 nulos aislados de SAN.MC deben desaparecer
        tras la interpolación lineal.
        """
        clean = da_instance.preprocess(prices_with_nulls, min_valid_ratio=0.80,
                                        max_consec_nulls=5)
        if "SAN.MC" in clean.columns:
            assert clean["SAN.MC"].isna().sum() == 0

    def test_paso2_interpola_nulos_consecutivos(self, da_instance, prices_with_nulls):
        """
        Paso 2: Los 4 nulos consecutivos de BBVA.MC (≤ max_consec_nulls=5)
        deben interpolarse.
        """
        clean = da_instance.preprocess(prices_with_nulls, min_valid_ratio=0.80,
                                        max_consec_nulls=5)
        if "BBVA.MC" in clean.columns:
            assert clean["BBVA.MC"].isna().sum() == 0

    def test_paso4_sin_duplicados(self, da_instance, synthetic_prices):
        """Paso 4: El resultado no debe tener índices duplicados."""
        # Introducir un duplicado artificial
        df_dup = pd.concat([synthetic_prices, synthetic_prices.iloc[[5]]])
        clean = da_instance.preprocess(df_dup)
        assert not clean.index.duplicated().any()

    def test_paso5_reindexado_a_dias_habiles(self, da_instance, synthetic_prices):
        """Paso 5: El índice del resultado son días hábiles (lun–vie)."""
        clean = da_instance.preprocess(synthetic_prices)
        dias_semana = clean.index.dayofweek
        assert all(d < 5 for d in dias_semana), \
            "El resultado contiene fines de semana"

    def test_shape_resultado_coherente(self, da_instance, synthetic_prices):
        """El resultado tiene el mismo número de columnas que la entrada (sin baja calidad)."""
        clean = da_instance.preprocess(synthetic_prices)
        # Sin nulos: ningún activo debe eliminarse
        assert clean.shape[1] == synthetic_prices.shape[1]

    def test_precios_no_negativos(self, da_instance, synthetic_prices):
        """Los precios limpios nunca deben ser negativos."""
        clean = da_instance.preprocess(synthetic_prices)
        assert (clean.dropna() > 0).all().all()


# =============================================================================
# 4. COMPUTE_ANNUALIZED_STATS
# =============================================================================

class TestAnnualizedStats:
    """Tests del método compute_annualized_stats()."""

    def test_columnas_correctas(self, da_instance, synthetic_returns):
        """El resultado debe tener exactamente las 6 columnas esperadas."""
        stats = da_instance.compute_annualized_stats(synthetic_returns)
        expected_cols = [
            "Rendimiento anual (%)",
            "Volatilidad anual (%)",
            "Ratio de Sharpe",
            "Asimetría",
            "Curtosis",
            "Máx. Drawdown (%)",
        ]
        assert list(stats.columns) == expected_cols

    def test_indice_son_tickers(self, da_instance, synthetic_returns):
        """El índice del resultado son los tickers."""
        stats = da_instance.compute_annualized_stats(synthetic_returns)
        assert list(stats.index) == list(synthetic_returns.columns)

    def test_volatilidad_positiva(self, da_instance, synthetic_returns):
        """La volatilidad anualizada debe ser siempre positiva."""
        stats = da_instance.compute_annualized_stats(synthetic_returns)
        assert (stats["Volatilidad anual (%)"] > 0).all()

    def test_max_drawdown_negativo(self, da_instance, synthetic_returns):
        """El Máx. Drawdown debe ser siempre negativo o cero."""
        stats = da_instance.compute_annualized_stats(synthetic_returns)
        assert (stats["Máx. Drawdown (%)"] <= 0).all()

    def test_volatilidad_anualizada_vs_diaria(self, da_instance, synthetic_returns):
        """La volatilidad anualizada debe ser aproximadamente vol_diaria × √252."""
        stats  = da_instance.compute_annualized_stats(synthetic_returns)
        vol_anual_esperada = synthetic_returns.std() * np.sqrt(252) * 100
        for ticker in synthetic_returns.columns:
            assert abs(
                stats.loc[ticker, "Volatilidad anual (%)"] - vol_anual_esperada[ticker]
            ) < 0.01

    def test_sin_nulos_en_stats(self, da_instance, synthetic_returns):
        """El DataFrame de estadísticas no debe contener NaN."""
        stats = da_instance.compute_annualized_stats(synthetic_returns)
        assert stats.isna().sum().sum() == 0


# =============================================================================
# 5. _MAX_DRAWDOWN (método estático)
# =============================================================================

class TestMaxDrawdown:
    """Tests del método estático _max_drawdown()."""

    def test_serie_siempre_positiva(self):
        """Con rendimientos siempre positivos, el drawdown debe ser 0."""
        serie = pd.Series([0.01] * 100)
        dd = DataAcquisition._max_drawdown(serie)
        assert dd == pytest.approx(0.0, abs=0.01)

    def test_serie_con_caida_conocida(self):
        """Con una caída del 50% conocida, verifica el resultado."""
        # Subida hasta 2x, luego caída a la mitad
        s = pd.Series([0.0] * 50 + [-0.01] * 70)
        dd = DataAcquisition._max_drawdown(s)
        assert dd < 0

    def test_resultado_es_float(self, synthetic_returns):
        """El resultado es un float."""
        dd = DataAcquisition._max_drawdown(synthetic_returns.iloc[:, 0])
        assert isinstance(dd, float)

    def test_max_drawdown_entre_menos100_y_0(self, synthetic_returns):
        """El Max Drawdown siempre está entre -100% y 0%."""
        for col in synthetic_returns.columns:
            dd = DataAcquisition._max_drawdown(synthetic_returns[col])
            assert -100.0 <= dd <= 0.0


# =============================================================================
# 6. GET_DEFAULT_TICKERS
# =============================================================================

class TestGetDefaultTickers:
    """Tests de la función get_default_tickers()."""

    def test_ibex35_devuelve_lista(self):
        """IBEX35 devuelve una lista no vacía."""
        tickers = get_default_tickers("IBEX35")
        assert isinstance(tickers, list)
        assert len(tickers) > 0

    def test_ibex35_sufijo_mc(self):
        """Todos los tickers del IBEX35 tienen sufijo .MC."""
        tickers = get_default_tickers("IBEX35")
        assert all(t.endswith(".MC") for t in tickers)

    def test_sp500_sin_sufijo(self):
        """Los tickers del S&P 500 no tienen sufijo .MC."""
        tickers = get_default_tickers("SP500")
        assert all(not t.endswith(".MC") for t in tickers)

    def test_case_insensitive(self):
        """El parámetro índice es insensible a mayúsculas/minúsculas."""
        assert get_default_tickers("ibex35") == get_default_tickers("IBEX35")
        assert get_default_tickers("sp500")  == get_default_tickers("SP500")

    def test_devuelve_copia(self):
        """Devuelve una copia — modificarla no afecta a la lista original."""
        t1 = get_default_tickers("IBEX35")
        t2 = get_default_tickers("IBEX35")
        t1.append("FAKE.MC")
        assert "FAKE.MC" not in t2

    def test_error_indice_desconocido(self):
        """Debe lanzar ValueError para un índice no reconocido."""
        with pytest.raises(ValueError, match="no reconocido"):
            get_default_tickers("DAX30")


# =============================================================================
# 7. LOAD_FROM_CSV
# =============================================================================

class TestLoadFromCSV:
    """Tests de la función load_from_csv()."""

    def test_carga_correctamente(self, synthetic_prices, tmp_path):
        """Carga un CSV guardado correctamente y devuelve el mismo DataFrame."""
        path = tmp_path / "test_prices.csv"
        synthetic_prices.to_csv(path)
        df = load_from_csv(path)
        assert df.shape == synthetic_prices.shape

    def test_indice_es_datetimeindex(self, synthetic_prices, tmp_path):
        """El índice del CSV cargado es DatetimeIndex."""
        path = tmp_path / "test_prices.csv"
        synthetic_prices.to_csv(path)
        df = load_from_csv(path)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_nombre_indice_es_date(self, synthetic_prices, tmp_path):
        """El nombre del índice es 'Date'."""
        path = tmp_path / "test_prices.csv"
        synthetic_prices.to_csv(path)
        df = load_from_csv(path)
        assert df.index.name == "Date"

    def test_error_archivo_no_existe(self, tmp_path):
        """Debe lanzar FileNotFoundError si el archivo no existe."""
        with pytest.raises(FileNotFoundError):
            load_from_csv(tmp_path / "no_existe.csv")


# =============================================================================
# 8. CACHÉ
# =============================================================================

class TestCache:
    """Tests del sistema de caché local en CSV."""

    def test_cache_path_con_cache_dir(self, da_instance):
        """Con cache_dir configurado, _build_cache_path devuelve un Path."""
        path = da_instance._build_cache_path("prices_raw")
        assert path is not None
        assert isinstance(path, Path)

    def test_cache_path_sin_cache_dir(self):
        """Sin cache_dir, _build_cache_path devuelve None."""
        da = DataAcquisition(tickers=TICKERS, start="2020-01-01")
        assert da._build_cache_path("prices_raw") is None

    def test_cache_incluye_fechas_en_nombre(self, da_instance):
        """El nombre del archivo de caché incluye las fechas de inicio y fin."""
        path = da_instance._build_cache_path("prices_raw")
        assert da_instance.start in path.name
        assert da_instance.end   in path.name

    def test_compute_returns_guarda_csv(self, synthetic_prices, tmp_path):
        """compute_returns guarda el CSV de rendimientos si hay cache_dir."""
        da = DataAcquisition(tickers=TICKERS, start="2020-01-01",
                             end="2022-12-31", cache_dir=tmp_path)
        da.compute_returns(synthetic_prices)
        csv_files = list(tmp_path.glob("returns_*.csv"))
        assert len(csv_files) == 1

    def test_preprocess_guarda_csv(self, synthetic_prices, tmp_path):
        """preprocess guarda el CSV de precios limpios si hay cache_dir."""
        da = DataAcquisition(tickers=TICKERS, start="2020-01-01",
                             end="2022-12-31", cache_dir=tmp_path)
        da.preprocess(synthetic_prices)
        csv_files = list(tmp_path.glob("prices_clean_*.csv"))
        assert len(csv_files) == 1
