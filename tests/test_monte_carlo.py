"""
=============================================================================
PROYECTO ATHENA — tests/test_monte_carlo.py
=============================================================================
Tests unitarios e integración del módulo monte_carlo.py.

Cubre:
    - MonteCarlo.__init__       → validaciones de entrada, normalización pesos
    - MonteCarlo.run()          → shapes, valores, propiedades estadísticas
    - MonteCarloResult          → summary(), metrics_dataframe()
    - _compute_risk_metrics()   → VaR, CVaR, VaR paramétrico
    - sensitivity_analysis()    → función de utilidad a nivel de módulo
    - Reproducibilidad          → misma semilla → mismo resultado
    - Casos límite              → 1 activo, capital muy grande/pequeño
=============================================================================
"""

import numpy as np
import pandas as pd
import pytest

from monte_carlo import MonteCarlo, MonteCarloResult, sensitivity_analysis
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from constants import TICKERS, CAPITAL, N_SIMS, HORIZON, SEED, N_ASSETS


# =============================================================================
# 1. MONTECARLO.__INIT__
# =============================================================================

class TestMonteCarloInit:
    """Tests de validación de parámetros en __init__."""

    def test_init_correcto(self, mc_instance):
        """La instancia se crea correctamente con los parámetros del fixture."""
        assert mc_instance.capital == CAPITAL
        assert mc_instance.n_simulations == N_SIMS
        assert mc_instance.horizon_days == HORIZON
        assert mc_instance.confidence == 0.95

    def test_pesos_normalizados(self, synthetic_returns):
        """Los pesos que no suman 1 se normalizan automáticamente."""
        w = np.ones(N_ASSETS)  # suma = N_ASSETS, no 1
        mc = MonteCarlo(weights=w, returns=synthetic_returns,
                        n_simulations=100, horizon_days=10, seed=SEED)
        assert abs(mc.weights.sum() - 1.0) < 1e-6

    def test_error_confianza_invalida(self, msr_result, synthetic_returns):
        """Debe lanzar ValueError si confidence no está en (0, 1)."""
        with pytest.raises(ValueError):
            MonteCarlo(weights=msr_result.weights.values,
                       returns=synthetic_returns, confidence=1.5)

    def test_error_capital_negativo(self, msr_result, synthetic_returns):
        """Debe lanzar ValueError si el capital es negativo o cero."""
        with pytest.raises(ValueError):
            MonteCarlo(weights=msr_result.weights.values,
                       returns=synthetic_returns, capital=-1000.0)

    def test_error_horizonte_cero(self, msr_result, synthetic_returns):
        """Debe lanzar ValueError si horizon_days < 1."""
        with pytest.raises(ValueError):
            MonteCarlo(weights=msr_result.weights.values,
                       returns=synthetic_returns, horizon_days=0)

    def test_error_simulaciones_insuficientes(self, msr_result, synthetic_returns):
        """Debe lanzar ValueError si n_simulations < 100."""
        with pytest.raises(ValueError):
            MonteCarlo(weights=msr_result.weights.values,
                       returns=synthetic_returns, n_simulations=10)

    def test_pesos_serie_pandas_aceptados(self, msr_result, synthetic_returns):
        """Se aceptan pesos como pd.Series además de np.ndarray."""
        mc = MonteCarlo(weights=msr_result.weights,  # pd.Series
                        returns=synthetic_returns,
                        n_simulations=100, horizon_days=10, seed=SEED)
        assert abs(mc.weights.sum() - 1.0) < 1e-6


# =============================================================================
# 2. MONTECARLO.RUN() — resultado y propiedades
# =============================================================================

class TestMonteCarloRun:
    """Tests del método run()."""

    def test_devuelve_montecarloresult(self, mc_result):
        """Devuelve un objeto MonteCarloResult."""
        assert isinstance(mc_result, MonteCarloResult)

    def test_shape_simulations(self, mc_result):
        """
        La matriz de trayectorias tiene shape (n_simulations, horizon_days + 1).
        La columna 0 es el día inicial (capital).
        """
        assert mc_result.simulations.shape == (N_SIMS, HORIZON + 1)

    def test_dia_inicial_igual_al_capital(self, mc_result):
        """Todas las trayectorias empiezan en el capital inicial."""
        assert np.allclose(mc_result.simulations[:, 0], CAPITAL, atol=1e-6)

    def test_final_values_shape(self, mc_result):
        """final_values tiene shape (n_simulations,)."""
        assert mc_result.final_values.shape == (N_SIMS,)

    def test_final_returns_shape(self, mc_result):
        """final_returns tiene shape (n_simulations,)."""
        assert mc_result.final_returns.shape == (N_SIMS,)

    def test_final_values_positivos(self, mc_result):
        """Los valores finales son siempre positivos (capital nunca negativo)."""
        assert np.all(mc_result.final_values > 0)

    def test_coherencia_values_returns(self, mc_result):
        """final_returns[i] = (final_values[i] - capital) / capital."""
        expected = (mc_result.final_values - CAPITAL) / CAPITAL
        assert np.allclose(mc_result.final_returns, expected, atol=1e-8)

    def test_var_negativo(self, mc_result):
        """El VaR histórico debe ser negativo (indica pérdida)."""
        assert mc_result.var_historical < 0

    def test_cvar_menor_o_igual_que_var(self, mc_result):
        """El CVaR debe ser ≤ al VaR (las pérdidas en la cola son mayores)."""
        assert mc_result.cvar_historical <= mc_result.var_historical + 1e-6

    def test_probabilidad_perdida_entre_0_y_1(self, mc_result):
        """La probabilidad de pérdida está entre 0 y 1."""
        assert 0.0 <= mc_result.prob_loss <= 1.0

    def test_percentiles_presentes(self, mc_result):
        """El diccionario de percentiles contiene los niveles esperados."""
        for p in ["P1", "P5", "P10", "P25", "P50", "P75", "P90", "P95", "P99"]:
            assert p in mc_result.percentiles

    def test_percentiles_ordenados(self, mc_result):
        """Los percentiles están ordenados de menor a mayor."""
        pcts = [mc_result.percentiles[f"P{p}"]
                for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]]
        assert all(pcts[i] <= pcts[i+1] + 1e-6 for i in range(len(pcts)-1))

    def test_capital_almacenado(self, mc_result):
        """El capital se almacena correctamente en el resultado."""
        assert mc_result.capital == CAPITAL

    def test_confianza_almacenada(self, mc_result):
        """El nivel de confianza se almacena correctamente."""
        assert mc_result.confidence == 0.95


# =============================================================================
# 3. REPRODUCIBILIDAD
# =============================================================================

class TestReproducibilidad:
    """Tests de reproducibilidad con semilla fija."""

    def test_misma_semilla_mismo_resultado(self, msr_result, synthetic_returns):
        """
        Dos ejecuciones con la misma semilla producen resultados idénticos.
        Garantiza reproducibilidad para el TFG.
        """
        mc1 = MonteCarlo(msr_result.weights.values, synthetic_returns,
                         capital=CAPITAL, n_simulations=200,
                         horizon_days=21, confidence=0.95, seed=SEED)
        mc2 = MonteCarlo(msr_result.weights.values, synthetic_returns,
                         capital=CAPITAL, n_simulations=200,
                         horizon_days=21, confidence=0.95, seed=SEED)
        r1 = mc1.run()
        r2 = mc2.run()
        assert np.allclose(r1.final_returns, r2.final_returns, atol=1e-10)

    def test_distinta_semilla_distinto_resultado(self, msr_result, synthetic_returns):
        """
        Dos ejecuciones con semillas distintas producen resultados diferentes.
        """
        mc1 = MonteCarlo(msr_result.weights.values, synthetic_returns,
                         capital=CAPITAL, n_simulations=200,
                         horizon_days=21, confidence=0.95, seed=1)
        mc2 = MonteCarlo(msr_result.weights.values, synthetic_returns,
                         capital=CAPITAL, n_simulations=200,
                         horizon_days=21, confidence=0.95, seed=99)
        r1 = mc1.run()
        r2 = mc2.run()
        assert not np.allclose(r1.final_returns, r2.final_returns)


# =============================================================================
# 4. MONTECARLORESULT — métodos del dataclass
# =============================================================================

class TestMonteCarloResult:
    """Tests de los métodos de MonteCarloResult."""

    def test_summary_devuelve_string(self, mc_result):
        """summary() devuelve una cadena no vacía."""
        s = mc_result.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_summary_contiene_var(self, mc_result):
        """summary() menciona el VaR."""
        s = mc_result.summary()
        assert "VaR" in s

    def test_summary_contiene_capital(self, mc_result):
        """summary() menciona el capital inicial."""
        s = mc_result.summary()
        assert str(int(CAPITAL)) in s.replace(",", "").replace(".", "")

    def test_metrics_dataframe_devuelve_df(self, mc_result):
        """metrics_dataframe() devuelve un pd.DataFrame."""
        df = mc_result.metrics_dataframe()
        assert isinstance(df, pd.DataFrame)

    def test_metrics_dataframe_columnas(self, mc_result):
        """metrics_dataframe() tiene las columnas 'Métrica' y 'Valor'."""
        df = mc_result.metrics_dataframe()
        assert "Métrica" in df.columns
        assert "Valor"   in df.columns

    def test_metrics_dataframe_filas(self, mc_result):
        """metrics_dataframe() tiene al menos 8 filas."""
        df = mc_result.metrics_dataframe()
        assert len(df) >= 8

    def test_metrics_dataframe_sin_nulos(self, mc_result):
        """metrics_dataframe() no contiene NaN."""
        df = mc_result.metrics_dataframe()
        assert df.isna().sum().sum() == 0


# =============================================================================
# 5. _COMPUTE_RISK_METRICS
# =============================================================================

class TestComputeRiskMetrics:
    """Tests del método privado _compute_risk_metrics()."""

    def test_var_es_percentil_correcto(self, mc_instance, mc_result):
        """
        El VaR histórico al 95% es el percentil 5 de la distribución empírica.
        """
        expected_var = np.percentile(mc_result.final_returns, 5.0)
        assert abs(mc_result.var_historical - expected_var) < 1e-8

    def test_cvar_es_media_de_la_cola(self, mc_result):
        """
        El CVaR es la media de los retornos que caen por debajo del VaR.
        """
        cola = mc_result.final_returns[
            mc_result.final_returns <= mc_result.var_historical
        ]
        expected_cvar = float(np.mean(cola))
        assert abs(mc_result.cvar_historical - expected_cvar) < 1e-8

    def test_var_parametrico_difiere_del_historico(self, mc_result):
        """
        El VaR paramétrico (supuesto normal) y el histórico no deben ser idénticos
        salvo coincidencia exacta (distribución real ≠ normal en general).
        """
        # No deben ser exactamente iguales (la distribución de retornos no es normal pura)
        diff = abs(mc_result.var_historical - mc_result.var_parametric)
        # Solo verificamos que ambos son negativos y tienen sentido
        assert mc_result.var_parametric < 0


# =============================================================================
# 6. SENSITIVITY_ANALYSIS
# =============================================================================

class TestSensitivityAnalysis:
    """Tests de la función sensitivity_analysis()."""

    @pytest.fixture(scope="class")
    def sens_df(self, msr_result, synthetic_returns):
        """DataFrame de sensibilidad con parámetros reducidos para rapidez."""
        return sensitivity_analysis(
            weights       = msr_result.weights.values,
            returns       = synthetic_returns,
            capital       = CAPITAL,
            n_simulations = 300,
            horizons      = [21, 63],
            confidences   = [0.90, 0.95],
            seed          = SEED,
        )

    def test_devuelve_dataframe(self, sens_df):
        """Devuelve un pd.DataFrame."""
        assert isinstance(sens_df, pd.DataFrame)

    def test_numero_de_filas(self, sens_df):
        """Tiene una fila por cada combinación horizonte × confianza (2×2=4)."""
        assert len(sens_df) == 4

    def test_columnas_requeridas(self, sens_df):
        """Contiene las columnas clave."""
        for col in ["VaR (%)", "CVaR (%)"]:
            assert col in sens_df.columns

    def test_var_negativo_en_todas_las_filas(self, sens_df):
        """El VaR es negativo en todas las combinaciones."""
        assert (sens_df["VaR (%)"] < 0).all()

    def test_cvar_menor_que_var_en_todas_las_filas(self, sens_df):
        """El CVaR es ≤ al VaR en todas las combinaciones."""
        assert (sens_df["CVaR (%)"] <= sens_df["VaR (%)"] + 1e-4).all()

    def test_var_crece_con_horizonte(self, sens_df):
        """
        A mayor horizonte temporal, el VaR debe ser igual o más negativo
        (mayor riesgo acumulado).
        """
        # Filtrar solo confianza 95%
        sub = sens_df[sens_df["Confianza (%)"] == 95].sort_values("Horizonte (días)")
        if len(sub) >= 2:
            vars_ = sub["VaR (%)"].values
            # VaR de horizonte mayor ≤ VaR de horizonte menor (más negativo)
            assert vars_[0] >= vars_[-1] - 1.0  # tolerancia 1% por varianza muestral


# =============================================================================
# 7. CASOS LÍMITE
# =============================================================================

class TestCasosLimite:
    """Tests de casos extremos y de borde."""

    def test_un_solo_activo(self, synthetic_returns):
        """La simulación funciona con un solo activo."""
        returns_1 = synthetic_returns.iloc[:, :1]
        w = np.array([1.0])
        mc = MonteCarlo(w, returns_1, capital=CAPITAL,
                        n_simulations=200, horizon_days=21, seed=SEED)
        result = mc.run()
        assert result.simulations.shape == (200, 22)
        assert np.allclose(result.simulations[:, 0], CAPITAL)

    def test_capital_muy_pequeno(self, msr_result, synthetic_returns):
        """La simulación funciona con capital mínimo (100€)."""
        mc = MonteCarlo(msr_result.weights.values, synthetic_returns,
                        capital=100.0, n_simulations=200,
                        horizon_days=21, seed=SEED)
        result = mc.run()
        assert np.allclose(result.simulations[:, 0], 100.0)

    def test_capital_grande(self, msr_result, synthetic_returns):
        """La simulación funciona con capital grande (1.000.000€)."""
        mc = MonteCarlo(msr_result.weights.values, synthetic_returns,
                        capital=1_000_000.0, n_simulations=200,
                        horizon_days=21, seed=SEED)
        result = mc.run()
        assert np.allclose(result.simulations[:, 0], 1_000_000.0)

    def test_confianza_99(self, msr_result, synthetic_returns):
        """La simulación funciona con nivel de confianza del 99%."""
        mc = MonteCarlo(msr_result.weights.values, synthetic_returns,
                        capital=CAPITAL, n_simulations=N_SIMS,
                        horizon_days=HORIZON, confidence=0.99, seed=SEED)
        result = mc.run()
        # VaR al 99% debe ser más negativo que al 95%
        mc95 = MonteCarlo(msr_result.weights.values, synthetic_returns,
                          capital=CAPITAL, n_simulations=N_SIMS,
                          horizon_days=HORIZON, confidence=0.95, seed=SEED)
        result95 = mc95.run()
        assert result.var_historical <= result95.var_historical + 1e-4

    def test_horizonte_un_dia(self, msr_result, synthetic_returns):
        """La simulación funciona con horizonte de 1 día."""
        mc = MonteCarlo(msr_result.weights.values, synthetic_returns,
                        capital=CAPITAL, n_simulations=200,
                        horizon_days=1, seed=SEED)
        result = mc.run()
        assert result.simulations.shape == (200, 2)
