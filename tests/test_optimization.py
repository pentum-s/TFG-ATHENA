"""
=============================================================================
PROYECTO ATHENA — tests/test_optimization.py
=============================================================================
Tests unitarios e integración del módulo optimization.py.

Cubre:
    - PortfolioResult           → to_dict(), summary()
    - PortfolioOptimizer.__init__ → validaciones de entrada, parámetros mu/sigma
    - min_variance()            → forma, restricciones, propiedades MVP
    - max_sharpe()              → Sharpe ≥ MVP, restricciones de pesos
    - efficient_frontier()      → monotonicidad, forma del DataFrame
    - equal_weight()            → pesos iguales, suma = 1
    - portfolio_metrics()       → métricas sobre pesos dados
    - portfolio_from_weights()  → función de utilidad a nivel de módulo
=============================================================================
"""

import numpy as np
import pandas as pd
import pytest

from optimization import PortfolioOptimizer, PortfolioResult, portfolio_from_weights
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from constants import TICKERS, RISK_FREE, N_ASSETS


# =============================================================================
# 1. PORTFOLIORESULT — dataclass de resultado
# =============================================================================

class TestPortfolioResult:
    """Tests de la dataclass PortfolioResult."""

    @pytest.fixture
    def sample_result(self, optimizer):
        """MVP de ejemplo para los tests."""
        return optimizer.min_variance()

    def test_atributos_presentes(self, sample_result):
        """Todos los atributos clave están presentes."""
        assert hasattr(sample_result, "weights")
        assert hasattr(sample_result, "expected_return")
        assert hasattr(sample_result, "volatility")
        assert hasattr(sample_result, "sharpe_ratio")
        assert hasattr(sample_result, "label")

    def test_weights_es_series(self, sample_result):
        """Los pesos son un pd.Series."""
        assert isinstance(sample_result.weights, pd.Series)

    def test_pesos_suman_uno(self, sample_result):
        """Los pesos deben sumar exactamente 1."""
        assert abs(sample_result.weights.sum() - 1.0) < 1e-6

    def test_to_dict_devuelve_dict(self, sample_result):
        """to_dict() devuelve un diccionario."""
        d = sample_result.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_claves_correctas(self, sample_result):
        """to_dict() contiene las claves esperadas."""
        d = sample_result.to_dict()
        for key in ["label", "weights", "expected_return", "volatility", "sharpe_ratio"]:
            assert key in d

    def test_to_dict_weights_es_dict(self, sample_result):
        """to_dict()['weights'] es un diccionario {ticker: peso}."""
        d = sample_result.to_dict()
        assert isinstance(d["weights"], dict)

    def test_summary_devuelve_string(self, sample_result):
        """summary() devuelve una cadena de texto."""
        s = sample_result.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_summary_contiene_label(self, sample_result):
        """summary() contiene el label de la cartera."""
        s = sample_result.summary()
        assert sample_result.label in s


# =============================================================================
# 2. PORTFOLIOOPTIMIZER.__INIT__
# =============================================================================

class TestPortfolioOptimizerInit:
    """Tests de inicialización y parámetros del optimizador."""

    def test_n_assets_correcto(self, optimizer):
        """El número de activos se detecta correctamente."""
        assert optimizer.n_assets == N_ASSETS

    def test_tickers_correctos(self, optimizer):
        """La lista de tickers se almacena correctamente."""
        assert optimizer.tickers == TICKERS

    def test_mu_shape(self, optimizer):
        """El vector mu tiene shape (n_assets,)."""
        assert optimizer.mu.shape == (N_ASSETS,)

    def test_sigma_shape(self, optimizer):
        """La matriz sigma tiene shape (n_assets, n_assets)."""
        assert optimizer.sigma.shape == (N_ASSETS, N_ASSETS)

    def test_sigma_simetrica(self, optimizer):
        """La matriz de covarianzas debe ser simétrica."""
        assert np.allclose(optimizer.sigma, optimizer.sigma.T, atol=1e-10)

    def test_sigma_semidefinida_positiva(self, optimizer):
        """La matriz de covarianzas debe ser semidefinida positiva (eigenvalores ≥ 0)."""
        eigenvalues = np.linalg.eigvalsh(optimizer.sigma)
        assert np.all(eigenvalues >= -1e-8)

    def test_error_dataframe_vacio(self):
        """Debe lanzar ValueError con un DataFrame vacío."""
        with pytest.raises(ValueError):
            PortfolioOptimizer(pd.DataFrame())

    def test_error_w_min_mayor_que_w_max(self, synthetic_returns):
        """Debe lanzar ValueError si w_min ≥ w_max."""
        with pytest.raises(ValueError):
            PortfolioOptimizer(synthetic_returns, w_min=0.5, w_max=0.3)

    def test_risk_free_almacenado(self, optimizer):
        """La tasa libre de riesgo se almacena correctamente."""
        assert optimizer.risk_free_rate == RISK_FREE


# =============================================================================
# 3. MIN_VARIANCE — Cartera de Mínima Varianza
# =============================================================================

class TestMinVariance:
    """Tests del método min_variance()."""

    @pytest.fixture(scope="class")
    def mvp(self, optimizer):
        return optimizer.min_variance()

    def test_devuelve_portfolio_result(self, mvp):
        """Devuelve un objeto PortfolioResult."""
        assert isinstance(mvp, PortfolioResult)

    def test_label_correcto(self, mvp):
        """El label contiene 'Varianza' o 'Mínima'."""
        assert "Varianza" in mvp.label or "Mínima" in mvp.label

    def test_pesos_suman_uno(self, mvp):
        """Los pesos suman 1."""
        assert abs(mvp.weights.sum() - 1.0) < 1e-5

    def test_pesos_dentro_de_bounds(self, mvp, optimizer):
        """Todos los pesos respetan los bounds configurados."""
        assert (mvp.weights >= optimizer.w_min - 1e-6).all()
        assert (mvp.weights <= optimizer.w_max + 1e-6).all()

    def test_pesos_no_negativos(self, mvp):
        """Sin ventas en corto (w_min=0), ningún peso es negativo."""
        assert (mvp.weights >= -1e-6).all()

    def test_volatilidad_positiva(self, mvp):
        """La volatilidad del MVP es positiva."""
        assert mvp.volatility > 0

    def test_volatilidad_menor_que_cualquier_activo_individual(
        self, mvp, optimizer
    ):
        """
        La volatilidad del MVP debe ser ≤ a la del activo individual
        menos volátil — la diversificación nunca puede empeorar el mínimo.
        """
        vols_individuales = np.sqrt(np.diag(optimizer.sigma))
        assert mvp.volatility <= vols_individuales.min() + 1e-6

    def test_reproducibilidad(self, optimizer):
        """Dos llamadas consecutivas devuelven el mismo resultado."""
        mvp1 = optimizer.min_variance()
        mvp2 = optimizer.min_variance()
        assert np.allclose(mvp1.weights.values, mvp2.weights.values, atol=1e-4)


# =============================================================================
# 4. MAX_SHARPE — Cartera de Máximo Sharpe
# =============================================================================

class TestMaxSharpe:
    """Tests del método max_sharpe()."""

    def test_devuelve_portfolio_result(self, msr_result):
        """Devuelve un objeto PortfolioResult."""
        assert isinstance(msr_result, PortfolioResult)

    def test_label_correcto(self, msr_result):
        """El label contiene 'Sharpe'."""
        assert "Sharpe" in msr_result.label

    def test_pesos_suman_uno(self, msr_result):
        """Los pesos suman 1."""
        assert abs(msr_result.weights.sum() - 1.0) < 1e-5

    def test_pesos_dentro_de_bounds(self, msr_result, optimizer):
        """Todos los pesos respetan los bounds configurados."""
        assert (msr_result.weights >= optimizer.w_min - 1e-6).all()
        assert (msr_result.weights <= optimizer.w_max + 1e-6).all()

    def test_sharpe_mayor_que_mvp(self, msr_result, optimizer):
        """
        El Sharpe de la MSR debe ser ≥ al Sharpe del MVP.
        La MSR es la cartera de máximo Sharpe por definición.
        """
        mvp = optimizer.min_variance()
        assert msr_result.sharpe_ratio >= mvp.sharpe_ratio - 1e-4

    def test_volatilidad_positiva(self, msr_result):
        """La volatilidad de la MSR es positiva."""
        assert msr_result.volatility > 0

    def test_tickers_en_weights(self, msr_result):
        """El índice de los pesos contiene los tickers configurados."""
        assert list(msr_result.weights.index) == TICKERS


# =============================================================================
# 5. EFFICIENT_FRONTIER — Frontera Eficiente
# =============================================================================

class TestEfficientFrontier:
    """Tests del método efficient_frontier()."""

    @pytest.fixture(scope="class")
    def frontier(self, optimizer):
        return optimizer.efficient_frontier(n_points=20)

    def test_devuelve_dataframe(self, frontier):
        """Devuelve un pd.DataFrame."""
        assert isinstance(frontier, pd.DataFrame)

    def test_columnas_requeridas(self, frontier):
        """Contiene las columnas mínimas esperadas."""
        for col in ["volatility", "expected_return", "sharpe_ratio"]:
            assert col in frontier.columns

    def test_columnas_de_pesos(self, frontier):
        """Contiene una columna de peso por cada ticker."""
        for ticker in TICKERS:
            assert f"w_{ticker}" in frontier.columns

    def test_numero_de_filas(self, frontier):
        """Tiene aproximadamente n_points filas (puede ser ligeramente menor)."""
        assert len(frontier) >= 10  # al menos la mitad de puntos válidos

    def test_volatilidades_positivas(self, frontier):
        """Todas las volatilidades de la frontera son positivas."""
        assert (frontier["volatility"] > 0).all()

    def test_pesos_suman_uno_en_cada_punto(self, frontier):
        """Los pesos de cada punto de la frontera suman 1."""
        weight_cols = [f"w_{t}" for t in TICKERS]
        for _, row in frontier.iterrows():
            total = sum(row[c] for c in weight_cols)
            assert abs(total - 1.0) < 1e-4

    def test_ordenado_por_volatilidad(self, frontier):
        """La frontera está ordenada de menor a mayor volatilidad."""
        vols = frontier["volatility"].values
        assert np.all(vols[1:] >= vols[:-1] - 1e-6)


# =============================================================================
# 6. EQUAL_WEIGHT — Cartera Equi-ponderada
# =============================================================================

class TestEqualWeight:
    """Tests del método equal_weight()."""

    @pytest.fixture(scope="class")
    def eq(self, optimizer):
        return optimizer.equal_weight()

    def test_devuelve_portfolio_result(self, eq):
        """Devuelve un objeto PortfolioResult."""
        assert isinstance(eq, PortfolioResult)

    def test_todos_pesos_iguales(self, eq):
        """Todos los pesos son iguales a 1/N."""
        expected_w = 1.0 / N_ASSETS
        assert np.allclose(eq.weights.values, expected_w, atol=1e-10)

    def test_pesos_suman_uno(self, eq):
        """Los pesos suman 1."""
        assert abs(eq.weights.sum() - 1.0) < 1e-10

    def test_label_contiene_1_N(self, eq):
        """El label menciona 1/N o Equi."""
        assert "1/N" in eq.label or "Equi" in eq.label

    def test_volatilidad_positiva(self, eq):
        """La volatilidad de la cartera equi-ponderada es positiva."""
        assert eq.volatility > 0


# =============================================================================
# 7. PORTFOLIO_METRICS — métricas sobre pesos arbitrarios
# =============================================================================

class TestPortfolioMetrics:
    """Tests del método portfolio_metrics()."""

    def test_devuelve_dict(self, optimizer):
        """Devuelve un diccionario."""
        w = np.ones(N_ASSETS) / N_ASSETS
        m = optimizer.portfolio_metrics(w)
        assert isinstance(m, dict)

    def test_claves_correctas(self, optimizer):
        """El diccionario contiene las claves esperadas."""
        w = np.ones(N_ASSETS) / N_ASSETS
        m = optimizer.portfolio_metrics(w)
        for key in ["expected_return", "volatility", "sharpe_ratio"]:
            assert key in m

    def test_volatilidad_positiva(self, optimizer):
        """La volatilidad calculada es positiva."""
        w = np.ones(N_ASSETS) / N_ASSETS
        m = optimizer.portfolio_metrics(w)
        assert m["volatility"] > 0

    def test_normaliza_pesos(self, optimizer):
        """Los pesos se normalizan si no suman exactamente 1."""
        w = np.ones(N_ASSETS) * 0.5  # suma > 1
        m = optimizer.portfolio_metrics(w)
        assert m["volatility"] > 0   # no debe lanzar error

    def test_coherencia_con_msr(self, optimizer, msr_result):
        """
        Las métricas calculadas manualmente sobre los pesos MSR
        deben coincidir con las del PortfolioResult.
        """
        m = optimizer.portfolio_metrics(msr_result.weights.values)
        assert abs(m["volatility"] - msr_result.volatility * 100) < 0.01
        assert abs(m["expected_return"] - msr_result.expected_return * 100) < 0.01


# =============================================================================
# 8. PORTFOLIO_FROM_WEIGHTS — función de utilidad
# =============================================================================

class TestPortfolioFromWeights:
    """Tests de la función portfolio_from_weights()."""

    def test_devuelve_portfolio_result(self, synthetic_returns):
        """Devuelve un PortfolioResult."""
        weights = {t: 1/N_ASSETS for t in TICKERS}
        pf = portfolio_from_weights(weights, synthetic_returns)
        assert isinstance(pf, PortfolioResult)

    def test_pesos_suman_uno(self, synthetic_returns):
        """Los pesos del resultado suman 1."""
        weights = {t: 0.2 for t in TICKERS}
        pf = portfolio_from_weights(weights, synthetic_returns)
        assert abs(pf.weights.sum() - 1.0) < 1e-6

    def test_error_ticker_no_encontrado(self, synthetic_returns):
        """Debe lanzar ValueError si un ticker no está en returns."""
        weights = {"FAKE.MC": 0.5, "SAN.MC": 0.5}
        with pytest.raises(ValueError, match="no encontrados"):
            portfolio_from_weights(weights, synthetic_returns)

    def test_pesos_parciales_se_normalizan(self, synthetic_returns):
        """Los pesos parciales (no todos los tickers) se normalizan correctamente."""
        weights = {"SAN.MC": 1.0, "BBVA.MC": 1.0}  # solo 2 de 5
        pf = portfolio_from_weights(weights, synthetic_returns)
        assert abs(pf.weights.sum() - 1.0) < 1e-6
