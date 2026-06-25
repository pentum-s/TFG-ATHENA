"""
=============================================================================
PROYECTO ATHENA — tests/conftest.py
=============================================================================
Fixtures compartidos entre todos los módulos de test.
Genera datos sintéticos reproducibles sin necesidad de conexión a la API
de Yahoo Finance, garantizando que los tests son rápidos y deterministas.
=============================================================================
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# Añadir el directorio raíz al path para que los tests encuentren los módulos
sys.path.insert(0, str(Path(__file__).parent.parent / "modules"))


# ---------------------------------------------------------------------------
# Parámetros globales reproducibles
# ---------------------------------------------------------------------------
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from constants import SEED, N_DAYS, N_ASSETS, TICKERS, CAPITAL, RISK_FREE, N_SIMS, HORIZON


# ---------------------------------------------------------------------------
# Fixture: fechas de días hábiles
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def trading_dates():
    """Rango de N_DAYS días hábiles a partir del 2020-01-02."""
    return pd.bdate_range(start="2020-01-02", periods=N_DAYS)


# ---------------------------------------------------------------------------
# Fixture: precios sintéticos realistas
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def synthetic_prices(trading_dates):
    """
    DataFrame de precios de cierre ajustados sintéticos.
    Generado con caminata aleatoria geométrica log-normal con correlaciones.
    """
    rng = np.random.default_rng(SEED)

    mu_daily  = np.array([0.0004, 0.0005, 0.0002, 0.0008, 0.0003])
    vol_daily = np.array([0.015,  0.017,  0.013,  0.018,  0.012])
    corr = np.array([
        [1.00, 0.75, 0.55, 0.30, 0.45],
        [0.75, 1.00, 0.60, 0.28, 0.50],
        [0.55, 0.60, 1.00, 0.22, 0.42],
        [0.30, 0.28, 0.22, 1.00, 0.35],
        [0.45, 0.50, 0.42, 0.35, 1.00],
    ])
    cov = np.diag(vol_daily) @ corr @ np.diag(vol_daily)
    returns_arr = rng.multivariate_normal(mu_daily, cov, size=N_DAYS)
    prices_arr  = np.cumprod(1 + returns_arr, axis=0) * 10.0

    return pd.DataFrame(prices_arr, index=trading_dates, columns=TICKERS)


# ---------------------------------------------------------------------------
# Fixture: precios con nulos para tests de preprocesado
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def prices_with_nulls(synthetic_prices):
    """
    Precios con nulos artificiales:
    - Nulos aislados en SAN.MC (días 10, 50, 200) → interpolables
    - 4 nulos consecutivos en BBVA.MC (días 100-103) → interpolables
    - 60% nulos en TEF.MC → baja calidad, debe eliminarse
    """
    df = synthetic_prices.copy()
    df.iloc[10,  0] = np.nan
    df.iloc[50,  0] = np.nan
    df.iloc[200, 0] = np.nan
    df.iloc[100:104, 1] = np.nan
    null_idx = np.random.default_rng(1).choice(
        N_DAYS, size=int(N_DAYS * 0.60), replace=False
    )
    df.iloc[null_idx, 2] = np.nan
    return df


# ---------------------------------------------------------------------------
# Fixture: rendimientos logarítmicos
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def synthetic_returns(synthetic_prices):
    """Rendimientos logarítmicos diarios sobre precios sintéticos."""
    returns = np.log(synthetic_prices / synthetic_prices.shift(1)).iloc[1:]
    return returns


# ---------------------------------------------------------------------------
# Fixture: instancia de DataAcquisition (sin API)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def da_instance(tmp_path_factory):
    """DataAcquisition con caché en directorio temporal."""
    from data_acquisition import DataAcquisition
    cache = tmp_path_factory.mktemp("cache")
    return DataAcquisition(
        tickers   = TICKERS,
        start     = "2020-01-02",
        end       = "2022-12-31",
        cache_dir = cache,
    )


# ---------------------------------------------------------------------------
# Fixture: instancia de PortfolioOptimizer
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def optimizer(synthetic_returns):
    """PortfolioOptimizer con rendimientos sintéticos."""
    from optimization import PortfolioOptimizer
    return PortfolioOptimizer(
        returns        = synthetic_returns,
        risk_free_rate = RISK_FREE,
        w_min          = 0.0,
        w_max          = 0.40,
    )


# ---------------------------------------------------------------------------
# Fixture: resultado MSR precalculado
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def msr_result(optimizer):
    """Cartera de Máximo Sharpe precalculada."""
    return optimizer.max_sharpe()


# ---------------------------------------------------------------------------
# Fixture: instancia de MonteCarlo
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def mc_instance(msr_result, synthetic_returns):
    """MonteCarlo con N_SIMS reducidas para rapidez en tests."""
    from monte_carlo import MonteCarlo
    return MonteCarlo(
        weights       = msr_result.weights.values,
        returns       = synthetic_returns,
        capital       = CAPITAL,
        n_simulations = N_SIMS,
        horizon_days  = HORIZON,
        confidence    = 0.95,
        seed          = SEED,
    )


# ---------------------------------------------------------------------------
# Fixture: resultado Monte Carlo precalculado
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def mc_result(mc_instance):
    """MonteCarloResult precalculado."""
    return mc_instance.run()
