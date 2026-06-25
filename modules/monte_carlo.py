"""
=============================================================================
PROYECTO ATHENA — Hito 5: Simulación Monte Carlo y Análisis de Riesgo
=============================================================================
Grado en Ciencia de Datos Aplicada — Universitat Oberta de Catalunya (UOC)

Autor   : Álvaro Sánchez Mendoza
Tutor   : Carlos Gaitán Poyatos

Descripción
-----------
Este módulo implementa el motor de simulación estocástica del Proyecto
Athena. Recibe los pesos de la cartera óptima producidos por el módulo
de optimización (Hito 4) y genera N escenarios futuros de evolución
del valor de la cartera mediante el método de Monte Carlo.

A partir de la distribución de escenarios calcula:

    1. Trayectorias simuladas de la cartera para un horizonte dado.
    2. Value at Risk (VaR): pérdida máxima esperada con un nivel de
       confianza dado (1 - alpha) en el horizonte temporal.
    3. Conditional Value at Risk (CVaR / Expected Shortfall): pérdida
       media esperada en los peores escenarios (cola de la distribución).
    4. Distribución final de rentabilidades acumuladas.
    5. Percentiles clave (P10, P50, P90) de la distribución de resultados.

Alcance del Hito 5
------------------
    Completado:
        - Simulación de N trayectorias por el método de Monte Carlo
          con rendimientos multivariantes (correlaciones conservadas)
        - Cálculo de VaR histórico y paramétrico
        - Cálculo de CVaR (Expected Shortfall)
        - Análisis de percentiles y estadísticas de la distribución
        - Probabilidad de pérdida y de superar el benchmark

Dependencias
------------
    pip install numpy pandas scipy

Uso rápido
----------
    from monte_carlo import MonteCarlo

    mc = MonteCarlo(
        weights        = msr.weights.values,   # pesos del Hito 4
        returns        = returns,              # rendimientos del Hito 3
        capital        = 10_000,               # capital inicial en euros
        n_simulations  = 10_000,
        horizon_days   = 252,
        confidence     = 0.95,
    )
    result = mc.run()
    print(result.summary())
=============================================================================
"""

# ---------------------------------------------------------------------------
# Importaciones
# ---------------------------------------------------------------------------
import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("athena.monte_carlo")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
TRADING_DAYS_YEAR: int = 252


# ---------------------------------------------------------------------------
# Dataclass de resultado de la simulación
# ---------------------------------------------------------------------------

@dataclass
class MonteCarloResult:
    """
    Contenedor con los resultados completos de la simulación Monte Carlo.

    Atributos
    ----------
    simulations : np.ndarray
        Matriz de trayectorias simuladas del valor de la cartera.
        Shape: (n_simulations, horizon_days + 1).
        La columna 0 es siempre igual al capital inicial.
    final_values : np.ndarray
        Vector de valores finales de la cartera en cada escenario.
        Shape: (n_simulations,). Expresado en la misma unidad que capital.
    final_returns : np.ndarray
        Vector de rentabilidades acumuladas finales (fracción).
        final_returns[i] = (final_values[i] - capital) / capital.
    var_historical : float
        VaR histórico: pérdida máxima al nivel de confianza (1-alpha).
        Expresado como fracción negativa del capital.
    cvar_historical : float
        CVaR histórico (Expected Shortfall): media de las pérdidas en
        los peores escenarios (cola izquierda de la distribución).
        Expresado como fracción negativa.
    var_parametric : float
        VaR paramétrico bajo supuesto de normalidad.
    percentiles : dict
        Percentiles P1, P5, P10, P25, P50, P75, P90, P95, P99
        de la distribución de rentabilidades finales.
    prob_loss : float
        Probabilidad de pérdida (rentabilidad final < 0).
    prob_target : float
        Probabilidad de superar la rentabilidad objetivo configurada.
    capital : float
        Capital inicial de la simulación.
    confidence : float
        Nivel de confianza del VaR (p.ej. 0.95).
    horizon_days : int
        Horizonte temporal de la simulación en días hábiles.
    n_simulations : int
        Número de trayectorias simuladas.
    """
    simulations:    np.ndarray
    final_values:   np.ndarray
    final_returns:  np.ndarray
    var_historical: float
    cvar_historical:float
    var_parametric: float
    percentiles:    dict
    prob_loss:      float
    prob_target:    float
    capital:        float
    confidence:     float
    horizon_days:   int
    n_simulations:  int
    target_return:  float = 0.0

    def summary(self) -> str:
        """
        Devuelve un resumen textual de los resultados de la simulación.

        Returns
        -------
        str
            Texto con las métricas clave de riesgo y rentabilidad.
        """
        alpha = 1.0 - self.confidence
        lines = [
            f"  {'─' * 50}",
            "  RESULTADOS — Simulación Monte Carlo",
            f"  {'─' * 50}",
            f"  Simulaciones     : {self.n_simulations:,}",
            f"  Horizonte        : {self.horizon_days} días hábiles "
            f"(~{self.horizon_days / TRADING_DAYS_YEAR:.1f} año/s)",
            f"  Capital inicial  : {self.capital:,.2f} €",
            f"  Nivel confianza  : {self.confidence * 100:.0f}%",
            "",
            "  ── Rentabilidad proyectada ──────────────────────────",
            f"  Mediana (P50)    : {self.percentiles['P50'] * 100:+.2f}%"
            f"  ({self.capital * (1 + self.percentiles['P50']):,.0f} €)",
            f"  P10 (pesimista)  : {self.percentiles['P10'] * 100:+.2f}%",
            f"  P90 (optimista)  : {self.percentiles['P90'] * 100:+.2f}%",
            "",
            "  ── Métricas de riesgo ───────────────────────────────",
            f"  VaR histórico    ({self.confidence:.0%}): "
            f"{self.var_historical * 100:+.2f}%"
            f"  ({self.capital * self.var_historical:,.0f} €)",
            f"  CVaR / ES        ({self.confidence:.0%}): "
            f"{self.cvar_historical * 100:+.2f}%"
            f"  ({self.capital * self.cvar_historical:,.0f} €)",
            f"  VaR paramétrico  ({self.confidence:.0%}): "
            f"{self.var_parametric * 100:+.2f}%",
            "",
            "  ── Probabilidades ───────────────────────────────────",
            f"  Prob. pérdida    : {self.prob_loss * 100:.1f}%",
            f"  Prob. ganancia   : {(1 - self.prob_loss) * 100:.1f}%",
            f"  {'─' * 50}",
        ]
        return "\n".join(lines)

    def metrics_dataframe(self) -> pd.DataFrame:
        """
        Devuelve las métricas de riesgo como DataFrame para el dashboard.

        Returns
        -------
        pd.DataFrame
            DataFrame con columnas 'Métrica' y 'Valor'.
        """
        rows = [
            ("VaR histórico (95%)",      f"{self.var_historical * 100:+.2f}%"),
            ("CVaR / ES (95%)",           f"{self.cvar_historical * 100:+.2f}%"),
            ("VaR paramétrico (95%)",     f"{self.var_parametric * 100:+.2f}%"),
            ("Rentabilidad mediana (P50)", f"{self.percentiles['P50'] * 100:+.2f}%"),
            ("Rentabilidad P10",          f"{self.percentiles['P10'] * 100:+.2f}%"),
            ("Rentabilidad P90",          f"{self.percentiles['P90'] * 100:+.2f}%"),
            ("Probabilidad de pérdida",   f"{self.prob_loss * 100:.1f}%"),
            ("Probabilidad de ganancia",  f"{(1 - self.prob_loss) * 100:.1f}%"),
            ("Capital inicial",           f"{self.capital:,.0f} €"),
            ("Valor mediano final",       f"{self.capital * (1 + self.percentiles['P50']):,.0f} €"),
            ("Peor escenario (P1)",       f"{self.percentiles['P1'] * 100:+.2f}%"),
            ("Mejor escenario (P99)",     f"{self.percentiles['P99'] * 100:+.2f}%"),
        ]
        return pd.DataFrame(rows, columns=["Métrica", "Valor"])


# ---------------------------------------------------------------------------
# Clase principal: MonteCarlo
# ---------------------------------------------------------------------------

class MonteCarlo:
    """
    Motor de simulación Monte Carlo para carteras de inversión.

    Genera N trayectorias estocásticas del valor de la cartera
    manteniendo la estructura de correlaciones entre activos.
    Calcula métricas de riesgo estándar: VaR, CVaR y distribución
    de rentabilidades finales.

    Parametros
    ----------
    weights : np.ndarray or pd.Series
        Pesos de la cartera a simular (deben sumar 1.0).
        Obtenidos del módulo de optimización (Hito 4).
    returns : pd.DataFrame
        DataFrame de rendimientos logarítmicos históricos (Hito 3).
        Se usa para estimar mu (media) y Sigma (covarianzas).
    capital : float, optional
        Capital inicial en euros. Por defecto 10.000 €.
    n_simulations : int, optional
        Número de trayectorias a simular. Por defecto 10.000.
        Más simulaciones = mayor precisión pero mayor tiempo.
        Rango típico: 1.000 – 50.000.
    horizon_days : int, optional
        Horizonte de simulación en días hábiles.
        Por defecto 252 (1 año).
    confidence : float, optional
        Nivel de confianza para el VaR. Por defecto 0.95 (95%).
    target_return : float, optional
        Rentabilidad objetivo anual para calcular la probabilidad
        de superarla. Por defecto 0.0 (probabilidad de no perder).
    seed : int, optional
        Semilla para reproducibilidad de los resultados. Por defecto 42.

    Ejemplos
    --------
    >>> mc = MonteCarlo(
    ...     weights       = msr.weights.values,
    ...     returns       = returns,
    ...     capital       = 10_000,
    ...     n_simulations = 10_000,
    ...     horizon_days  = 252,
    ...     confidence    = 0.95,
    ... )
    >>> result = mc.run()
    >>> print(result.summary())
    """

    def __init__(
        self,
        weights:       np.ndarray | pd.Series,
        returns:       pd.DataFrame,
        capital:       float = 10_000.0,
        n_simulations: int   = 10_000,
        horizon_days:  int   = 252,
        confidence:    float = 0.95,
        target_return: float = 0.0,
        seed:          int   = 42,
    ) -> None:

        # Validaciones de entrada
        if not (0.0 < confidence < 1.0):
            raise ValueError(
                f"El nivel de confianza debe estar en (0, 1). "
                f"Recibido: {confidence}"
            )
        if capital <= 0:
            raise ValueError(
                f"El capital inicial debe ser positivo. Recibido: {capital}"
            )
        if horizon_days < 1:
            raise ValueError("El horizonte debe ser al menos 1 día.")
        if n_simulations < 100:
            raise ValueError(
                "Se recomienda un mínimo de 100 simulaciones para "
                "obtener estimaciones estadísticamente significativas."
            )

        # Normalizar pesos
        w = np.asarray(weights, dtype=float)
        if not np.isclose(w.sum(), 1.0, atol=1e-6):
            logger.warning(
                "Los pesos no suman exactamente 1 (suma=%.8f). "
                "Se normalizan automáticamente.", w.sum()
            )
            w = w / w.sum()
        self.weights = w

        self.tickers       = list(returns.columns)
        self.n_assets      = len(self.tickers)
        self.capital       = capital
        self.n_simulations = n_simulations
        self.horizon_days  = horizon_days
        self.confidence    = confidence
        self.target_return = target_return
        self.seed          = seed

        # Estimación de parámetros diarios (NO anualizados)
        # para que las simulaciones sean en escala diaria
        self.mu_daily    = returns.mean().values
        self.sigma_daily = returns.cov().values

        logger.info(
            "MonteCarlo inicializado — %d simulaciones | %d días | "
            "confianza: %.0f%% | capital: %.0f€ | seed: %d",
            n_simulations, horizon_days, confidence * 100, capital, seed,
        )

    # ------------------------------------------------------------------ #
    # MÉTODO PRINCIPAL: RUN                                                #
    # ------------------------------------------------------------------ #

    def run(self) -> MonteCarloResult:
        """
        Ejecuta la simulación Monte Carlo completa.

        Genera N trayectorias de rendimientos multivariantes conservando
        la estructura de correlaciones histórica entre activos, acumula
        los rendimientos para obtener el valor de la cartera en cada
        paso temporal, y calcula todas las métricas de riesgo.

        Returns
        -------
        MonteCarloResult
            Objeto con todas las trayectorias y métricas de riesgo.

        Notas
        -----
        La simulación usa el método de **Cholesky** para generar
        rendimientos multivariantes correlacionados:

            1. Descomponer Sigma = L · L^T  (descomposición de Cholesky).
            2. Generar Z ~ N(0, I)  (normales independientes).
            3. Obtener r = mu + L · Z  (normales correlacionadas).

        Este enfoque preserva exactamente las correlaciones históricas
        estimadas, lo que es más realista que simular activos
        independientes y más eficiente computacionalmente que la
        simulación de escenarios históricos completos.

        El rendimiento de la cartera en cada día es:
            r_portfolio_t = w^T · r_t

        El valor acumulado:
            V_t = V_0 · exp(sum_{s=1}^{t} r_portfolio_s)
        """
        logger.info(
            "Iniciando simulación — %d trayectorias × %d días...",
            self.n_simulations, self.horizon_days,
        )

        rng = np.random.default_rng(seed=self.seed)

        # ── Descomposición de Cholesky de la matriz de covarianzas ──────
        # Garantiza que los rendimientos simulados conservan las
        # correlaciones históricas entre activos.
        # Se añade una pequeña perturbación (jitter) para garantizar
        # que la matriz es definida positiva (estabilidad numérica).
        sigma_jitter = self.sigma_daily + np.eye(self.n_assets) * 1e-8
        try:
            L = np.linalg.cholesky(sigma_jitter)
        except np.linalg.LinAlgError:
            # Si Cholesky falla, usar la pseudo-inversa
            logger.warning(
                "Cholesky falló. Usando descomposición SVD como fallback."
            )
            U, s, _ = np.linalg.svd(sigma_jitter)
            L = U @ np.diag(np.sqrt(np.maximum(s, 0)))

        # ── Generar rendimientos simulados ───────────────────────────────
        # Shape de Z: (n_simulations, horizon_days, n_assets)
        Z = rng.standard_normal(
            (self.n_simulations, self.horizon_days, self.n_assets)
        )
        # Aplicar Cholesky: r = mu + L @ z para cada día y simulación
        # Dimensiones: (N, T, n_assets) = mu + Z @ L^T
        returns_sim = self.mu_daily + Z @ L.T

        # ── Rendimientos diarios de la cartera ───────────────────────────
        # r_portfolio[i, t] = w^T · returns_sim[i, t, :]
        # Shape: (n_simulations, horizon_days)
        portfolio_returns_daily = returns_sim @ self.weights

        # ── Trayectorias de valor acumulado ──────────────────────────────
        # V_t = V_0 · exp(sum_{s<=t} r_s)  — rendimientos logarítmicos
        # cumsum da los rendimientos acumulados hasta cada día t
        cumulative_log_returns = np.cumsum(portfolio_returns_daily, axis=1)

        # Añadir el día 0 (valor inicial = capital) al principio
        # Shape: (n_simulations, horizon_days + 1)
        cum_with_start = np.hstack([
            np.zeros((self.n_simulations, 1)),
            cumulative_log_returns,
        ])
        simulations = self.capital * np.exp(cum_with_start)

        # ── Valores y rentabilidades finales ─────────────────────────────
        final_values   = simulations[:, -1]
        final_returns  = (final_values - self.capital) / self.capital

        # ── Métricas de riesgo ───────────────────────────────────────────
        var_h, cvar_h, var_p = self._compute_risk_metrics(final_returns)

        # ── Percentiles de la distribución de rentabilidades ─────────────
        pct_levels = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentiles = {
            f"P{p}": float(np.percentile(final_returns, p))
            for p in pct_levels
        }

        # ── Probabilidades ───────────────────────────────────────────────
        # Probabilidad de pérdida: escenarios con rentabilidad < 0
        prob_loss   = float(np.mean(final_returns < 0))

        # Probabilidad de superar el objetivo de rentabilidad
        # Convertir la rentabilidad objetivo anual a escala del horizonte
        target_horizon = (
            (1.0 + self.target_return) **
            (self.horizon_days / TRADING_DAYS_YEAR) - 1.0
        )
        prob_target = float(np.mean(final_returns >= target_horizon))

        logger.info(
            "Simulación completada — "
            "VaR(%.0f%%): %.2f%% | CVaR: %.2f%% | P(pérdida): %.1f%%",
            self.confidence * 100,
            var_h * 100,
            cvar_h * 100,
            prob_loss * 100,
        )

        return MonteCarloResult(
            simulations     = simulations,
            final_values    = final_values,
            final_returns   = final_returns,
            var_historical  = var_h,
            cvar_historical = cvar_h,
            var_parametric  = var_p,
            percentiles     = percentiles,
            prob_loss       = prob_loss,
            prob_target     = prob_target,
            capital         = self.capital,
            confidence      = self.confidence,
            horizon_days    = self.horizon_days,
            n_simulations   = self.n_simulations,
            target_return   = self.target_return,
        )

    # ------------------------------------------------------------------ #
    # MÉTODOS AUXILIARES                                                   #
    # ------------------------------------------------------------------ #

    def _compute_risk_metrics(
        self,
        final_returns: np.ndarray,
    ) -> tuple[float, float, float]:
        """
        Calcula VaR histórico, CVaR histórico y VaR paramétrico.

        Parametros
        ----------
        final_returns : np.ndarray
            Vector de rentabilidades acumuladas finales de todos los
            escenarios simulados (shape: n_simulations,).

        Returns
        -------
        tuple[float, float, float]
            (var_historical, cvar_historical, var_parametric).
            Todos expresados como fracción del capital (valores negativos
            indican pérdida).

        Notas
        -----
        **VaR histórico**: percentil alpha de la distribución de
        rentabilidades finales. Por ejemplo, VaR(95%) es el percentil
        5 — en el 5% de los peores escenarios, la pérdida supera este
        valor.

            VaR_hist = percentile(returns, (1 - confidence) * 100)

        **CVaR / Expected Shortfall**: media de los rendimientos que
        quedan por debajo del VaR (cola izquierda). Captura la severidad
        de los peores escenarios, no solo su umbral.

            CVaR = E[returns | returns < VaR]

        **VaR paramétrico**: asume distribución normal de rendimientos.
        Útil como referencia cuando los datos son escasos, pero puede
        subestimar el riesgo en distribuciones con colas pesadas
        (leptocúrticas), habituales en series financieras.

            VaR_param = mu - z_{alpha} * sigma
            donde z_{alpha} = percentil alpha de la N(0,1)
        """
        alpha = 1.0 - self.confidence

        # VaR histórico — percentil alpha de la distribución empírica
        var_hist = float(np.percentile(final_returns, alpha * 100))

        # CVaR (Expected Shortfall) — media de la cola izquierda
        tail_mask = final_returns <= var_hist
        cvar_hist = float(np.mean(final_returns[tail_mask]))

        # VaR paramétrico bajo supuesto de normalidad
        mu_ret    = float(np.mean(final_returns))
        sigma_ret = float(np.std(final_returns, ddof=1))
        z_alpha   = float(stats.norm.ppf(alpha))  # cuantil normal
        var_param = mu_ret + z_alpha * sigma_ret   # negativo = pérdida

        return var_hist, cvar_hist, var_param


# ---------------------------------------------------------------------------
# Funciones de utilidad a nivel de módulo
# ---------------------------------------------------------------------------

def sensitivity_analysis(
    weights:       np.ndarray,
    returns:       pd.DataFrame,
    capital:       float       = 10_000.0,
    n_simulations: int         = 5_000,
    horizons:      list[int]   = None,
    confidences:   list[float] = None,
    seed:          int         = 42,
) -> pd.DataFrame:
    """
    Análisis de sensibilidad del VaR ante distintos horizontes y confianzas.

    Ejecuta múltiples simulaciones variando el horizonte temporal y el
    nivel de confianza, para mostrar cómo evoluciona el VaR en distintos
    escenarios. Útil para el panel de análisis de riesgo del dashboard.

    Parametros
    ----------
    weights : np.ndarray
        Pesos de la cartera a analizar.
    returns : pd.DataFrame
        Rendimientos históricos (Hito 3).
    capital : float, optional
        Capital inicial. Por defecto 10.000 €.
    n_simulations : int, optional
        Número de simulaciones por combinación. Por defecto 5.000.
    horizons : list[int], optional
        Horizontes en días hábiles. Por defecto [21, 63, 126, 252].
        (≈ 1 mes, 3 meses, 6 meses, 1 año).
    confidences : list[float], optional
        Niveles de confianza. Por defecto [0.90, 0.95, 0.99].
    seed : int, optional
        Semilla de reproducibilidad. Por defecto 42.

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas 'Horizonte', 'Confianza', 'VaR (%)',
        'CVaR (%)' y 'Prob. pérdida (%)'.
        Una fila por cada combinación horizonte × confianza.

    Ejemplos
    --------
    >>> df_sens = sensitivity_analysis(msr.weights.values, returns)
    >>> print(df_sens.to_string())
    """
    if horizons is None:
        horizons = [21, 63, 126, 252]    # 1m, 3m, 6m, 1a
    if confidences is None:
        confidences = [0.90, 0.95, 0.99]

    records = []
    logger.info(
        "Análisis de sensibilidad: %d horizontes × %d confianzas = %d runs",
        len(horizons), len(confidences), len(horizons) * len(confidences),
    )

    for horizon in horizons:
        for conf in confidences:
            mc = MonteCarlo(
                weights       = weights,
                returns       = returns,
                capital       = capital,
                n_simulations = n_simulations,
                horizon_days  = horizon,
                confidence    = conf,
                seed          = seed,
            )
            res = mc.run()
            records.append({
                "Horizonte (días)":   horizon,
                "Horizonte":          f"{horizon}d (~{horizon//21}m)",
                "Confianza (%)":      int(conf * 100),
                "VaR (%)":            round(res.var_historical * 100, 2),
                "CVaR (%)":           round(res.cvar_historical * 100, 2),
                "Prob. pérdida (%)":  round(res.prob_loss * 100, 1),
                "Rentabilidad P50 (%)": round(res.percentiles["P50"] * 100, 2),
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Bloque de demostración
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Demostración del pipeline del Hito 5.
    Usa datos sintéticos para no requerir conexión a la API.
    """
    print("=" * 65)
    print("  PROYECTO ATHENA — Hito 5: Simulación Monte Carlo")
    print("=" * 65)

    # ── Datos sintéticos reproducibles ──────────────────────────────
    np.random.seed(42)
    N_DAYS  = 1260
    TICKERS = ["SAN.MC", "BBVA.MC", "TEF.MC", "ITX.MC",
               "IBE.MC", "REP.MC", "ACS.MC", "FER.MC"]
    mu_d    = np.array([0.00048, 0.00060, 0.00027, 0.00089,
                        0.00037, 0.00042, 0.00035, 0.00045])
    cov_d   = np.eye(len(TICKERS)) * 0.0003 + 0.00015
    ret_arr = np.random.multivariate_normal(mu_d, cov_d, N_DAYS)
    dates   = pd.bdate_range("2019-01-02", periods=N_DAYS)
    returns = pd.DataFrame(ret_arr, index=dates, columns=TICKERS)

    # Pesos de la cartera MSR (simulados)
    weights_msr = np.array([0.05, 0.10, 0.15, 0.20, 0.15, 0.15, 0.10, 0.10])

    # ── Simulación principal ─────────────────────────────────────────
    print("\n[1/2] Ejecutando simulación Monte Carlo (10.000 escenarios)...")
    mc = MonteCarlo(
        weights       = weights_msr,
        returns       = returns,
        capital       = 10_000.0,
        n_simulations = 10_000,
        horizon_days  = 252,
        confidence    = 0.95,
        target_return = 0.0,
        seed          = 42,
    )
    result = mc.run()
    print(result.summary())

    print("\n  Tabla de métricas para el dashboard:")
    print(result.metrics_dataframe().to_string(index=False))

    # ── Análisis de sensibilidad ─────────────────────────────────────
    print("\n[2/2] Análisis de sensibilidad (horizonte × confianza)...")
    df_sens = sensitivity_analysis(
        weights       = weights_msr,
        returns       = returns,
        capital       = 10_000.0,
        n_simulations = 2_000,
        seed          = 42,
    )
    print("\n  Tabla de sensibilidad del VaR:")
    print(df_sens[["Horizonte", "Confianza (%)", "VaR (%)", "CVaR (%)",
                   "Prob. pérdida (%)"]].to_string(index=False))

    print("\n" + "=" * 65)
    print("  Hito 5 completado.")
    print("  -> result   : MonteCarloResult con trayectorias y métricas")
    print("  -> df_sens  : análisis de sensibilidad VaR")
    print("  -> Listos para integración en el Dashboard (Hito 6)")
    print("=" * 65)