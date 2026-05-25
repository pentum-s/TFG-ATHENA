"""
=============================================================================
PROYECTO ATHENA — Hito 4: Módulo de Optimización de Carteras (Markowitz)
=============================================================================
Grado en Ciencia de Datos Aplicada — Universitat Oberta de Catalunya (UOC)

Autor   : Álvaro Sánchez Mendoza
Tutor   : Carlos Gaitán Poyatos
Versión : 1.0.0

Descripción
-----------
Este módulo implementa el núcleo analítico de optimización cuantitativa
del Proyecto Athena. Recibe los rendimientos logarítmicos producidos por
el módulo de adquisición de datos (Hito 3) y aplica la Teoría Moderna de
Carteras de Markowitz (1952) para calcular:

    1. El vector de rendimientos esperados (mu) y la matriz de
       covarianzas (Sigma) a partir de los rendimientos históricos.
    2. La Frontera Eficiente: conjunto de carteras que maximizan el
       rendimiento esperado para cada nivel de riesgo dado.
    3. La cartera de Mínima Varianza Global (MVP): menor riesgo posible.
    4. La cartera de Máximo Ratio de Sharpe (MSR): mejor relación
       rendimiento/riesgo para una tasa libre de riesgo dada.
    5. Un conjunto de carteras eficientes parametrizadas por nivel
       objetivo de rendimiento, para trazar la frontera completa.

Alcance del Hito 4
------------------
    Completado:
        - Estimación de parámetros (mu, Sigma) anualizados
        - Optimización con scipy.optimize.minimize (SLSQP)
        - Cartera de Mínima Varianza Global (MVP)
        - Cartera de Máximo Ratio de Sharpe (MSR)
        - Frontera Eficiente (N puntos parametrizados)
        - Restricciones de pesos: suma=1, pesos en [w_min, w_max]
        - Métricas de cartera: rendimiento, volatilidad, Sharpe
    Pendiente (Hito 5):
        - Simulación de Monte Carlo sobre la cartera óptima
        - Cálculo de VaR y CVaR

Dependencias
------------
    pip install numpy pandas scipy

Uso rápido
----------
    from data_acquisition import DataAcquisition
    from optimization import PortfolioOptimizer

    # Obtener rendimientos del Hito 3
    da      = DataAcquisition(tickers=[...], start="2019-01-01")
    prices  = da.download()
    returns = da.compute_returns(prices)

    # Optimizar cartera
    opt = PortfolioOptimizer(returns, risk_free_rate=0.035)
    mvp = opt.min_variance()
    msr = opt.max_sharpe()
    frontier = opt.efficient_frontier(n_points=100)
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
from scipy.optimize import minimize, OptimizeResult

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("athena.optimization")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
TRADING_DAYS_YEAR: int = 252   # días hábiles anuales estándar


# ---------------------------------------------------------------------------
# Dataclass de resultado de cartera
# ---------------------------------------------------------------------------

@dataclass
class PortfolioResult:
    """
    Contenedor de los resultados de una cartera optimizada.

    Agrupa los pesos óptimos junto con las métricas financieras clave
    de la cartera resultante, facilitando su uso en el dashboard y
    en el módulo de simulación Monte Carlo (Hito 5).

    Attributes
    ----------
    weights : pd.Series
        Pesos óptimos asignados a cada activo.
        Índice: tickers. Valores: fracción del capital [0, 1].
        La suma de los pesos es siempre igual a 1.
    expected_return : float
        Rendimiento anual esperado de la cartera (fracción, no porcentaje).
        Por ejemplo: 0.148 representa un 14,8 % anual.
    volatility : float
        Volatilidad anual (desviación estándar) de la cartera.
        Por ejemplo: 0.123 representa un 12,3 % anual.
    sharpe_ratio : float
        Ratio de Sharpe de la cartera, calculado como:
            (rendimiento_esperado - risk_free_rate) / volatilidad
    label : str
        Etiqueta descriptiva del tipo de cartera optimizada.
        Ejemplos: "Mínima Varianza", "Máximo Sharpe".
    """
    weights:         pd.Series
    expected_return: float
    volatility:      float
    sharpe_ratio:    float
    label:           str = "Cartera óptima"

    def to_dict(self) -> dict:
        """
        Serializa el resultado a un diccionario para uso en el dashboard.

        Returns
        -------
        dict
            Claves: 'label', 'weights', 'expected_return',
                    'volatility', 'sharpe_ratio'.
        """
        return {
            "label":           self.label,
            "weights":         self.weights.to_dict(),
            "expected_return": round(self.expected_return * 100, 4),
            "volatility":      round(self.volatility * 100, 4),
            "sharpe_ratio":    round(self.sharpe_ratio, 4),
        }

    def summary(self) -> str:
        """
        Devuelve un resumen legible de la cartera.

        Returns
        -------
        str
            Texto con las métricas clave de la cartera.
        """
        lines = [
            f"  {'─' * 45}",
            f"  Cartera      : {self.label}",
            f"  Rendimiento  : {self.expected_return * 100:.2f} %",
            f"  Volatilidad  : {self.volatility * 100:.2f} %",
            f"  Sharpe       : {self.sharpe_ratio:.4f}",
            f"  {'─' * 45}",
            f"  Pesos:",
        ]
        for ticker, w in self.weights.items():
            lines.append(f"    {ticker:<14}: {w * 100:.2f} %")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Clase principal: PortfolioOptimizer
# ---------------------------------------------------------------------------

class PortfolioOptimizer:
    """
    Optimizador de carteras basado en la Teoría Moderna de Carteras (Markowitz).

    Implementa la optimización cuadrática de carteras mediante
    ``scipy.optimize.minimize`` con el método SLSQP (Sequential Least
    Squares Programming), que maneja correctamente restricciones de
    igualdad (suma de pesos = 1) y de desigualdad (pesos en [w_min, w_max]).

    El optimizador calcula:
        - Cartera de Mínima Varianza Global (MVP)
        - Cartera de Máximo Ratio de Sharpe (MSR)
        - Frontera Eficiente completa (N puntos)

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame de rendimientos logarítmicos diarios, tal como
        devuelve ``DataAcquisition.compute_returns()``.
        Índice: DatetimeIndex. Columnas: tickers.
    risk_free_rate : float, optional
        Tasa libre de riesgo anual (en fracción, no porcentaje).
        Por ejemplo: 0.035 para el 3,5 % anual.
        Se usa para calcular el Ratio de Sharpe. Por defecto 0.0.
    w_min : float, optional
        Peso mínimo por activo. Por defecto 0.0 (sin ventas en corto).
    w_max : float, optional
        Peso máximo por activo. Por defecto 1.0 (sin límite superior).

    Attributes
    ----------
    tickers : list[str]
        Lista de tickers del universo de activos.
    n_assets : int
        Número de activos en el universo.
    mu : np.ndarray
        Vector de rendimientos esperados anualizados (shape: n_assets,).
    sigma : np.ndarray
        Matriz de covarianzas anualizadas (shape: n_assets × n_assets).
    risk_free_rate : float
        Tasa libre de riesgo configurada.

    Examples
    --------
    >>> opt = PortfolioOptimizer(returns, risk_free_rate=0.035)
    >>> mvp = opt.min_variance()
    >>> msr = opt.max_sharpe()
    >>> print(msr.summary())
    """

    def __init__(
        self,
        returns:        pd.DataFrame,
        risk_free_rate: float = 0.0,
        w_min:          float = 0.0,
        w_max:          float = 1.0,
    ) -> None:

        if returns.empty:
            raise ValueError("El DataFrame de rendimientos está vacío.")
        if not (0.0 <= w_min < w_max <= 1.0):
            raise ValueError(
                f"Restricciones de pesos inválidas: "
                f"w_min={w_min}, w_max={w_max}. "
                f"Se requiere 0 ≤ w_min < w_max ≤ 1."
            )

        self.tickers        = list(returns.columns)
        self.n_assets       = len(self.tickers)
        self.risk_free_rate = risk_free_rate
        self.w_min          = w_min
        self.w_max          = w_max

        # --- Estimación de parámetros (anualizados) ---
        # mu: vector de rendimientos esperados anualizados
        # Usamos la media histórica como estimador puntual.
        # Formula: mu_i = mean(r_i) * 252
        self.mu = returns.mean().values * TRADING_DAYS_YEAR

        # sigma: matriz de covarianzas anualizadas
        # Formula: Sigma = Cov(r) * 252
        self.sigma = returns.cov().values * TRADING_DAYS_YEAR

        logger.info(
            "PortfolioOptimizer inicializado — %d activos | "
            "rf=%.2f%% | w=[%.0f%%, %.0f%%]",
            self.n_assets,
            risk_free_rate * 100,
            w_min * 100,
            w_max * 100,
        )
        logger.info(
            "Rendimientos esperados anualizados (mu): "
            "min=%.2f%%, max=%.2f%%, media=%.2f%%",
            self.mu.min() * 100,
            self.mu.max() * 100,
            self.mu.mean() * 100,
        )

    # ------------------------------------------------------------------ #
    # MÉTODOS PÚBLICOS DE OPTIMIZACIÓN                                     #
    # ------------------------------------------------------------------ #

    def min_variance(self) -> PortfolioResult:
        """
        Calcula la cartera de Mínima Varianza Global (MVP).

        La cartera MVP es el punto más a la izquierda de la frontera
        eficiente: minimiza la varianza de la cartera sin imponer
        ningún objetivo de rendimiento.

        Problema de optimización:
            min  w^T · Sigma · w
            s.t. sum(w) = 1
                 w_min ≤ w_i ≤ w_max  para todo i

        Returns
        -------
        PortfolioResult
            Resultado con pesos óptimos y métricas de la MVP.

        Notes
        -----
        La MVP es el punto de partida de la frontera eficiente.
        Todas las carteras eficientes tienen mayor rendimiento
        esperado que la MVP pero también mayor volatilidad.

        Examples
        --------
        >>> mvp = opt.min_variance()
        >>> print(f"Volatilidad MVP: {mvp.volatility * 100:.2f}%")
        """
        logger.info("Optimizando: Cartera de Mínima Varianza Global (MVP)...")

        # Función objetivo: varianza de la cartera
        # w^T · Sigma · w — forma cuadrática
        def portfolio_variance(w: np.ndarray) -> float:
            return float(w @ self.sigma @ w)

        result = self._optimize(
            objective   = portfolio_variance,
            label       = "Mínima Varianza",
        )
        return result

    def max_sharpe(self) -> PortfolioResult:
        """
        Calcula la cartera de Máximo Ratio de Sharpe (MSR).

        La cartera MSR maximiza la pendiente de la Línea del Mercado
        de Capitales (CML), representando la mejor relación entre
        rendimiento excedente sobre la tasa libre de riesgo y riesgo.

        Problema de optimización (equivalente a maximizar Sharpe):
            min  -( w^T·mu - rf ) / sqrt(w^T·Sigma·w)
            s.t. sum(w) = 1
                 w_min ≤ w_i ≤ w_max  para todo i

        Returns
        -------
        PortfolioResult
            Resultado con pesos óptimos y métricas de la MSR.

        Notes
        -----
        Se minimiza el negativo del ratio de Sharpe porque scipy solo
        implementa minimización. La protección ante volatilidad cero
        (denominador) evita divisiones por cero en carteras degeneradas.

        El ratio de Sharpe de la MSR es el máximo alcanzable dentro
        del universo de activos dado y las restricciones de pesos.

        Examples
        --------
        >>> msr = opt.max_sharpe()
        >>> print(f"Sharpe MSR: {msr.sharpe_ratio:.4f}")
        """
        logger.info("Optimizando: Cartera de Máximo Ratio de Sharpe (MSR)...")

        rf = self.risk_free_rate  # capturar en scope local

        # Función objetivo: negativo del ratio de Sharpe
        def neg_sharpe(w: np.ndarray) -> float:
            ret  = float(w @ self.mu)
            vol  = float(np.sqrt(w @ self.sigma @ w))
            if vol < 1e-10:
                # Evitar división por cero en carteras de volatilidad nula
                return 0.0
            return -(ret - rf) / vol

        result = self._optimize(
            objective   = neg_sharpe,
            label       = "Máximo Sharpe",
        )
        return result

    def efficient_frontier(
        self,
        n_points: int = 100,
    ) -> pd.DataFrame:
        """
        Calcula la Frontera Eficiente con N carteras parametrizadas.

        Resuelve N problemas de optimización, cada uno con un objetivo
        de rendimiento distinto entre el mínimo (MVP) y el máximo
        alcanzable, generando los puntos que trazan la frontera eficiente
        en el espacio riesgo-rendimiento.

        Parameters
        ----------
        n_points : int, optional
            Número de carteras a calcular en la frontera. Por defecto 100.
            Valores típicos: 50–200. Más puntos = frontera más suave
            pero mayor tiempo de cómputo.

        Returns
        -------
        pd.DataFrame
            DataFrame con ``n_points`` filas y las columnas:
                - ``volatility``       : volatilidad anual de cada cartera.
                - ``expected_return``  : rendimiento esperado anual.
                - ``sharpe_ratio``     : ratio de Sharpe.
                - ``w_{ticker}``       : peso de cada activo (una columna
                                        por ticker).
            Ordenado de menor a mayor volatilidad.

        Notes
        -----
        Se usa la parametrización por rendimiento objetivo (target return).
        Para cada nivel objetivo r* se resuelve:
            min  w^T · Sigma · w
            s.t. w^T · mu = r*
                 sum(w) = 1
                 w_min ≤ w_i ≤ w_max

        El rango de rendimientos se determina automáticamente:
            - Límite inferior: rendimiento de la MVP.
            - Límite superior: 95% del rendimiento máximo individual,
              para evitar concentraciones extremas en un solo activo.

        Examples
        --------
        >>> frontier = opt.efficient_frontier(n_points=100)
        >>> print(frontier.head(3))
           volatility  expected_return  sharpe_ratio
        0       0.103            0.082        0.795
        1       0.105            0.091        0.867
        2       0.108            0.100        0.928
        """
        logger.info(
            "Calculando Frontera Eficiente con %d puntos...", n_points
        )

        # Obtener MVP para fijar el límite inferior de rendimientos
        mvp = self.min_variance()
        r_min = mvp.expected_return
        r_max = self.mu.max() * 0.95   # evitar concentración en un activo

        if r_min >= r_max:
            logger.warning(
                "Rango de rendimientos inválido (r_min=%.4f >= r_max=%.4f). "
                "Ajustando automáticamente.", r_min, r_max
            )
            r_max = r_min * 1.5

        # Niveles de rendimiento objetivo equiespaciados
        target_returns = np.linspace(r_min, r_max, n_points)

        records = []

        for i, target_r in enumerate(target_returns):

            # Función objetivo: minimizar varianza
            def portfolio_variance(w: np.ndarray) -> float:
                return float(w @ self.sigma @ w)

            # Restricción adicional: rendimiento objetivo fijo
            def return_constraint(w: np.ndarray) -> float:
                return float(w @ self.mu) - target_r

            # Resolver con restricción de rendimiento
            result = self._optimize(
                objective   = portfolio_variance,
                label       = f"Frontera r*={target_r:.4f}",
                extra_constraints = [
                    {"type": "eq", "fun": return_constraint}
                ],
                log_result = False,
            )

            if result is not None:
                row = {
                    "volatility":      result.volatility,
                    "expected_return": result.expected_return,
                    "sharpe_ratio":    result.sharpe_ratio,
                }
                # Añadir pesos como columnas individuales
                for ticker, w in result.weights.items():
                    row[f"w_{ticker}"] = w
                records.append(row)

        if not records:
            raise RuntimeError(
                "No se pudo calcular ningún punto de la Frontera Eficiente. "
                "Comprueba los datos de entrada y las restricciones de pesos."
            )

        frontier_df = (
            pd.DataFrame(records)
            .sort_values("volatility")
            .reset_index(drop=True)
        )

        logger.info(
            "Frontera Eficiente calculada — %d/%d puntos válidos",
            len(frontier_df), n_points,
        )
        return frontier_df

    def equal_weight(self) -> "PortfolioResult":
        """
        Devuelve la cartera equi-ponderada (1/N) como benchmark de referencia.

        La estrategia 1/N asigna el mismo peso a cada activo del universo.
        Sirve como benchmark de referencia para evaluar si la optimización
        añade valor respecto a la diversificación ingenua (DeMiguel et al., 2009).

        Returns
        -------
        PortfolioResult
            Cartera con pesos iguales para todos los activos.
        """
        w = np.ones(self.n_assets) / self.n_assets
        ret = float(w @ self.mu)
        vol = float(np.sqrt(w @ self.sigma @ w))
        sr  = (ret - self.risk_free_rate) / vol if vol > 1e-10 else 0.0
        return PortfolioResult(
            weights         = pd.Series(w, index=self.tickers),
            expected_return = ret,
            volatility      = vol,
            sharpe_ratio    = sr,
            label           = "Equi-ponderada (1/N)",
        )

    def portfolio_metrics(self, weights: np.ndarray) -> dict:
        """
        Calcula las métricas financieras de una cartera dada por sus pesos.

        Método de conveniencia para evaluar cualquier vector de pesos,
        no necesariamente el resultado de una optimización. Utilizado
        en el dashboard para mostrar métricas de carteras personalizadas.

        Parameters
        ----------
        weights : np.ndarray
            Vector de pesos de la cartera (shape: n_assets,).
            Debe sumar 1.0 (se normaliza automáticamente si no).

        Returns
        -------
        dict
            Diccionario con claves:
                - ``expected_return`` : rendimiento anual esperado (%).
                - ``volatility``      : volatilidad anual (%).
                - ``sharpe_ratio``    : ratio de Sharpe.

        Examples
        --------
        >>> w_equal = np.ones(opt.n_assets) / opt.n_assets
        >>> metrics = opt.portfolio_metrics(w_equal)
        >>> print(metrics)
        {'expected_return': 11.23, 'volatility': 16.45, 'sharpe_ratio': 0.682}
        """
        w = np.asarray(weights, dtype=float)

        # Normalizar si la suma no es exactamente 1
        total = w.sum()
        if not np.isclose(total, 1.0, atol=1e-6):
            logger.warning("Pesos no suman 1 (suma=%.6f). Normalizando.", total)
            w = w / total

        ret = float(w @ self.mu)
        vol = float(np.sqrt(w @ self.sigma @ w))
        sr  = (ret - self.risk_free_rate) / vol if vol > 1e-10 else 0.0

        return {
            "expected_return": round(ret * 100, 4),
            "volatility":      round(vol * 100, 4),
            "sharpe_ratio":    round(sr, 4),
        }

    # ------------------------------------------------------------------ #
    # MÉTODOS PRIVADOS                                                     #
    # ------------------------------------------------------------------ #

    def _optimize(
        self,
        objective:         callable,
        label:             str,
        extra_constraints: list  = None,
        log_result:        bool  = True,
    ) -> Optional[PortfolioResult]:
        """
        Ejecuta la optimización numérica con scipy.optimize.minimize.

        Envuelve la llamada a scipy con las restricciones estándar
        (suma de pesos = 1, bounds por activo) más cualquier restricción
        adicional específica de cada problema.

        Parameters
        ----------
        objective : callable
            Función objetivo a minimizar. Recibe un array de pesos (w)
            y devuelve un escalar float.
        label : str
            Etiqueta descriptiva para el log y el PortfolioResult.
        extra_constraints : list, optional
            Lista de diccionarios de restricciones adicionales en formato
            scipy (claves 'type' y 'fun'). Por defecto None.
        log_result : bool, optional
            Si True, registra las métricas del resultado en el log.

        Returns
        -------
        PortfolioResult or None
            Resultado de la optimización, o None si no converge.

        Notes
        -----
        Se usa el método SLSQP (Sequential Least Squares Programming)
        porque es el único método de scipy que soporta simultáneamente:
            - Restricciones de igualdad (sum(w) = 1)
            - Restricciones de desigualdad (w_min ≤ w_i ≤ w_max)
            - Función objetivo diferenciable (varianza cuadrática)

        Se prueban múltiples puntos de inicio para aumentar la robustez
        frente a mínimos locales.
        """
        # Restricción estándar: suma de pesos = 1
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        if extra_constraints:
            constraints.extend(extra_constraints)

        # Bounds: cada peso en [w_min, w_max]
        bounds = [(self.w_min, self.w_max)] * self.n_assets

        # Intentar con múltiples puntos de inicio para robustez
        # 1) Cartera equi-ponderada
        # 2) Punto aleatorio normalizado (semilla fija para reproducibilidad)
        w0_candidates = [
            np.ones(self.n_assets) / self.n_assets,
        ]
        rng = np.random.default_rng(seed=42)
        for _ in range(3):
            w_rand = rng.dirichlet(np.ones(self.n_assets))
            w_rand = np.clip(w_rand, self.w_min, self.w_max)
            w_rand /= w_rand.sum()
            w0_candidates.append(w_rand)

        best_result: Optional[OptimizeResult] = None

        for w0 in w0_candidates:
            res = minimize(
                fun         = objective,
                x0          = w0,
                method      = "SLSQP",   # Sequential Least Squares
                bounds      = bounds,
                constraints = constraints,
                options     = {
                    "ftol":   1e-12,   # tolerancia de convergencia
                    "maxiter": 1000,   # máximo de iteraciones
                    "disp":   False,
                },
            )
            if res.success:
                if best_result is None or res.fun < best_result.fun:
                    best_result = res

        if best_result is None:
            logger.warning(
                "La optimización '%s' no convergió. Usando equi-ponderada.",
                label,
            )
            # Devolver cartera equi-ponderada como fallback
            w_fallback = np.ones(self.n_assets) / self.n_assets
            best_result = type("Result", (), {"x": w_fallback})()

        # Construir resultado
        w_opt   = np.clip(best_result.x, 0.0, 1.0)
        w_opt  /= w_opt.sum()   # re-normalizar para garantizar suma=1

        ret = float(w_opt @ self.mu)
        vol = float(np.sqrt(w_opt @ self.sigma @ w_opt))
        sr  = (ret - self.risk_free_rate) / vol if vol > 1e-10 else 0.0

        portfolio = PortfolioResult(
            weights         = pd.Series(w_opt, index=self.tickers),
            expected_return = ret,
            volatility      = vol,
            sharpe_ratio    = sr,
            label           = label,
        )

        if log_result:
            logger.info(
                "Resultado '%s' — Ret: %.2f%% | Vol: %.2f%% | Sharpe: %.4f",
                label, ret * 100, vol * 100, sr,
            )

        return portfolio


# ---------------------------------------------------------------------------
# Funciones de utilidad a nivel de módulo
# ---------------------------------------------------------------------------

def portfolio_from_weights(
    weights:        dict,
    returns:        pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> PortfolioResult:
    """
    Construye un PortfolioResult a partir de pesos definidos manualmente.

    Útil para evaluar carteras de referencia (equi-ponderada, benchmark)
    o para reconstruir carteras guardadas desde el dashboard.

    Parameters
    ----------
    weights : dict
        Diccionario {ticker: peso}. Los pesos se normalizan automáticamente.
    returns : pd.DataFrame
        DataFrame de rendimientos para estimar mu y Sigma.
    risk_free_rate : float, optional
        Tasa libre de riesgo anual. Por defecto 0.0.

    Returns
    -------
    PortfolioResult
        Resultado con pesos y métricas calculadas.

    Raises
    ------
    ValueError
        Si algún ticker de weights no está en returns.

    Examples
    --------
    >>> pf = portfolio_from_weights(
    ...     {"SAN.MC": 0.25, "BBVA.MC": 0.25, "TEF.MC": 0.25, "ITX.MC": 0.25},
    ...     returns,
    ...     risk_free_rate=0.035,
    ... )
    >>> print(pf.summary())
    """
    missing = [t for t in weights if t not in returns.columns]
    if missing:
        raise ValueError(
            f"Tickers no encontrados en returns: {missing}. "
            f"Disponibles: {list(returns.columns)}"
        )

    # Crear optimizador para acceder a mu y sigma
    opt = PortfolioOptimizer(returns, risk_free_rate=risk_free_rate)

    # Construir y normalizar vector de pesos en el orden correcto
    w = np.array([weights.get(t, 0.0) for t in opt.tickers], dtype=float)
    w = np.clip(w, 0.0, 1.0)
    w /= w.sum()

    ret = float(w @ opt.mu)
    vol = float(np.sqrt(w @ opt.sigma @ w))
    sr  = (ret - risk_free_rate) / vol if vol > 1e-10 else 0.0

    return PortfolioResult(
        weights         = pd.Series(w, index=opt.tickers),
        expected_return = ret,
        volatility      = vol,
        sharpe_ratio    = sr,
        label           = "Cartera personalizada",
    )


# ---------------------------------------------------------------------------
# Bloque de demostración
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Demostración del pipeline completo del Hito 4.

    Genera datos sintéticos para ilustrar el funcionamiento del
    optimizador sin necesidad de conexión a la API de Yahoo Finance.

    Uso:
        python optimization.py
    """
    print("=" * 65)
    print("  PROYECTO ATHENA — Hito 4: Optimización de Carteras")
    print("=" * 65)

    # ── Datos sintéticos reproducibles (sin API) ──────────────────────
    np.random.seed(42)
    N_DAYS   = 1260   # ~5 años de días hábiles
    TICKERS  = ["SAN.MC", "BBVA.MC", "TEF.MC", "ITX.MC",
                "IBE.MC", "REP.MC", "ACS.MC", "FER.MC"]

    # Simular rendimientos con correlaciones realistas
    mu_daily  = np.array([0.00048, 0.00060, 0.00027, 0.00089,
                           0.00037, 0.00042, 0.00035, 0.00045])
    cov_daily = np.eye(len(TICKERS)) * 0.0003 + 0.00015
    returns_arr = np.random.multivariate_normal(
        mean=mu_daily, cov=cov_daily, size=N_DAYS
    )
    dates   = pd.bdate_range("2019-01-02", periods=N_DAYS)
    returns = pd.DataFrame(returns_arr, index=dates, columns=TICKERS)

    print(f"\n  Activos    : {TICKERS}")
    print(f"  Observac.  : {len(returns)} días")

    # ── Inicializar optimizador ───────────────────────────────────────
    print("\n[1/4] Inicializando PortfolioOptimizer...")
    opt = PortfolioOptimizer(
        returns        = returns,
        risk_free_rate = 0.035,   # tasa libre de riesgo: 3.5%
        w_min          = 0.0,     # sin ventas en corto
        w_max          = 0.40,    # máximo 40% por activo
    )

    # ── Cartera de Mínima Varianza ────────────────────────────────────
    print("\n[2/4] Calculando Cartera de Mínima Varianza (MVP)...")
    mvp = opt.min_variance()
    print(mvp.summary())

    # ── Cartera de Máximo Sharpe ──────────────────────────────────────
    print("\n[3/4] Calculando Cartera de Máximo Sharpe (MSR)...")
    msr = opt.max_sharpe()
    print(msr.summary())

    # ── Frontera Eficiente ────────────────────────────────────────────
    print("\n[4/4] Calculando Frontera Eficiente (100 puntos)...")
    frontier = opt.efficient_frontier(n_points=100)
    print(f"\n  Frontera calculada — {len(frontier)} puntos válidos")
    print(f"  Rango volatilidad : "
          f"{frontier['volatility'].min()*100:.2f}% — "
          f"{frontier['volatility'].max()*100:.2f}%")
    print(f"  Rango rendimiento : "
          f"{frontier['expected_return'].min()*100:.2f}% — "
          f"{frontier['expected_return'].max()*100:.2f}%")
    print(f"\n  Primeras 5 filas de la Frontera Eficiente:")
    print(
        frontier[["volatility","expected_return","sharpe_ratio"]]
        .head(5)
        .applymap(lambda x: f"{x*100:.2f}%" if x < 10 else f"{x:.4f}")
        .to_string()
    )

    # ── Cartera equi-ponderada (referencia) ──────────────────────────
    print("\n  Cartera equi-ponderada (referencia):")
    w_eq = {t: 1/len(TICKERS) for t in TICKERS}
    eq_pf = portfolio_from_weights(w_eq, returns, risk_free_rate=0.035)
    print(eq_pf.summary())

    print("\n" + "=" * 65)
    print("  Hito 4 completado.")
    print("  → mvp      : cartera de mínima varianza")
    print("  → msr      : cartera de máximo Sharpe")
    print("  → frontier : DataFrame con la Frontera Eficiente")
    print("  → Listos para el módulo Monte Carlo (Hito 5)")
    print("=" * 65)