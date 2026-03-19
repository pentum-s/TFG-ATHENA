"""
=============================================================================
PROYECTO ATHENA — Hito 3: Módulo de Adquisición de Datos Financieros
=============================================================================
Grado en Ciencia de Datos Aplicada — Universitat Oberta de Catalunya (UOC)

Autor   : Álvaro Sánchez Mendoza
Tutor   : Carlos Gaitán Poyatos
Versión : 1.0.0

Descripción
-----------
Este módulo implementa el pipeline de adquisición de datos financieros
para el Proyecto Athena. Cubre las actividades completadas del Hito 3:

    1. Conexión con la API pública de Yahoo Finance mediante la librería
       yfinance y validación de la conectividad.
    2. Descarga de series históricas de precios de cierre ajustados para
       los activos del IBEX 35 y/o S&P 500, con gestión de errores y
       sistema de caché local para evitar descargas redundantes.
    3. Transformación de precios brutos a rendimientos logarítmicos
       diarios, que constituyen el input directo del módulo de
       optimización de Markowitz (Hito 4).

Alcance del Hito 3
------------------
    Completado:
        - Conexión con la API de Yahoo Finance (yfinance)
        - Descarga de series históricas de precios de cierre ajustados
        - Cálculo de rendimientos logarítmicos diarios
        - Sistema de caché local en CSV
        - Estadísticas descriptivas básicas (EDA)
    Pendiente (Hito siguiente):
        - Preprocesado avanzado: tratamiento de nulos, outliers,
          reindexado a días hábiles

Dependencias
------------
    pip install yfinance pandas numpy

Uso rápido
----------
    from data_acquisition import DataAcquisition

    da = DataAcquisition(
        tickers = ["SAN.MC", "BBVA.MC", "TEF.MC"],
        start   = "2019-01-01",
        end     = "2024-12-31",
    )
    prices  = da.download()
    returns = da.compute_returns(prices)
=============================================================================
"""

# ---------------------------------------------------------------------------
# Importaciones estándar de la librería de Python
# ---------------------------------------------------------------------------
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Importaciones de terceros
# ---------------------------------------------------------------------------
import numpy as np          # álgebra lineal y operaciones vectoriales
import pandas as pd         # manipulación de DataFrames y series temporales
import yfinance as yf       # wrapper de la API de Yahoo Finance

# Suprimir avisos de deprecación de dependencias externas irrelevantes
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuración del sistema de logging del módulo
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("athena.data_acquisition")


# ---------------------------------------------------------------------------
# Constantes del módulo
# ---------------------------------------------------------------------------

# Selección representativa de tickers del IBEX 35 en formato Yahoo Finance.
# El sufijo ".MC" corresponde a la bolsa española (Mercado Continuo).
IBEX35_TICKERS: list[str] = [
    "SAN.MC", "BBVA.MC", "TEF.MC", "ITX.MC", "IBE.MC",
    "REP.MC", "ACS.MC", "FER.MC", "CLNX.MC", "AENA.MC",
    "AMS.MC", "ACX.MC", "ELE.MC", "GRF.MC", "IAG.MC",
    "IDR.MC", "INM.MC", "MAP.MC", "MRL.MC", "NTGY.MC",
    "PHM.MC", "RED.MC", "ROVI.MC", "SAB.MC", "SGRE.MC",
    "SLR.MC", "VIS.MC", "BKT.MC", "MEL.MC", "MTS.MC",
]

# Selección representativa de tickers del S&P 500.
# Sin sufijo de bolsa (mercado estadounidense por defecto en Yahoo Finance).
SP500_TICKERS: list[str] = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",
    "NVDA", "TSLA", "BRK-B", "UNH", "JNJ",
    "JPM", "V", "PG", "XOM", "HD",
    "MA", "CVX", "ABBV", "MRK", "KO",
    "PEP", "LLY", "AVGO", "TMO", "COST",
    "MCD", "DHR", "WMT", "ACN", "NKE",
]

# Días hábiles anuales estándar en finanzas cuantitativas
TRADING_DAYS_YEAR: int = 252


# ---------------------------------------------------------------------------
# Clase principal del módulo
# ---------------------------------------------------------------------------

class DataAcquisition:
    """
    Pipeline de adquisición de datos financieros desde Yahoo Finance.

    Gestiona la conexión con la API pública de Yahoo Finance, la descarga
    de series históricas de precios de cierre ajustados y su
    transformación a rendimientos logarítmicos diarios.

    Los datos descargados pueden persistirse en un directorio de caché
    local en formato CSV, evitando llamadas redundantes a la API en
    ejecuciones posteriores dentro del mismo rango de fechas.

    Parameters
    ----------
    tickers : list[str]
        Lista de símbolos bursátiles en formato Yahoo Finance.
        Para el IBEX 35 usar sufijo ".MC" (ej. "SAN.MC").
        Para el S&P 500 sin sufijo (ej. "AAPL").
    start : str
        Fecha de inicio del período histórico, formato "YYYY-MM-DD".
    end : str, optional
        Fecha de fin del período histórico, formato "YYYY-MM-DD".
        Si no se especifica, se usa la fecha actual del sistema.
    interval : str, optional
        Frecuencia temporal de los datos. Por defecto "1d" (diario).
        Valores posibles: "1d" (diario), "1wk" (semanal), "1mo" (mensual).
    cache_dir : str or Path, optional
        Ruta al directorio de caché local donde guardar los datos en CSV.
        Si es None, los datos no se persisten en disco.

    Attributes
    ----------
    tickers : list[str]
        Lista de tickers configurada (normalizada a mayúsculas).
    start : str
        Fecha de inicio configurada.
    end : str
        Fecha de fin configurada.
    interval : str
        Frecuencia temporal configurada.

    Examples
    --------
    >>> da = DataAcquisition(
    ...     tickers = ["SAN.MC", "BBVA.MC", "TEF.MC"],
    ...     start   = "2020-01-01",
    ...     end     = "2024-12-31",
    ... )
    >>> prices = da.download()
    >>> returns = da.compute_returns(prices)
    """

    def __init__(
        self,
        tickers:   list[str],
        start:     str = "2019-01-01",
        end:       Optional[str] = None,
        interval:  str = "1d",
        cache_dir: Optional[str | Path] = None,
    ) -> None:

        # Validación: la lista de tickers no puede estar vacía
        if not tickers:
            raise ValueError(
                "La lista de tickers no puede estar vacía. "
                "Proporciona al menos un símbolo bursátil válido."
            )

        # Normalizamos tickers: mayúsculas y sin espacios en blanco
        self.tickers  = [t.upper().strip() for t in tickers]
        self.start    = start
        self.end      = end or datetime.today().strftime("%Y-%m-%d")
        self.interval = interval

        # Configuramos y creamos directorio de caché si se especifica
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

        logger.info(
            "DataAcquisition inicializado — %d activos | %s → %s | "
            "frecuencia: %s | caché: %s",
            len(self.tickers),
            self.start,
            self.end,
            self.interval,
            str(self.cache_dir) if self.cache_dir else "desactivada",
        )

    # ------------------------------------------------------------------ #
    # MÉTODO 1: DESCARGA DE DATOS DESDE YAHOO FINANCE                      #
    # ------------------------------------------------------------------ #

    def download(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Descarga los precios de cierre ajustados desde Yahoo Finance.

        Si ``use_cache=True`` y existe un archivo CSV previo para el
        mismo rango de fechas en el directorio de caché, lo carga
        directamente sin conectar con la API. En caso contrario,
        realiza la llamada a la API y, si hay caché configurada,
        persiste el resultado en disco para futuras ejecuciones.

        Parameters
        ----------
        use_cache : bool, optional
            Si True, intenta cargar desde caché antes de descargar.
            Por defecto True.

        Returns
        -------
        pd.DataFrame
            DataFrame con:
                - Índice   : DatetimeIndex con fechas de cotización.
                - Columnas : un ticker por columna.
                - Valores  : precios de cierre ajustados por dividendos
                             y splits (Adjusted Close).
            Ejemplo de shape: (1509, 8) para 8 activos y ~6 años.

        Raises
        ------
        ConnectionError
            Si la conexión con Yahoo Finance falla completamente.
        ValueError
            Si ningún ticker devuelve datos válidos.

        Notes
        -----
        Se usa ``auto_adjust=True`` en yfinance, que devuelve
        directamente los precios ajustados (equivalente a "Adj Close")
        en la columna "Close". Esto garantiza la comparabilidad
        temporal de los precios, requisito indispensable del modelo
        de Markowitz, ya que elimina las discontinuidades artificiales
        causadas por splits de acciones y pagos de dividendos.

        La latencia de los datos de Yahoo Finance es inferior a
        15 minutos, lo que cumple el KPI de latencia del Hito 3.

        Examples
        --------
        >>> prices = da.download()
        >>> print(prices.shape)
        (1509, 8)
        >>> print(prices.tail(2))
                      SAN.MC  BBVA.MC  TEF.MC
        Date
        2024-12-30    3.82     9.41    3.68
        2024-12-31    3.79     9.38    3.65
        """
        # Construir ruta de caché para este dataset
        cache_path = self._build_cache_path("prices_raw")

        # Intentar cargar desde caché si está disponible y activada
        if use_cache and cache_path is not None and cache_path.exists():
            logger.info(
                "Cargando precios desde caché local: %s", cache_path.name
            )
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            df.index.name = "Date"
            logger.info(
                "Caché cargada — %d días × %d activos",
                len(df), df.shape[1],
            )
            return df

        # Conectar con la API de Yahoo Finance
        logger.info(
            "Conectando con Yahoo Finance — %d activos | %s → %s",
            len(self.tickers), self.start, self.end,
        )

        try:
            raw = yf.download(
                tickers     = self.tickers,
                start       = self.start,
                end         = self.end,
                interval    = self.interval,
                auto_adjust = True,   # precios ajustados automáticamente
                progress    = False,  # sin barra de progreso en terminal
                threads     = True,   # descarga paralela de múltiples tickers
            )
        except Exception as exc:
            raise ConnectionError(
                f"Error al conectar con Yahoo Finance: {exc}. "
                "Verifica tu conexión a internet e inténtalo de nuevo."
            ) from exc

        # Extraer precios de cierre del DataFrame de yfinance
        # Para múltiples tickers yfinance devuelve MultiIndex (campo, ticker)
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"].copy()
        else:
            # Un solo ticker: renombrar columna con el ticker
            prices = raw[["Close"]].copy()
            prices.columns = self.tickers[:1]

        # Validar que la descarga contiene datos
        if prices.empty:
            raise ValueError(
                "La descarga no devolvió datos. Verifica que los tickers "
                "son válidos en Yahoo Finance y que el rango de fechas "
                "no corresponde a un período sin cotizaciones."
            )

        # Normalizar índice: DatetimeIndex con nombre estándar
        prices.index = pd.to_datetime(prices.index)
        prices.index.name = "Date"

        # Registrar informe de calidad en el log
        self._log_download_report(prices)

        # Persistir en caché si está configurada
        if cache_path is not None:
            prices.to_csv(cache_path)
            logger.info("Datos guardados en caché: %s", cache_path.name)

        return prices

    # ------------------------------------------------------------------ #
    # MÉTODO 2: CÁLCULO DE RENDIMIENTOS LOGARÍTMICOS                       #
    # ------------------------------------------------------------------ #

    def compute_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula los rendimientos logarítmicos diarios a partir de precios.

        El rendimiento logarítmico en el período t se define como:

            r_t = ln(P_t / P_{t-1})

        donde P_t es el precio de cierre ajustado en la fecha t y
        P_{t-1} es el precio del día hábil anterior.

        Parameters
        ----------
        prices : pd.DataFrame
            DataFrame de precios de cierre ajustados, tal como
            devuelve el método ``download()``.

        Returns
        -------
        pd.DataFrame
            DataFrame de rendimientos logarítmicos diarios con:
                - Mismo número de columnas que ``prices``.
                - Una fila menos (la primera siempre es NaN, se elimina).
                - Índice DatetimeIndex idéntico al de ``prices`` sin la
                  primera fila.

        Notes
        -----
        Se elige el rendimiento logarítmico frente al aritmético
        (``pct_change()``) por tres razones fundamentales:

        1. **Aditividad temporal**: la suma de rendimientos logarítmicos
           diarios es igual al rendimiento logarítmico del período
           completo. Esto simplifica la anualización de estadísticas.

        2. **Simetría estadística**: una caída y una subida del mismo
           porcentaje producen rendimientos simétricos, lo que mejora
           el ajuste de modelos estadísticos.

        3. **Estándar en la literatura financiera**: el modelo de
           Markowitz y la mayoría de trabajos académicos sobre
           optimización de carteras utilizan rendimientos logarítmicos
           como input para la estimación de parámetros.

        Los rendimientos están expresados en escala diaria. La
        anualización (×252) se aplica en ``compute_annualized_stats()``.

        Examples
        --------
        >>> returns = da.compute_returns(prices)
        >>> print(returns.shape)
        (1508, 8)
        >>> print(returns.head(2).round(5))
                      SAN.MC   BBVA.MC
        Date
        2019-01-03  0.00234  0.00312
        2019-01-04 -0.00893 -0.01245
        """
        logger.info(
            "Calculando rendimientos logarítmicos para %d activos...",
            prices.shape[1],
        )

        # Calcular rendimientos: ln(P_t / P_{t-1})
        # prices.shift(1) desplaza una posición hacia abajo (precio anterior)
        # .iloc[1:] elimina la primera fila, que siempre es NaN
        returns = np.log(prices / prices.shift(1)).iloc[1:]

        # Validar que el resultado tiene datos suficientes
        if returns.empty:
            raise ValueError(
                "No se pueden calcular rendimientos. El DataFrame de "
                "precios debe tener al menos 2 filas."
            )

        logger.info(
            "Rendimientos calculados — %d obs × %d activos | "
            "media diaria: %.6f | std diaria: %.6f",
            len(returns),
            returns.shape[1],
            returns.mean().mean(),
            returns.std().mean(),
        )

        # Persistir rendimientos en caché si está configurada
        cache_path = self._build_cache_path("returns")
        if cache_path is not None:
            returns.to_csv(cache_path)
            logger.info("Rendimientos guardados: %s", cache_path.name)

        return returns

    # ------------------------------------------------------------------ #
    # MÉTODO 3: ESTADÍSTICAS DESCRIPTIVAS ANUALIZADAS                      #
    # ------------------------------------------------------------------ #

    def compute_annualized_stats(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula las métricas estadísticas anualizadas de los rendimientos.

        Genera la tabla de estadísticas descriptivas utilizada en el
        panel de Análisis Exploratorio de Datos (EDA) del dashboard.

        Parameters
        ----------
        returns : pd.DataFrame
            DataFrame de rendimientos logarítmicos diarios, tal como
            devuelve ``compute_returns()``.

        Returns
        -------
        pd.DataFrame
            DataFrame con índice = tickers y columnas:
                - Rendimiento anual (%) : media diaria × 252 × 100.
                - Volatilidad anual (%) : std diaria × √252 × 100.
                - Ratio de Sharpe       : rendimiento / volatilidad
                                         (sin tasa libre de riesgo,
                                         solo para comparación relativa).
                - Asimetría             : skewness de la distribución.
                - Curtosis              : exceso de curtosis (normal = 0).
                - Máx. Drawdown (%)     : mayor caída desde máximo.

        Notes
        -----
        El Ratio de Sharpe calculado aquí usa rf = 0 para permitir
        la comparación rápida entre activos. En el módulo de
        optimización (Hito 4) se incorporará la tasa libre de riesgo
        real configurada por el usuario en el dashboard.

        La curtosis reportada es la curtosis en exceso (scipy/pandas
        convention): una distribución normal tiene curtosis = 0.
        Valores positivos indican colas más pesadas que la normal
        (leptocúrtica), común en series financieras.

        Examples
        --------
        >>> stats = da.compute_annualized_stats(returns)
        >>> print(stats[["Rendimiento anual (%)", "Volatilidad anual (%)"]])
                Rendimiento anual (%)  Volatilidad anual (%)
        SAN.MC                  12.3                   18.4
        BBVA.MC                 15.1                   21.2
        """
        logger.info(
            "Calculando estadísticas anualizadas para %d activos...",
            returns.shape[1],
        )

        stats = pd.DataFrame(index=returns.columns)

        # Rendimiento medio anualizado
        # Media de rendimientos diarios × 252 días hábiles × 100 (a porcentaje)
        stats["Rendimiento anual (%)"] = (
            returns.mean() * TRADING_DAYS_YEAR * 100
        ).round(2)

        # Volatilidad anualizada
        # Desviación estándar diaria × √252 × 100 (a porcentaje)
        stats["Volatilidad anual (%)"] = (
            returns.std() * np.sqrt(TRADING_DAYS_YEAR) * 100
        ).round(2)

        # Ratio de Sharpe sin tasa libre de riesgo (rf = 0)
        # Protección ante volatilidad cero para evitar división por cero
        stats["Ratio de Sharpe"] = (
            stats["Rendimiento anual (%)"]
            / stats["Volatilidad anual (%)"].replace(0, np.nan)
        ).round(4)

        # Asimetría (skewness): valores negativos indican colas izquierdas
        # más largas, es decir, mayor frecuencia de pérdidas extremas
        stats["Asimetría"] = returns.skew().round(4)

        # Curtosis en exceso: valores > 0 indican colas más pesadas
        # que la distribución normal (leptocúrtica)
        stats["Curtosis"] = returns.kurt().round(4)

        # Máximo Drawdown: mayor caída porcentual desde un máximo histórico
        stats["Máx. Drawdown (%)"] = (
            returns.apply(self._max_drawdown)
        ).round(2)

        logger.info("Estadísticas calculadas para %d activos.", len(stats))
        return stats

    # ------------------------------------------------------------------ #
    # MÉTODOS PRIVADOS AUXILIARES                                          #
    # ------------------------------------------------------------------ #

    def _build_cache_path(self, dataset_name: str) -> Optional[Path]:
        """
        Construye la ruta del archivo de caché para un dataset.

        El nombre del archivo incluye el dataset, el rango de fechas
        y la frecuencia para evitar colisiones entre configuraciones
        distintas de DataAcquisition en el mismo directorio de caché.

        Parameters
        ----------
        dataset_name : str
            Nombre del dataset: "prices_raw" o "returns".

        Returns
        -------
        Path or None
            Ruta completa al archivo CSV de caché, o None si
            cache_dir no está configurado.
        """
        if self.cache_dir is None:
            return None

        # Nombre único por configuración (evita colisiones)
        filename = (
            f"{dataset_name}_{self.start}_{self.end}_{self.interval}.csv"
        )
        return self.cache_dir / filename

    def _log_download_report(self, prices: pd.DataFrame) -> None:
        """
        Registra un informe detallado de calidad de la descarga.

        Imprime en el log el número de activos, el período efectivo,
        las observaciones totales y el porcentaje de valores nulos
        por activo. Los tickers sin ningún dato se registran como error.

        Parameters
        ----------
        prices : pd.DataFrame
            DataFrame de precios recién descargados desde Yahoo Finance.
        """
        logger.info("─── Informe de descarga ─────────────────────────────")
        logger.info("  Activos solicitados : %d", len(self.tickers))
        logger.info("  Activos descargados : %d", prices.shape[1])
        logger.info(
            "  Período efectivo    : %s → %s",
            prices.index.min().date(),
            prices.index.max().date(),
        )
        logger.info("  Observaciones       : %d días", len(prices))

        # Nulos por activo: advertencia si hay datos faltantes
        null_pct = prices.isna().mean() * 100
        for ticker, pct in null_pct.items():
            if pct == 0:
                logger.info("  %-12s : Sin datos nulos", ticker)
            else:
                logger.warning(
                    "  %-12s : %.1f%% de valores nulos — "
                    "pendiente de tratamiento en el preprocesado.",
                    ticker, pct,
                )

        # Tickers sin ningún dato: probable error en el símbolo
        empty = prices.columns[prices.isna().all()].tolist()
        if empty:
            logger.error(
                "  Sin datos: %s — verifica estos tickers en "
                "finance.yahoo.com", empty
            )
        logger.info("─────────────────────────────────────────────────────")

    @staticmethod
    def _max_drawdown(series: pd.Series) -> float:
        """
        Calcula el Máximo Drawdown de una serie de rendimientos.

        El Máximo Drawdown (MDD) mide la mayor pérdida porcentual
        acumulada desde un máximo histórico hasta el mínimo
        subsiguiente antes de que se alcance un nuevo máximo.
        Es una medida estándar de riesgo en gestión de carteras.

        Parameters
        ----------
        series : pd.Series
            Serie de rendimientos logarítmicos diarios de un activo.

        Returns
        -------
        float
            Máximo Drawdown como porcentaje negativo.
            Por ejemplo: -32.1 significa una caída máxima del 32.1%.
        """
        # Valor acumulado de la inversión a partir de los rendimientos
        cum_returns = (1 + series).cumprod()

        # Máximo rolling: valor máximo alcanzado hasta cada fecha
        rolling_max = cum_returns.cummax()

        # Drawdown en cada punto: caída relativa desde el máximo
        drawdown = (cum_returns - rolling_max) / rolling_max

        # El MDD es el valor mínimo del drawdown (mayor caída)
        return float(drawdown.min() * 100)


# ---------------------------------------------------------------------------
# Funciones de utilidad a nivel de módulo
# ---------------------------------------------------------------------------

def get_default_tickers(index: str = "IBEX35") -> list[str]:
    """
    Devuelve la lista de tickers predefinidos para un índice bursátil.

    Parameters
    ----------
    index : str
        Nombre del índice: "IBEX35" o "SP500". Por defecto "IBEX35".

    Returns
    -------
    list[str]
        Lista de tickers en formato Yahoo Finance.

    Raises
    ------
    ValueError
        Si el índice especificado no está definido en el módulo.

    Examples
    --------
    >>> tickers = get_default_tickers("IBEX35")
    >>> print(tickers[:3])
    ['SAN.MC', 'BBVA.MC', 'TEF.MC']
    """
    index_norm = index.upper().replace(" ", "").replace("-", "")
    if index_norm == "IBEX35":
        return IBEX35_TICKERS.copy()
    elif index_norm in ("SP500", "S&P500"):
        return SP500_TICKERS.copy()
    else:
        raise ValueError(
            f"Índice '{index}' no reconocido. "
            "Opciones: 'IBEX35', 'SP500'."
        )


def load_from_csv(path: str | Path) -> pd.DataFrame:
    """
    Carga un DataFrame de precios o rendimientos desde un CSV de caché.

    Parameters
    ----------
    path : str or Path
        Ruta al archivo CSV generado por DataAcquisition.

    Returns
    -------
    pd.DataFrame
        DataFrame con índice DatetimeIndex y columnas de activos.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe.

    Examples
    --------
    >>> prices = load_from_csv("data/cache/prices_raw_2019-01-01_2024-12-31_1d.csv")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "Date"
    logger.info("CSV cargado: %s — shape: %s", path.name, df.shape)
    return df


# ---------------------------------------------------------------------------
# Bloque de ejecución directa — demostración del pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  PROYECTO ATHENA — Hito 3: Adquisición de Datos")
    print("=" * 65)

    # Configuración de la demostración
    TICKERS_DEMO = [
        "SAN.MC", "BBVA.MC", "TEF.MC", "ITX.MC",
        "IBE.MC", "REP.MC", "ACS.MC", "FER.MC",
    ]
    START_DATE = "2019-01-01"
    END_DATE   = "2024-12-31"

    # Paso 1 — Inicializar pipeline
    print("\n[1/3] Inicializando pipeline...")
    pipeline = DataAcquisition(
        tickers   = TICKERS_DEMO,
        start     = START_DATE,
        end       = END_DATE,
        cache_dir = Path("data/cache"),
    )

    # Paso 2 — Descargar precios
    print("\n[2/3] Descargando precios de cierre ajustados...")
    prices = pipeline.download(use_cache=True)
    print(f"      Shape          : {prices.shape}")
    print(f"      Período real   : {prices.index.min().date()} → "
          f"{prices.index.max().date()}")
    print(f"      Nulos totales  : {prices.isna().sum().sum()}")
    print("\n      Últimas 3 filas de precios:")
    print(prices.tail(3).to_string())

    # Paso 3 — Calcular rendimientos logarítmicos
    print("\n[3/3] Calculando rendimientos logarítmicos diarios...")
    returns = pipeline.compute_returns(prices)
    print(f"      Shape          : {returns.shape}")
    print(f"      Media diaria   : {returns.mean().mean():.6f}")
    print(f"      Std diaria     : {returns.std().mean():.6f}")

    # Estadísticas anualizadas
    print("\n      Estadísticas anualizadas:")
    stats = pipeline.compute_annualized_stats(returns)
    print(stats.to_string())

    print("\n" + "=" * 65)
    print("  Pipeline Hito 3 completado.")
    print("  → prices  : precios de cierre ajustados (input Hito 4)")
    print("  → returns : rendimientos logarítmicos diarios (input Hito 4)")
    print("=" * 65)