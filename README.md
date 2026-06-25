# Proyecto Athena: Inteligencia de Datos para la Gestión Financiera

Trabajo Final de Grado — Grado en Ciencia de Datos Aplicada  
Universitat Oberta de Catalunya (UOC) — Curso 2025-2026

Autor: Álvaro Sánchez Mendoza  
Tutor: Carlos Gaitán Poyatos  
Director: Antonio Pita Lozano

---

## Descripción

Proyecto Athena es un sistema interactivo de análisis y optimización de carteras de inversión desarrollado como Trabajo Final de Grado en Ciencia de Datos Aplicada. El objetivo del proyecto es aplicar técnicas de ciencia de datos al ámbito de las finanzas cuantitativas, combinando la Teoría Moderna de Carteras de Markowitz con simulación estocástica de Monte Carlo para ofrecer al usuario una herramienta de apoyo a la toma de decisiones de inversión.

El sistema se articula en torno a un dashboard web interactivo construido con Streamlit que integra cuatro módulos diferenciados: adquisición y preprocesado de datos financieros históricos, optimización de carteras mediante programación cuadrática, simulación de escenarios futuros y análisis de riesgo. Todo el código está escrito en Python y hace uso exclusivo de librerías de código abierto.

Este proyecto tiene carácter estrictamente educativo y no constituye asesoramiento financiero ni gestión de carteras en el sentido de la Directiva MiFID II.

---

## Requisitos del sistema

- Python 3.11 o superior
- Conexión a internet para la descarga de datos desde Yahoo Finance
- Sistema operativo: Windows, macOS o Linux

---

## Instalación

Clonar el repositorio e instalar las dependencias dentro de un entorno virtual:

```bash
git clone https://github.com/pentum-s/TFG-ATHENA.git
cd TFG-ATHENA
python -m venv .venv
```

Activar el entorno virtual:

```bash
# En Windows
.venv\Scripts\activate

# En macOS / Linux
source .venv/bin/activate
```

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

---

## Uso

### Lanzar el dashboard

El dashboard se ejecuta desde la raíz del proyecto:

```bash
streamlit run modules/app.py
```

Se abrirá automáticamente el navegador en `http://localhost:8501`. Si no se abre, copia la dirección manualmente.

### Flujo de trabajo del dashboard

1. Seleccionar el índice bursátil (IBEX 35 o S&P 500) y los activos a incluir en la cartera.
2. Configurar el capital a invertir, el período histórico, el número de simulaciones Monte Carlo, la tasa libre de riesgo y el nivel de confianza del VaR.
3. Ajustar las restricciones de pesos por activo y las opciones adicionales de visualización.
4. Pulsar el botón "Ejecutar Análisis Completo" para lanzar el pipeline.
5. Explorar los resultados en las cuatro secciones del dashboard: análisis exploratorio, optimización, simulación y métricas de riesgo.

### Ejecutar los módulos de forma independiente

Cada módulo puede ejecutarse de forma autónoma desde la carpeta `modules/`:

```bash
cd modules

# Pipeline de adquisición y preprocesado de datos
python data_acquisition.py

# Optimización de carteras (Markowitz)
python optimization.py

# Simulación Monte Carlo y análisis de riesgo
python monte_carlo.py
```

Cada script incluye un bloque de demostración con datos de ejemplo que se ejecuta al llamarlo directamente.

---

## Estructura del proyecto

```
TFG-ATHENA/
│
├── modules/                        # Código fuente principal
│   ├── data/
│   │    └── cache/                 # Caché local de datos descargados (CSV)
│   ├── __init__.py        
│   ├── app.py                      # Dashboard Streamlit (punto de entrada)
│   ├── data_acquisition.py         # Hito 3: adquisición y preprocesado
│   ├── optimization.py             # Hito 4: optimización de Markowitz
│   └── monte_carlo.py              # Hito 5: simulación Monte Carlo y VaR
│
├── notebooks/                      # Notebooks de desarrollo por hito
│   ├── data/
│   │    └── cache/                 # Caché local de datos descargados (CSV)
│   ├── data_acquisition.ipynb
│   ├── optimization.ipynb
│   └── monte_carlo.ipynb
│
├── tests/                          # Suite de tests unitarios e integración
│   ├── conftest.py                 # Fixtures y datos sintéticos compartidos
│   ├── constants.py                # Constantes globales de los tests
│   ├── test_data_acquisition.py    # 53 tests del módulo de datos
│   ├── test_optimization.py        # 48 tests del módulo de optimización
│   └── test_monte_carlo.py         # 43 tests del módulo de simulación
│
│
├── conftest.py
├── pytest.ini                      # Configuración de pytest
├── requirements.txt                # Dependencias del proyecto
├── LICENSE
└── README.md
```

---

## Módulos

### data_acquisition.py

Implementa el pipeline completo de adquisición y preprocesado de datos financieros. Se conecta a la API pública de Yahoo Finance mediante la librería `yfinance` y descarga series históricas de precios de cierre ajustados por dividendos y splits.

El preprocesado se aplica en cinco pasos secuenciales: eliminación de activos con más de un 20% de valores nulos, interpolación lineal de huecos aislados de hasta cinco días consecutivos, eliminación de filas enteramente vacías correspondientes a festivos globales, detección y eliminación de duplicados en el índice temporal, y reindexado al calendario completo de días hábiles.

A partir de los precios limpios se calculan los rendimientos logarítmicos diarios, que son el input directo del módulo de optimización. El módulo también calcula estadísticas descriptivas anualizadas para el panel de análisis exploratorio del dashboard.

Clases y funciones principales:

- `DataAcquisition` — clase principal con métodos `download()`, `preprocess()`, `compute_returns()` y `compute_annualized_stats()`
- `get_default_tickers(index)` — devuelve la lista de tickers predefinidos para IBEX 35 o S&P 500
- `load_from_csv(path)` — carga datos previamente guardados desde la caché local

### optimization.py

Implementa la optimización de carteras basada en la Teoría Moderna de Carteras de Markowitz (1952). A partir de la matriz de covarianzas y el vector de rendimientos esperados anualizados, resuelve tres problemas de optimización cuadrática mediante `scipy.optimize.minimize` con el método SLSQP:

- Cartera de Mínima Varianza Global (MVP): minimiza la varianza de la cartera sin restricción de rendimiento.
- Cartera de Máximo Ratio de Sharpe (MSR): maximiza el cociente entre el rendimiento excedente sobre la tasa libre de riesgo y la volatilidad.
- Frontera Eficiente: calcula N carteras óptimas parametrizadas por rendimiento objetivo, trazando la curva completa en el espacio riesgo-rendimiento.

El optimizador utiliza múltiples puntos de inicio para aumentar la robustez frente a mínimos locales y respeta las restricciones de pesos configuradas por el usuario.

Clases y funciones principales:

- `PortfolioOptimizer` — clase principal con métodos `min_variance()`, `max_sharpe()`, `efficient_frontier()`, `equal_weight()` y `portfolio_metrics()`
- `PortfolioResult` — dataclass que agrupa los pesos óptimos y las métricas de la cartera resultante
- `portfolio_from_weights(weights, returns)` — evalúa carteras de referencia con las mismas métricas

### monte_carlo.py

Implementa la simulación estocástica de Monte Carlo sobre la cartera óptima. Genera N trayectorias de rendimientos futuros multivariantes preservando la estructura de correlaciones históricas entre activos mediante descomposición de Cholesky. A partir de la distribución empírica de rentabilidades acumuladas al final del horizonte calcula las siguientes métricas de riesgo:

- VaR histórico: percentil alpha de la distribución empírica de rentabilidades finales.
- CVaR (Expected Shortfall): media de las rentabilidades que caen por debajo del VaR, capturando la severidad de los peores escenarios.
- VaR paramétrico: estimación bajo supuesto de normalidad, como referencia comparativa.
- Percentiles P1 a P99 de la distribución final.
- Probabilidad de pérdida y de superar una rentabilidad objetivo.

Clases y funciones principales:

- `MonteCarlo` — clase principal con método `run()` que devuelve un `MonteCarloResult`
- `MonteCarloResult` — dataclass con todas las trayectorias y métricas, con métodos `summary()` y `metrics_dataframe()`
- `sensitivity_analysis()` — tabla de VaR y CVaR para distintos horizontes temporales y niveles de confianza

### app.py

Punto de entrada del dashboard. Integra los tres módulos anteriores en una aplicación Streamlit de página única con scroll, organizada en cuatro secciones:

- Sección 1: configuración del análisis (selección de activos, capital, parámetros y restricciones).
- Sección 2: análisis exploratorio de datos con gráfico de precios normalizados, histograma de rendimientos, tabla de estadísticas descriptivas y mapa de calor de correlaciones.
- Sección 3: optimización de cartera con gráfico de Frontera Eficiente, tarjetas de métricas KPI, gráfico comparativo de pesos y gráfico de tarta.
- Sección 4: simulación Monte Carlo con trayectorias proyectadas, histograma de distribución final con zona VaR, tabla de métricas de riesgo y análisis de sensibilidad.

El pipeline principal está cacheado mediante `@st.cache_data` con un TTL de 15 minutos para evitar llamadas redundantes a la API de Yahoo Finance entre ejecuciones consecutivas con los mismos parámetros.

---

## Tests

La suite de tests cubre los tres módulos principales con un total de 144 tests organizados en clases por funcionalidad. Todos los tests utilizan datos sintéticos generados de forma determinista con semilla fija, por lo que no requieren conexión a internet y se ejecutan en menos de 15 segundos.

Para ejecutar la suite completa desde la raíz del proyecto:

```bash
python -m pytest tests/ -v
```

Para ejecutar solo los tests de un módulo concreto:

```bash
python -m pytest tests/test_data_acquisition.py -v
python -m pytest tests/test_optimization.py -v
python -m pytest tests/test_monte_carlo.py -v
```

Para ejecutar un test específico por nombre:

```bash
python -m pytest tests/ -v -k "test_pesos_suman_uno"
```

Resumen de la cobertura:

| Archivo de test | Módulo cubierto | Tests |
|---|---|---|
| test_data_acquisition.py | data_acquisition.py | 53 |
| test_optimization.py | optimization.py | 48 |
| test_monte_carlo.py | monte_carlo.py | 43 |
| Total | | 144 |

---

## Dependencias

Las dependencias principales del proyecto son:

| Librería | Versión mínima | Uso |
|---|---|---|
| streamlit | 1.28.0 | Dashboard web interactivo |
| plotly | 5.15.0 | Visualizaciones interactivas |
| pandas | 2.0.0 | Manipulación de series temporales |
| numpy | 1.26.0 | Operaciones matriciales |
| scipy | 1.11.0 | Optimización cuadrática (SLSQP) |
| yfinance | 0.2.28 | Descarga de datos de Yahoo Finance |
| matplotlib | 3.7.0 | Visualizaciones en notebooks |
| pytest | 7.0.0 | Suite de tests |

---

## Bibliografía

Las referencias principales que fundamentan el trabajo son las siguientes. La lista completa se encuentra en la memoria del TFG.

Artzner, P., Delbaen, F., Eber, J.-M., & Heath, D. (1999). Coherent Measures of Risk. *Mathematical Finance*, 9(3), 203-228. https://doi.org/10.1111/1467-9965.00068

Basel Committee on Banking Supervision. (2019). *Minimum Capital Requirements for Market Risk*. Bank for International Settlements. https://www.bis.org/bcbs/publ/d457.htm

DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy? *The Review of Financial Studies*, 22(5), 1915-1953. https://doi.org/10.1093/rfs/hhm075

Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer. https://doi.org/10.1007/978-0-387-21617-1

Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array Programming with NumPy. *Nature*, 585(7825), 357-362. https://doi.org/10.1038/s41586-020-2649-2

Ledoit, O., & Wolf, M. (2004). A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices. *Journal of Multivariate Analysis*, 88(2), 365-411. https://doi.org/10.1016/S0047-259X(03)00096-4

Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*, 7(1), 77-91. https://doi.org/10.2307/2975974

Martin, R. A. (2021). PyPortfolioOpt: Portfolio Optimization in Python. *Journal of Open Source Software*, 6(61), 3066. https://doi.org/10.21105/joss.03066

McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*, 56-61. https://doi.org/10.25080/Majora-92bf1922-00a

Rockafellar, R. T., & Uryasev, S. (2000). Optimization of Conditional Value-at-Risk. *Journal of Risk*, 2(3), 21-42. https://doi.org/10.21314/JOR.2000.038

Sharpe, W. F. (1966). Mutual Fund Performance. *The Journal of Business*, 39(1), 119-138. https://doi.org/10.1086/294846

Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. *Nature Methods*, 17(3), 261-272. https://doi.org/10.1038/s41592-019-0686-2

---

## Licencia

Este proyecto se distribuye bajo la licencia MIT. Consulta el archivo LICENSE para más información.

---

*Proyecto Athena — Trabajo Final de Grado — Grado en Ciencia de Datos Aplicada — UOC 2026*