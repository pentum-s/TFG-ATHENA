"""
=============================================================================
PROYECTO ATHENA — app.py: Dashboard Interactivo
=============================================================================
Grado en Ciencia de Datos Aplicada — Universitat Oberta de Catalunya (UOC)

Autor   : Álvaro Sánchez Mendoza
Tutor   : Carlos Gaitán Poyatos

Descripción
-----------
Aplicación Streamlit de página única que integra los cuatro módulos del
Proyecto Athena en un dashboard interactivo accesible sin instalación:

    Sección 1 — Configuración del Análisis      (inputs del usuario)
    Sección 2 — Análisis Exploratorio (EDA)     (precios, retornos, correlaciones)
    Sección 3 — Optimización de Cartera         (Markowitz, Frontera Eficiente)
    Sección 4 — Simulación Monte Carlo          (VaR, CVaR, trayectorias)

Despliegue
----------
    streamlit run app.py

Dependencias
------------
    pip install streamlit plotly pandas numpy scipy yfinance
=============================================================================
"""

# ---------------------------------------------------------------------------
# Importaciones
# ---------------------------------------------------------------------------
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_acquisition import DataAcquisition, get_default_tickers
from monte_carlo import MonteCarlo
from optimization import PortfolioOptimizer

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuración general de la página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Proyecto Athena",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Estilos CSS personalizados
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Cabecera principal */
    .athena-header {
        background: linear-gradient(135deg, #1F4E79 0%, #2E75B6 100%);
        padding: 1.8rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .athena-header h1 { margin: 0; font-size: 2.2rem; font-weight: 700; }
    .athena-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.95rem; }

    /* Barras de sección */
    .section-bar {
        background: #EBF3FB;
        border-left: 6px solid #2E75B6;
        padding: 0.6rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1.5rem 0 1rem;
        font-weight: 700;
        font-size: 1.05rem;
        color: #1F4E79;
    }

    /* Tarjetas KPI */
    .kpi-card {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border: 1.5px solid #D6E4F0;
        text-align: center;
    }
    .kpi-value { font-size: 1.7rem; font-weight: 700; margin: 0; }
    .kpi-label { font-size: 0.80rem; color: #595959; margin: 0; }

    /* Aviso legal */
    .disclaimer {
        background: #FFF8E1;
        border: 1.5px solid #F9A825;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        font-size: 0.82rem;
        color: #5D4037;
        margin-top: 2rem;
    }

    /* Separador */
    hr.athena { border: none; border-top: 2px solid #D6E4F0; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# CONSTANTES
# ---------------------------------------------------------------------------
TRADING_DAYS = 252
CACHE_DIR    = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Paleta de colores coherente con el mockup del Hito 2
COLORS = [
    "#2E75B6", "#C55A11", "#375623", "#C00000",
    "#7030A0", "#E67E22", "#2ECC71", "#1F4E79",
]

# ---------------------------------------------------------------------------
# CABECERA
# ---------------------------------------------------------------------------
st.markdown("""
<div class="athena-header">
  <h1>⚡ PROYECTO ATHENA</h1>
  <p>Inteligencia de Datos para la Gestión Financiera &nbsp;·&nbsp;
     Teoría Moderna de Carteras &nbsp;·&nbsp; Simulación Monte Carlo &nbsp;·&nbsp;
     Análisis de Riesgo</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# ══════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — CONFIGURACIÓN DEL ANÁLISIS
# ══════════════════════════════════════════════════════════════════════════
# ---------------------------------------------------------------------------
st.markdown('<div class="section-bar">Sección 1 — Configuración del Análisis</div>',
            unsafe_allow_html=True)
st.caption("Introduce los parámetros del análisis y ejecuta el sistema.")

with st.container():
    col_l, col_r = st.columns([1.1, 0.9], gap="large")

    with col_l:
        st.markdown("**Selección de activos**")

        # Selector de índice
        indice = st.radio(
            "Índice bursátil",
            ["IBEX 35", "S&P 500"],
            horizontal=True,
            help="Selecciona el universo de activos para el análisis.",
        )
        tickers_disponibles = get_default_tickers(
            "IBEX35" if indice == "IBEX 35" else "SP500"
        )
        labels_disp = [t.replace(".MC", "") for t in tickers_disponibles]
        label_to_ticker = dict(zip(labels_disp, tickers_disponibles))

        # Multiselect de activos
        default_labels = labels_disp[:8]
        selected_labels = st.multiselect(
            f"Activos a incluir (mín. 2, máx. {len(tickers_disponibles)})",
            options=labels_disp,
            default=default_labels,
            help="Selecciona los activos que formarán el universo de optimización.",
        )
        selected_tickers = [label_to_ticker[l] for l in selected_labels]

        if len(selected_tickers) < 2:
            st.warning("Selecciona al menos 2 activos para continuar.")

        # Capital a invertir
        capital = st.number_input(
            "Capital a invertir (€)",
            min_value=100.0,
            max_value=10_000_000.0,
            value=10_000.0,
            step=1_000.0,
            format="%.0f",
            help="Capital inicial sobre el que se calculan las proyecciones monetarias.",
        )

    with col_r:
        st.markdown("**Parámetros de análisis**")

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            start_date = st.selectbox(
                "Período histórico",
                ["1 año", "2 años", "3 años", "5 años", "10 años"],
                index=3,
            )
            years_map  = {"1 año": 1, "2 años": 2, "3 años": 3, "5 años": 5, "10 años": 10}
            n_years    = years_map[start_date]
            end_dt     = pd.Timestamp.today()
            start_dt   = end_dt - pd.DateOffset(years=n_years)
            start_str  = start_dt.strftime("%Y-%m-%d")
            end_str    = end_dt.strftime("%Y-%m-%d")

            n_simulations = st.selectbox(
                "Simulaciones Monte Carlo",
                [1_000, 5_000, 10_000, 20_000],
                index=2,
                help="Más simulaciones = mayor precisión pero más tiempo de cómputo.",
            )

        with col_p2:
            risk_free = st.slider(
                "Tasa libre de riesgo (%)",
                min_value=0.0, max_value=10.0, value=3.5, step=0.1,
                help="Tasa anual de referencia para el cálculo del Ratio de Sharpe.",
            )
            var_confidence = st.slider(
                "Nivel de confianza VaR (%)",
                min_value=90, max_value=99, value=95, step=1,
                help="Nivel de confianza para el cálculo del VaR histórico.",
            )

        st.markdown("**Restricciones de pesos**")
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            w_min = st.number_input("Peso mínimo por activo (%)", 0.0, 20.0, 0.0, 1.0) / 100
        with col_w2:
            w_max = st.number_input("Peso máximo por activo (%)", 10.0, 100.0, 40.0, 5.0) / 100

        st.markdown("**Opciones adicionales**")
        show_mvp       = st.checkbox("Mostrar cartera de Mínima Varianza (MVP)", value=True)
        show_pct       = st.checkbox("Mostrar percentiles P10 / P50 / P90 en Monte Carlo", value=True)
        show_stats_tbl = st.checkbox("Mostrar tabla de estadísticas descriptivas", value=True)

# Botones de acción
col_btn1, col_btn2, _ = st.columns([1.2, 0.8, 3])
with col_btn1:
    run_btn = st.button("▶  Ejecutar Análisis Completo", type="primary",
                        use_container_width=True)
with col_btn2:
    reset_btn = st.button("↺  Restablecer", use_container_width=True)

if reset_btn:
    st.rerun()

st.markdown('<hr class="athena">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# LÓGICA PRINCIPAL — se ejecuta al pulsar el botón
# ---------------------------------------------------------------------------
if not run_btn:
    # Estado inicial: KPIs vacíos y mensaje de bienvenida
    k1, k2, k3, k4 = st.columns(4)
    for col, val, lbl, col_css in [
        (k1, "—", "Activos cargados",        "#2E75B6"),
        (k2, "—", "Período analizado",       "#375623"),
        (k3, "—", "Datos descargados",       "#375623"),
        (k4, "—", "Última actualización",    "#C55A11"),
    ]:
        col.markdown(f"""
        <div class="kpi-card">
          <p class="kpi-value" style="color:{col_css}">{val}</p>
          <p class="kpi-label">{lbl}</p>
        </div>""", unsafe_allow_html=True)
    st.info("Configura los parámetros en la Sección 1 y pulsa **Ejecutar Análisis Completo**.")
    st.stop()

# ---------------------------------------------------------------------------
# VALIDACIONES PREVIAS
# ---------------------------------------------------------------------------
if len(selected_tickers) < 2:
    st.error("Selecciona al menos 2 activos para continuar.")
    st.stop()

if w_min >= w_max:
    st.error("El peso mínimo debe ser menor que el peso máximo.")
    st.stop()

# ---------------------------------------------------------------------------
# PIPELINE DE DATOS (con spinner y caché de sesión)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=900)   # caché 15 min (latencia ≤ 15 min)
def run_pipeline(tickers, start, end, wmin, wmax, rf, n_sims, var_conf, cap):
    """
    Ejecuta el pipeline completo de Athena y devuelve todos los resultados.
    Cacheado 15 minutos para evitar llamadas redundantes a la API.
    """
    # ── Hito 3: Adquisición y preprocesado ──────────────────────────
    da = DataAcquisition(tickers=tickers, start=start, end=end,
                         cache_dir=CACHE_DIR)
    prices_raw   = da.download(use_cache=True)
    prices_clean = da.preprocess(prices_raw)
    returns      = da.compute_returns(prices_clean)
    stats        = da.compute_annualized_stats(returns)

    # ── Hito 4: Optimización de Markowitz ───────────────────────────
    opt     = PortfolioOptimizer(returns, risk_free_rate=rf/100,
                                 w_min=wmin, w_max=wmax)
    mvp     = opt.min_variance()
    msr     = opt.max_sharpe()
    eq_pf   = opt.equal_weight()
    frontier = opt.efficient_frontier(n_points=80)

    # ── Hito 5: Simulación Monte Carlo ──────────────────────────────
    mc     = MonteCarlo(weights=msr.weights.values, returns=returns,
                        capital=cap, n_simulations=n_sims,
                        horizon_days=TRADING_DAYS,
                        confidence=var_conf/100, seed=42)
    mc_res = mc.run()

    return {
        "prices_raw":   prices_raw,
        "prices_clean": prices_clean,
        "returns":      returns,
        "stats":        stats,
        "mvp":          mvp,
        "msr":          msr,
        "eq_pf":        eq_pf,
        "frontier":     frontier,
        "mc_res":       mc_res,
        "tickers":      list(returns.columns),
        "labels":       [t.replace(".MC","") for t in returns.columns],
    }

with st.spinner("⏳ Descargando datos y ejecutando el análisis completo..."):
    try:
        data = run_pipeline(
            tickers=selected_tickers,
            start=start_str, end=end_str,
            wmin=w_min, wmax=w_max,
            rf=risk_free, n_sims=n_simulations,
            var_conf=var_confidence, cap=capital,
        )
    except ConnectionError as e:
        st.error(f"Error de conexión con Yahoo Finance: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error inesperado: {e}")
        st.stop()

# Extraer objetos del resultado
prices_raw   = data["prices_raw"]
prices_clean = data["prices_clean"]
returns      = data["returns"]
stats_df     = data["stats"]
mvp          = data["mvp"]
msr          = data["msr"]
eq_pf        = data["eq_pf"]
frontier     = data["frontier"]
mc_res       = data["mc_res"]
tickers      = data["tickers"]
labels       = data["labels"]

# KPIs de estado
k1, k2, k3, k4 = st.columns(4)
kpi_data = [
    (str(len(tickers)),              "Activos cargados",      "#2E75B6"),
    (f"{n_years} año/s",             "Período analizado",     "#375623"),
    ("✓  Correctos",                 "Datos descargados",     "#375623"),
    (pd.Timestamp.today().strftime("%d/%m/%Y %H:%M"), "Actualización", "#C55A11"),
]
for col, (val, lbl, clr) in zip([k1,k2,k3,k4], kpi_data):
    col.markdown(f"""
    <div class="kpi-card">
      <p class="kpi-value" style="color:{clr}">{val}</p>
      <p class="kpi-label">{lbl}</p>
    </div>""", unsafe_allow_html=True)

st.success(f"Análisis completado — {len(returns)} observaciones × {len(tickers)} activos "
           f"({returns.index.min().date()} → {returns.index.max().date()})")

# ---------------------------------------------------------------------------
# ══════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ══════════════════════════════════════════════════════════════════════════
# ---------------------------------------------------------------------------
st.markdown('<div class="section-bar">Sección 2 — Análisis Exploratorio de Datos (EDA)</div>',
            unsafe_allow_html=True)
st.caption("Evolución histórica de precios, distribución de rendimientos y correlaciones entre activos.")

# ── Gráfico de precios normalizados ──────────────────────────────────────
prices_norm = prices_clean / prices_clean.iloc[0] * 100

fig_prices = go.Figure()
for ticker, label, color in zip(tickers, labels, COLORS):
    fig_prices.add_trace(go.Scatter(
        x=prices_norm.index, y=prices_norm[ticker],
        mode="lines", name=label, line=dict(color=color, width=1.5),
        hovertemplate=f"<b>{label}</b><br>%{{x|%d/%m/%Y}}: %{{y:.1f}}<extra></extra>",
    ))
fig_prices.add_hline(y=100, line_dash="dash", line_color="gray",
                     line_width=1, opacity=0.5, annotation_text="Base 100")
fig_prices.update_layout(
    title="Evolución Histórica de Precios Normalizados (base 100)",
    xaxis_title="Fecha", yaxis_title="Precio normalizado",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    height=380, margin=dict(t=60, b=40),
    hovermode="x unified",
)
st.plotly_chart(fig_prices, use_container_width=True)

# ── Fila de tres paneles ──────────────────────────────────────────────────
col_eda1, col_eda2, col_eda3 = st.columns(3, gap="medium")

# Panel 1: Histograma de rendimientos del activo seleccionado
with col_eda1:
    activo_sel = st.selectbox("Activo para histograma",
                               labels, index=0, key="hist_sel")
    ticker_sel = tickers[labels.index(activo_sel)]
    ret_serie  = returns[ticker_sel].dropna() * 100

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=ret_serie, nbinsx=60, name="Rendimientos",
        marker_color="#2E75B6", opacity=0.75,
    ))
    var_line = float(np.percentile(ret_serie, (1 - var_confidence/100) * 100))
    fig_hist.add_vline(x=var_line, line_dash="dash", line_color="red",
                       annotation_text=f"VaR {var_confidence}%", annotation_position="top left")
    fig_hist.update_layout(
        title=f"Rendimientos Diarios — {activo_sel}",
        xaxis_title="Rendimiento (%)", yaxis_title="Frecuencia",
        height=300, margin=dict(t=50, b=30), showlegend=False,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# Panel 2: Tabla de estadísticas descriptivas
with col_eda2:
    st.markdown("**Estadísticas Descriptivas**")
    if show_stats_tbl:
        stats_show = stats_df.copy()
        stats_show.index = labels
        stats_show.columns = ["Rend. anual (%)", "Volat. anual (%)",
                               "Sharpe", "Asimetría", "Curtosis", "Max DD (%)"]
        st.dataframe(
            stats_show.style
            .format("{:.2f}")
            .background_gradient(subset=["Rend. anual (%)"], cmap="Greens")
            .background_gradient(subset=["Volat. anual (%)"], cmap="Reds_r")
            .background_gradient(subset=["Sharpe"], cmap="Blues"),
            height=300,
            use_container_width=True,
        )
    else:
        st.info("Activa 'Mostrar tabla de estadísticas' en la configuración.")

# Panel 3: Heatmap de correlaciones
with col_eda3:
    corr = returns.corr()
    fig_corr = px.imshow(
        corr.values,
        x=labels, y=labels,
        color_continuous_scale="RdYlGn",
        zmin=-1, zmax=1,
        text_auto=".2f",
        title="Matriz de Correlaciones",
        aspect="auto",
    )
    fig_corr.update_layout(
        height=320, margin=dict(t=50, b=10),
        coloraxis_showscale=False,
        xaxis=dict(tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
    )
    fig_corr.update_traces(textfont_size=8)
    st.plotly_chart(fig_corr, use_container_width=True)

st.markdown('<hr class="athena">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# ══════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — OPTIMIZACIÓN DE CARTERA (MARKOWITZ)
# ══════════════════════════════════════════════════════════════════════════
# ---------------------------------------------------------------------------
st.markdown('<div class="section-bar">Sección 3 — Optimización de Cartera (Modelo de Markowitz)</div>',
            unsafe_allow_html=True)
st.caption("Frontera Eficiente, cartera de Mínima Varianza y cartera de Máximo Sharpe.")

col_fe, col_kpis = st.columns([1.55, 1], gap="large")

# ── Gráfico Frontera Eficiente ────────────────────────────────────────────
with col_fe:
    # Nube de carteras aleatorias para contexto visual
    np.random.seed(0)
    N_RAN = 2000
    rng   = np.random.default_rng(0)
    mu_v  = returns.mean().values * TRADING_DAYS
    sig_v = returns.cov().values  * TRADING_DAYS
    ran_vols, ran_rets, ran_sr = [], [], []
    for _ in range(N_RAN):
        w = rng.dirichlet(np.ones(len(tickers)))
        w = np.clip(w, w_min, w_max); w /= w.sum()
        r_ = float(w @ mu_v)
        v_ = float(np.sqrt(w @ sig_v @ w))
        s_ = (r_ - risk_free/100) / v_ if v_ > 1e-10 else 0
        ran_vols.append(v_ * 100)
        ran_rets.append(r_ * 100)
        ran_sr.append(s_)

    fig_fe = go.Figure()
    # Nube
    fig_fe.add_trace(go.Scatter(
        x=ran_vols, y=ran_rets, mode="markers",
        marker=dict(size=4, color=ran_sr, colorscale="YlOrRd",
                    opacity=0.35, colorbar=dict(title="Sharpe", thickness=12)),
        name="Carteras simuladas",
        hovertemplate="Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<extra></extra>",
    ))
    # Frontera eficiente
    fig_fe.add_trace(go.Scatter(
        x=frontier["volatility"]*100, y=frontier["expected_return"]*100,
        mode="lines", line=dict(color="#2E75B6", width=2.5),
        name="Frontera Eficiente",
    ))
    # MVP
    if show_mvp:
        fig_fe.add_trace(go.Scatter(
            x=[mvp.volatility*100], y=[mvp.expected_return*100],
            mode="markers+text",
            marker=dict(size=14, color="#1F4E79", symbol="diamond"),
            text=["MVP"], textposition="top center",
            name=f"Mínima Varianza (Sharpe={mvp.sharpe_ratio:.3f})",
            hovertemplate=(f"<b>MVP</b><br>Vol: {mvp.volatility*100:.2f}%<br>"
                           f"Ret: {mvp.expected_return*100:.2f}%<extra></extra>"),
        ))
    # MSR
    fig_fe.add_trace(go.Scatter(
        x=[msr.volatility*100], y=[msr.expected_return*100],
        mode="markers+text",
        marker=dict(size=18, color="#C00000", symbol="star"),
        text=["MSR ★"], textposition="top right",
        name=f"Máximo Sharpe (Sharpe={msr.sharpe_ratio:.3f})",
        hovertemplate=(f"<b>MSR</b><br>Vol: {msr.volatility*100:.2f}%<br>"
                       f"Ret: {msr.expected_return*100:.2f}%<extra></extra>"),
    ))
    fig_fe.update_layout(
        title="Frontera Eficiente — Modelo de Markowitz",
        xaxis_title="Volatilidad anual (%)",
        yaxis_title="Rendimiento esperado anual (%)",
        height=420, margin=dict(t=55, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="closest",
    )
    st.plotly_chart(fig_fe, use_container_width=True)

# ── KPIs de carteras optimizadas ─────────────────────────────────────────
with col_kpis:
    st.markdown("**Métricas de las carteras óptimas**")
    carteras = [msr, eq_pf] + ([mvp] if show_mvp else [])
    for pf in carteras:
        color = "#375623" if "Sharpe" in pf.label else (
                "#C55A11" if "Equi" in pf.label else "#2E75B6")
        st.markdown(f"""
        <div class="kpi-card" style="border-color:{color};margin-bottom:0.6rem">
          <p style="font-weight:700;color:{color};margin:0 0 0.4rem">{pf.label}</p>
          <div style="display:flex;justify-content:space-between;font-size:0.88rem">
            <span>Rentabilidad</span><b style="color:{color}">{pf.expected_return*100:.2f}%</b>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:0.88rem">
            <span>Volatilidad</span><b style="color:{color}">{pf.volatility*100:.2f}%</b>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:0.88rem">
            <span>Sharpe</span><b style="color:{color}">{pf.sharpe_ratio:.4f}</b>
          </div>
        </div>""", unsafe_allow_html=True)

# ── Distribución de pesos ─────────────────────────────────────────────────
st.markdown("**Distribución de Pesos — Cartera Óptima (Máximo Sharpe)**")
col_bar, col_pie = st.columns([2.2, 1], gap="medium")

with col_bar:
    # Comparativa barras: MSR vs MVP vs Equi-ponderada
    carteras_cmp  = [msr, eq_pf] + ([mvp] if show_mvp else [])
    colores_cmp   = ["#C00000", "#888888", "#1F4E79"]
    fig_pesos = go.Figure()
    for pf, clr in zip(carteras_cmp, colores_cmp):
        fig_pesos.add_trace(go.Bar(
            x=labels,
            y=pf.weights.values * 100,
            name=pf.label,
            marker_color=clr,
            opacity=0.82,
        ))
    fig_pesos.update_layout(
        barmode="group",
        yaxis_title="Peso (%)",
        xaxis_title="Activo",
        height=300, margin=dict(t=20, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_pesos, use_container_width=True)

with col_pie:
    fig_pie = px.pie(
        values=msr.weights.values * 100,
        names=labels,
        color_discrete_sequence=COLORS,
        hole=0.35,
    )
    fig_pie.update_layout(height=300, margin=dict(t=10, b=10),
                           showlegend=True,
                           legend=dict(font=dict(size=9)))
    fig_pie.update_traces(textinfo="percent+label", textfont_size=9)
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown('<hr class="athena">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# ══════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — SIMULACIÓN MONTE CARLO Y ANÁLISIS DE RIESGO
# ══════════════════════════════════════════════════════════════════════════
# ---------------------------------------------------------------------------
st.markdown('<div class="section-bar">Sección 4 — Simulación Monte Carlo y Análisis de Riesgo</div>',
            unsafe_allow_html=True)
st.caption(f"Proyección de {n_simulations:,} escenarios futuros · "
           f"Horizonte: {TRADING_DAYS} días hábiles (~1 año) · "
           f"Confianza VaR: {var_confidence}%")

# ── Gráfico de trayectorias ───────────────────────────────────────────────
N_SHOW = min(300, n_simulations)
dates_mc = pd.bdate_range(start=pd.Timestamp.today(), periods=TRADING_DAYS + 1)
rng_plot  = np.random.default_rng(1)
idx_show  = rng_plot.choice(n_simulations, N_SHOW, replace=False)

p10_traj = np.percentile(mc_res.simulations, 10, axis=0)
p50_traj = np.percentile(mc_res.simulations, 50, axis=0)
p90_traj = np.percentile(mc_res.simulations, 90, axis=0)

fig_traj = go.Figure()
# Muestra de trayectorias
for i in idx_show:
    color = "rgba(192,0,0,0.07)" if mc_res.final_values[i] < capital else "rgba(46,117,182,0.07)"
    fig_traj.add_trace(go.Scatter(
        x=dates_mc, y=mc_res.simulations[i],
        mode="lines", line=dict(color=color, width=0.6),
        showlegend=False, hoverinfo="skip",
    ))
# Percentiles
for pct_arr, color, name in [
    (p90_traj, "#375623", f"Percentil 90 ({capital*(1+mc_res.percentiles['P90']):,.0f}€)"),
    (p50_traj, "#1F4E79", f"Percentil 50 ({capital*(1+mc_res.percentiles['P50']):,.0f}€)"),
    (p10_traj, "#C00000", f"Percentil 10 ({capital*(1+mc_res.percentiles['P10']):,.0f}€)"),
]:
    fig_traj.add_trace(go.Scatter(
        x=dates_mc, y=pct_arr, mode="lines",
        line=dict(color=color, width=2.2), name=name,
    ))
# Capital inicial
fig_traj.add_hline(y=capital, line_dash="dash", line_color="gray",
                   line_width=1.2, opacity=0.6,
                   annotation_text=f"Capital inicial ({capital:,.0f}€)")
fig_traj.update_layout(
    title=f"Trayectorias Simuladas de la Cartera "
          f"({n_simulations:,} escenarios · horizonte {TRADING_DAYS} días)",
    xaxis_title="Fecha proyectada",
    yaxis_title="Valor de la cartera (€)",
    height=400, margin=dict(t=55, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    hovermode="x unified",
    yaxis=dict(tickformat=",.0f", ticksuffix="€"),
)
st.plotly_chart(fig_traj, use_container_width=True)

# ── Tres paneles inferiores ───────────────────────────────────────────────
col_mc1, col_mc2, col_mc3 = st.columns(3, gap="medium")

# Panel 1: Histograma de rentabilidades finales
with col_mc1:
    final_ret_pct = mc_res.final_returns * 100
    var_pct       = mc_res.var_historical * 100
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=final_ret_pct, nbinsx=70,
        marker=dict(color=[
            "rgba(192,0,0,0.75)" if v < var_pct else "rgba(46,117,182,0.65)"
            for v in np.linspace(final_ret_pct.min(), final_ret_pct.max(), 70)
        ]),
        name="Distribución",
    ))
    fig_dist.add_vline(x=var_pct, line_dash="dash", line_color="red",
                       annotation_text=f"VaR {var_confidence}% = {var_pct:.1f}%",
                       annotation_position="top left")
    fig_dist.add_vline(x=mc_res.cvar_historical*100, line_dash="dot",
                       line_color="darkred",
                       annotation_text=f"CVaR = {mc_res.cvar_historical*100:.1f}%",
                       annotation_position="bottom left")
    fig_dist.update_layout(
        title="Distribución Final de Rentabilidades",
        xaxis_title="Rentabilidad acumulada (%)",
        yaxis_title="Frecuencia",
        height=320, margin=dict(t=50, b=30), showlegend=False,
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# Panel 2: Tabla de métricas de riesgo
with col_mc2:
    st.markdown("**Métricas de Riesgo**")
    metrics_df = mc_res.metrics_dataframe()
    # Añadir columna de valor monetario para las métricas de rentabilidad
    def add_euros(row):
        try:
            pct = float(row["Valor"].replace("%","").replace(",",".").replace("+",""))
            return f"({capital*(1+pct/100):,.0f}€)"
        except Exception:
            return ""

    st.dataframe(
        metrics_df,
        hide_index=True,
        use_container_width=True,
        height=320,
        column_config={
            "Métrica": st.column_config.TextColumn("Métrica", width="medium"),
            "Valor":   st.column_config.TextColumn("Valor",   width="small"),
        },
    )

# Panel 3: Análisis de sensibilidad VaR
with col_mc3:
    st.markdown("**Sensibilidad del VaR por Horizonte**")
    horizons_sens = [21, 63, 126, 252]
    horizon_labels = ["1 mes", "3 meses", "6 meses", "1 año"]

    @st.cache_data(show_spinner=False, ttl=900)
    def compute_sensitivity(tickers, start, end, weights_arr, cap, n_sims, conf):
        returns_s = run_pipeline(
            tickers, start, end, 0.0, 1.0, 0.0, n_sims, int(conf*100), cap
        )["returns"]
        records = []
        for h, hl in zip(horizons_sens, horizon_labels):
            mc_s = MonteCarlo(weights=weights_arr, returns=returns_s,
                               capital=cap, n_simulations=max(n_sims//3, 1000),
                               horizon_days=h, confidence=conf, seed=42)
            r = mc_s.run()
            records.append({
                "Horizonte":    hl,
                f"VaR {int(conf*100)}%": f"{r.var_historical*100:.1f}%",
                "CVaR":          f"{r.cvar_historical*100:.1f}%",
                "P(pérdida)":    f"{r.prob_loss*100:.1f}%",
            })
        return pd.DataFrame(records)

    try:
        df_sens = compute_sensitivity(
            tuple(selected_tickers), start_str, end_str,
            tuple(msr.weights.values.tolist()),
            capital, n_simulations, var_confidence/100,
        )
        st.dataframe(df_sens, hide_index=True,
                     use_container_width=True, height=320)
    except Exception:
        st.info("El análisis de sensibilidad requiere conexión. "
                "Inténtalo de nuevo.")

st.markdown('<hr class="athena">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# AVISO LEGAL
# ---------------------------------------------------------------------------
st.markdown("""
<div class="disclaimer">
  <b>⚠ AVISO LEGAL</b> — Esta herramienta tiene exclusivamente carácter educativo
  y no constituye asesoramiento financiero ni gestión de carteras en el sentido de la
  Directiva MiFID II (2014/65/UE). Los resultados se basan en datos históricos;
  las rentabilidades pasadas no garantizan resultados futuros. Herramienta no
  registrada como Empresa de Servicios de Inversión (ESI) ante la CNMV.
  Uso estrictamente educativo — Proyecto Athena · TFG UOC · 2026.
</div>
""", unsafe_allow_html=True)