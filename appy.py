import os
import pandas as pd
import streamlit as st
from prophet import Prophet
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import plotly.graph_objects as go

# ----------------------------------------
# CONFIGURACIÓN GENERAL
# ----------------------------------------
st.set_page_config(
    page_title="Forecast Financiero con Prophet",
    page_icon="📈",
    layout="wide"
)

# ----------------------------------------
# CARGAR DATOS
# ----------------------------------------
folder = r"C:\Users\estel\PycharmProjects\Forecast activos\indicadores_por_activo"

if not os.path.exists(folder):
    st.error("❌ No se encuentra la carpeta de datos 'indicadores_por_activo'.")
    st.stop()

dataframes = {}
archivos = [f for f in os.listdir(folder) if f.endswith(".csv")]
activos = []

for file in archivos:
    try:
        path = os.path.join(folder, file)
        df = pd.read_csv(path)
        if 'Date' in df.columns and 'Close' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            dataframes[os.path.splitext(file)[0]] = df
            activos.append(os.path.splitext(file)[0])
        else:
            st.warning(f"⚠️ {file} no tiene columnas 'Date' y 'Close'. Se omitirá.")
    except Exception as e:
        st.warning(f"⚠️ No se pudo procesar {file}: {e}")

if not activos:
    st.error("No se cargaron activos válidos.")
    st.stop()

# ----------------------------------------
# DIVIDIR ACTIVOS POR GRUPOS AUTOMÁTICAMENTE
# ----------------------------------------
acciones = [a for a in activos if a.upper() in ["MSFT", "NVDA", "AAPL"]]
criptos = [a for a in activos if a.upper() in ["BTC_USD", "ETH_USD"]]
commodities = [a for a in activos if a.upper() in ["GC_F", "SI_F", "HG_F"]]
etfs = [a for a in activos if a.upper() in ["SPY", "ICLN", "HYG"]]
bonos = [a for a in activos if a.upper() in ["TNX"]]

# ----------------------------------------
# PESTAÑAS PRINCIPALES
# ----------------------------------------
tabs = st.tabs(["🏁 Introducción", "📊 Exploración de Datos", "🔮 Modelo Prophet", "🧠 Conclusiones"])

# ----------------------------------------
# TAB 1: INTRODUCCIÓN
# ----------------------------------------
with tabs[0]:
    st.title("📈 Predicción de Activos Financieros con Prophet")
    st.markdown("""
    Este proyecto utiliza **Facebook Prophet** para generar pronósticos de precios de distintos activos financieros
    (acciones, criptomonedas, ETFs, bonos, etc.).  

    Prophet está diseñado para capturar:
    - Tendencias de largo plazo 📈  
    - Estacionalidades semanales/anuales 📆  
    - Comportamientos irregulares de series temporales 💹  

    ---  
    **Objetivos:**  
    1. Analizar la evolución histórica de los activos.  
    2. Entrenar un modelo Prophet para estimar su comportamiento futuro.  
    3. Evaluar la precisión de las predicciones mediante métricas de error.  
    """)

    st.header("📊 Activos organizados por grupo")
    if acciones: st.write("💼 Acciones:", ", ".join(acciones))
    if criptos: st.write("💰 Criptomonedas:", ", ".join(criptos))
    if commodities: st.write("⚙️ Commodities:", ", ".join(commodities))
    if etfs: st.write("📊 ETFs:", ", ".join(etfs))
    if bonos: st.write("💵 Bonos:", ", ".join(bonos))
    st.balloons()

# ----------------------------------------
# TAB 2: EXPLORACIÓN DE DATOS
# ----------------------------------------
with tabs[1]:
    st.header("📊 Exploración Comparativa de Datos")
    col_select = st.columns(2)
    with col_select[0]:
        activo_1 = st.selectbox("Selecciona el primer activo:", activos, key="activo_1")
    with col_select[1]:
        activo_2 = st.selectbox("Selecciona el segundo activo para comparar:", activos, key="activo_2")

    df1 = dataframes[activo_1].copy()
    df2 = dataframes[activo_2].copy()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"📈 {activo_1}")
        indicadores_1 = [c for c in df1.columns if c not in ['Date', 'Close']]
        seleccion_1 = st.multiselect("Indicadores a mostrar:", options=['Close'] + indicadores_1, default=['Close'], key="indicadores_1")
        fig1 = go.Figure()
        for col in seleccion_1: fig1.add_trace(go.Scatter(x=df1['Date'], y=df1[col], mode='lines', name=col))
        fig1.update_layout(title=f"Evolución de {activo_1}", xaxis_title="Fecha", yaxis_title="Valor", template="plotly_white")
        st.plotly_chart(fig1, use_container_width=True)
        with st.expander("Tabla completa"): st.dataframe(df1)

    with col2:
        st.subheader(f"📊 {activo_2}")
        indicadores_2 = [c for c in df2.columns if c not in ['Date', 'Close']]
        seleccion_2 = st.multiselect("Indicadores a mostrar:", options=['Close'] + indicadores_2, default=['Close'], key="indicadores_2")
        fig2 = go.Figure()
        for col in seleccion_2: fig2.add_trace(go.Scatter(x=df2['Date'], y=df2[col], mode='lines', name=col))
        fig2.update_layout(title=f"Evolución de {activo_2}", xaxis_title="Fecha", yaxis_title="Valor", template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)
        with st.expander("Tabla completa"): st.dataframe(df2)

# ----------------------------------------
# TAB 3: MODELO PROPHET
# ----------------------------------------
with tabs[2]:
    st.header("🔮 Comparativa entre dos modelos Prophet")

    if st.button("🚀 Cargar todos los modelos Prophet"):
        with st.spinner("⚙️ Entrenando Prophet para todos los activos..."):
            @st.cache_data(show_spinner=True)
            def entrenar_todos_los_modelos(dataframes, periodos=30):
                resultados = {}
                epsilon = 1e-8
                for activo, df in dataframes.items():
                    try:
                        dfp = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
                        model = Prophet()
                        model.fit(dfp)
                        future = model.make_future_dataframe(periods=periodos)
                        forecast = model.predict(future)

                        test = dfp.iloc[-30:]
                        pred_test = model.predict(test[['ds']])
                        y_true = test['y'].values
                        y_pred = pred_test['yhat'].values

                        mae = mean_absolute_error(y_true, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                        mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

                        resultados[activo] = {"modelo": model, "forecast": forecast, "MAE": mae, "RMSE": rmse, "MAPE": mape}
                    except Exception as e:
                        st.warning(f"⚠️ Error al entrenar {activo}: {e}")
                return resultados

            st.session_state["resultados_modelos"] = entrenar_todos_los_modelos(dataframes)
            st.success("✅ Todos los modelos Prophet han sido entrenados y almacenados.")

    resultados_modelos = st.session_state.get("resultados_modelos", {})

    def ejecutar_modelo(activo):
        res = resultados_modelos.get(activo)
        if not res: st.warning(f"No se encontraron resultados para {activo}."); return
        forecast = res["forecast"]; mae = res["MAE"]; rmse = res["RMSE"]; mape = res["MAPE"]
        forecast_futuro = forecast[forecast['ds'] >= pd.Timestamp(datetime.today())]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_futuro['ds'], y=forecast_futuro['yhat'], mode='lines', name=f"Pronóstico {activo}"))
        fig.add_trace(go.Scatter(x=forecast_futuro['ds'], y=forecast_futuro['yhat_upper'], mode='lines', line=dict(dash='dot'), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast_futuro['ds'], y=forecast_futuro['yhat_lower'], mode='lines', line=dict(dash='dot'), fill='tonexty', fillcolor='rgba(0,100,80,0.2)', showlegend=False))
        fig.update_layout(title=f"Pronóstico {activo}", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        metrics = pd.DataFrame({'MAE': [mae], 'RMSE': [rmse], 'MAPE (%)': [mape]})
        st.subheader("📊 Métricas del modelo")
        st.plotly_chart(go.Figure([go.Bar(x=metrics.columns, y=metrics.iloc[0], marker_color='crimson')]), use_container_width=True)
        st.dataframe(metrics.style.format("{:.3f}"))

    col1, col2 = st.columns(2)
    with col1:
        activo_a = st.selectbox("Selecciona activo A:", list(resultados_modelos.keys()), key="activoA")
        if "ejecutar_a" not in st.session_state: st.session_state["ejecutar_a"] = False
        if st.button("🚀 Ejecutar Modelo A"): st.session_state["ejecutar_a"] = True
        if st.session_state["ejecutar_a"]: ejecutar_modelo(activo_a)

    with col2:
        activo_b = st.selectbox("Selecciona activo B:", list(resultados_modelos.keys()), key="activoB")
        if "ejecutar_b" not in st.session_state: st.session_state["ejecutar_b"] = False
        if st.button("🚀 Ejecutar Modelo B"): st.session_state["ejecutar_b"] = True
        if st.session_state["ejecutar_b"]: ejecutar_modelo(activo_b)

# ----------------------------------------
# TAB 4: CONCLUSIONES AUTOMÁTICAS
# ----------------------------------------
with tabs[3]:
    st.title("🧠 Conclusiones Automáticas del Modelo Prophet")
    resultados = st.session_state.get("resultados_modelos", {})
    if not resultados: st.warning("⚠️ No se han entrenado modelos."); st.stop()

    st.markdown("### Selecciona los activos a mostrar conclusiones:")
    activos_seleccionados = []

    if acciones:
        activos_acciones = st.multiselect("Acciones", options=acciones, default=acciones)
        activos_seleccionados.extend(activos_acciones)
    if etfs:
        activos_etfs = st.multiselect("ETFs", options=etfs, default=etfs)
        activos_seleccionados.extend(activos_etfs)
    if criptos:
        activos_criptos = st.multiselect("Criptomonedas", options=criptos, default=criptos)
        activos_seleccionados.extend(activos_criptos)
    if commodities:
        activos_commodities = st.multiselect("Commodities", options=commodities, default=commodities)
        activos_seleccionados.extend(activos_commodities)
    if bonos:
        activos_bonos = st.multiselect("Bonos", options=bonos, default=bonos)
        activos_seleccionados.extend(activos_bonos)


    for activo in activos_seleccionados:
        res = resultados[activo]; df = dataframes[activo]; y = df['Close']
        std_rel = y.std() / y.mean(); max_rel_change = (y.diff().abs() / y).max(); mean_price = y.mean()

        if std_rel > 0.08 or max_rel_change > 0.15: grupo = "Criptomoneda"; recomendacion = "💰 Alta volatilidad detectada."
        elif mean_price < 50: grupo = "Commodity"; recomendacion = "⚙️ Precio bajo: sensible a shocks externos."
        else: grupo = "Acción/ETF o índice"; recomendacion = "💼 Precio estable y tendencia definida."

        mae = res["MAE"]; rmse = res["RMSE"]; mape = res["MAPE"]
        if mae < 5 and mape < 5: conclusion = "Excelente ajuste."; nivel = "✅ Bueno"
        elif mae < 10 and mape < 10: conclusion = "Buen ajuste."; nivel = "⚖️ Moderado"
        elif mae < 20 or mape < 20: conclusion = "Ajuste moderado."; nivel = "⚖️ Moderado"
        else: conclusion = "Baja precisión."; nivel = "❌ Malo"

        st.subheader(f"📊 {activo} ")
        st.metric("MAE", f"{mae:.3f}"); st.metric("RMSE", f"{rmse:.3f}"); st.metric("MAPE (%)", f"{mape:.2f}%")
        st.write(f"📌 Conclusión automática: {conclusion}")
        st.info(f"Nivel de confianza: {nivel}\n{recomendacion}")
