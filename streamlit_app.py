import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import datetime

# Configurar la página de Streamlit
st.set_page_config(layout='centered', initial_sidebar_state='expanded')

# Cargar datos desde los archivos
@st.cache_data
def load_data():
    data = pd.read_excel('Rendimiento_Imputado.xlsx')
    data['Fecha'] = pd.to_datetime(data['Fecha'])
    return data

@st.cache_data
def load_litros_data():
    data = pd.read_csv('LitrosFiltrada (1).csv')
    data['Registrado'] = pd.to_datetime(data['Registrado'])
    return data

data2 = load_data()
litros_data = load_litros_data()

# Lista de pinturas disponibles para predicción
available_paintings = [
    '0021-UNIVERSAL PRIMER',
    '0023-CLEAR EPOXICO P/ESPUMA',
    '0028-CLEAR P/ESPUMA',
    '0038-PRIMER 917',
    '0400-GRIS FONDO',
    '0404-GRIS FONDO MC',
    '0079-BECKRYPRIM 246',
    '0435-APOLLO GRAY KRYSTAL KOTE',
    '0601-VERDE PRIMSA',
]

# Selección de pintura
selected_painting = st.selectbox("Seleccione la pintura para predecir:", available_paintings)

selected_linea = ['Pintado 1 UNI']

# Definir el rango de selección de fechas
min_year = 2021
max_year = 2025

# Colocar los selectores de fechas en una sola fila
col1, col2, col3, col4 = st.columns(4)

with col1:
    start_year = st.selectbox('Año de inicio', list(range(min_year, max_year + 1)), index=0)
with col2:
    start_month = st.selectbox('Mes de inicio', list(range(1, 13)), index=0)
with col3:
    end_year = st.selectbox('Año de fin', list(range(min_year, max_year + 1)), index=3)  # Default 2024
with col4:
    end_month = st.selectbox('Mes de fin', list(range(1, 13)), index=7)  # Default August

start_date = pd.Timestamp(datetime.date(start_year, start_month, 1))
end_date = pd.Timestamp(datetime.date(end_year, end_month, 1) + pd.DateOffset(months=1) - pd.DateOffset(days=1))

def plot_forecast(data, date_col, value_col, title, ylabel, xlabel='Fecha'):
    # Crear la visualización con la serie temporal filtrada
    fig, ax = plt.subplots(figsize=(4, 3), dpi=80)  # Tamaño en pulgadas (más pequeño)

    # Configurar fondo negro, quitar gridlines y línea roja
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.plot(data[date_col], data[value_col], label='Datos Reales', color='red', marker='o', linewidth=1, markersize=4)
    ax.grid(False)  # Quitar gridlines
    # Rotar las etiquetas del eje x
    plt.xticks(rotation=90)

    # Si el rango de fechas incluye periodos futuros, realizar la predicción
    if end_date > data[date_col].max():
        series = data.set_index(date_col)[value_col]

        # Definir parámetros SARIMA (esto debe ser ajustado según tu modelo específico)
        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 12)
        steps = (end_date - data[date_col].max()).days // 30

        if steps > 0:
            try:
                # Ajustar el modelo SARIMA
                model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
                model_fit = model.fit(disp=False)

                # Predecir los próximos meses hasta el end_date
                forecast = model_fit.get_forecast(steps=steps)
                predicted_mean = forecast.predicted_mean
                pred_ci = forecast.conf_int()

                # Mostrar predicciones y los intervalos de confianza
                predicted_mean.index = pd.date_range(start=series.index[-1] + pd.DateOffset(months=1), periods=steps, freq='M')
                pred_ci.index = predicted_mean.index

                ax.plot(predicted_mean.index, predicted_mean, label='Predicción', color='yellow', marker='x', linestyle='--', linewidth=1, markersize=4)
                ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='gray', alpha=0.1)
            except Exception as e:
                st.error(f'Error al ajustar el modelo SARIMA: {e}')

    # Ajustar tamaños de las fuentes y colores
    ax.set_xlabel(xlabel, fontsize=4, color='white')
    ax.set_ylabel(ylabel, fontsize=4, color='white')
    ax.legend(fontsize=5, facecolor='black', edgecolor='white', labelcolor='white', markerscale=0.8, handletextpad=0.5)
    ax.tick_params(axis='both', which='major', labelsize=5, colors='white')

    plt.tight_layout()  # Ajusta el layout para evitar solapamiento

    return fig

# Filtrar los datos para la pintura seleccionada y el rango de fechas
filtered_data_all_lines = data2[(data2['Pintura'] == selected_painting) &
                                (data2['Línea'].isin(selected_linea)) &
                                (data2['Fecha'] >= start_date) &
                                (data2['Fecha'] <= end_date)]

filtered_litros_data = litros_data[(litros_data['Texto breve de material'] == selected_painting) &
                                   (litros_data['Línea'].isin(selected_linea)) &
                                   (litros_data['Registrado'] >= start_date) &
                                   (litros_data['Registrado'] <= end_date)]

if filtered_data_all_lines.empty:
    st.warning('No hay datos disponibles para el rango de fechas seleccionado.')
else:
    # Sumar los litros para la pintura seleccionada por cada mes
    summary_all_lines = filtered_data_all_lines.groupby('Fecha')['Rendimiento Litros/M2'].sum().reset_index()

    # Renombrar las columnas para mayor claridad
    summary_all_lines.columns = ['Fecha', 'Rendimiento']

    # Convertir la columna 'Fecha' a tipo datetime
    summary_all_lines['Fecha'] = pd.to_datetime(summary_all_lines['Fecha'])

    # Mostrar la figura en Streamlit con tamaño ajustado
    fig = plot_forecast(summary_all_lines, 'Fecha', 'Rendimiento', f'Predicción de Rendimiento de Pintura - {selected_painting}', 'Rendimiento')
    st.pyplot(fig, use_container_width=True)

# Filtrar los datos de litros para la gráfica de rendimiento
if filtered_litros_data.empty:
    st.warning('No hay datos disponibles para el rango de fechas seleccionado para el modelo de rendimiento.')
else:
    # Sumar el rendimiento para la pintura seleccionada por cada mes
    summary_litros_data = filtered_litros_data.groupby('Registrado')['Ctd.total reg.'].sum().reset_index()

    # Renombrar las columnas para mayor claridad
    summary_litros_data.columns = ['Fecha', 'Ctd_total_reg']

    # Convertir la columna 'Fecha' a tipo datetime
    summary_litros_data['Fecha'] = pd.to_datetime(summary_litros_data['Fecha'])

    # Mostrar la figura en Streamlit con tamaño ajustado
    fig2 = plot_forecast(summary_litros_data, 'Fecha', 'Ctd_total_reg', f'Predicción de Litros - {selected_painting}', 'Litros')
    st.pyplot(fig2, use_container_width=True)
