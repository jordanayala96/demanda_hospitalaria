###### GRUPO 10 ######

###### VISUALIZADOR DE DEMANDA HOSPITALARIA ######

###### Trabajo previo a la obtención de título de Magister ######

##### LIBRERÍAS #####

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

###### 1. Configuración de la página ######
st.set_page_config(page_title="Sistema de Alerta Hospitalaria", layout="wide")
st.title("🏥 Sistema de Alerta Temprana de Saturación Hospitalaria")
st.markdown("**Desarrollado con Arquitectura LSTM para el Distrito Metropolitano de Quito (2024)**")
st.markdown("---")

###### 2. Cargar los datos predichos ######
@st.cache_data
def cargar_datos():
    df = pd.read_csv("escenarios_predichos_2024.csv")
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    return df

df_escenarios = cargar_datos()

###### 3. Interfaz Lateral (Sidebar) para interacción ######
st.sidebar.header("Panel de Control")
fecha_seleccionada = st.sidebar.date_input(
    "Seleccione una fecha de evaluación:",
    min_value=df_escenarios['Fecha'].min().date(),
    max_value=df_escenarios['Fecha'].max().date(),
    value=df_escenarios['Fecha'].min().date()
)

# Filtrar el dato exacto del día seleccionado
dato_dia = df_escenarios[df_escenarios['Fecha'].dt.date == fecha_seleccionada]

###### 4. Panel de Métricas Superiores ######
if not dato_dia.empty:
    riesgo = dato_dia['Escenario_IA'].values[0]
    prediccion = int(dato_dia['Demanda_Predicha'].values[0])

    col1, col2, col3 = st.columns(3)
    col1.metric("📅 Fecha Seleccionada", str(fecha_seleccionada))
    col2.metric("🛏️ Demanda Predicha (Egresos)", prediccion)

    # Asignar color según el riesgo
    if riesgo == 'Operación Normal':
        st.success(f"🟢 Estado del Sistema: {riesgo}")
    elif riesgo == 'Alerta Amarilla':
        st.warning(f"🟠 Estado del Sistema: {riesgo} (Cerca del Límite)")
    else:
        st.error(f"🔴 Estado del Sistema: {riesgo} (Colapso)")

st.markdown("---")

###### 5. Gráfico Principal Interactivo ######
st.subheader("📈 Curva de Demanda y Umbrales")

fig, ax = plt.subplots(figsize=(12, 4))
sns.set_theme(style="whitegrid")

# Línea base de predicciones
ax.plot(df_escenarios['Fecha'], df_escenarios['Demanda_Predicha'], color='gray', alpha=0.5, label="Predicción LSTM")

# Destacar el día seleccionado
if not dato_dia.empty:
    ax.scatter(dato_dia['Fecha'], dato_dia['Demanda_Predicha'], color='blue', s=100, zorder=5, label='Día Evaluado')

# Umbrales
ax.axhline(748, color='#FF8C00', linestyle='--', label='Umbral Alerta (748)')
ax.axhline(790, color='#D62728', linestyle='--', label='Umbral Crítico (790)')

ax.set_xlabel("Fecha")
ax.set_ylabel("Ingresos Proyectados")
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.xticks(rotation=45)

st.pyplot(fig)

st.markdown("*Modelo de Deep Learning implementado para la optimización de recursos en salud pública.*")
