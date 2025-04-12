import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.font_manager import FontManager
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from mplsoccer import VerticalPitch
import matplotlib.patheffects as path_effects
import plotly.graph_objects as go
import os
import base64
import matplotlib.font_manager as fm
from matplotlib import rcParams
from mplsoccer import VerticalPitch, PyPizza
import img2pdf
from datetime import datetime
import matplotlib.colors as mcolors
import io
from PIL import Image as PILImage 
import gc
from scipy.ndimage import gaussian_filter
import matplotlib.patches as mpatches
import io
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as ReportLabImage, Spacer, Table, TableStyle
from pathlib import Path
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_LEFT, TA_CENTER
import matplotlib
matplotlib.use('Agg') 

# Debe ser lo primero después de los imports
st.set_page_config(
    page_title="Estadísticas de Jugadores",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None
)

# Health check endpoint
if 'healthz' in st.query_params:
    st.write('OK')
    st.stop()

# Configuración de caché y memoria
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'visualization_cache' not in st.session_state:
    st.session_state.visualization_cache = {}

def get_cached_visualization(player_id, season_ids):
    """
    Recupera una visualización del caché si existe.
    """
    try:
        cache_key = f"{player_id}_{'-'.join(map(str, season_ids))}"
        return st.session_state.visualization_cache.get(cache_key)
    except Exception as e:
        print(f"Error al recuperar del caché: {e}")
        return None

def clear_memory():
    """
    Limpia la memoria y el caché si es necesario.
    """
    try:
        if hasattr(st.session_state, 'visualization_cache'):
            if len(st.session_state.visualization_cache) > 5:  # Mantener solo las últimas 5
                oldest_key = next(iter(st.session_state.visualization_cache))
                del st.session_state.visualization_cache[oldest_key]
        plt.close('all')
        gc.collect()
    except Exception as e:
        print(f"Error al limpiar memoria: {e}")

# Base directory and asset paths setup
BASE_DIR = Path(__file__).parent.parent
icon_path = os.path.join(BASE_DIR, 'assets', 'icono_player.png')
logo_path = os.path.join(BASE_DIR, 'assets', 'escudo_alaves_original.png')
jugador_path = os.path.join(BASE_DIR, 'assets', 'jugador_alaves.png')
fondo_path = os.path.join(BASE_DIR, 'assets', 'fondo_alaves.png')
banner_path = os.path.join(BASE_DIR, 'assets', 'bunner_alaves.png')

# Global colors
BACKGROUND_COLOR = '#0E1117'
LINE_COLOR = '#FFFFFF'
TEXT_COLOR = '#FFFFFF'
HIGHLIGHT_COLOR = '#4BB3FD'

# CSS personalizado para fondo oscuro y texto blanco
st.markdown("""
    <style>
        .stApp {
            background-color: #0E1117;
        }
        .st-emotion-cache-16idsys {
            background-color: #0E1117;
        }
        .st-emotion-cache-1cypcdb {
            background-color: #262730;
        }
        /* Estilo para las etiquetas de métricas */
        .st-emotion-cache-q8sbsg p {
            color: white !important;
        }
        /* Estilo para los valores de métricas */
        .st-emotion-cache-q8sbsg p:first-child {
            color: white !important;
        }
        /* Estilo general para texto */
        .st-emotion-cache-10trblm {
            color: white !important;
        }
        div[data-testid="stMetricLabel"] > div {
            color: white !important;
        }
        div[data-testid="stMetricValue"] > div {
            color: white !important;
        }
        
        /* Estilos para el contenedor de gráficas */
        .chart-container {
            background-color: #1E1E1E;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .plot-title {
            color: white;
            font-size: 14px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 1rem;
            padding: 0.5rem;
        }
        
        /* Estilo para la pantalla de carga */
        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background-color: #0E1117;
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 9999;
        }

        .loading-content {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }

        .loading-logo {
            width: 200px;
            height: 200px;
            animation: pulse 2s infinite, glow 3s infinite alternate;
            border-radius: 50%; /* Hace el GIF circular */
            box-shadow: 0 0 20px rgba(75, 179, 253, 0.8); /* Brillo moderno */
        }

        .loading-text {
            color: white;
            font-size: 1.5rem;
            margin-top: 20px;
            text-align: center;
            animation: text-fade 2s infinite alternate;
        }

        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }

        @keyframes glow {
            0% { box-shadow: 0 0 20px rgba(75, 179, 253, 0.8); }
            50% { box-shadow: 0 0 40px rgba(75, 179, 253, 1); }
            100% { box-shadow: 0 0 20px rgba(75, 179, 253, 0.8); }
        }

        @keyframes text-fade {
            0% { opacity: 0.8; }
            100% { opacity: 1; }
        }

        /* Animación de difuminado al finalizar */
        @keyframes fadeOut {
            0% { opacity: 1; }
            100% { opacity: 0; }
        }

        .fade-out {
            animation: fadeOut 1s ease-out forwards;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def get_image_base64_optimized(path, max_size=(1024, 1024), quality=70):
    try:
        image = PILImage.open(path)  # Cambiado de Image.open a PILImage.open
        image.thumbnail(max_size, PILImage.LANCZOS)  # También aquí cambia Image.LANCZOS a PILImage.LANCZOS
        # Convertir la imagen a RGB si tiene canal alfa
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        st.error(f"Error cargando imagen: {e}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_image_assets():
    assets = {}
    for name, path in {
        'logo': logo_path,
        'icon': icon_path,
        'jugador': jugador_path,
        'fondo': fondo_path,
        'banner': banner_path
    }.items():
        try:
            with open(path, "rb") as f:
                assets[name] = base64.b64encode(f.read()).decode()
        except Exception as e:
            print(f"Error loading {name}: {e}")
            assets[name] = None
    return assets

def load_data():
    """
    Carga los datos iniciales de los archivos parquet
    """
    try:
        from data_manager import DataManager
        return DataManager.get_filtered_data()
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return None, None

def store_figure_as_buffer(fig, key, dpi=150):
    """ Convierte una figura de Matplotlib en un buffer PNG y la almacena en session_state.visualization_cache. """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    # Cierra la figura para liberar memoria
    plt.close(fig)
    st.session_state.visualization_cache[key] = buf
    return buf

def get_cached_figure_buffer(key):
    cached = st.session_state.visualization_cache.get(key)
    if cached and isinstance(cached, io.BytesIO):
        return cached
    return None

def get_processed_player_data(df_jugadores, player_id, season_ids):
    """
    Filtra los datos para el jugador y la temporada, y calcula las métricas agregadas.
    Retorna (df_jugador, df_season). Si no hay datos para el jugador, retorna (None, None).
    """
    # Filtrar datos para el jugador y para la temporada
    df_jugador = df_jugadores[
        (df_jugadores['player_id'] == player_id) & 
        (df_jugadores['season_id'].isin(season_ids))
    ].copy()
    
    df_season = df_jugadores[df_jugadores['season_id'].isin(season_ids)].copy()

    if df_jugador.empty:
        return None, None

    # Calcular métricas agregadas
    df_jugador = calcular_metricas_agregadas(df_jugador)
    df_season = calcular_metricas_agregadas(df_season)

    return df_jugador, df_season

def draw_circular_progress(ax, x, y, percentage, total_events, is_left=True, radius=0.8):
    # Fondo del círculo (gris claro)
    background_circle = patches.Circle((x, y), radius, color='lightgray', alpha=0.3)
    ax.add_patch(background_circle)
    
    # Ángulos para el arco de progreso
    start_angle = 90  # Comienza desde la parte superior
    end_angle = 90 - (360 * percentage / 100)  # Avanza en sentido horario
    
    # Crear el arco de progreso (ajustado para que sea un círculo perfecto)
    arc = patches.Arc((x, y), radius * 12, radius * 2, angle=0, 
                    theta1=end_angle, theta2=start_angle, color='#4169E1', lw=2)
    ax.add_patch(arc)
    
    # Rellenar el arco de progreso
    theta = np.linspace(np.deg2rad(start_angle), np.deg2rad(end_angle), 100)
    x_arc = x + radius * np.cos(theta)
    y_arc = y + radius * np.sin(theta)
    ax.fill(np.append(x_arc, x), np.append(y_arc, y), color='#4169E1', alpha=0.8)
    
    # Texto del porcentaje en el centro
    ax.text(x, y, f'{percentage:.0f}%', ha='center', va='center',
            fontsize=8, fontweight='bold', color='white')
    
    # Número total de eventos (en azul con fondo blanco)
    x_offset = -radius - 10 if is_left else radius + 10
    ax.text(x + x_offset, y, str(total_events), ha='center', va='center',
            fontsize=8, fontweight='bold', color='blue',
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))


def calcular_metricas_agregadas(df):
    """
    Calcula métricas agregadas asegurando la calidad de los datos.
    
    Args:
        df: DataFrame con los datos originales
        
    Returns:
        DataFrame con las métricas agregadas calculadas
    """
    # Verificar que tenemos datos
    if df is None or df.empty:
        print("ADVERTENCIA: DataFrame vacío o None en calcular_metricas_agregadas")
        return pd.DataFrame()
    
    # Copia para evitar modificar el original
    df = df.copy()
    
    # Definir columnas métricas
    columnas_metricas = [
        'duelos_aereos_ganados_zona_area', 'duelos_aereos_ganados_zona_baja', 
        'duelos_aereos_ganados_zona_media', 'duelos_aereos_ganados_zona_alta',
        'duelos_aereos_perdidos_zona_area', 'duelos_aereos_perdidos_zona_baja',
        'duelos_aereos_perdidos_zona_media', 'duelos_aereos_perdidos_zona_alta',
        'pases_largos_exitosos', 'cambios_orientacion_exitosos',
        'pases_largos_fallidos', 'cambios_orientacion_fallidos',
        'duelos_suelo_ganados_zona_area', 'entradas_ganadas_zona_area',
        'duelos_suelo_perdidos_zona_area', 'entradas_perdidas_zona_area',
        'pases_adelante_inicio', 'pases_adelante_creacion',
        'pases_horizontal_inicio', 'pases_horizontal_creacion',
        'pases_atras_inicio', 'pases_atras_creacion',
        'pases_exitosos_campo_propio', 'pases_fallidos_campo_propio',
        'pases_exitosos_campo_contrario', 'pases_fallidos_campo_contrario',
        'recuperaciones_zona_baja', 'recuperaciones_zona_media', 'recuperaciones_zona_alta',
        'entradas_ganadas_zona_baja', 'entradas_ganadas_zona_media', 'entradas_ganadas_zona_alta'
    ]
    
    # Verificar qué columnas existen realmente
    columnas_metricas_existentes = [col for col in columnas_metricas if col in df.columns]
    columnas_faltantes = [col for col in columnas_metricas if col not in df.columns]
    
    if columnas_faltantes:
        print(f"ADVERTENCIA: Faltan las siguientes columnas: {columnas_faltantes}")
    
    # Convertir a numérico todas las columnas métricas que existen y manejar valores no válidos
    for col in columnas_metricas_existentes:
        # Guardar valores originales para verificación
        orig_values = df[col].copy()
        
        # Convertir a numérico
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Contar valores que cambiaron (no eran numéricos)
        non_numeric_count = (pd.isna(df[col]) & ~pd.isna(orig_values)).sum()
        if non_numeric_count > 0:
            print(f"ADVERTENCIA: {non_numeric_count} valores no numéricos en columna '{col}'")
        
        # Contar valores NaN
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            print(f"ADVERTENCIA: {nan_count} valores NaN en columna '{col}'")
        
        # Rellenar NaN con 0
        df[col] = df[col].fillna(0)
        
        # Verificar valores negativos (podrían ser errores)
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            print(f"ADVERTENCIA: {neg_count} valores negativos en columna '{col}'")
            # Opcionalmente, corregir valores negativos si representan errores
            # df.loc[df[col] < 0, col] = 0
    
    # Definir las métricas a calcular y las columnas necesarias
    metricas_agregadas = {
        'duelos_aereos_ganados_total': [
            'duelos_aereos_ganados_zona_area', 
            'duelos_aereos_ganados_zona_baja', 
            'duelos_aereos_ganados_zona_media', 
            'duelos_aereos_ganados_zona_alta'
        ],
        'duelos_aereos_perdidos_total': [
            'duelos_aereos_perdidos_zona_area',
            'duelos_aereos_perdidos_zona_baja',
            'duelos_aereos_perdidos_zona_media',
            'duelos_aereos_perdidos_zona_alta'
        ],
        'pases_largos_cambios_exitosos': [
            'pases_largos_exitosos',
            'cambios_orientacion_exitosos'
        ],
        'pases_largos_cambios_fallidos': [
            'pases_largos_fallidos',
            'cambios_orientacion_fallidos'
        ],
        'duelos_area_ganados': [
            'duelos_aereos_ganados_zona_area',
            'duelos_suelo_ganados_zona_area',
            'entradas_ganadas_zona_area'
        ],
        'duelos_area_perdidos': [
            'duelos_aereos_perdidos_zona_area',
            'duelos_suelo_perdidos_zona_area',
            'entradas_perdidas_zona_area'
        ],
        'pases_adelante_inicio_creacion': [
            'pases_adelante_inicio',
            'pases_adelante_creacion'
        ],
        'pases_horizontal_inicio_creacion': [
            'pases_horizontal_inicio',
            'pases_horizontal_creacion'
        ],
        'recuperaciones_zona_baja_total': [
            'recuperaciones_zona_baja'
        ],
        'recuperaciones_zona_media_alta': [
            'recuperaciones_zona_media',
            'recuperaciones_zona_alta'
        ],
        'entradas_ganadas_zona_baja_total': [
            'entradas_ganadas_zona_baja'
        ],
        'entradas_ganadas_zona_media_alta': [
            'entradas_ganadas_zona_media',
            'entradas_ganadas_zona_alta'
        ]
    }
    
    # Calcular cada métrica agregada verificando que existan las columnas necesarias
    for metrica, columnas_necesarias in metricas_agregadas.items():
        # Verificar si todas las columnas necesarias existen
        columnas_disponibles = [col for col in columnas_necesarias if col in df.columns]
        
        if columnas_disponibles:  # Siempre que haya al menos una columna disponible
            # Mostrar advertencia si faltan algunas columnas
            if len(columnas_disponibles) < len(columnas_necesarias):
                columnas_faltantes = [col for col in columnas_necesarias if col not in df.columns]
                print(f"ADVERTENCIA: Para calcular '{metrica}' faltan las columnas: {columnas_faltantes}")
            
            # Realizar la suma de las columnas disponibles
            df[metrica] = df[columnas_disponibles].sum(axis=1)
            
            # Verificar resultados después de la agregación
            if df[metrica].isna().any():
                print(f"ADVERTENCIA: La métrica '{metrica}' tiene valores NaN después de agregar")
                df[metrica] = df[metrica].fillna(0)
        else:
            # Si no hay ninguna columna disponible, crear la columna con ceros
            print(f"ERROR: No hay columnas disponibles para calcular '{metrica}'. Se creará con valores 0.")
            df[metrica] = 0
    
    # Verificar y reportar valores negativos inesperados
    for col in metricas_agregadas.keys():
        if col in df.columns and (df[col] < 0).any():
            neg_count = (df[col] < 0).sum()
            print(f"ADVERTENCIA: {neg_count} valores negativos en la métrica '{col}'")
    
    return df

# Para gráfica de métricas avanzadas
def plot_player_metrics_modern(ax, df_jugadores, player_id, season_ids): 
    """
    Visualiza métricas de jugador con barras comparativas
    
    Args:
        ax: Axes de matplotlib donde dibujar
        df_jugadores: DataFrame con datos de jugadores
        player_id: ID del jugador a visualizar
        season_ids: Lista de IDs de temporadas a considerar
    """
    # Usar la función auxiliar para obtener datos procesados 
    df_jugador, df_season = get_processed_player_data(df_jugadores, player_id, season_ids) 
    
    # Verificar que tenemos datos válidos
    if df_jugador is None or len(df_jugador) == 0: 
        ax.text(0.5, 0.5, "No hay datos para mostrar", ha='center', va='center', fontsize=24) 
        return
    
    # Definir etiquetas y pares de métricas para graficar
    metric_labels = {
        'pases_largos_cambios_exitosos': 'Pases Largos Exitosos',
        'pases_largos_cambios_fallidos': 'Pases Largos Fallidos',
        'pases_adelante_inicio_creacion': 'Pases Adelante',
        'pases_horizontal_inicio_creacion': 'Pases Atrás/Horizontal',
        'pases_exitosos_campo_propio': 'Pases Propio Exitosos',
        'pases_fallidos_campo_propio': 'Pases Propio Fallidos',
        'pases_exitosos_campo_contrario': 'Pases Contrario Exitosos',
        'pases_fallidos_campo_contrario': 'Pases Contrario Fallidos',
        'duelos_aereos_ganados_total': 'Duelos Aéreos Ganados',
        'duelos_aereos_perdidos_total': 'Duelos Aéreos Perdidos',
        'duelos_area_ganados': 'Duelos Área Ganados',
        'duelos_area_perdidos': 'Duelos Área Perdidos',
        'recuperaciones_zona_media_alta': 'Recuperaciones Zona Media/Alta',
        'recuperaciones_zona_baja_total': 'Recuperaciones Zona Baja',
        'entradas_ganadas_zona_media_alta': 'Entradas Ganadas Zona Media/Alta',
        'entradas_ganadas_zona_baja_total': 'Entradas Ganadas Zona Baja'
    }

    # Los pares de métricas que se visualizarán enfrentados
    metric_pairs = [
        ('pases_largos_cambios_exitosos', 'pases_largos_cambios_fallidos'),
        ('pases_adelante_inicio_creacion', 'pases_horizontal_inicio_creacion'),
        ('pases_exitosos_campo_propio', 'pases_fallidos_campo_propio'),
        ('pases_exitosos_campo_contrario', 'pases_fallidos_campo_contrario'),
        ('duelos_aereos_ganados_total', 'duelos_aereos_perdidos_total'),
        ('duelos_area_ganados', 'duelos_area_perdidos'),
        ('recuperaciones_zona_media_alta', 'recuperaciones_zona_baja_total'),
        ('entradas_ganadas_zona_media_alta', 'entradas_ganadas_zona_baja_total')
    ]

    # Ajustado para dar más espacio vertical entre barras
    y_positions = np.arange(len(metric_pairs)) * 2.0
    bar_height = 1.0
    avg_bar_height = 0.4  # Altura más pequeña para la barra de promedio
    avg_bar_offset = -0.4  # Ajustado para mejor posicionamiento vertical

    def draw_fixed_width_bar(ax, y_pos, label, percentage, total_events, is_left=True, is_average=False):
        """
        Dibuja una barra de ancho fijo con la etiqueta y porcentaje
        
        Args:
            ax: Axes donde dibujar
            y_pos: Posición vertical
            label: Etiqueta a mostrar
            percentage: Porcentaje para la barra (0-100)
            total_events: Número total de eventos
            is_left: Si es True, barra a la izquierda, si es False, a la derecha
            is_average: Si es True, se dibuja como promedio (más pequeña)
        """
        # Verificar valores válidos para evitar errores
        if not isinstance(percentage, (int, float)) or np.isnan(percentage):
            print(f"ADVERTENCIA: Porcentaje no válido ({percentage}) para '{label}'")
            percentage = 0
        
        # Asegurar que el porcentaje está en el rango [0, 100]
        percentage = max(0, min(100, percentage))
        
        width = 100
        x_start = 0
        actual_y_pos = y_pos + (avg_bar_offset if is_average else 0)
        
        if not is_average:
            # Barra de fondo
            ax.barh(actual_y_pos, width if is_left else -width, height=bar_height, 
                    color='#E0E0E0', alpha=0.1)
            # Barra con el porcentaje
            fill_width = (width * percentage / 100) if is_left else -(width * percentage / 100)
            ax.barh(actual_y_pos, fill_width, height=bar_height, 
                    color='#2196F3' if is_left else '#F44336', alpha=0.99)
            # Texto de la etiqueta
            ax.text(x_start + (width/2) if is_left else x_start - (width/2), 
                    actual_y_pos, label, ha='center', va='center', color='white', 
                    fontsize=9, fontweight='bold')
            
            # Indicador circular de progreso
            x_circle = width + 15 if is_left else -width - 15
            draw_circular_progress(ax, x_circle, actual_y_pos, percentage, total_events, is_left)
        else:
            # Barra de promedio - aumentada la opacidad
            fill_width = (width * percentage / 100) if is_left else -(width * percentage / 100)
            ax.barh(actual_y_pos, fill_width, height=avg_bar_height, 
                    color='white', alpha=0.35)

    # Procesar cada par de métricas
    valid_pairs_count = 0  # Contador de pares válidos para ajustar ylim
    
    for i, (metric1, metric2) in enumerate(metric_pairs):
        # Verificar que ambas métricas existen
        if metric1 not in df_jugador.columns or metric2 not in df_jugador.columns:
            print(f"ADVERTENCIA: Par de métricas '{metric1}' y/o '{metric2}' no disponible")
            continue
        
        try:
            # Sumar los valores del jugador
            value1 = float(df_jugador[metric1].sum())
            value2 = float(df_jugador[metric2].sum())
            total = value1 + value2
            
            # Calcular valores promedio de la temporada
            avg_value1 = float(df_season[metric1].mean())
            avg_value2 = float(df_season[metric2].mean())
            avg_total = avg_value1 + avg_value2
            
            # Solo dibujar si hay datos válidos
            if total > 0:
                # Normalizar como porcentajes
                norm_value1 = value1 / total * 100
                norm_value2 = value2 / total * 100
                
                # Verificar que la suma es 100%
                if abs(norm_value1 + norm_value2 - 100) > 0.1:
                    print(f"  ADVERTENCIA: Suma de porcentajes = {norm_value1 + norm_value2}%, debería ser 100%")
                
                # Dibujar barras para el jugador
                draw_fixed_width_bar(ax, y_positions[i], metric_labels.get(metric1, metric1), 
                                    norm_value1, int(value1), True)
                draw_fixed_width_bar(ax, y_positions[i], metric_labels.get(metric2, metric2), 
                                    norm_value2, int(value2), False)
                
                # Dibujar barras promedio si hay datos
                if avg_total > 0:
                    avg_norm_value1 = avg_value1 / avg_total * 100
                    avg_norm_value2 = avg_value2 / avg_total * 100
                    
                    # Verificar suma de porcentajes promedio
                    if abs(avg_norm_value1 + avg_norm_value2 - 100) > 0.1:
                        print(f"  ADVERTENCIA: Suma de porcentajes promedio = {avg_norm_value1 + avg_norm_value2}%")
                    
                    draw_fixed_width_bar(ax, y_positions[i], "", avg_norm_value1, 
                                        int(avg_value1), True, True)
                    draw_fixed_width_bar(ax, y_positions[i], "", avg_norm_value2, 
                                        int(avg_value2), False, True)
                
                valid_pairs_count += 1
            else:
                print(f"  Omitiendo par {i+1} porque el total es 0")
        except Exception as e:
            print(f"ERROR procesando par {i+1} ({metric1}, {metric2}): {str(e)}")
    
    # Ajustar límites solo si hay datos válidos
    if valid_pairs_count > 0:
        # Ajustar límites para dar más espacio a los círculos
        ax.set_xlim(-140, 140)
        ax.set_ylim(min(y_positions[:valid_pairs_count]) - 1.8, max(y_positions[:valid_pairs_count]) + 1.8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axvline(x=0, color='white', alpha=0.3, linestyle='--', linewidth=1)
        
        for spine in ax.spines.values():
            spine.set_visible(False)
    else:
        # No hay datos válidos para mostrar
        ax.text(0.5, 0.5, "No hay métricas válidas para mostrar", 
               ha='center', va='center', fontsize=24)
        ax.set_xticks([])
        ax.set_yticks([])
        
        for spine in ax.spines.values():
            spine.set_visible(False)

# Función para la gráfica evolutiva del KPI
def create_kpi_evolution_chart(ax, df_estadisticas, player_id, season_ids):
    """
    Creates a bar chart for KPI_rendimiento across matchdays with colors based on player positions.
    """
    # Asegurarse de que player_id es float
    player_id = float(player_id)
    
    # Obtener los datos filtrados
    filtered_data = df_estadisticas[
        (df_estadisticas['player_id'] == player_id) &
        (df_estadisticas['season_id'].isin(season_ids))
    ].copy()
    
    # if not filtered_data.empty:
    #     print("\nTipo de datos de la columna jornada:")
    #     print(filtered_data['jornada'].dtype)
    #     print("\nPrimeras filas de jornada:")
    #     print(filtered_data['jornada'].head())
    
    try:
        # Convertir la columna jornada a string explícitamente
        filtered_data['jornada'] = filtered_data['jornada'].astype(str)
        
        # Intentar diferentes métodos de extracción del número
        if filtered_data['jornada'].iloc[0].startswith('Jornada'):
            # Si el formato es "Jornada X"
            filtered_data['jornada_num'] = filtered_data['jornada'].str.split().str[1].astype(float)
        else:
            # Si es solo el número
            filtered_data['jornada_num'] = pd.to_numeric(filtered_data['jornada'], errors='coerce')
        
        # Convertir KPI_rendimiento a numérico
        filtered_data['KPI_rendimiento'] = pd.to_numeric(filtered_data['KPI_rendimiento'], errors='coerce')
        
        # Eliminar filas con valores NaN
        filtered_data = filtered_data.dropna(subset=['jornada_num', 'KPI_rendimiento'])
        
        # print("\nDatos después de procesar:")
        # print(filtered_data[['jornada', 'jornada_num', 'KPI_rendimiento']])
        
        if not filtered_data.empty:
            # Ordenar por número de jornada
            filtered_data = filtered_data.sort_values('jornada_num')
            
            # Generar paleta de colores para posiciones
            unique_positions = filtered_data['demarcacion'].unique()
            
            # Paleta de colores claros
            light_colors = [
                mcolors.to_rgba('#B0E0E6', alpha=0.7),  # Pale turquoise
                mcolors.to_rgba('#98FB98', alpha=0.7),  # Pale green
                mcolors.to_rgba('#FFA07A', alpha=0.7),  # Light salmon
                mcolors.to_rgba('#87CEFA', alpha=0.7),  # Light sky blue
                mcolors.to_rgba('#DDA0DD', alpha=0.7),  # Plum
                mcolors.to_rgba('#F0E68C', alpha=0.7),  # Khaki
                mcolors.to_rgba('#E6E6FA', alpha=0.7),  # Lavender
                mcolors.to_rgba('#FFE4E1', alpha=0.7),  # Misty rose
                mcolors.to_rgba('#00CED1', alpha=0.7),  # Dark turquoise
                mcolors.to_rgba('#98FB98', alpha=0.7),  # Light green
            ]
            
            # Si más posiciones que colores predefinidos, generar más
            if len(unique_positions) > len(light_colors):
                additional_colors = [mcolors.to_rgba(plt.cm.Pastel1(np.random.rand()), alpha=0.7) for _ in range(len(unique_positions) - len(light_colors))]
                light_colors.extend(additional_colors)
            
            # Diccionario de mapeo de posiciones a colores
            position_color_map = dict(zip(unique_positions, light_colors[:len(unique_positions)]))
            
            # Crear el gráfico de barras con colores por posición
            bars = []
            for pos in unique_positions:
                pos_data = filtered_data[filtered_data['demarcacion'] == pos]
                pos_bars = ax.bar(
                    pos_data['jornada_num'], 
                    pos_data['KPI_rendimiento'],
                    color=position_color_map[pos], 
                    width=0.8,
                    label=pos
                )
                bars.extend(pos_bars)
            
            # Añadir etiquetas sobre las barras
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2., 
                    height, 
                    f'{height:.1f}',
                    ha='center', 
                    va='bottom', 
                    color='white', 
                    fontsize=8
                )
    
    except Exception as e:
        print(f"Error procesando los datos: {e}")
        return
    
    # Personalizar el gráfico
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.grid(True, color='white', alpha=0.2, linestyle='--')
    ax.set_xlabel('Jornada', color='white', fontsize=12)
    ax.set_ylabel('KPI Rendimiento', color='white', fontsize=12)
    ax.tick_params(colors='white')
    
    # Configurar el eje X para mostrar todas las jornadas del 1 al 38
    ax.set_xticks(range(1, 39, 2))
    ax.set_xlim(0.5, 38.5)
    
    # Configurar el eje Y
    if not filtered_data.empty:
        y_max = max(10, filtered_data['KPI_rendimiento'].max() * 1.1)
    else:
        y_max = 10
    ax.set_ylim(0, y_max)
    
    # Personalizar los bordes
    for spine in ax.spines.values():
        spine.set_color('white')
        spine.set_alpha(0.3)
    
    # Añadir leyenda de posiciones
    ax.legend(title='Posición', title_fontsize=10, loc='best', facecolor=BACKGROUND_COLOR, edgecolor='white', labelcolor='white')
    
    # Añadir padding para las etiquetas
    ax.margins(y=0.1)

# Función para la Pizza de Kpi jugador
def create_pizza_chart(ax, df_estadisticas, player_id, season_ids):
    player_id = float(player_id)
    
    # Get player data
    player_data = df_estadisticas[
        (df_estadisticas['player_id'] == player_id) &
        (df_estadisticas['season_id'].isin(season_ids))
    ]
    
    if player_data.empty:
        print("No hay datos para el jugador")
        return
    
    # Get player's position and league
    player_position = player_data['demarcacion'].iloc[0]
    player_league = player_data['liga'].iloc[0]
    
    # Encontrar el mejor jugador de la misma demarcación y liga basado en KPI_rendimiento
    df_filtrado = df_estadisticas[
        (df_estadisticas['demarcacion'] == player_position) &
        (df_estadisticas['liga'] == player_league) &
        (df_estadisticas['season_id'].isin(season_ids)) &
        (df_estadisticas['player_id'] != player_id)  # Excluimos al jugador actual
    ].copy()
    
    # Calcular el KPI promedio para cada jugador sin usar groupby
    kpi_promedios = []
    jugadores_unicos = df_filtrado['player_id'].unique()
    
    for pid in jugadores_unicos:
        datos_jugador = df_filtrado[df_filtrado['player_id'] == pid]
        kpi_promedio = datos_jugador['KPI_rendimiento'].mean()
        nombre_jugador = datos_jugador['jugador'].iloc[0]
        kpi_promedios.append({
            'player_id': pid,
            'jugador': nombre_jugador,
            'KPI_promedio': kpi_promedio
        })
    
    # Convertir a DataFrame y encontrar el mejor
    df_promedios = pd.DataFrame(kpi_promedios)
    mejor_jugador = df_promedios.loc[df_promedios['KPI_promedio'].idxmax()]
    
    # Obtener los datos completos del mejor jugador
    best_player_data = df_estadisticas[
        (df_estadisticas['player_id'] == mejor_jugador['player_id']) &
        (df_estadisticas['season_id'].isin(season_ids))
    ]
    
    params = [
        "Construcción\nde Ataque", 
        "Progresión", 
        "Habilidad\nIndividual",
        "Peligro\nGenerado", 
        "Finalización", 
        "Eficacia\nDefensiva",
        "Juego\nAéreo", 
        "Capacidad de\nRecuperación", 
        "Posicionamiento\nTáctico", 
        "Rendimiento"
    ]
    
    kpi_columns = [
        'KPI_construccion_ataque', 'KPI_progresion', 'KPI_habilidad_individual',
        'KPI_peligro_generado', 'KPI_finalizacion', 'KPI_eficacia_defensiva',
        'KPI_juego_aereo', 'KPI_capacidad_recuperacion', 'KPI_posicionamiento_tactico',
        'KPI_rendimiento'
    ]
    
    def calculate_player_values(data):
        values = []
        partidos_jugados = data['match_id'].nunique()
        
        for col in kpi_columns:
            if col in data.columns and partidos_jugados > 0:
                kpi_values = data[col]
                mean_val = kpi_values.mean()
                normalized_val = (mean_val) if mean_val <= 100 else 10
                normalized_val = max(0, min(10, normalized_val))
                normalized_val = round(normalized_val, 2)
                values.append(normalized_val)
            else:
                values.append(0)
        return values
    
    # Calculate values for both players
    player_values = calculate_player_values(player_data)
    best_player_values = calculate_player_values(best_player_data)
    
            # Get player names for title
    player_name = player_data['jugador'].iloc[0]
    best_player_name = best_player_data['jugador'].iloc[0]
    
    baker = PyPizza(
        params=params,
        straight_line_color="#FFFFFF",
        straight_line_lw=1,
        last_circle_lw=1,
        other_circle_lw=1,
        other_circle_ls="-.",
        background_color=BACKGROUND_COLOR,
        straight_line_limit=10
    )
    
    try:
        # Draw the pizza with comparison
        baker.make_pizza(
            player_values,
            compare_values=best_player_values,
            ax=ax,
            param_location=10,
            kwargs_slices=dict(
                facecolor="#4BB3FD",  # Azul para el jugador filtrado
                edgecolor="#FFFFFF",
                zorder=3,  # Mayor zorder para estar por encima
                linewidth=1,
                alpha=0.7
            ),
            kwargs_compare=dict(
                facecolor="#FFFFFF",  # Blanco para el mejor jugador
                edgecolor="#000000",  # Borde negro para mejor contraste
                zorder=2,  # Menor zorder para estar por detrás
                linewidth=1,
                alpha=0.7
            ),
            kwargs_params=dict(
                color="#FFFFFF",
                fontsize=16,
                va="center",
                zorder=1
            ),
            kwargs_values=dict(
                color="#FFFFFF",
                fontsize=15,
                zorder=5,
                va='bottom',  # Valores del jugador filtrado arriba
                bbox=dict(
                    edgecolor="#FFFFFF",
                    facecolor="#4BB3FD",
                    boxstyle="round,pad=0.2",
                    lw=1,
                    alpha=0.7
                )
            ),
            kwargs_compare_values=dict(
                color="#000000",  # Color negro para los valores del mejor jugador
                fontsize=12,
                zorder=4,
                va='top',  # Valores del mejor jugador abajo
                bbox=dict(
                    edgecolor="#000000",
                    facecolor="#FFFFFF",
                    boxstyle="round,pad=0.2",
                    lw=1,
                    alpha=0.7
                )
            )
        )

        # Obtener el equipo del mejor jugador
        best_player_team = best_player_data['equipo'].iloc[0]

        # Añadir subtítulo con colores personalizados
        ax.text(0.5, 1.18, player_name,
            color='#4BB3FD', fontsize=14, fontweight='bold',
            ha='center', va='center', transform=ax.transAxes)
        ax.text(0.5, 1.13, 'vs',
            color='white', fontsize=12,
            ha='center', va='center', transform=ax.transAxes)
        ax.text(0.5, 1.08, f"{best_player_name} ({best_player_team})",
            color='white', fontsize=14, fontweight='bold',
            ha='center', va='center', transform=ax.transAxes)
        
    except Exception as e:
        print(f"Error al crear el gráfico de pizza: {e}")

# Función para mapa de calor con tamaño fijo independiente de demarcaciones
def create_heatmap(ax, df_jugadores, player_id, season_ids, pitch):
    df_acciones = df_jugadores[
        (df_jugadores['player_id'] == player_id) &
        (df_jugadores['season_id'].isin(season_ids))
    ].copy()
    
    for col in ['xstart', 'ystart']:
        df_acciones[col] = pd.to_numeric(df_acciones[col], errors='coerce')
    
    if not df_acciones.empty:
        # Configuración fija para el campo
        pitch.draw(ax=ax)
        
        # Mapa de calor estándar
        bin_statistic = pitch.bin_statistic(
            df_acciones['ystart'].astype(float),
            df_acciones['xstart'].astype(float),
            statistic='count',
            bins=(15, 15)  # Reducir para que sea más limpio
        )
        
        bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
        
        heatmap_cmap = LinearSegmentedColormap.from_list(
            "custom_heatmap",
            [BACKGROUND_COLOR, HIGHLIGHT_COLOR]
        )
        
        pitch.heatmap(
            bin_statistic,
            ax=ax,
            cmap=heatmap_cmap,
            alpha=0.6
        )
        
        # Obtener todas las demarcaciones que juega el jugador
        demarcaciones = df_acciones['demarcacion'].unique()
        
        # Diccionarios estándar para colores y abreviaturas
        demarcacion_colors = {
            'Portero': '#FF5733',
            'Lateral Derecho': '#33FF57',
            'Lateral Izquierdo': '#3357FF',
            'Defensa Central': '#F3FF33',
            'Mediocentro Defensivo': '#FF33F3',
            'Mediocentro': '#33FFF3',
            'Interior Derecho': '#FF8333',
            'Interior Izquierdo': '#8333FF',
            'Mediapunta': '#FF3383',
            'Extremo Derecho': '#33FF83',
            'Extremo Izquierdo': '#3383FF',
            'Delantero Centro': '#FFFFFF',
            'Delantero Izquierdo': '#FF33A2',
            'Delantero Derecho': '#A2FF33'
        }
        
        abreviaturas = {
            'Portero': 'POR', 'Lateral Derecho': 'LD', 'Lateral Izquierdo': 'LI',
            'Defensa Central': 'CT', 'Mediocentro Defensivo': 'MCD',
            'Mediocentro': 'MC', 'Interior Derecho': 'ID',
            'Interior Izquierdo': 'II', 'Mediapunta': 'MP',
            'Extremo Derecho': 'ED', 'Extremo Izquierdo': 'EI',
            'Delantero Centro': 'DC', 'Delantero Izquierdo': 'DI', 'Delantero Derecho': 'DD'
        }
        
        # Procesar cada demarcación - limitando a máximo 4 para mostrar en el gráfico
        # Ordenamos por porcentaje para mostrar las más importantes
        demarcacion_stats = []
        
        for demarcacion in demarcaciones:
            if pd.notna(demarcacion) and demarcacion.strip() != '':
                df_filtered = df_acciones[df_acciones['demarcacion'] == demarcacion]
                
                if not df_filtered.empty:
                    porcentaje = (len(df_filtered) / len(df_acciones)) * 100
                    pos_media = df_filtered.agg({
                        'ystart': 'mean',
                        'xstart': 'mean'
                    })
                    
                    demarcacion_stats.append({
                        'demarcacion': demarcacion,
                        'porcentaje': porcentaje,
                        'pos_media': pos_media,
                        'color': demarcacion_colors.get(demarcacion, 'white'),
                        'abreviatura': abreviaturas.get(demarcacion, demarcacion[:2])
                    })
        
        # Ordenar por porcentaje y tomar las top 4
        demarcacion_stats = sorted(demarcacion_stats, key=lambda x: x['porcentaje'], reverse=True)[:4]
        
        # Dibujar solo las top demarcaciones con tamaño fijo
        legend_elements = []
        for stats in demarcacion_stats:
            # Dibujar punto para la posición media
            ax.plot(
                stats['pos_media']['xstart'],
                stats['pos_media']['ystart'],
                'o',
                color=stats['color'],
                markersize=4,  # Tamaño fijo más pequeño
                zorder=4,
                markeredgecolor='white',
                markeredgewidth=0.3
            )
            
            # Añadir texto con la abreviatura
            ax.text(
                stats['pos_media']['xstart'],
                stats['pos_media']['ystart'] - 5,
                f"{stats['abreviatura']}",
                color='white',
                ha='center',
                va='bottom',
                fontsize=5,
                fontweight='bold',
                zorder=9,
                bbox=dict(facecolor=BACKGROUND_COLOR, alpha=0.3, pad=0.5, boxstyle='round')
            )
            
            # Añadir texto con el porcentaje
            ax.text(
                stats['pos_media']['xstart'],
                stats['pos_media']['ystart'] + 4,
                f"{stats['porcentaje']:.0f}%",
                color='white',
                ha='center',
                va='top',
                fontsize=4,
                zorder=9
            )

# Función para el panel de información del jugador
def create_player_info_panel(df_jugadores, df_estadisticas, player_id, season_ids):
    """
    Crea un panel informativo con datos del jugador: equipos, demarcaciones,
    KPI por posición, goles, asistencias y tarjetas.
    """
    
    # Convertir player_id a float
    player_id = float(player_id)
    
    # Filtrar datos para el jugador seleccionado
    df_jugador = df_jugadores[
        (df_jugadores['player_id'] == player_id) & 
        (df_jugadores['season_id'].isin(season_ids))
    ].copy() if df_jugadores is not None else pd.DataFrame()
    
    df_stats = df_estadisticas[
        (df_estadisticas['player_id'] == player_id) & 
        (df_estadisticas['season_id'].isin(season_ids))
    ].copy() if df_estadisticas is not None else pd.DataFrame()
    
    # Verificar si hay datos suficientes
    if df_jugador.empty and df_stats.empty:
        st.warning("No hay datos para este jugador en las temporadas seleccionadas.")
        return
    
    # Iniciar el panel visual
    st.markdown("""
    <div class="chart-container" style="padding: 1.5rem;">
        <div class="plot-title" style="font-size: 16px; margin-bottom: 1rem;">Información del Jugador</div>
    """, unsafe_allow_html=True)
    
    # Fila 1: Información general con solo dos columnas
    cols = st.columns(2)
    
    # Columna 1: Equipos y demarcación
    with cols[0]:
        # Estilo mejorado para los títulos de sección
        st.markdown("""
        <h4 style='color: white; font-size: 14px; 
                   background-color: rgba(75, 179, 253, 0.3); 
                   padding: 6px 10px; 
                   border-left: 3px solid #4BB3FD;
                   border-radius: 3px;'>
            Equipos
        </h4>
        """, unsafe_allow_html=True)
        
        if not df_jugador.empty:
            # Si tenemos datos en df_jugador, usarlos para equipos
            equipos = df_jugador['equipo'].unique()
            for equipo in equipos:
                st.markdown(f"<div style='margin-bottom: 5px; color: white; padding-left: 10px;'>• {equipo}</div>", unsafe_allow_html=True)
        elif not df_stats.empty:
            # Si solo tenemos datos en df_stats, usar esa información
            equipos = df_stats['equipo'].unique() if 'equipo' in df_stats.columns else []
            for equipo in equipos:
                st.markdown(f"<div style='margin-bottom: 5px; color: white; padding-left: 10px;'>• {equipo}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='margin-bottom: 5px; color: white; padding-left: 10px;'>Sin datos de equipos</div>", unsafe_allow_html=True)
        
        # Listado de demarcaciones con media KPI
        if not df_stats.empty and 'demarcacion' in df_stats.columns and 'KPI_rendimiento' in df_stats.columns:
            # Convertimos KPI_rendimiento a numérico en caso de que no lo sea
            df_stats['KPI_rendimiento'] = pd.to_numeric(df_stats['KPI_rendimiento'], errors='coerce')
            
            # Agrupar por demarcación y calcular la media
            kpi_por_demarcacion = df_stats.groupby('demarcacion')['KPI_rendimiento'].mean().reset_index()
            
            # Filtrar valores NaN
            kpi_por_demarcacion = kpi_por_demarcacion[~pd.isna(kpi_por_demarcacion['KPI_rendimiento'])]
            
            if not kpi_por_demarcacion.empty:
                # Ordenar de mayor a menor KPI
                kpi_por_demarcacion = kpi_por_demarcacion.sort_values('KPI_rendimiento', ascending=False)
                
                # Estilo mejorado para el título de KPI
                st.markdown("""
                <h4 style='color: white; 
                           font-size: 14px; 
                           margin-top: 20px; 
                           background-color: rgba(255, 87, 51, 0.3); 
                           padding: 6px 10px; 
                           border-left: 3px solid #FF5733;
                           border-radius: 3px;'>
                    KPI por Demarcación
                </h4>
                """, unsafe_allow_html=True)
                
                for _, row in kpi_por_demarcacion.iterrows():
                    demarcacion = row['demarcacion']
                    kpi = row['KPI_rendimiento']
                    
                    # Color basado en el valor del KPI (verde más intenso para valores más altos)
                    kpi_color = f"rgba(0, {min(255, int(kpi * 25))}, 0, 0.8)"
                    
                    # Modificado para acercar el indicador de KPI al texto
                    st.markdown(
                        f"""
                        <div style='margin-bottom: 8px; display: flex; align-items: center; padding-left: 10px;'>
                            <span style='color: white;'>{demarcacion}</span>
                            <span style='color: white; 
                                   background-color: {kpi_color}; 
                                   padding: 2px 6px; 
                                   border-radius: 2px; 
                                   font-weight: bold; 
                                   margin-left: 5px; 
                                   min-width: 36px; 
                                   text-align: center;'>
                                {kpi:.1f}
                            </span>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
    
    # Columna 2: Estadísticas básicas
    with cols[1]:
        # Estilo mejorado para el título de estadísticas
        st.markdown("""
        <h4 style='color: white; 
                   font-size: 14px; 
                   background-color: rgba(108, 117, 125, 0.3); 
                   padding: 6px 10px; 
                   border-left: 3px solid #6C757D;
                   border-radius: 3px;'>
            Estadísticas
        </h4>
        """, unsafe_allow_html=True)
        
        if not df_stats.empty:
            # Calcular minutos por partido
            minutos_totales = 0
            minutos_por_partido = 0
            partidos_jugados = 0
            
            if 'minutos_jugados' in df_stats.columns:
                minutos_jugados = df_stats['minutos_jugados']
                minutos_jugados = pd.to_numeric(minutos_jugados, errors='coerce').fillna(0)
                minutos_totales = int(minutos_jugados.sum())
                
                # Contar partidos jugados (cada fila es un partido)
                partidos_jugados = len(df_stats)
                
                # Calcular minutos por partido
                minutos_por_partido = minutos_totales / partidos_jugados if partidos_jugados > 0 else 0
                minutos_por_partido = round(minutos_por_partido, 1)
            
            estadisticas = {
                'Goles': df_stats['goles'].sum() if 'goles' in df_stats.columns else 'N/A',
                'Asistencias': df_stats['asistencias'].sum() if 'asistencias' in df_stats.columns else 'N/A',
                'Tarjetas Amarillas': df_stats['tarjetas_amarillas'].sum() if 'tarjetas_amarillas' in df_stats.columns else 'N/A',
                'Tarjetas Rojas': df_stats['tarjetas_rojas'].sum() if 'tarjetas_rojas' in df_stats.columns else 'N/A',
                'Minutos Jugados': minutos_totales,
                'Minutos por Partido': minutos_por_partido,
                'Partidos Jugados': partidos_jugados
            }
            
            # Estilo mejorado para las estadísticas
            for stat, value in estadisticas.items():
                if value == 'N/A':
                    st.markdown(f"""
                    <div style='margin-bottom: 8px; color: white; padding-left: 10px; display: flex; justify-content: space-between;'>
                        <span><b>{stat}:</b></span>
                        <span>{value}</span>
                    </div>
                    """, unsafe_allow_html=True)
                elif stat == 'Minutos Jugados' and isinstance(value, (int, float)) and value > 0:
                    # Mostrar solo los minutos
                    st.markdown(f"""
                    <div style='margin-bottom: 8px; color: white; padding-left: 10px; display: flex; justify-content: space-between;'>
                        <span><b>{stat}:</b></span>
                        <span>{value} min</span>
                    </div>
                    """, unsafe_allow_html=True)
                elif stat == 'Minutos por Partido':
                    st.markdown(f"""
                    <div style='margin-bottom: 8px; color: white; padding-left: 10px; display: flex; justify-content: space-between;'>
                        <span><b>{stat}:</b></span>
                        <span>{value:.1f} min</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='margin-bottom: 8px; color: white; padding-left: 10px; display: flex; justify-content: space-between;'>
                        <span><b>{stat}:</b></span>
                        <span>{value if isinstance(value, str) else value:.0f}</span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("<div style='margin-bottom: 5px; color: white; padding-left: 10px;'>Sin datos estadísticos</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Definir colores modernos para la visualización
BACKGROUND_COLOR = '#1E1E1E'  # Fondo oscuro moderno
COLOR_LOCAL = '#2ECC71'       # Verde brillante para local
COLOR_VISITANTE = '#E74C3C'   # Rojo brillante para visitante
COLOR_GENERAL = '#3498DB'     # Azul para la línea general

def create_positive_actions_chart(df_jugador, ax=None):
    """
    Crea una gráfica que muestra las acciones positivas por tramos de 5 minutos,
    con líneas separadas para partidos de local y visitante.
    
    Acciones positivas:
    - Pases exitosos - fallidos
    - Duelos exitosos - fallidos
    - Suma de eventos positivos (Intercepción, Parada, etc.)
    
    Parámetros:
    - df_jugador: DataFrame con los datos del jugador
    - ax: Axes de matplotlib donde dibujar (opcional)
    
    Retorna:
    - Figura de matplotlib
    """
    # Crear figura si no se proporciona un eje
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12), facecolor=BACKGROUND_COLOR)
        ax.set_facecolor(BACKGROUND_COLOR)
    else:
        fig = ax.figure
    
    # Verificar columnas necesarias
    required_cols = ['event_time', 'periodo', 'tipo_evento', 'resultado', 'partido', 'equipo']
    missing_cols = [col for col in required_cols if col not in df_jugador.columns]
    
    if missing_cols:
        ax.text(0.5, 0.5, 
                f"Faltan columnas: {', '.join(missing_cols)}", 
                ha='center', va='center', color='white', fontsize=12)
        return fig
    
    # Preparar datos
    df_chart = df_jugador.copy()
    
    # Extraer el nombre del equipo para identificar local/visitante
    equipo_actual = df_chart['equipo'].iloc[0] if not df_chart.empty else ""
    
    # Determinar si es local o visitante basado en la columna partido
    df_chart['es_local'] = df_chart['partido'].apply(
        lambda x: equipo_actual in str(x).split('-')[0].strip() if not pd.isna(x) else False
    )
    
    df_chart['local_visitante'] = df_chart['es_local'].apply(
        lambda x: 'Local' if x == True else 'Visitante'
    )
    
    # Función para procesar minutos según el período
    def procesar_tiempo_minutos(tiempo_str, periodo_str):
        try:
            if pd.isna(tiempo_str) or pd.isna(periodo_str):
                return 30 if 'a_parte' in str(periodo_str).lower() else 75  # Valor predeterminado
            
            # Convertir a string y limpiar
            tiempo_str = str(tiempo_str).strip()
            
            # Manejar formatos especiales
            if "'" in tiempo_str:
                tiempo_str = tiempo_str.replace("'", "")
            
            # Manejar tiempo añadido (45+2, etc.)
            if "+" in tiempo_str:
                tiempo_str = tiempo_str.split("+")[0]
            
            # Extraer minutos (ignorando segundos)
            minutos = None
            if ':' in tiempo_str:
                # Formato MM:SS - extraemos solo minutos
                minutos = int(tiempo_str.split(':')[0])
            else:
                # Intentar convertir directamente
                try:
                    minutos = float(tiempo_str)
                except ValueError:
                    return 30 if 'a_parte' in str(periodo_str).lower() else 75  # Valor predeterminado
            
            # Verificar si es un número válido
            if minutos is None or np.isnan(minutos):
                return 30 if 'a_parte' in str(periodo_str).lower() else 75  # Valor predeterminado
            
            # Convertir a entero
            minutos = int(minutos)
            
            # Ajustar según el periodo
            es_segunda_parte = any(p in str(periodo_str).lower() for p in ['2a', 'second', '2'])
            
            if es_segunda_parte:
                # Si es segunda parte, sumamos 45 minutos
                return minutos
            else:
                # Si es primera parte, lo dejamos tal cual
                return minutos
            
        except Exception:
            return 30 if 'a_parte' in str(periodo_str).lower() else 75  # Valor predeterminado
    
    # Aplicar la función mejorada para obtener minutos
    df_chart['minuto'] = df_chart.apply(
        lambda row: procesar_tiempo_minutos(row['event_time'], row['periodo']), axis=1
    )
    
    # Función para asignar tramos de 5 minutos con caso especial para tiempos añadidos
    def asignar_tramo_5min(minuto):
        if pd.isna(minuto):
            return "0-5"  # Valor predeterminado
        
        # Casos especiales para tiempos añadidos
        if 40 <= minuto < 45:
            return '40-45'  # Tiempo añadido primera parte
        elif minuto >= 85:
            return '85-90'  # Tiempo añadido segunda parte
        
        # Para el resto de minutos, asignar tramos normales de 5 minutos
        inicio = (minuto // 5) * 5
        fin = inicio + 5
        
        # Crear etiqueta del tramo
        tramo = f"{int(inicio)}-{int(fin)}"
        
        return tramo
    
    # Aplicar la función de asignación de tramos
    df_chart['tramo_5min'] = df_chart['minuto'].apply(asignar_tramo_5min)
    
    # Definir eventos positivos
    eventos_positivos = [
        'Intercepción', 'Parada', 'Recuperación', 'Regate', 'Disparo',
        'Pase Recibido', 'Conduccion', 'Apoyo a la Línea Defensiva',
        'Intervención', 'Bloqueo'
    ]
    
    # Calcular acciones positivas
    def calcular_accion_positiva(row):
        tipo_evento = str(row['tipo_evento']) if not pd.isna(row['tipo_evento']) else ''
        resultado = str(row['resultado']) if not pd.isna(row['resultado']) else ''
        
        # Casos para Pase y Duelo
        if tipo_evento == 'Pase':
            if resultado == 'Exitoso':
                return 1
            elif resultado == 'Fallido':
                return -1
        
        elif tipo_evento == 'Duelo':
            if resultado == 'Exitoso':
                return 1
            elif resultado == 'Fallido':
                return -1
        
        # Otros eventos positivos (siempre suman 1)
        elif tipo_evento in eventos_positivos:
            return 1
        
        # Cualquier otro evento no suma ni resta
        return 0
    
    df_chart['accion_positiva'] = df_chart.apply(calcular_accion_positiva, axis=1)
    
    # Definir todos los tramos posibles de 5 minutos
    todos_tramos = [
        '0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-35', '35-40', '40-45',
        '45-50', '50-55', '55-60', '60-65', '65-70', '70-75', '75-80', '80-85', '85-90'
    ]
    
    # Agrupar por tramo y local/visitante para el análisis
    # Usamos groupby y agg para calcular las estadísticas de manera más eficiente
    agg_general = df_chart.groupby('tramo_5min')['accion_positiva'].sum().to_dict()
    agg_local = df_chart[df_chart['local_visitante'] == 'Local'].groupby('tramo_5min')['accion_positiva'].sum().to_dict()
    agg_visitante = df_chart[df_chart['local_visitante'] == 'Visitante'].groupby('tramo_5min')['accion_positiva'].sum().to_dict()
    
    # Asegurarnos de que todos los tramos están representados con valor 0 si no hay datos
    for tramo in todos_tramos:
        if tramo not in agg_general:
            agg_general[tramo] = 0
        if tramo not in agg_local:
            agg_local[tramo] = 0
        if tramo not in agg_visitante:
            agg_visitante[tramo] = 0
    
    # Extraer valores ordenados para la gráfica
    valores_general = [agg_general.get(tramo, 0) for tramo in todos_tramos]
    valores_local = [agg_local.get(tramo, 0) for tramo in todos_tramos]
    valores_visitante = [agg_visitante.get(tramo, 0) for tramo in todos_tramos]
    
    # Convertir etiquetas de tramos a posiciones numéricas para el eje X
    posiciones_x = []
    for tramo in todos_tramos:
        inicio, _ = map(int, tramo.split('-'))
        # Punto medio del tramo
        posiciones_x.append(inicio + 2.5)
    
    # Crear líneas en la gráfica
    ax.plot(posiciones_x, valores_general, 'o-', color=COLOR_GENERAL, linewidth=2.5, markersize=7, 
            label='General', zorder=10)
    ax.plot(posiciones_x, valores_local, 's-', color=COLOR_LOCAL, linewidth=2.5, markersize=7, 
            label='Local', zorder=8)
    ax.plot(posiciones_x, valores_visitante, '^-', color=COLOR_VISITANTE, linewidth=2.5, markersize=7, 
            label='Visitante', zorder=9)
    
    # Añadir valor en cada punto de la línea general
    for i, (x, y) in enumerate(zip(posiciones_x, valores_general)):
        if y != 0:
            ax.text(x, y + 0.5, f"{y:.0f}", ha='center', va='bottom', color='white', 
                    fontsize=9, fontweight='bold', zorder=11)
    
    # Detectar si hay un posible problema con los datos de tiempo
    tramo_85_90_count = df_chart['tramo_5min'].value_counts().get('85-90', 0)
    total_filas = len(df_chart)
    if tramo_85_90_count > total_filas * 0.4:  # Si más del 40% está en ese tramo, hay un posible problema
        titulo += ' (ADVERTENCIA: Posible problema con datos de tiempo)'
    
    ax.set_xlabel('Minutos de Partido', color='#FFFFFF', fontsize=14, fontweight='bold', labelpad=12)
    ax.set_ylabel('Balance de Acciones Positivas', color='#FFFFFF', fontsize=14, fontweight='bold', labelpad=12)
    
    # Configurar límites y ticks del eje X para mostrar el tiempo completo (0-90)
    ax.set_xlim(-2.5, 92.5)
    
    # Usar las etiquetas de los tramos para el eje X
    ax.set_xticks(posiciones_x)
    ax.set_xticklabels(todos_tramos, rotation=45)
    
    # Definir límites del eje Y dinámicamente
    todos_valores = valores_general + valores_local + valores_visitante
    max_valor = max(todos_valores) if todos_valores else 5
    min_valor = min(todos_valores) if todos_valores else -5
    
    # Añadir un poco de espacio para las etiquetas
    margen = max(10, (max_valor - min_valor) * 1.0)
    ax.set_ylim(min_valor - margen, max_valor + margen)
    
    # Personalizar ejes con mayor contraste
    ax.tick_params(axis='both', colors='#FFFFFF', labelsize=11)
    
    # Añadir línea horizontal en 0 más visible
    ax.axhline(y=0, color='#FFFFFF', linestyle='-', alpha=0.5, linewidth=1.5)
    
    # Personalizar grid más visible
    ax.grid(True, axis='y', color='#FFFFFF', alpha=0.2, linestyle='--')
    ax.grid(True, axis='x', color='#FFFFFF', alpha=0.1, linestyle='--')
    
    # Dividir visualmente primera y segunda parte con mayor contraste
    ax.axvline(x=45, color='#FFFFFF', linestyle='-', alpha=0.7, linewidth=2.0)
    ax.text(45, ax.get_ylim()[1], 'MEDIO TIEMPO', ha='center', va='top', 
            color='#FFFFFF', fontsize=14, fontweight='bold', alpha=0.9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#333333', alpha=0.7, edgecolor='#FFFFFF'))
    
    # Sombrear ligeramente los tramos para mejor visualización
    for i in range(0, 90, 10):
        ax.axvspan(i, i+5, color='#FFFFFF', alpha=0.04)
    
    # Estilo de bordes más destacados
    for spine in ax.spines.values():
        spine.set_color('#4BB3FD')  # Azul destacado para los bordes
        spine.set_alpha(0.8)
        spine.set_linewidth(2.0)
    
    # Añadir leyenda más visible
    legend = ax.legend(loc='upper right', 
                     facecolor='#333333', edgecolor='#4BB3FD', labelcolor='#FFFFFF',
                     framealpha=0.9, fontsize=12)
    legend.get_frame().set_linewidth(2)
    
    # Calcular información de resumen
    total_acciones = df_chart.shape[0]
    total_positivas = df_chart[df_chart['accion_positiva'] > 0].shape[0]
    total_negativas = df_chart[df_chart['accion_positiva'] < 0].shape[0]
    balance_general = df_chart['accion_positiva'].sum()
    
    # Calcular balance por local/visitante
    balance_local = df_chart[df_chart['local_visitante'] == 'Local']['accion_positiva'].sum()
    balance_visitante = df_chart[df_chart['local_visitante'] == 'Visitante']['accion_positiva'].sum()
    
    # Contar partidos de local y visitante
    partidos_local = df_chart[df_chart['local_visitante'] == 'Local']['partido'].nunique()
    partidos_visitante = df_chart[df_chart['local_visitante'] == 'Visitante']['partido'].nunique()
    
    # Añadir texto con estadísticas en un cuadro destacado
    info_text = (
        f"Resumen de Acciones\n"
        f"Total evaluadas: {total_acciones}\n"
        f"Positivas: {total_positivas} | Negativas: {total_negativas}\n"
        f"Balance general: {balance_general:.0f}\n"
        f"Local ({partidos_local} partidos): {balance_local:.0f}\n"
        f"Visitante ({partidos_visitante} partidos): {balance_visitante:.0f}"
    )
    
    ax.text(0.02, 0.72, info_text, transform=ax.transAxes,
            color='#FFFFFF', fontsize=11, alpha=1.0, va='bottom', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='#333333', 
                     edgecolor='#4BB3FD', alpha=0.8, linewidth=2))
    
    # Ajustar layout
    plt.tight_layout()
    
    return fig

# Definición de la función draw_combined_passes
def draw_combined_passes(ax, df_jugadores, player_id, season_ids, pitch):
    """
    Función que muestra secuencias de pases con un diseño moderno de flecha,
    utilizando gradientes y un estilo contemporáneo con líneas cuyo grosor aumenta
    gradualmente desde el inicio hasta el final. Incluye estadísticas de POP del jugador.
    """
    
    # Diccionario de abreviaturas para demarcaciones
    abreviaturas_demarcacion = {
        'Portero': 'POR', 
        'Lateral Derecho': 'LD', 
        'Lateral Izquierdo': 'LI',
        'Defensa Central': 'CT', 
        'Mediocentro Defensivo': 'MCD',
        'Mediocentro': 'MC', 
        'Interior Derecho': 'ID',
        'Interior Izquierdo': 'II', 
        'Mediapunta': 'MP',
        'Extremo Derecho': 'ED', 
        'Extremo Izquierdo': 'EI',
        'Delantero Centro': 'DC', 
        'Delantero Izquierdo': 'DI', 
        'Delantero Derecho': 'DD'
    }
    
    # Filtrar datos para el jugador y temporadas específicas
    df_pases = df_jugadores[
        (df_jugadores['player_id'] == player_id) &
        (df_jugadores['season_id'].isin(season_ids))
    ].copy()
    
    # Convertir columnas de coordenadas a numéricas
    for col in ['xstart', 'ystart', 'xend', 'yend']:
        df_pases[col] = pd.to_numeric(df_pases[col], errors='coerce')
    
    # Convertir la columna POP a numérica
    if 'POP' in df_pases.columns:
        df_pases.loc[:, 'POP'] = pd.to_numeric(df_pases['POP'], errors='coerce').fillna(0)
    
    # Función para convertir event_time a segundos totales
    def time_to_seconds(time_str):
        try:
            if pd.isna(time_str):
                return None
            # Verificar si el formato es minutos:segundos
            if isinstance(time_str, str) and ':' in time_str:
                match = re.match(r'(\d+):(\d+)', time_str)
                if match:
                    minutes, seconds = map(int, match.groups())
                    return minutes * 60 + seconds
            # Si es un número, devolver como está
            return float(time_str)
        except Exception as e:
            return None
    
    # Procesar event_time para ordenar cronológicamente
    if 'event_time' in df_pases.columns:
        df_pases['event_seconds'] = df_pases['event_time'].apply(time_to_seconds)
        # Ordenar por tiempo
        df_pases = df_pases.sort_values('event_seconds').reset_index(drop=False)
    else:
        print("⚠️ ADVERTENCIA: No se encontró columna 'event_time'. La secuencia de eventos puede ser incorrecta.")
    
    # Diccionarios para almacenar datos para estadísticas
    stats = {}
    secuencias_info = []
    
    # Dibujar el campo de fútbol
    pitch.draw(ax=ax)
    
    # Crear un colormap personalizado para los gradientes de la flecha
    custom_cmap = LinearSegmentedColormap.from_list('modern_gradient', 
                                                   ['#3b82f6', '#ef4444', '#10b981'], N=256)
    
    # Variables para almacenar leyendas
    legend_labels = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3b82f6', markersize=8, 
               label='Inicio', path_effects=[path_effects.withStroke(linewidth=2, foreground='black')]),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ef4444', markersize=10, 
               label='Recepción (POP)', path_effects=[path_effects.withStroke(linewidth=2, foreground='black')]),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#10b981', markersize=8, 
               label='Destino', path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
    ]
    
    # Contador para seguimiento
    secuencias_encontradas = 0
    pop_acumulado = 0
    alturas_pop = []
    demarcaciones_previas = []
    eventos_finales = []
    
    # Recorrer el DataFrame ordenado para encontrar secuencias
    for i in range(1, len(df_pases) - 2):  # Necesitamos que exista i+2 para obtener el evento posterior
        try:
            # Verificar el patrón "Pase → Pase Recibido → Pase"
            prev_row = df_pases.iloc[i-1]
            curr_row = df_pases.iloc[i]
            next_row = df_pases.iloc[i+1]
            post_row = df_pases.iloc[i+2]  # Evento dos filas después del POP
            
            # Verificar si tenemos la secuencia correcta de eventos
            if (prev_row['tipo_evento'] == "Pase" and 
                curr_row['tipo_evento'] == "Pase Recibido" and 
                next_row['tipo_evento'] == "Pase" and
                curr_row['POP'] > 0.5):
                
                # Punto 1: Inicio del primer pase (Azul)
                x1 = prev_row['xstart']
                y1 = prev_row['ystart']
                
                # Punto 2: Ubicación del POP (Pase Recibido) (Rojo)
                x2 = curr_row['xstart']
                y2 = curr_row['ystart']
                pop_value = curr_row['POP']
                
                # Punto 3: Final del pase posterior (Verde)
                x3 = next_row['xend']
                y3 = next_row['yend']
                
                # Actualizar estadísticas
                pop_acumulado += pop_value
                alturas_pop.append(y2)  # Almacenar la altura ystart del POP
                
                # Almacenar demarcación del pase previo
                demarcacion = prev_row.get('demarcacion', '')
                demarcaciones_previas.append(demarcacion)
                
                # Almacenar tipo de evento posterior al POP
                tipo_evento_posterior = post_row.get('tipo_evento', '')
                eventos_finales.append(tipo_evento_posterior)
                
                # Almacenar información de la secuencia para análisis posterior
                secuencia_info = {
                    'demarcacion_previa': demarcacion,
                    'altura_pop': y2,
                    'evento_final': tipo_evento_posterior,
                    'pop_value': pop_value
                }
                secuencias_info.append(secuencia_info)
                
                # Verificar que todas las coordenadas son válidas
                if (pd.notna(x1) and pd.notna(y1) and np.isfinite(x1) and np.isfinite(y1) and
                    pd.notna(x2) and pd.notna(y2) and np.isfinite(x2) and np.isfinite(y2) and
                    pd.notna(x3) and pd.notna(y3) and np.isfinite(x3) and np.isfinite(y3)):
                    
                    # Crear puntos intermedios para el primer segmento con grosor variable
                    # Usamos más puntos para crear un efecto de grosor que aumenta gradualmente
                    num_points = 10  # Número de puntos intermedios
                    x_points1 = np.linspace(x1, x2, num_points)
                    y_points1 = np.linspace(y1, y2, num_points)
                    widths1 = np.linspace(2, 4, num_points-1)  # Grosor aumenta de 2 a 4
                    
                    # Crear puntos intermedios para el segundo segmento con grosor variable
                    x_points2 = np.linspace(x2, x3, num_points)
                    y_points2 = np.linspace(y2, y3, num_points)
                    widths2 = np.linspace(4, 6, num_points-1)  # Grosor aumenta de 4 a 6
                    
                    # Dibujar segmentos del primer tramo con grosor variable
                    for j in range(num_points-1):
                        points = np.array([
                            [x_points1[j], y_points1[j]], 
                            [x_points1[j+1], y_points1[j+1]]
                        ]).reshape(-1, 1, 2)
                        segments = np.concatenate([points[:-1], points[1:]], axis=1)
                        lc = LineCollection(segments, linewidths=widths1[j], color=custom_cmap(j/(num_points-1)*0.5), alpha=0.8)
                        ax.add_collection(lc)
                    
                    # Dibujar segmentos del segundo tramo con grosor variable
                    for j in range(num_points-1):
                        points = np.array([
                            [x_points2[j], y_points2[j]], 
                            [x_points2[j+1], y_points2[j+1]]
                        ]).reshape(-1, 1, 2)
                        segments = np.concatenate([points[:-1], points[1:]], axis=1)
                        lc = LineCollection(segments, linewidths=widths2[j], color=custom_cmap(0.5 + j/(num_points-1)*0.5), alpha=0.8)
                        ax.add_collection(lc)
                    
                    # Dibujar los puntos con bordes negros para mejor visibilidad
                    # Punto inicial (azul)
                    arrow_size = 120
                    ax.scatter(x1, y1, s=100, color='#3b82f6', edgecolor='black', linewidth=1.5, zorder=5, alpha=0.9)
                    
                    # Punto medio (rojo)
                    ax.scatter(x2, y2, s=160, color='#ef4444', edgecolor='black', linewidth=1.5, zorder=5, alpha=0.9)
                    
                    # Dibujar un círculo verde en el punto final
                    ax.scatter(x3, y3, s=arrow_size, color='#10b981', marker='o', 
                              edgecolor='black', linewidth=1.5, zorder=5, alpha=0.9)
                    
                    # Obtener abreviatura de la demarcación del jugador que hace el primer pase (punto azul)
                    abreviatura_dem = abreviaturas_demarcacion.get(demarcacion, 
                                                              demarcacion[:3].upper() if isinstance(demarcacion, str) else '')
                    
                    # Obtener el tipo de evento posterior al POP (tres primeras letras) para el punto verde
                    abreviatura_evt = tipo_evento_posterior[:3].upper() if isinstance(tipo_evento_posterior, str) else ''
                    
                    # Añadir abreviatura de demarcación en el punto azul (inicial)
                    text_dem = ax.text(x1, y1, abreviatura_dem, color='white', fontsize=6, 
                                     ha='center', va='center', fontweight='bold', zorder=6)
                    text_dem.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='black')])
                    
                    # Añadir abreviatura del tipo de evento en el punto verde (final) con letra más pequeña
                    text_evt = ax.text(x3, y3, abreviatura_evt, color='white', fontsize=5, 
                                     ha='center', va='center', fontweight='bold', zorder=6)
                    text_evt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='black')])
                    
                    # Añadir el valor POP multiplicado por 10 como texto dentro del punto rojo
                    pop_text = str(int(pop_value * 10))
                    text = ax.text(x2, y2, pop_text, color='white', fontsize=8, 
                                  ha='center', va='center', fontweight='bold', zorder=6)
                    text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
                    
                    secuencias_encontradas += 1
                else:
                    print(f"❌ Coordenadas no válidas para la secuencia en filas {i-1},{i},{i+1}")
        
        except Exception as e:
            print(f"Error al procesar secuencia en fila {i}: {e}")
        
    # Calcular estadísticas para mostrar en el gráfico
    stats['pop_acumulado'] = pop_acumulado
    stats['secuencias_encontradas'] = secuencias_encontradas
    
    # Calcular altura media del POP (ystart)
    if alturas_pop:
        stats['altura_media_pop'] = sum(alturas_pop) / len(alturas_pop)
    
    # Encontrar la demarcación más frecuente del pase previo
    if demarcaciones_previas:
        from collections import Counter
        demarcacion_counts = Counter(demarcaciones_previas)
        stats['demarcacion_mas_frecuente'] = demarcacion_counts.most_common(1)[0][0]
        abreviatura_demarcacion = abreviaturas_demarcacion.get(stats['demarcacion_mas_frecuente'], 
                                                      stats['demarcacion_mas_frecuente'][:3].upper() 
                                                      if isinstance(stats['demarcacion_mas_frecuente'], str) else '')
        stats['abreviatura_demarcacion'] = abreviatura_demarcacion
    
    # Encontrar el tipo de evento más frecuente al final
    if eventos_finales:
        evento_counts = Counter(eventos_finales)
        stats['evento_final_mas_frecuente'] = evento_counts.most_common(1)[0][0]
        stats['abreviatura_evento'] = stats['evento_final_mas_frecuente'][:3].upper() if isinstance(stats['evento_final_mas_frecuente'], str) else ''
    
    # Añadir estadísticas de POP del jugador encima del campo
    if stats:
        # Obtener el nombre del jugador si está disponible
        nombre_jugador = ""
        if 'jugador' in df_pases.columns:
            nombres_jugador = df_pases[df_pases['player_id'] == player_id]['jugador'].unique()
            if len(nombres_jugador) > 0:
                nombre_jugador = nombres_jugador[0]
        
        # Crear el texto de estadísticas
        stats_text = f"Jugador: {nombre_jugador if nombre_jugador else player_id}\n"
        
        if 'pop_acumulado' in stats:
            stats_text += f"POP Acumulado: {stats['pop_acumulado']:.3f}\n"
        
        if 'demarcacion_mas_frecuente' in stats:
            stats_text += f"Dem. frecuente: {stats['demarcacion_mas_frecuente']}\n"
        
        if 'altura_media_pop' in stats:
            stats_text += f"Altura media POP: {stats['pop_acumulado']:.1f}\n"
        
        if 'evento_final_mas_frecuente' in stats:
            stats_text += f"Evento final frecuente: {stats['evento_final_mas_frecuente']}"
        
        # Añadir un recuadro con las estadísticas en la parte superior (más cerca del campo)
        ax.text(50, 105, stats_text, 
                horizontalalignment='center', 
                verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#1e293b", edgecolor="white", alpha=0.8),
                color="white", 
                fontsize=4,
                weight='bold',
                transform=ax.transData)
            
    # Dibujar leyenda con estilo moderno
    try:
        legend = ax.legend(
            handles=legend_labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.01),  # Acercamos la leyenda al campo
            facecolor='#1e293b',
            edgecolor='white',
            fontsize=6,
            labelcolor='white',
            ncol=3,
            framealpha=0.9,
            borderpad=1.0,
            handletextpad=1
        )
        # Añadir contorno a la leyenda para mejor visibilidad
        frame = legend.get_frame()
        frame.set_linewidth(2)
    except Exception as e:
        print(f"Error al crear la leyenda: {e}")

def capturar_visualizaciones(fig, dpi=150):
    """
    Convierte una figura de matplotlib en un objeto Image compatible con ReportLab,
    con manejo especial para campogramas verticales.
    """
    # Obtener dimensiones originales para determinar si es un campograma vertical
    fig_width_inches, fig_height_inches = fig.get_size_inches()
    aspect_ratio = fig_width_inches / fig_height_inches
    
    # Detectar campogramas verticales (más altos que anchos)
    es_campograma_vertical = aspect_ratio < 0.8 and fig_height_inches > fig_width_inches
    
    # Guardar con diferentes configuraciones según el tipo de figura
    buf = io.BytesIO()
    
    if es_campograma_vertical:
        # Para campogramas verticales, desactivar bbox_inches para evitar recortes
        fig.savefig(buf, format='png', dpi=dpi, pad_inches=0.5)
    else:
        # Para otras figuras, usar configuración normal
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    
    buf.seek(0)
    
    # Ajustar tamaños para diferentes tipos de figuras
    if es_campograma_vertical:
        # Asignar un ancho fijo ligeramente mayor para asegurar que se vean los bordes
        width = 2.5 * inch
        # Ajustar la altura manteniendo la proporción del campograma
        height = width / aspect_ratio
        
        # Limitar la altura máxima si es demasiado grande
        max_height = 4.5 * inch
        if height > max_height:
            height = max_height
            width = height * aspect_ratio
    else:
        # Para otras figuras, usar dimensiones estándar
        max_width = 7 * inch
        max_height = 4 * inch
        
        # Calcular dimensiones manteniendo la relación de aspecto
        if aspect_ratio > 1.75:  # Figuras muy anchas
            width = max_width
            height = width / aspect_ratio
        else:
            height = max_height
            width = min(max_width, height * aspect_ratio)
    
    # Crear la imagen ReportLab con dimensiones ajustadas
    return ReportLabImage(buf, width=width, height=height)

def generar_analisis_jugador(df_jugador, df_estadisticas, player_id, season_ids):
    """
    Genera un análisis simple del jugador basado en sus datos.
    Extrae: Porcentaje de pases, Minutos Jugados, Goles, xG, POP, KPI
    """
    # Filtrar datos del jugador
    player_id = float(player_id) if isinstance(player_id, (int, str)) else player_id
    
    datos_jugador = df_jugador[df_jugador['player_id'] == player_id].copy()
    stats_jugador = df_estadisticas[
        (df_estadisticas['player_id'] == player_id) &
        (df_estadisticas['season_id'].isin(season_ids))
    ].copy()
    
    # Verificar si hay datos básicos para trabajar
    if datos_jugador.empty and stats_jugador.empty:
        return "No hay datos disponibles para este jugador."
    
    # Obtener información básica
    if not datos_jugador.empty:
        nombre = datos_jugador['jugador'].iloc[0]
        posicion = datos_jugador['demarcacion'].iloc[0] if 'demarcacion' in datos_jugador.columns else "No especificada"
        equipo = datos_jugador['equipo'].iloc[0] if 'equipo' in datos_jugador.columns else "No especificado"
    elif not stats_jugador.empty:
        nombre = stats_jugador['jugador'].iloc[0]
        posicion = stats_jugador['demarcacion'].iloc[0] if 'demarcacion' in stats_jugador.columns else "No especificada"
        equipo = stats_jugador['equipo'].iloc[0] if 'equipo' in stats_jugador.columns else "No especificado"
    else:
        nombre = "Jugador"
        posicion = "No especificada"
        equipo = "No especificado"
    
    # Inicializar valores
    minutos_jugados = 0
    porcentaje_pases = "N/A"
    goles = 0
    asistencias = 0
    xg = "N/A"
    pop = "N/A"
    kpi = "N/A"
    
    # Estadísticas de df_estadisticas
    if not stats_jugador.empty:
        # Minutos jugados
        if 'minutos_jugados' in stats_jugador.columns:
            min_jugados = pd.to_numeric(stats_jugador['minutos_jugados'], errors='coerce')
            minutos_jugados = int(min_jugados.sum())
        
        # Goles
        if 'goles' in stats_jugador.columns:
            goles = int(stats_jugador['goles'].sum())
            
        # Asistencias
        if 'asistencias' in stats_jugador.columns:
            asistencias = int(stats_jugador['asistencias'].sum())
        
        # xG (expected goals)
        if 'xG' in stats_jugador.columns:
            xg_values = pd.to_numeric(stats_jugador['xG'], errors='coerce')
            if not xg_values.isna().all():
                xg = f"{xg_values.sum():.2f}"
        
        # POP (expected progresón)
        if 'xG' in stats_jugador.columns:
            xg_values = pd.to_numeric(stats_jugador['POP'], errors='coerce')
            if not xg_values.isna().all():
                xg = f"{xg_values.sum():.2f}"
        
        # KPI
        if 'KPI_rendimiento' in stats_jugador.columns:
            kpi_values = pd.to_numeric(stats_jugador['KPI_rendimiento'], errors='coerce')
            if not kpi_values.isna().all():
                kpi = f"{kpi_values.mean():.2f}"
    
    # Revisar en df_jugador para datos adicionales
    if not datos_jugador.empty:
        # Calcular porcentaje de pases
        pases_exitosos_total = 0
        pases_fallidos_total = 0
        
        # Verificar diferentes columnas que podrían contener datos de pases
        posibles_columnas_exitosos = [
            'pases_exitosos_campo_propio', 'pases_exitosos_campo_contrario',
            'pases_adelante_inicio', 'pases_adelante_creacion',
            'pases_horizontal_inicio', 'pases_horizontal_creacion',
            'pases_largos_exitosos', 'cambios_orientacion_exitosos'
        ]
        
        posibles_columnas_fallidos = [
            'pases_fallidos_campo_propio', 'pases_fallidos_campo_contrario',
            'pases_largos_fallidos', 'cambios_orientacion_fallidos'
        ]
        
        for col in posibles_columnas_exitosos:
            if col in datos_jugador.columns:
                pases_exitosos_total += datos_jugador[col].sum()
        
        for col in posibles_columnas_fallidos:
            if col in datos_jugador.columns:
                pases_fallidos_total += datos_jugador[col].sum()
        
        total_pases = pases_exitosos_total + pases_fallidos_total
        if total_pases > 0:
            porcentaje_pases = f"{(pases_exitosos_total / total_pases * 100):.1f}%"
        
        # POP
        if 'POP' in datos_jugador.columns:
            pop_values = pd.to_numeric(datos_jugador['POP'], errors='coerce')
            if not pop_values.isna().all() and pop_values.sum() > 0:
                pop = f"{pop_values.mean():.3f}"
    
    # Generar el análisis en un formato más compacto para que quepa mejor en el PDF
    analisis = f""" {posicion} | Equipo: {equipo} |
Minutos: {minutos_jugados} | Goles: {goles} | Asistencias: {asistencias} | xG: {xg} | % Pases: {porcentaje_pases} | POP: {pop} | KPI: {kpi}"""
    
    return analisis
    

def generar_pdf(df_jugador, df_estadisticas, player_id, season_ids, figuras):
    """
    Genera un PDF con las visualizaciones y el análisis del jugador.
    Todo el contenido debe caber en una sola página con formato horizontal.
    Fondo completamente negro sin márgenes blancos y texto en color blanco.
    Incluye los títulos de las gráficas centrados.
    """
    # Obtener información del jugador para el título
    df_jugador_filtrado = df_jugador[df_jugador['player_id'] == player_id]
    if not df_jugador_filtrado.empty:
        nombre_jugador = df_jugador_filtrado['jugador'].iloc[0]
        posicion = df_jugador_filtrado['demarcacion'].iloc[0] if 'demarcacion' in df_jugador_filtrado.columns else ""
        equipo = df_jugador_filtrado['equipo'].iloc[0] if 'equipo' in df_jugador_filtrado.columns else ""
    else:
        # Intentar buscar en estadísticas si no está en jugadores
        df_stats_filtrado = df_estadisticas[df_estadisticas['player_id'] == player_id]
        if not df_stats_filtrado.empty:
            nombre_jugador = df_stats_filtrado['jugador'].iloc[0]
            posicion = df_stats_filtrado['demarcacion'].iloc[0] if 'demarcacion' in df_stats_filtrado.columns else ""
            equipo = df_stats_filtrado['equipo'].iloc[0] if 'equipo' in df_stats_filtrado.columns else ""
        else:
            nombre_jugador = f"Jugador {player_id}"
            posicion = ""
            equipo = ""
    
    # Definir los títulos de las gráficas
    titulos_graficas = [
        "Métricas del Jugador",
        "Comparativa KPI con el mejor en esa demarcación",
        "Evolución KPI Rendimiento",
        "Mapa de Calor",
        "Evolución de Acciones Exitosas",
        "Mapa de Pases Orientados Progresivos"
    ]
    
    # Preparar el PDF (formato horizontal con márgenes absolutamente mínimos)
    buffer = io.BytesIO()
    
    # Eliminar márgenes completamente para evitar bordes blancos
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=0,
        leftMargin=0,
        topMargin=0,
        bottomMargin=0
    )
    
    # Obtener estilos de reportlab
    styles = getSampleStyleSheet()
    
    # Crear estilos personalizados con texto blanco sobre fondo negro
    styles.add(ParagraphStyle(
        name='Analisis',
        parent=styles['Heading1'],
        fontSize=12,
        textColor=colors.white,
        backColor=colors.black,
        alignment=TA_LEFT
    ))
    
    # Estilo para títulos centrados
    styles.add(ParagraphStyle(
        name='TituloGrafica',
        parent=styles['Heading2'],
        fontSize=10,
        textColor=colors.white,
        backColor=colors.black,
        alignment=TA_CENTER,  # Centrar los títulos
        spaceAfter=6  # Añadir un poco de espacio después del título
    ))
    
    styles.add(ParagraphStyle(
        name='SubtituloArtistico',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=5,
        textColor=colors.HexColor('#FFD700'),  # Color dorado
        backColor=colors.black,
        fontName='Helvetica-Bold',
    ))
    
    # Crear elementos del PDF
    elementos = []
    
    # Fondo negro para toda la página
    page_width, page_height = landscape(A4)
    black_background = Paragraph(
        "<para></para>",
        ParagraphStyle(
            'background',
            parent=styles['Normal'],
            backColor=colors.black,  # Fondo negro
        )
    )
    elementos.append(black_background)
    
    # Crear un encabezado con título e imagen usando una tabla
    try:
        # Crear imagen para escudo
        try:
            # Abrir la imagen del escudo
            escudo_buffer = io.BytesIO()
            with open(logo_path, 'rb') as f:
                escudo_buffer.write(f.read())
            escudo_buffer.seek(0)
            
            # Crear la imagen ReportLab con el buffer
            img_reportlab_escudo = ReportLabImage(escudo_buffer, width=0.6*inch, height=0.6*inch)
        except Exception as e:
            print(f"Error al cargar el escudo: {e}")
            img_reportlab_escudo = None
        
        # Crear un estilo para texto blanco y negrita
        white_bold_style = ParagraphStyle(
            name='WhiteBold',
            parent=styles['Heading1'],
            fontSize=12,
            textColor=colors.white,
            fontName='Helvetica-Bold',
            alignment=TA_LEFT,
            backColor=None,  # Transparent background
        )
        
        # Crear tabla para colocar título e imagen lado a lado
        # Ajustamos los anchos para poner el escudo más a la derecha
        header_data = [[
            Paragraph(f"Informe de Rendimiento - {nombre_jugador}", white_bold_style),
            img_reportlab_escudo
        ]]
        
        # Ajustar ancho para maximizar espacio y mover el escudo más a la derecha
        header_table = Table(header_data, colWidths=[8.0*inch, 3.4*inch], hAlign='LEFT')
        header_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (0, 0), 'MIDDLE'),
            ('ALIGN', (0, 0), (0, 0), 'LEFT'),
            ('ALIGN', (1, 0), (1, 0), 'RIGHT'),  # Alineación a la derecha para el escudo
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
            ('BACKGROUND', (0, 0), (-1, -1), colors.black),  # Fondo negro
        ]))
        
        # Agregar elementos
        elementos.append(header_table)
        
    except Exception as e:
        # Si hay algún problema con la imagen, usar solo el título
        print(f"Error al cargar la imagen: {e}")
        elementos.append(Paragraph(f"Informe de Rendimiento - {nombre_jugador}", styles['Analisis']))
        
    # Análisis del jugador (en formato compacto)
    analisis = generar_analisis_jugador(df_jugador, df_estadisticas, player_id, season_ids)
    elementos.append(Paragraph(analisis, styles['Analisis']))
    
    # Añadir un pequeño espacio después del análisis
    elementos.append(Spacer(1, 5))
    
    # Calcular dimensiones disponibles
    viz_width = page_width - 20  # Dejar un pequeño margen
    viz_height = page_height - 60  # Reservamos espacio para título, análisis y márgenes
    
    # Crear una única tabla para todas las visualizaciones
    if len(figuras) >= 6:
        # Definir proporciones de ancho más equilibradas para ambas filas
        # Aumentar el primer valor de la segunda fila para dar más espacio a la izquierda
        estructura = [
            [2.0, 1.2, 1.8],  # Primera fila: más ancho para la primera y tercera gráfica
            [1.0, 2.8, 1.0]   # Segunda fila: más ancho para la del medio, más espacio para la izquierda
        ]
        
        # Calcular el ancho de la unidad base (para un total de 5 unidades)
        unidad_base = viz_width / 5.0
        
        # Calcular los anchos de columnas para cada fila separadamente
        anchos_col_fila1 = [unidad_base * estructura[0][i] for i in range(3)]
        anchos_col_fila2 = [unidad_base * estructura[1][i] for i in range(3)]
        
        # Altura por fila - un poco menos para dejar espacio entre filas
        altura_fila = viz_height / 2.2
        
        # Preparar las imágenes y títulos
        images = []
        image_titles = []
        
        # Procesar las 6 gráficas
        for row in range(2):
            for col in range(3):
                idx = row * 3 + col
                fig = figuras[idx]
                
                # Obtener el título correspondiente (si está disponible)
                titulo = titulos_graficas[idx] if idx < len(titulos_graficas) else f"Gráfica {idx+1}"
                
                # Asegurar que el fondo de la gráfica sea transparente o negro
                fig.patch.set_facecolor('#0E1117')  # Fondo oscuro
                
                # Determinar el ancho según la estructura de la fila específica
                ancho_unidades = estructura[row][col]
                ancho_puntos = unidad_base * ancho_unidades
                
                # Crear buffer y guardar imagen
                img_buffer = io.BytesIO()
                fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', 
                        pad_inches=0.05, facecolor='#0E1117', transparent=True)  # Añadir padding interno
                img_buffer.seek(0)
                
                # Convertir a imagen de ReportLab con altura ajustada pero respetando proporciones
                img = ReportLabImage(img_buffer, width=ancho_puntos, height=altura_fila - 25)  # Reducir altura para dejar espacio al título
                images.append(img)
                
                # Crear párrafo para el título
                image_titles.append(Paragraph(titulo, styles['TituloGrafica']))
        
        # Organizar las visualizaciones en dos tablas separadas, una para cada fila
        # Esto nos permite usar diferentes anchos de columna para cada fila
        
        # Primera fila
        data_row1 = [[image_titles[0], image_titles[1], image_titles[2]], 
                     [images[0], images[1], images[2]]]
        
        table_row1 = Table(data_row1, 
                          colWidths=anchos_col_fila1,
                          rowHeights=[20, altura_fila - 25])  # Altura específica para títulos y gráficas
        
        # Segunda fila
        data_row2 = [[image_titles[3], image_titles[4], image_titles[5]], 
                     [images[3], images[4], images[5]]]
        
        table_row2 = Table(data_row2, 
                          colWidths=anchos_col_fila2,  # Usar anchos específicos para la segunda fila
                          rowHeights=[20, altura_fila - 25])
        
        # Estilo para ambas tablas
        table_style = TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),  # Centrar los títulos
            ('ALIGN', (0, 1), (-1, 1), 'CENTER'),  # Centrar las imágenes
            ('LEFTPADDING', (0, 0), (-1, -1), 5),  # Padding izquierdo
            ('RIGHTPADDING', (0, 0), (-1, -1), 5), # Padding derecho
            ('TOPPADDING', (0, 0), (-1, 0), 2),    # Padding superior para títulos
            ('BOTTOMPADDING', (0, 0), (-1, 0), 2), # Padding inferior para títulos
            ('TOPPADDING', (0, 1), (-1, 1), 0),    # Sin padding superior para imágenes
            ('BOTTOMPADDING', (0, 1), (-1, 1), 0), # Sin padding inferior para imágenes
            ('BACKGROUND', (0, 0), (-1, -1), colors.black),  # Fondo negro
        ])
        
        table_row1.setStyle(table_style)
        table_row2.setStyle(table_style)
        
        # Crear una tabla contenedora para las dos filas
        data_main = [[table_row1], [table_row2]]
        main_table = Table(data_main, colWidths=[viz_width])
        
        main_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (0, 1), 'MIDDLE'),
            ('ALIGN', (0, 0), (0, 1), 'CENTER'),
            ('LEFTPADDING', (0, 0), (0, 1), 0),
            ('RIGHTPADDING', (0, 0), (0, 1), 0),
            ('TOPPADDING', (0, 0), (0, 0), 0),    
            ('BOTTOMPADDING', (0, 0), (0, 0), 5), # Espacio entre filas
            ('TOPPADDING', (0, 1), (0, 1), 5),    
            ('BOTTOMPADDING', (0, 1), (0, 1), 0),
            ('BACKGROUND', (0, 0), (0, 1), colors.black),
        ]))
        
        elementos.append(main_table)
    
    # Función para fondo negro de toda la página
    def black_canvas(canvas, doc):
        canvas.setFillColor(colors.black)
        canvas.rect(0, 0, doc.width, doc.height, fill=1)
        
    # Construir el PDF con el fondo negro
    doc.build(elementos, onFirstPage=black_canvas, onLaterPages=black_canvas)
    buffer.seek(0)
    return buffer

# Banner personalizado
banner_base64 = get_image_base64_optimized(banner_path)
st.markdown(
    f"""
    <div style='
        padding: 2rem; 
        border-radius: 5px; 
        margin-bottom: 2rem; 
        background-image: url(data:image/png;base64,{banner_base64}); 
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        height: 150px;
        display: flex;
        align-items: center;
        justify-content: center;
    '>
        <h1 style='color: white; text-align: center;'>Jugadores Academia</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Gestión del estado
if 'df_jugadores' not in st.session_state:
    st.session_state.df_jugadores = None
if 'df_estadisticas' not in st.session_state:
    st.session_state.df_estadisticas = None

# Cargar datos
df_jugadores, df_estadisticas = load_data()

if df_jugadores is not None and df_estadisticas is not None:
    # Filtros
    col1, col2, col3 = st.columns(3)
    
    with col1:
        equipos = df_jugadores[df_jugadores['equipo'].str.contains('Alav', case=False, na=False)]['equipo'].unique()
        equipo = st.selectbox('Equipo:', ['Toda la Academia'] + list(equipos))

    with col2:
        # Crear un diccionario para mapear season_id -> temporada
        season_mapping = (
            df_jugadores[['season_id', 'temporada']]
            .dropna()
            .drop_duplicates()
            .astype(str)
            .set_index('season_id')['temporada']
            .to_dict()
        )

        # Filtrar por equipo seleccionado
        df_temp = df_jugadores.copy()
        if equipo != 'Toda la Academia':
            df_temp = df_temp[df_temp['equipo'] == equipo]

        # Mapear la temporada
        df_temp['temporada'] = df_temp['season_id'].astype(str).map(season_mapping)

        # Obtener y ordenar las temporadas únicas
        temporadas = sorted(
            df_temp['temporada'].dropna().unique(), 
            key=lambda x: int(x.split('/')[0]) if '/' in x else 0, 
            reverse=True
        )

        temporada = st.selectbox('Temporada:', ['Todas'] + temporadas)
        
    with col3:
        posiciones = df_jugadores[
            (df_jugadores['demarcacion'].notna()) & 
            (df_jugadores['demarcacion'] != '0') & 
            (df_jugadores['demarcacion'] != '') & 
            (df_jugadores['demarcacion'].str.strip() != '')
        ]['demarcacion'].unique()
        posiciones = sorted([pos for pos in posiciones if pos is not None])
        posicion = st.selectbox('Posición:', ['Todas'] + list(posiciones))

    # Filtrar datos para el selectbox de jugadores
    df_filtrado = df_jugadores.copy()
    
    # Filtrar por equipo
    if equipo != 'Toda la Academia':
        df_filtrado = df_filtrado[df_filtrado['equipo'] == equipo]
    else:
        df_filtrado = df_filtrado[df_filtrado['equipo'].str.contains('Alav', case=False, na=False)]

    # Filtrar por temporada
    if temporada != 'Todas':
        df_filtrado = df_filtrado[df_filtrado['temporada'] == temporada]

    # Filtrar por posición
    if posicion != 'Todas':
        df_filtrado = df_filtrado[df_filtrado['demarcacion'] == posicion]

    # Lista de jugadores filtrados
    jugadores = df_filtrado[['player_id', 'jugador']].drop_duplicates().sort_values('jugador')
    
    if not jugadores.empty:
        jugador_seleccionado = st.selectbox(
            'Jugador:',
            options=jugadores['player_id'].tolist(),
            format_func=lambda x: jugadores[jugadores['player_id'] == x]['jugador'].iloc[0]
        )

        if jugador_seleccionado:
            info_jugador = df_filtrado[df_filtrado['player_id'] == jugador_seleccionado].iloc[0]
            
            st.markdown("---")
            cols_info = st.columns(4)
            with cols_info[0]:
                st.metric("Jugador", info_jugador['jugador'])
            with cols_info[1]:
                st.metric("Equipo", info_jugador['equipo'])
            with cols_info[2]:
                st.metric("Posición", info_jugador['demarcacion'])
            with cols_info[3]:
                st.metric("Temporada", info_jugador['temporada'])

            # Botones para generar visualización y exportar
            button_cols = st.columns([1, 1, 4])
            with button_cols[0]:
                generar_viz = st.button('Generar Visualización', type='primary')
                
            # El botón de exportar PDF ahora siempre estará visible
            with button_cols[1]:
                exportar_pdf = st.button('Exportar a PDF', type='secondary')
            
            # Contenedor para la visualización
            viz_container = st.container()
            
            # Lógica al presionar el botón de generar visualización
            if generar_viz:  # Verificamos si se presionó el botón "Generar Visualización"
                st.components.v1.html(
                    """
                    <script>
                        // Cambiar el texto y aplicar el difuminado
                        const loadingText = document.getElementById("loading-text");
                        const loadingGif = document.getElementById("loading-gif");

                        loadingText.textContent = "¡Listo!";
                        loadingGif.classList.add("fade-out");

                        // Eliminar el contenedor después de la animación
                        setTimeout(() => {
                            loadingGif.remove();
                        }, 1000);  // 1000ms = 1 segundo (duración de la animación)
                    </script>
                    """,
                    height=0,  
                )                
                # Mostrar overlay de carga mientras se generan las visualizaciones
                overlay_container = st.empty()
                
                # Mostrar la pantalla de carga
                overlay_container.markdown(
                    f"""
                    <div class="loading-overlay" id="loading-gif">
                        <div class="loading-content">
                            <img src="data:image/png;base64,{get_image_base64_optimized(logo_path)}" 
                                class="loading-logo" alt="Cargando...">
                            <div class="loading-text" id="loading-text">Generando análisis...</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                with st.spinner('Generando visualización...'):
                    try:
                        # Obtenemos los season_id correspondientes a la temporada seleccionada
                        if temporada != 'Todas':
                            season_ids = df_jugadores[
                                (df_jugadores['player_id'] == jugador_seleccionado) & 
                                (df_jugadores['temporada'] == temporada)
                            ]['season_id'].unique()
                        else:
                            season_ids = df_jugadores[
                                df_jugadores['player_id'] == jugador_seleccionado
                            ]['season_id'].unique()
                        
                        with viz_container:
                            # NUEVO: Panel informativo del jugador antes de todas las gráficas
                            create_player_info_panel(df_jugadores, df_estadisticas, jugador_seleccionado, season_ids)

                            # Definir alturas consistentes para cada fila
                            ALTURA_FILA_1 = 8  # Altura para gráficas de la primera fila
                            ALTURA_FILA_2 = 8  # Altura para gráficas de la segunda fila
                            ALTURA_FILA_3 = 8  # Altura para gráficas de la tercera fila
                            
                            # Primera fila - Usamos diferentes proporciones en las columnas
                            row1_col1_2, row1_col3 = st.columns([2, 1])  # Primera columna tiene 2/3 del ancho
                            
                            # Gráfica 1: Métricas avanzadas (ocupa columnas 1 y 2)
                            with row1_col1_2:
                                st.markdown("""
                                <div class="chart-container">
                                    <div class="plot-title">Métricas del Jugador</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Mantener relación de aspecto pero ajustar altura a constante de fila 1
                                fig, ax = plt.subplots(figsize=(12, ALTURA_FILA_1), facecolor=BACKGROUND_COLOR)
                                ax.set_facecolor(BACKGROUND_COLOR)
                                plot_player_metrics_modern(ax, df_jugadores, jugador_seleccionado, season_ids)
                                st.pyplot(fig)
                                plt.close(fig)  # Cerrar figura para liberar memoria

                            # Gráfica 2: Gráfico de Pizza (columna 3 de la fila 1)
                            with row1_col3:
                                st.markdown("""
                                <div class="chart-container">
                                    <div class="plot-title">Comparativa KPI con el mejor en esa demarcación</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Ajustar la altura del gráfico de pizza para que coincida con la fila 1
                                fig_pizza, ax_pizza = plt.subplots(figsize=(6, ALTURA_FILA_1), subplot_kw=dict(polar=True), facecolor=BACKGROUND_COLOR)
                                ax_pizza.set_facecolor(BACKGROUND_COLOR)
                                create_pizza_chart(ax_pizza, df_estadisticas, jugador_seleccionado, season_ids)
                                st.pyplot(fig_pizza)
                                plt.close(fig_pizza)  # Cerrar figura para liberar memoria

                            # Segunda fila - También usando proporciones 2:1
                            row2_col1_2, row2_col3 = st.columns([2, 1])

                            # Gráfica 3: Evolución KPI (ocupa columnas 1 y 2)
                            with row2_col1_2:
                                st.markdown("""
                                <div class="chart-container">
                                    <div class="plot-title">Evolución KPI Rendimiento</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Ajustar la altura para que coincida con la fila 2
                                fig2, ax2 = plt.subplots(figsize=(12, ALTURA_FILA_2), facecolor=BACKGROUND_COLOR)
                                ax2.set_facecolor(BACKGROUND_COLOR)
                                create_kpi_evolution_chart(ax2, df_estadisticas, jugador_seleccionado, season_ids)
                                st.pyplot(fig2)
                                plt.close(fig2)  # Cerrar figura para liberar memoria

                            # Gráfica 4: Gráfico de Mapa Calor (columna 3)
                            with row2_col3:
                                st.markdown("""
                                <div class="chart-container" style="padding: 0.5rem;">
                                    <div class="plot-title" style="margin-bottom: 0.3rem; padding: 0.2rem; font-size: 12px;">Mapa de Calor</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Para mantener la consistencia vertical pero preservar la proporción del campo
                                # Nota: El mapa de calor necesita proporciones específicas para verse bien
                                fig_heatmap, ax_heatmap = plt.subplots(figsize=(2.5, ALTURA_FILA_2), facecolor=BACKGROUND_COLOR)
                                ax_heatmap.set_facecolor(BACKGROUND_COLOR)
                                
                                # Configurar el campo vertical
                                pitch = VerticalPitch(
                                    pitch_type='wyscout',
                                    axis=False, 
                                    label=False,
                                    pitch_color=BACKGROUND_COLOR,
                                    line_color='white',
                                    stripe=False,
                                    linewidth=0.5,
                                    pad_top=0,
                                    pad_bottom=0,
                                    pad_left=0,
                                    pad_right=0
                                )
                                
                                # Crear el mapa de calor
                                create_heatmap(ax_heatmap, df_jugadores, jugador_seleccionado, season_ids, pitch)
                                
                                # Ajustar los márgenes para maximizar el espacio del campo
                                plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.05)
                                st.pyplot(fig_heatmap)
                                plt.close(fig_heatmap)

                            # Tercera fila - Con proporciones 2:1 igual que las filas anteriores
                            # Definimos las columnas para la tercera fila
                            row3_col1_2, row3_col3 = st.columns([2, 1])

                            # Gráfica 5: Evolución de Acciones Exitosas (ocupa columnas 1 y 2 de la fila 3)
                            with row3_col1_2:
                                st.markdown("""
                                <div class="chart-container">
                                    <div class="plot-title">Evolución de Acciones Exitosas</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Obtener los datos del jugador
                                df_jugador_filtrado = df_jugadores[
                                    (df_jugadores['player_id'] == jugador_seleccionado) & 
                                    (df_jugadores['season_id'].isin(season_ids))
                                ].copy()
                                
                                # Ajustar la altura para que coincida con la fila 3
                                fig_events, ax_events = plt.subplots(figsize=(12, ALTURA_FILA_3), facecolor=BACKGROUND_COLOR)
                                ax_events.set_facecolor(BACKGROUND_COLOR)
                                create_positive_actions_chart(df_jugador_filtrado, ax_events)
                                st.pyplot(fig_events)
                                plt.close(fig_events)  # Cerrar figura para liberar memoria

                            # Gráfica 6: Mapa de Pases Progresivos (columna 3 de la fila 3)
                            with row3_col3:
                                st.markdown("""
                                <div class="chart-container" style="padding: 0.5rem;">
                                    <div class="plot-title" style="margin-bottom: 0.3rem; padding: 0.2rem; font-size: 12px;">Mapa de Pases Orientados Progresivos</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                try:
                                    # Ajustar para mantener la consistencia vertical con las otras gráficas de la fila 3
                                    fig_passes, ax_passes = plt.subplots(figsize=(2.5, ALTURA_FILA_3), facecolor=BACKGROUND_COLOR)
                                    ax_passes.set_facecolor(BACKGROUND_COLOR)
                                    
                                    # Definir el campo vertical con las configuraciones exactas
                                    pitch = VerticalPitch(
                                        pitch_type='wyscout',
                                        axis=False, 
                                        label=False,
                                        pitch_color='#22312b',
                                        line_color='white',
                                        stripe=False,
                                        linewidth=0.5,
                                        pad_top=0,
                                        pad_bottom=0,
                                        pad_left=0,
                                        pad_right=0
                                    )
                                    
                                    # Llamar a la función para dibujar los pases combinados
                                    draw_combined_passes(ax_passes, df_jugadores, jugador_seleccionado, season_ids, pitch)
                                    
                                    # Ajustar los márgenes para maximizar el espacio del campo
                                    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.05)
                                    
                                    # Mostrar el gráfico
                                    st.pyplot(fig_passes)
                                    plt.close(fig_passes)
                                    
                                except Exception as e:
                                    st.error(f"Error al generar el mapa de pases combinados: {str(e)}")
                            
                            # Añadir titulos de las gráficas:
                            titulos_graficas = [
                                "Métricas del Jugador",
                                "Comparativa KPI con el mejor en esa demarcación",
                                "Evolución KPI Rendimiento",
                                "Mapa de Calor",
                                "Evolución de Acciones Exitosas",
                                "Mapa de Pases Orientados Progresivos"
                            ]

                            st.session_state.figuras_generadas = [fig, fig_pizza, fig2, fig_heatmap, fig_events, fig_passes]
                            st.session_state.titulos_graficas = titulos_graficas  # Guardar los títulos
                            st.session_state.season_ids = season_ids  # Guardar season_ids para uso posterior
                            st.session_state.visualizacion_generada = True
                           
                            # Limpiar la pantalla de carga cuando todo esté listo
                            overlay_container.empty()
                            
                    except Exception as e:
                        # También limpiar la pantalla de carga en caso de error
                        overlay_container.empty()
                        st.error(f"Error al generar visualización: {e}")
                        import traceback
                        st.text(traceback.format_exc())

            # Lógica para exportar a PDF
            if 'visualizacion_generada' in st.session_state and st.session_state.visualizacion_generada:
                if 'exportar_pdf' in locals() and exportar_pdf:
                    with st.spinner('Generando PDF...'):
                        try:
                            # Generar el PDF con las figuras guardadas y los season_ids guardados
                            if 'titulos_graficas' in st.session_state:
                                # Usar títulos almacenados en la sesión
                                pdf_buffer = generar_pdf(
                                    df_jugadores, 
                                    df_estadisticas, 
                                    jugador_seleccionado, 
                                    st.session_state.season_ids,
                                    st.session_state.figuras_generadas
                                )
                            else:
                                # Si no hay títulos almacenados, usar predeterminados
                                titulos_graficas = [
                                    "Métricas del Jugador",
                                    "Comparativa KPI con el mejor en esa demarcación",
                                    "Evolución KPI Rendimiento",
                                    "Mapa de Calor",
                                    "Evolución de Acciones Exitosas",
                                    "Mapa de Pases Orientados Progresivos"
                                ]
                                pdf_buffer = generar_pdf(
                                    df_jugadores, 
                                    df_estadisticas, 
                                    jugador_seleccionado, 
                                    st.session_state.season_ids,
                                    st.session_state.figuras_generadas
                                )
                            
                            # Ofrecer la descarga del PDF
                            nombre_archivo = f"Informe_{info_jugador['jugador'].replace(' ', '_')}.pdf"
                            
                            # Usar el objeto BytesIO directamente con st.download_button
                            st.download_button(
                                label="Descargar PDF",
                                data=pdf_buffer,
                                file_name=nombre_archivo,
                                mime="application/pdf",
                                key='download_pdf'
                            )
                            
                            st.success(f"¡PDF generado! Haz clic en 'Descargar PDF' para guardarlo.")
                            
                        except Exception as e:
                            st.error(f"Error al generar el PDF: {str(e)}")
                            import traceback
                            st.text(traceback.format_exc())    
                    