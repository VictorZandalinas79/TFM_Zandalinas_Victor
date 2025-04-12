import numpy as np
import streamlit as st
import common.menu as menu
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch
from matplotlib.colors import LinearSegmentedColormap, to_rgba
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import base64
import os
from pathlib import Path
from data_manager_match import DataManagerMatch
import re
import seaborn as sns
from thefuzz import fuzz
from PIL import Image
import io
import math
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as ReportLabImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from PIL import Image

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Partidos",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None
)

# Limpiar caché de Streamlit
st.cache_data.clear()

# Inicialización del caché si no existe
if 'visualization_cache' not in st.session_state:
    st.session_state.visualization_cache = {}

# Base directory and asset paths setup
BASE_DIR = Path(__file__).parent.parent
icon_path = os.path.join(BASE_DIR, 'assets', 'icono_player.png')
logo_path = os.path.join(BASE_DIR, 'assets', 'escudo_alaves_original.png')
jugador_path = os.path.join(BASE_DIR, 'assets', 'jugador_alaves.png')
fondo_path = os.path.join(BASE_DIR, 'assets', 'fondo_alaves.png')
banner_path = os.path.join(BASE_DIR, 'assets', 'bunner_alaves.png')

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
        /* Oculta la lista automática de páginas en el sidebar */ 
        [data-testid="stSidebar"] 
        .css-1d391kg { display: none; }
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

# Función para obtener imagen base64
def get_image_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache_data 
def get_image_base64_optimized(path, max_size=(1024, 1024), quality=70): 
    try: 
        image = Image.open(path) 
        image.thumbnail(max_size, Image.LANCZOS) 
        if image.mode in ("RGBA", "P"): 
            image = image.convert("RGB") 
            buffer = io.BytesIO() 
            image.save(buffer, format="JPEG", quality=quality) 
            buffer.seek(0) 
            return base64.b64encode(buffer.getvalue()).decode() 
    except Exception as e: 
        st.error(f"Error cargando imagen: {e}") 
        return None

icon_base64 = get_image_base64_optimized(icon_path)
fondo_base64 = get_image_base64_optimized(fondo_path)
banner_base64 = get_image_base64_optimized(banner_path)

# Custom CSS
st.markdown(
    f"""
    <div style='
        padding: 2rem; 
        border-radius: 5px; 
        margin-bottom: 2rem; 
        background-image: url(data:image/png;base64,{get_image_base64_optimized("assets/bunner_alaves.png")}); 
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        height: 150px;
        display: flex;
        align-items: center;
        justify-content: center;
    '>
        <h1 style='color: white; text-align: center;'>Análisis de Partidos</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Global colors
BACKGROUND_COLOR = '#0E1117'
LINE_COLOR = '#FFFFFF'
TEXT_COLOR = '#FFFFFF'
HIGHLIGHT_COLOR = '#4BB3FD'

# Función para extraer equipos local y visitante de la columna 'partido'
def extract_teams(partido):
    match = re.match(r"^\d+\s*-\s*(.*?)\s*\d+.*\d+\s*(.*)$", partido)
    if match:
        local_team = match.group(1).strip()
        away_team = match.group(2).strip()
        return local_team, away_team
    return None, None

# Función para encontrar el nombre más similar en la columna 'equipo'
def find_similar_team(team_name, equipo_list):
    if not team_name:
        return None
    best_match = None
    best_score = 0
    for equipo in equipo_list:
        score = fuzz.ratio(team_name, equipo)
        if score > best_score and score > 70:  # Umbral de similitud
            best_score = score
            best_match = equipo
    return best_match

# Función para generar la visualización del partido
def calculate_team_kpi(df_eventos, df_estadisticas, team):
    """
    Calcula el KPI de rendimiento para un equipo en un partido específico.
    
    Parámetros:
    - df_eventos: DataFrame de eventos del partido
    - df_estadisticas: DataFrame de estadísticas del partido
    - team: Nombre del equipo
    
    Retorna:
    - Valor de KPI de rendimiento (escalado de 0 a 10)
    """
    try:
        # 1. Obtener el match_id del partido para el equipo
        match_id = df_eventos[df_eventos['equipo'] == team]['match_id'].iloc[0]
        
        # 2. Filtrar estadísticas para el equipo específico en este partido
        team_stats = df_estadisticas[
            (df_estadisticas['match_id'] == match_id) & 
            (df_estadisticas['equipo'] == team)
        ]
        
        # 3. Si no hay estadísticas para el equipo, retornar un valor por defecto
        if team_stats.empty:
            st.error(f"No hay estadísticas para el equipo {team} en este partido")
            return 5.0  # Valor por defecto
        
        # 4. Extraer el KPI de rendimiento
        kpi_rendimiento = team_stats['KPI_rendimiento'].values[0]
        
        # 5. Verificar si el KPI ya está en la escala correcta (0 a 10)
        # Si el KPI está en una escala de 0 a 1, escalarlo a 0-10
        if kpi_rendimiento <= 1.0:  # Asumimos que si el valor es <= 1, está en escala 0-1
            kpi_escalado = kpi_rendimiento * 10
        else:
            kpi_escalado = kpi_rendimiento  # Si ya está en escala 0-10, no escalar
        
        # 6. Asegurarse de que el KPI esté dentro del rango 0-10
        kpi_escalado = max(0, min(kpi_escalado, 10))  # Limitar entre 0 y 10
        
        # 7. Depuración: Mostrar valores en la interfaz de Streamlit
        st.write(f"Equipo: {team}")
        st.write(f"Match ID: {match_id}")
        st.write(f"KPI Rendimiento (crudo): {kpi_rendimiento}")
        st.write(f"KPI Rendimiento (escalado): {kpi_escalado}")
        
        return kpi_escalado
    
    except Exception as e:
        st.error(f"Error calculando KPI para {team}: {e}")
        return 5.0  # Valor por defecto si hay un error

def calculate_team_kpi(df_eventos, df_estadisticas, team):
    """
    Calcula el KPI de rendimiento para un equipo en un partido específico.
    
    Parámetros:
    - df_eventos: DataFrame de eventos del partido
    - df_estadisticas: DataFrame de estadísticas del partido
    - team: Nombre del equipo
    
    Retorna:
    - Valor de KPI de rendimiento (escalado de 0 a 10)
    """
    try:
        # Filtrar estadísticas para el equipo específico
        team_stats = df_estadisticas[df_estadisticas['equipo'] == team]
        
        # Si no hay estadísticas para el equipo, retornar un valor por defecto
        if team_stats.empty:
            return 5.0
        
        # Extraer el KPI de rendimiento
        kpi_rendimiento = team_stats['KPI_rendimiento'].iloc[0]
        
        # Escalar el KPI a una escala de 0 a 10
        # Asumiendo que el KPI original puede estar en una escala diferente
        kpi_escalado = min(max(kpi_rendimiento, 0), 10)
        
        return kpi_escalado
    
    except Exception as e:
        print(f"Error calculando KPI para {team}: {e}")
        return 5.0 
    
def draw_single_kpi_square(ax, team, kpi_value):
    """Helper function to draw a single KPI square"""
    ax.set_facecolor(BACKGROUND_COLOR)
    
    def get_color(value):
        """
        Devuelve un color basado en el valor del KPI.git
        - Rojo disminuye a medida que el valor aumenta.
        - Verde aumenta a medida que el valor aumenta.
        """
        normalized_value = value / 10  # Normalizar el valor entre 0 y 1
        red = 1 - normalized_value  # Rojo disminuye
        green = normalized_value    # Verde aumenta
        return (red, green, 0)      # Azul fijo en 0
    
    square_size = 1.5
    corner_radius = 0.2
    
    # Center the square in the available space
    square = patches.FancyBboxPatch(
        (-square_size/2, -square_size/2),
        square_size, square_size,
        boxstyle=f"round,pad=0,rounding_size={corner_radius}",
        facecolor=get_color(kpi_value),
        edgecolor='white',
        linewidth=3
    )
    ax.add_patch(square)
    
    # Add text with shadow effect
    text_effect = [path_effects.withStroke(linewidth=3, foreground='black')]
    
    ax.text(0, 0.8, f"{team}",
        ha='center', va='center',
        color='white', fontsize=18,
        fontweight='bold', fontstyle='italic',
        path_effects=text_effect)
    
    ax.text(0, -0.1, f"{kpi_value:.2f}",
        ha='center', va='center',
        color='white', fontsize=60,
        fontweight='bold', fontstyle='italic',
        path_effects=text_effect)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

# Función pases progresivos
def draw_passing_map(ax, df_eventos, team_name, match_id, color):
    """
    Dibuja un mapa de pases progresivos para un equipo específico.
    
    Args:
        ax: El eje de matplotlib donde dibujar
        df_eventos: DataFrame con los eventos del partido
        team_name: Nombre del equipo
        match_id: ID del partido
        color: Color para las flechas y el texto
    """
    # Filtrado de eventos para el equipo y partido específico
    # Asegurarse de que tipo_evento es un string antes de aplicar string methods
    team_events = df_eventos[
        (df_eventos['equipo'] == team_name) & 
        (df_eventos['match_id'] == match_id) &
        (df_eventos['tipo_evento'].astype(str).str.strip().str.lower() == 'pase')
    ].copy()
        
    if team_events.empty:
        print(f"No hay datos de pases para {team_name}")
        ax.text(0.5, 0.5, f"No hay datos de pases para {team_name}",
               ha='center', va='center', color='white', fontsize=16)
        ax.set_facecolor(BACKGROUND_COLOR)
        ax.axis('off')
        return
    
    # Configuración del pitch
    pitch = VerticalPitch(pitch_type='wyscout', pitch_color='None', line_color=LINE_COLOR, half=False)
    pitch.draw(ax=ax)
    
    # Dibujar líneas verticales discontinuas para dividir los carriles (izquierdo, central, derecho)
    # Posicionamos las líneas en x=30 y x=70
    ax.axvline(x=30, color=LINE_COLOR, linestyle='--', alpha=0.5, linewidth=2)
    ax.axvline(x=70, color=LINE_COLOR, linestyle='--', alpha=0.5, linewidth=2)
    
    # Añadir etiquetas para los carriles
    ax.text(15, 95, "CARRIL IZQUIERDO", color=LINE_COLOR, ha='center', va='bottom', 
            fontsize=8, fontweight='bold', alpha=0.9)
    ax.text(50, 95, "CARRIL CENTRAL", color=LINE_COLOR, ha='center', va='bottom', 
            fontsize=8, fontweight='bold', alpha=0.9)
    ax.text(85, 95, "CARRIL DERECHO", color=LINE_COLOR, ha='center', va='bottom', 
            fontsize=8, fontweight='bold', alpha=0.9)
    
    # Convertir columnas a numéricas
    numeric_cols = ['xstart', 'ystart', 'xend', 'yend',
                    'pases_progresivos_inicio', 'pases_progresivos_creacion',
                    'pases_progresivos_finalizacion']
    
    for col in numeric_cols:
        if col in team_events.columns:
            team_events[col] = pd.to_numeric(team_events[col], errors='coerce')
        else:
            print(f"Columna {col} no encontrada en el DataFrame")
            team_events[col] = np.nan
    
    # Eliminar filas con valores NaN en las coordenadas
    team_events = team_events.dropna(subset=['xstart', 'ystart', 'xend', 'yend'])
        
    if team_events.empty:
        print(f"No hay datos válidos de pases para {team_name} después de limpiar NaNs")
        ax.text(0.5, 0.5, f"No hay datos válidos de pases para {team_name}",
               ha='center', va='center', color='white', fontsize=16)
        ax.set_facecolor(BACKGROUND_COLOR)
        ax.axis('off')
        return
    
    # Dibujar flechas para los diferentes tipos de pases progresivos
    for tipo in ['pases_progresivos_inicio', 'pases_progresivos_creacion', 'pases_progresivos_finalizacion']:
        if tipo not in team_events.columns:
            print(f"Columna {tipo} no encontrada")
            continue
            
        prog_passes = team_events[team_events[tipo] == 1.0]
        if not prog_passes.empty:
            pitch.arrows(
                prog_passes['ystart'], prog_passes['xstart'],
                prog_passes['yend'], prog_passes['xend'],
                ax=ax, color=color, width=4, alpha=0.7
            )
    
    # Calcular el total de pases progresivos
    if all(col in team_events.columns for col in ['pases_progresivos_inicio', 'pases_progresivos_creacion', 'pases_progresivos_finalizacion']):
        # Obtener todos los pases progresivos
        pases_progresivos = team_events[
            (team_events['pases_progresivos_inicio'] == 1.0) |
            (team_events['pases_progresivos_creacion'] == 1.0) |
            (team_events['pases_progresivos_finalizacion'] == 1.0)
        ]
        
        total_progresivos = len(pases_progresivos)
                
        if total_progresivos > 0:
            # Definir los carriles laterales por rango de coordenada X
            carriles = {
                'Izquierdo': (0, 30, 15),    # Carril izquierdo: x entre 0 y 30
                'Central': (30, 70, 50),      # Carril central: x entre 30 y 70
                'Derecho': (70, 100, 85)      # Carril derecho: x entre 70 y 100
            }
            
            import matplotlib.patches as patches
            
            for carril, (min_x, max_x, x_pos) in carriles.items():
                # Calcular pases por carril basado en la coordenada X inicial (xstart)
                pases_carril = len(pases_progresivos[
                    (pases_progresivos['xend'] >= min_x) & 
                    (pases_progresivos['xend'] < max_x)
                ])
                
                porcentaje = (pases_carril/total_progresivos) * 100 if total_progresivos > 0 else 0
                                
                # Crear una flecha blanca con 75% de transparencia
                arrow = patches.FancyArrow(
                    x=x_pos,          # Posición x inicial
                    y=2,              # Posición y inicial (punto de inicio más bajo para que apunte hacia arriba)
                    dx=0,             # Desplazamiento en x (0 para que vaya vertical)
                    dy=6,             # Desplazamiento en y (positivo para que vaya hacia arriba)
                    width=19,          # Ancho de la flecha
                    head_width=22,     # Ancho de la punta de la flecha
                    head_length=8,    # Longitud de la punta de la flecha
                    facecolor='white',
                    edgecolor=color,
                    alpha=0.80,       # 80% de opacidad
                    linewidth=6,
                    zorder=10         # Asegura que esté por encima del fondo
                )
                ax.add_patch(arrow)
                
                # Añadir el texto del porcentaje encima del círculo
                # Usar rojo intenso para el equipo local o azul intenso para el visitante
                text_color = '#0000FF' if color == '#3498db' else '#FF0000'  # Rojo para local, Azul para visitante
                
                ax.text(x_pos, 6, f"{porcentaje:.1f}%\n({pases_carril})",
                       ha='center', va='center', color=text_color, 
                       fontsize=22, fontweight='bold', zorder=11)  # zorder mayor que el círculo
    else:
        print(f"Columnas de pases progresivos no encontradas para {team_name}")
    
# Función red de pases
def create_pass_network(df_combined, equipo, temporada, custom_node_color=None):
    """
    Versión optimizada del mapa de red de pases entre demarcaciones.
    - Usa abreviaturas en español para las demarcaciones
    - Permite visualizar pases entre jugadores de la misma demarcación
    - Unifica laterales y carrileros
    
    Args:
        df_combined: DataFrame con los datos combinados
        equipo: Nombre del equipo a analizar
        temporada: Temporada a analizar
        custom_node_color: Color personalizado para los nodos (círculos)
        
    Returns:
        Figura de matplotlib con el mapa de red de pases
    """
    try:
        # Verificar si existe la columna 'demarcacion'
        if 'demarcacion' not in df_combined.columns:
            fig, ax = plt.subplots(figsize=(5, 7), facecolor=BACKGROUND_COLOR)
            ax.text(0.5, 0.5, "No hay datos de demarcación\ndisponibles para este análisis",
                   ha='center', va='center', color='white', fontsize=12)
            ax.set_facecolor(BACKGROUND_COLOR)
            ax.axis('off')
            return fig
        
        # Paso 1: Filtramos solo los datos necesarios
        df_filtered = df_combined[
            (df_combined['equipo'] == equipo) & 
            (df_combined['temporada'] == temporada) &
            (df_combined['tipo_evento'] == 'Pase')
        ].copy()
        
        # Seleccionar solo las columnas que necesitamos
        try:
            df_passes = df_filtered[['demarcacion', 'xstart', 'ystart', 'xend', 'yend']].copy()
        except KeyError as e:
            # Si hay un error con alguna columna, mostrar mensaje y retornar
            fig, ax = plt.subplots(figsize=(5, 7), facecolor=BACKGROUND_COLOR)
            ax.text(0.5, 0.5, f"Error al acceder a columnas:\n{str(e)}",
                   ha='center', va='center', color='white', fontsize=12)
            ax.set_facecolor(BACKGROUND_COLOR)
            ax.axis('off')
            return fig
        
        # Paso 2: Aplicar una muestra si hay demasiados datos
        if len(df_passes) > 500:  # Reducido de 1000 a 500
            df_passes = df_passes.sample(n=500, random_state=42)
        
        # Si no hay suficientes pases, mostrar mensaje
        if len(df_passes) < 10:
            fig, ax = plt.subplots(figsize=(5, 7), facecolor=BACKGROUND_COLOR)
            ax.text(0.5, 0.5, f"No hay suficientes pases para crear la red\npara {equipo} en {temporada}",
                   ha='center', va='center', color='white', fontsize=12)
            ax.set_facecolor(BACKGROUND_COLOR)
            ax.axis('off')
            return fig
        
        # El resto del código sigue igual...
        # Paso 3: Convertir a tipos numéricos de una vez
        for col in ['xstart', 'ystart', 'xend', 'yend']:
            df_passes[col] = pd.to_numeric(df_passes[col], errors='coerce', downcast='float')
        
        # Paso 4: Eliminar valores nulos
        df_passes = df_passes.dropna(subset=['xstart', 'ystart', 'xend', 'yend', 'demarcacion'])
        
        # Paso 5: Unificar demarcaciones y crear abreviaturas en español
        # Mapeo de posiciones originales a unificadas
        position_mapping = {
            'LF': 'Delantero Izquierdo', 'CF': 'Delantero Centro', 'RF': 'Delantero Derecho',
            'LW': 'Extremo Izquierdo', 'CAM': 'Mediapunta', 'RW': 'Extremo Derecho',
            'LM': 'Interior Izquierdo', 'CM': 'Mediocentro', 'RM': 'Interior Derecho',
            'CDM': 'Mediocentro Defensivo', 
            'LWB': 'Lateral Izquierdo', 'LB': 'Lateral Izquierdo',  # Unificados
            'RWB': 'Lateral Derecho', 'RB': 'Lateral Derecho',      # Unificados
            'CB': 'Defensa Central', 'GK': 'Portero'
        }
        
        # Mapeo de posiciones unificadas a abreviaturas en español
        abbr_mapping_es = {
            'Delantero Izquierdo': 'DI', 'Delantero Centro': 'DC', 'Delantero Derecho': 'DD',
            'Extremo Izquierdo': 'EI', 'Mediapunta': 'MP', 'Extremo Derecho': 'ED',
            'Interior Izquierdo': 'II', 'Mediocentro': 'MC', 'Interior Derecho': 'ID',
            'Mediocentro Defensivo': 'MD', 
            'Lateral Izquierdo': 'LI',  # Unificado
            'Lateral Derecho': 'LD',    # Unificado
            'Defensa Central': 'CT',    # Abreviatura en español para Defensa Central
            'Portero': 'PO'
        }
        
        # Crear un campo con demarcaciones unificadas
        df_passes['demarcacion_unificada'] = df_passes['demarcacion'].map(
            lambda x: position_mapping.get(x, x)
        )
        
        # Paso 6: Configurar campo y crear figura
        pitch = VerticalPitch(
            pitch_type='wyscout',
            axis=False, 
            label=False,  
            pitch_color='none',
            line_color='white',
            stripe=False
        )
        
        fig, ax = plt.subplots(figsize=(5, 7), facecolor=BACKGROUND_COLOR)
        ax.set_facecolor(BACKGROUND_COLOR)
        pitch.draw(ax=ax)
        
        # Paso 7: Calcular posición media para cada demarcación
        demarcation_positions = {}
        
        for demarcacion in df_passes['demarcacion_unificada'].unique():
            # Filtrar pases de esta demarcación
            positions_df = df_passes[df_passes['demarcacion_unificada'] == demarcacion]
            
            # Calcular posición media para cada demarcación
            avg_pos = (positions_df['xstart'].mean(), positions_df['ystart'].mean())
            demarcation_positions[demarcacion] = avg_pos
        
        # Paso 8: Preparar datos para los nodos
        node_data = []
        for key, (x, y) in demarcation_positions.items():
            count = df_passes[df_passes['demarcacion_unificada'] == key].shape[0]
            
            # Añadimos a la lista de nodos
            node_data.append({
                'demarcacion': key,
                'xstart': x, 
                'ystart': y,
                'count': count
            })
        
        # Convertir a DataFrame
        node_df = pd.DataFrame(node_data)
        
        # Paso 9: Simplificar el cálculo del tamaño de nodos
        min_count = node_df['count'].min()
        max_count = node_df['count'].max()
        # Escalar entre 0.3 y 1.5 para un tamaño más consistente
        node_df['size'] = 0.3 + ((node_df['count'] - min_count) / max(1, max_count - min_count)) * 1.2
        
        # Paso 10: Crear conexiones - analizar los pases para identificar conexiones
        connections = []
        
        # Crear diccionario para búsqueda rápida de posiciones de nodos
        position_lookup = {dem: (x, y) for dem, (x, y) in demarcation_positions.items()}
        
        # Para cada pase en el conjunto de datos
        for _, pass_row in df_passes.iterrows():
            origin_demarcation = pass_row['demarcacion_unificada']
            end_x, end_y = pass_row['xend'], pass_row['yend']
            
            # Determinar origen específico (si hay múltiples para esta demarcación)
            origin_options = [k for k in position_lookup.keys() if k.startswith(origin_demarcation)]
            
            if len(origin_options) > 1:
                # Si hay múltiples opciones, elegir la más cercana al punto de inicio
                start_x, start_y = pass_row['xstart'], pass_row['ystart']
                origin = min(origin_options, key=lambda k: 
                             ((position_lookup[k][0] - start_x)**2 + 
                              (position_lookup[k][1] - start_y)**2)**0.5)
            else:
                origin = origin_options[0]
            
            # Encontrar el destino más cercano
            min_dist = float('inf')
            target = None
            
            for dem, (pos_x, pos_y) in position_lookup.items():
                # Ahora permitimos conexiones a la misma demarcación
                dist = ((pos_x - end_x)**2 + (pos_y - end_y)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    target = dem
            
            if target is not None:
                connections.append((origin, target))
        
        # Paso 11: Contar conexiones 
        connection_counts = {}
        for origin, target in connections:
            key = (origin, target)
            connection_counts[key] = connection_counts.get(key, 0) + 1
        
        # Paso 12: Filtrar solo las conexiones más frecuentes
        sorted_connections = sorted(connection_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Tomar solo las top 20 conexiones
        top_connections = sorted_connections[:20]
        
        # Paso 13: Dibujar las conexiones principales
        for (origin, target), count in top_connections:
            # Obtener coordenadas
            origin_x, origin_y = position_lookup[origin]
            target_x, target_y = position_lookup[target]
            
            # Calcular grosor de línea - simplificado
            width = 1 + (count / max(connection_counts.values()) * 4)
            alpha = 0.2 + (count / max(connection_counts.values()) * 0.7)
            
            if origin == target:
                # Autoconexión (pases dentro de la misma demarcación)
                # Crear un arco para pases a la misma posición
                arc = patches.Arc(
                    (origin_x, origin_y), 
                    width=2.0, 
                    height=2.0, 
                    angle=0,
                    theta1=30, 
                    theta2=330, 
                    linewidth=width,
                    color=HIGHLIGHT_COLOR,  # Usar el color global que se pasa como HIGHLIGHT_COLOR
                    alpha=alpha,
                    zorder=1
                )
                ax.add_patch(arc)
            else:
                # Dibujar línea recta para pases entre diferentes posiciones
                pitch.lines(
                    origin_y, origin_x,  # Intercambiamos x e y para el campo vertical
                    target_y, target_x,
                    lw=width,
                    color=HIGHLIGHT_COLOR,  # Usar el color global que se pasa como HIGHLIGHT_COLOR
                    alpha=alpha,
                    zorder=1,
                    ax=ax
                )
        
        # Paso 14: Dibujar nodos con tamaños simplificados
        for _, row in node_df.iterrows():
            # Obtenemos la abreviatura en español
            base_demarcation = row.demarcacion
            abreviatura = abbr_mapping_es.get(base_demarcation, base_demarcation[:2])
            
            # Usar el color personalizado para los nodos si se proporciona
            node_color = custom_node_color if custom_node_color else '#0066CC'
            
            # Color para los círculos con borde blanco
            circle = plt.Circle(
                (row.xstart, row.ystart),
                row.size,  # Tamaño más razonable
                color=node_color,  # Color personalizado o por defecto azul
                alpha=0.9,
                zorder=2,
                edgecolor='white',  # Borde blanco
                linewidth=1.2  # Grosor del borde
            )
            ax.add_patch(circle)
            
            # Añadir etiqueta en negrita
            ax.annotate(
                abreviatura,
                xy=(row.xstart, row.ystart),
                c='white',
                va='center',
                ha='center',
                fontsize=9,
                fontweight='bold',  # Aseguramos que esté en negrita
                zorder=3
            )
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        # Si hay un error, creamos un gráfico con el mensaje de error
        fig, ax = plt.subplots(figsize=(5, 7), facecolor=BACKGROUND_COLOR)
        ax.text(
            0.5, 0.5, 
            f"Error al crear la red de pases:\n{str(e)}",
            ha='center', 
            va='center', 
            color='red', 
            fontsize=10
        )
        print(f"Error detallado: {e}")  # Para depuración
        import traceback
        print(traceback.format_exc())  # Agrega trazas completas para mejor depuración
        ax.set_facecolor(BACKGROUND_COLOR)
        return fig
    
# Función para dibujar el heatmap defensivo
def draw_defensive_heatmap(ax, df_eventos, team_name, match_id, color):
    team_events = df_eventos[
        (df_eventos['equipo'] == team_name) & 
        (df_eventos['match_id'] == match_id) &
        (df_eventos['tipo_evento'].isin(['Recuperación', 'Intercepción', 'Entrada']))
    ].copy()
    
    if team_events.empty:
        ax.text(0.5, 0.5, f"No hay datos defensivos para {team_name}",
               ha='center', va='center', color='white', fontsize=16)
        ax.set_facecolor(BACKGROUND_COLOR)
        ax.axis('off')
        return
    
    pitch = VerticalPitch(pitch_type='wyscout', pitch_color='None', line_color=LINE_COLOR, line_zorder=2, half=False)
    pitch.draw(ax=ax)
    
    team_events['ystart'] = pd.to_numeric(team_events['ystart'], errors='coerce')
    team_events['xstart'] = pd.to_numeric(team_events['xstart'], errors='coerce')
    team_events = team_events.dropna(subset=['xstart', 'ystart'])
    
    if not team_events.empty:
        # Configuramos colores según si es local (azul) o visitante (rojo)
        if color == 'Blues':
            # Para equipo local (azul)
            dot_color = '#4BB3FD'
            fill_color = 'Blues'
        else:
            # Para equipo visitante (rojo)
            dot_color = '#ff3333'
            fill_color = 'Reds'
        
        # Usar la paleta apropiada con fill=True para llenar con colores
        sns.kdeplot(
            data=team_events,
            x='xstart',
            y='ystart',
            fill=True,
            alpha=0.5,
            cmap=fill_color,
            levels=20,
            ax=ax
        )
        
        # Añadir puntos individuales
        ax.scatter(
            team_events['xstart'],
            team_events['ystart'],
            c=dot_color,
            alpha=0.6,
            s=25
        )
        
        total_acciones = len(team_events)
        ax.text(50, -5, f'Total acciones defensivas exitosas: {total_acciones}',
               ha='center', va='top', color='white', fontsize=10)

# Función para los disparos (xg)import math
def draw_shot_analysis(ax, df_eventos, team_name, match_id, color):
    """
    Dibuja un mapa de disparos para un equipo específico y un mapa de centros debajo.
    
    Args:
        ax: El eje de matplotlib donde dibujar el mapa de disparos
        df_eventos: DataFrame con los eventos del partido
        team_name: Nombre del equipo
        match_id: ID del partido
        color: Color base para los elementos
    """
    # Preparar una copia del DataFrame para evitar modificar el original
    df_eventos = df_eventos.copy()
    
    # Convertir la columna 'centro' a un formato más manejable si existe
    if 'centro' in df_eventos.columns:
        # Convertir a string primero para manejar cualquier tipo de datos
        df_eventos['centro_str'] = df_eventos['centro'].astype(str)
        
        # Crear una columna booleana basada en valores comunes para "verdadero"
        df_eventos['centro_bool'] = df_eventos['centro_str'].str.lower().isin(['true', '1', 'yes', 'y', 'si', 's', 'sí'])
    else:
        # Si no existe la columna, crear una vacía
        df_eventos['centro_bool'] = False
        print("Advertencia: No se encontró la columna 'centro' en el DataFrame")
    
    # Crear figura con dos subplots, maximizando espacio disponible
    fig = ax.figure
    # Eliminar hspace completamente para maximizar espacio
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.01)
    ax_shots = fig.add_subplot(gs[0])
    ax_crosses = fig.add_subplot(gs[1])
    
    # Ajustar el tamaño de la figura para maximizar espacio
    fig.set_size_inches(14, 20)  # Aumentar tamaño para más espacio
    
    # Eliminar todo el espacio adicional alrededor de los ejes
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    
    team_shots = df_eventos[
        (df_eventos['equipo'] == team_name) & 
        (df_eventos['tipo_evento'] == 'Disparo') &
        (df_eventos['match_id'] == match_id)
    ].copy()
    
    # Filtrar centros del equipo usando nuestra nueva columna booleana
    team_crosses = df_eventos[
        (df_eventos['equipo'] == team_name) &
        (df_eventos['centro_bool'] == True) &
        (df_eventos['match_id'] == match_id)
    ].copy()
    
    # PARTE 1: MAPA DE DISPAROS
    if team_shots.empty:
        ax_shots.text(0.5, 0.5, f"No hay datos de disparos para {team_name}",
               ha='center', va='center', color='white', fontsize=16)
        ax_shots.set_facecolor(BACKGROUND_COLOR)
        ax_shots.axis('off')
    else:
        # Crear el campo de fútbol (solo la mitad) con tamaño maximizado
        pitch_shots = VerticalPitch(
            pitch_type='wyscout', 
            pitch_color='None', 
            line_color=LINE_COLOR, 
            line_zorder=2, 
            half=True,
            pad_top=0,       # Eliminar padding completamente
            pad_bottom=0,
            pad_left=0,
            pad_right=0
        )
        pitch_shots.draw(ax=ax_shots)
        
        # Convertir columnas numéricas
        numeric_cols = ['ystart', 'xstart', 'xg']
        for col in numeric_cols:
            if col in team_shots.columns:
                team_shots[col] = pd.to_numeric(team_shots[col], errors='coerce')
            else:
                print(f"Columna {col} no encontrada en el DataFrame")
                team_shots[col] = np.nan
        
        # Eliminar filas con valores NaN en las coordenadas
        team_shots = team_shots.dropna(subset=['xstart', 'ystart'])
        
        if team_shots.empty:
            ax_shots.text(0.5, 0.5, f"No hay datos válidos de disparos para {team_name}",
                   ha='center', va='center', color='white', fontsize=16)
            ax_shots.set_facecolor(BACKGROUND_COLOR)
            ax_shots.axis('off')
        else:
            # Tamaño de los puntos basado en xG
            min_size = 100  # Aumentamos el tamaño mínimo (era 70)
            max_size = 700  # Aumentamos el tamaño máximo (era 550)
            if 'xg' not in team_shots.columns or team_shots['xg'].isna().all():
                # Si no hay datos de xG, usar un valor por defecto
                team_shots['xg'] = 0.05
            
            sizes = min_size + (team_shots['xg'].fillna(0.05) * (max_size - min_size))
            
            # Obtener resultados únicos
            if 'resultado' not in team_shots.columns:
                team_shots['resultado'] = 'Disparo'
            
            resultados = team_shots['resultado'].unique()
            
            if len(resultados) > 0:  # Solo si hay resultados
                # Paleta de colores para resultados (usamos 'tab20' para más variedad)
                colores = plt.cm.tab20.colors
                color_map = {}
                
                for i, resultado in enumerate(resultados):
                    if resultado == 'Gol':
                        color_map[resultado] = 'yellow'
                    elif resultado == 'Poste':
                        color_map[resultado] = 'orange'
                    elif resultado == 'Parada':
                        color_map[resultado] = 'lightblue'
                    elif resultado == 'Fuera':
                        color_map[resultado] = 'lightgray'
                    elif resultado == 'Bloqueado':
                        color_map[resultado] = 'darkgray'
                    else:
                        color_map[resultado] = colores[i % len(colores)]
                        
                handles = []
                labels = []
                
                for resultado in resultados:
                    mask = team_shots['resultado'] == resultado
                    if sum(mask) > 0:
                        scatter = ax_shots.scatter(
                            team_shots[mask]['xstart'],
                            team_shots[mask]['ystart'],
                            s=sizes[mask],
                            c=color_map[resultado],
                            edgecolors='black',
                            alpha=0.7,
                            label=f'{resultado} ({sum(mask)})'
                        )
                        handles.append(scatter)
                        labels.append(f'{resultado} ({sum(mask)})')
                        
                # Mostrar el xG total
                total_xg = team_shots['xg'].sum()
                ax_shots.text(70, 47, f'Total xG: {total_xg:.2f}',
                       ha='center', va='center', color='white',
                       fontsize=14, fontweight='bold')
                        
                if handles:
                    ax_shots.legend(
                        handles=handles,
                        labels=labels,
                        loc='upper center', 
                        bbox_to_anchor=(0.5, 0.15),
                        ncol=2,
                        fontsize=16,  # Aumentado de 12 a 16
                        frameon=True,
                        facecolor='white',
                        edgecolor='black',
                        labelcolor='black',
                        markerscale=1.5  # Hacer los marcadores de la leyenda 1.5 veces más grandes
                    )
            
            ax_shots.set_ylim(45, 100)
            ax_shots.set_title(f'Mapa de disparos - {team_name}', color='white', fontsize=18, pad=10)
    
    # PARTE 2: MAPA DE CENTROS
    # Imprimir valores únicos de la columna 'centro' para diagnóstico
    if 'centro' in df_eventos.columns:
        valores_centro = df_eventos[df_eventos['match_id'] == match_id]['centro'].unique()
        
        # Contar eventos marcados como centros para diagnóstico
        num_centros = df_eventos[(df_eventos['match_id'] == match_id) & (df_eventos['centro_bool'] == True)].shape[0]
    
    if team_crosses.empty:
        ax_crosses.text(0.5, 0.5, f"No hay datos de centros para {team_name}",
               ha='center', va='center', color='white', fontsize=16)
        ax_crosses.set_facecolor(BACKGROUND_COLOR)
        ax_crosses.axis('off')
    else:
        # Crear el campo de fútbol para los centros, con tamaño maximizado
        pitch_crosses = VerticalPitch(
            pitch_type='wyscout', 
            pitch_color='None', 
            line_color=LINE_COLOR, 
            line_zorder=2, 
            half=True,
            pad_top=0,       # Eliminar padding completamente
            pad_bottom=0,
            pad_left=0,
            pad_right=0
        )
        pitch_crosses.draw(ax=ax_crosses)
        
        # Convertir columnas numéricas para los centros
        numeric_cols = ['xstart', 'ystart', 'xend', 'yend']
        for col in numeric_cols:
            if col in team_crosses.columns:
                team_crosses[col] = pd.to_numeric(team_crosses[col], errors='coerce')
            else:
                print(f"Columna {col} no encontrada en el DataFrame de centros")
                team_crosses[col] = np.nan
        
        # Eliminar filas con valores NaN en las coordenadas
        team_crosses = team_crosses.dropna(subset=['xstart', 'ystart', 'xend', 'yend'])
        
        if team_crosses.empty:
            ax_crosses.text(0.5, 0.5, f"No hay datos válidos de centros para {team_name}",
                   ha='center', va='center', color='white', fontsize=16)
            ax_crosses.set_facecolor(BACKGROUND_COLOR)
            ax_crosses.axis('off')
        else:
            # Ajustar los límites del eje para mostrar solo media cancha (como en el mapa de disparos)
            ax_crosses.set_ylim(45, 100)
            
            # Dibujar vectores de centros
            for idx, row in team_crosses.iterrows():
                # Convertir a números si no lo son ya
                x_start = float(row['xstart'])
                y_start = float(row['ystart'])
                x_end = float(row['xend'])
                y_end = float(row['yend'])
                
                # Ajustar puntos finales si están fuera del rango visible
                # Si el punto final está más allá de y=100, ajustamos la proporción para mantenerlo dentro
                if y_end > 100:
                    # Calcular la proporción para mantener la dirección
                    factor = (100 - y_start) / (y_end - y_start)
                    x_end = x_start + factor * (x_end - x_start)
                    y_end = 100
                
                # Calcular el punto de control para la curva convexa
                dx = x_end - x_start
                dy = y_end - y_start
                
                # Punto medio de la línea recta
                mid_x = (x_start + x_end) / 2
                mid_y = (y_start + y_end) / 2
                
                # Calcular la distancia para el punto de control (determina cuán convexa es la curva)
                dist = max(5, min(15, (dx**2 + dy**2)**0.5 / 4))
                
                # Vector perpendicular para el punto de control
                if abs(dy) < 0.001:  # Casi horizontal
                    control_x = mid_x
                    control_y = mid_y + dist
                elif abs(dx) < 0.001:  # Casi vertical
                    control_x = mid_x + dist
                    control_y = mid_y
                else:
                    # Vector perpendicular normalizado
                    length = math.sqrt(dx**2 + dy**2)
                    nx, ny = -dy/length, dx/length
                    control_x = mid_x + dist * nx
                    control_y = mid_y + dist * ny
                
                # Asegurarse de que el punto de control esté dentro del campo visible
                if control_y > 100:
                    control_y = 100
                elif control_y < 0:
                    control_y = 0
                
                # Generar la curva de Bézier
                t = np.linspace(0, 1, 50)  # Menos puntos para mejor rendimiento
                x = (1-t)**2 * x_start + 2 * (1-t) * t * control_x + t**2 * x_end
                y = (1-t)**2 * y_start + 2 * (1-t) * t * control_y + t**2 * y_end
                
                # Crear un degradado de color más moderno
                base_color = plt.cm.colors.to_rgb(color)
                
                # Crear un cmap personalizado para un efecto más moderno
                # Color base con menor saturación al inicio y más intenso al final
                start_color = np.array(base_color) * 0.7  # Más tenue al inicio
                end_color = np.array(base_color) * 1.2    # Más brillante al final
                end_color = np.clip(end_color, 0, 1)      # Asegurar que no exceda 1
                
                custom_cmap = LinearSegmentedColormap.from_list(
                    'custom_fade', 
                    [start_color, end_color],
                    N=100
                )
                
                # Obtener colores del gradiente
                colors = [custom_cmap(t_val) for t_val in t]
                
                # Ancho variable para un aspecto más moderno (más delgado al inicio, más ancho al final)
                # Aumentamos el ancho para que sea más visible
                linewidths = [6 + 8 * t_val for t_val in t]
                
                # Dibujar la curva con segmentos de ancho variable y color gradual
                # Solo dibujar puntos que estén dentro del campo visible
                inside_field = (y >= 0) & (y <= 100)
                
                # Identificar segmentos donde al menos un punto está dentro del campo
                segments = []
                for i in range(len(t)-1):
                    if inside_field[i] or inside_field[i+1]:
                        segments.append(i)
                
                # Dibujar los segmentos válidos
                for i in segments:
                    ax_crosses.plot(
                        [x[i], x[i+1]], 
                        [y[i], y[i+1]], 
                        color=colors[i], 
                        linewidth=linewidths[i],
                        alpha=0.85,
                        solid_capstyle='round'
                    )
                
                # Solo dibujar la punta de flecha si el punto final está dentro del campo
                if 0 <= y_end <= 100:
                    # Dibujar una punta de flecha más moderna
                    # Usar los últimos puntos visibles para determinar la dirección
                    visible_indices = np.where(inside_field)[0]
                    if len(visible_indices) >= 2:
                        last_idx = visible_indices[-1]
                        prev_idx = visible_indices[-2]
                        
                        end_x, end_y = x[last_idx], y[last_idx]
                        pre_end_x, pre_end_y = x[prev_idx], y[prev_idx]
                        
                        # Vector dirección normalizado
                        dx_end = end_x - pre_end_x
                        dy_end = end_y - pre_end_y
                        length = math.sqrt(dx_end**2 + dy_end**2)
                        if length > 0:
                            dx_end /= length
                            dy_end /= length
                        
                        # Tamaño de la punta de flecha
                        arrow_size = 3
                        arrow_width = 2.0
                        
                        # Calcular los puntos para el triángulo de la punta
                        # Punto de la punta
                        tip_x, tip_y = end_x, end_y
                        
                        # Calcular los puntos base del triángulo
                        # Vector perpendicular normalizado
                        perp_x, perp_y = -dy_end, dx_end
                        
                        # Puntos base de la flecha (más ancha y moderna)
                        base1_x = end_x - arrow_size * dx_end + arrow_width * perp_x
                        base1_y = end_y - arrow_size * dy_end + arrow_width * perp_y
                        base2_x = end_x - arrow_size * dx_end - arrow_width * perp_x
                        base2_y = end_y - arrow_size * dy_end - arrow_width * perp_y
                        
                        # Dibujar el triángulo de la punta
                        arrow_triangle = plt.Polygon(
                            [[tip_x, tip_y], [base1_x, base1_y], [base2_x, base2_y]],
                            closed=True,
                            facecolor=end_color,  # Usar el color final del degradado
                            edgecolor='none',     # Sin borde para un aspecto más moderno
                            alpha=0.95,
                            zorder=10             # Asegurar que esté encima de la línea
                        )
                        ax_crosses.add_patch(arrow_triangle)
            
            # Añadir información sobre la cantidad de centros
            total_crosses = len(team_crosses)
            ax_crosses.text(50, 50, f'Total centros: {total_crosses}',
                   ha='center', va='center', color='white',
                   fontsize=20, fontweight='bold', bbox=dict(facecolor='black', alpha=0.6))
            
            ax_crosses.set_title(f'Mapa de centros - {team_name}', color='white', fontsize=18, pad=10)
    
    # Asegurarse de que la figura original está vacía para evitar contenido duplicado
    ax.clear()
    ax.axis('off')
    
    # Eliminar cualquier ajuste automático adicional que pueda reducir el tamaño
    # No usamos tight_layout que puede reajustar y reducir los campos
    
    return fig

# Función para crear el radar chart
def create_radar_chart(ax, df_estadisticas, local_team, away_team, match_id):
    """
    Creates a radar chart showing KPIs for both teams with improved visibility.
    """
    try:
        # Obtener datos específicos del partido
        match_stats = df_estadisticas[df_estadisticas['match_id'] == match_id].copy()
        
        # Filtrar por equipos
        local_stats = match_stats[match_stats['equipo'] == local_team]
        away_stats = match_stats[match_stats['equipo'] == away_team]
        
        # Verificar que tenemos datos
        if local_stats.empty or away_stats.empty:
            print(f"No hay datos disponibles para uno o ambos equipos: {local_team} o {away_team}")
            ax.text(0.5, 0.5, "No hay datos KPI suficientes para crear el radar",
                  ha='center', va='center', color='white', fontsize=14)
            ax.set_facecolor(BACKGROUND_COLOR)
            return
        
        # Lista de KPIs a visualizar
        kpi_columns = [
            'KPI_construccion_ataque', 'KPI_progresion', 'KPI_habilidad_individual',
            'KPI_peligro_generado', 'KPI_finalizacion', 'KPI_eficacia_defensiva',
            'KPI_juego_aereo', 'KPI_capacidad_recuperacion', 'KPI_posicionamiento_tactico'
        ]
        
        # Verificar que las columnas existen
        available_columns = [col for col in kpi_columns if col in local_stats.columns and col in away_stats.columns]
        
        if not available_columns:
            print("No hay columnas de KPI disponibles para ninguno de los equipos")
            ax.text(0.5, 0.5, "No hay columnas KPI disponibles",
                  ha='center', va='center', color='white', fontsize=14)
            ax.set_facecolor(BACKGROUND_COLOR)
            return
        
        # Si solo algunas columnas están disponibles, usar solo esas
        kpi_columns = available_columns
        
        # Nombres de parámetros para ejes
        param_mapping = {
            'KPI_construccion_ataque': "Construcción\nde Ataque", 
            'KPI_progresion': "Progresión", 
            'KPI_habilidad_individual': "Habilidad\nIndividual",
            'KPI_peligro_generado': "Peligro\nGenerado", 
            'KPI_finalizacion': "Finalización", 
            'KPI_eficacia_defensiva': "Eficacia\nDefensiva",
            'KPI_juego_aereo': "Juego\nAéreo", 
            'KPI_capacidad_recuperacion': "Capacidad de\nRecuperación", 
            'KPI_posicionamiento_tactico': "Posicionamiento\nTáctico"
        }
        
        params = [param_mapping[col] for col in kpi_columns]
        
        # Obtener valores para cada equipo
        values_team1 = local_stats[kpi_columns].mean().values
        values_team2 = away_stats[kpi_columns].mean().values
        
        # Configuración del gráfico
        num_vars = len(params)
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]  # Cerrar el polígono
        
        # Extender los valores para cerrar el polígono
        values_team1 = np.concatenate((values_team1, [values_team1[0]]))
        values_team2 = np.concatenate((values_team2, [values_team2[0]]))
        
        # Configurar el gráfico
        ax.set_facecolor(BACKGROUND_COLOR)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(0)
        
        # Definir colores específicos para cada equipo
        local_color = '#3498db'  # Azul para el equipo local
        away_color = '#e74c3c'   # Rojo para el equipo visitante
        
        # Dibujar el radar con líneas más gruesas
        ax.plot(angles, values_team1, color=local_color, linewidth=4, label=local_team)
        ax.fill(angles, values_team1, color=local_color, alpha=0.3)
        ax.plot(angles, values_team2, color=away_color, linewidth=4, label=away_team)
        ax.fill(angles, values_team2, color=away_color, alpha=0.3)
        
        # Configurar etiquetas y escala con texto más grande
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(params, color='white', size=12, fontweight='bold')
        ax.set_ylim(0, 10)
        
        # Añadir líneas de grid más visibles
        ax.grid(True, color='white', alpha=0.4, linewidth=1.5)
        ax.spines['polar'].set_color('white')
        ax.spines['polar'].set_alpha(0.4)
        ax.spines['polar'].set_linewidth(2)
        
        # Etiquetas numéricas más grandes
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_yticklabels(['2', '4', '6', '8', '10'], color='white', size=12, fontweight='bold')
        
        # Título y leyenda con texto más grande
        legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                ncol=2, facecolor=BACKGROUND_COLOR, edgecolor='white',
                labelcolor='white', fontsize=14, framealpha=0.7)
                
        # Hacer más gruesos los bordes de la leyenda
        legend.get_frame().set_linewidth(2)
    
    except Exception as e:
        print(f"Error en create_radar_chart: {e}")
        import traceback
        traceback.print_exc()
        ax.text(0.5, 0.5, f"Error al crear radar: {str(e)}",
              ha='center', va='center', color='red', fontsize=14)
        ax.set_facecolor(BACKGROUND_COLOR)

# Función para crear el gráfico de comparación de métricas de zona
def create_zone_metrics_comparison(df_eventos, local_team, away_team, match_id):
    """
    Crea un gráfico de barras horizontales comparando métricas por zonas entre dos equipos.
    
    Args:
        df_eventos: DataFrame con los eventos del partido
        local_team: Nombre del equipo local
        away_team: Nombre del equipo visitante
        match_id: ID del partido
    
    Returns:
        Una figura matplotlib con la visualización comparativa
    """
    
    # Filtrar datos para el partido y equipos específicos
    local_events = df_eventos[(df_eventos['match_id'] == match_id) & 
                             (df_eventos['equipo'] == local_team)]
    away_events = df_eventos[(df_eventos['match_id'] == match_id) & 
                            (df_eventos['equipo'] == away_team)]
    
    # Definir las métricas a visualizar
    metrics_groups = {
        'Pases Progresivos': [
            'pases_progresivos_inicio',
            'pases_progresivos_creacion',
            'pases_progresivos_finalizacion'
        ],
        'Recuperaciones': [
            'recuperaciones_zona_baja',
            'recuperaciones_zona_media',
            'recuperaciones_zona_alta'
        ],
        'Duelos Aéreos': [
            'duelos_aereos_ganados_zona_baja',
            'duelos_aereos_ganados_zona_media',
            'duelos_aereos_ganados_zona_alta',
            'duelos_aereos_ganados_zona_area'
        ],
        'Entradas': [
            'entradas_ganadas_zona_baja',
            'entradas_ganadas_zona_media'
        ],
        'Pases Largos': [
            'pases_largos_exitosos'
        ]
    }
    
    # Configurar colores
    local_color = '#3498db'  # Azul para equipo local
    away_color = '#e74c3c'   # Rojo para equipo visitante
    background_color = '#0E1117'
    text_color = '#FFFFFF'
    
    # Calcular altura total de la fila basada en el número de métricas
    total_metrics = sum(len(metrics) for metrics in metrics_groups.values())
    row_height = 0.4
    total_height = (total_metrics + len(metrics_groups)) * row_height * 1.5
    
    # Crear figura con tamaño apropiado - fondo transparente
    fig = plt.figure(figsize=(12, total_height), facecolor='none')  # Fondo transparente
    ax = fig.add_subplot(111, facecolor='none')  # Fondo del área del gráfico transparente
    
    # Inicializar posición y contador de filas para alternar colores
    y_position = 0
    row_counter = 0
    all_labels = []
    
    # Definir la posición central para las métricas
    center = 50  # Columna central donde se ubicarán los nombres de las métricas
    
    # Procesar cada grupo de métricas
    for group_name, metrics in metrics_groups.items():
        # Añadir encabezado de grupo
        ax.text(center, y_position, group_name, color=text_color, fontsize=14, fontweight='bold',
                ha='center', va='center')
        y_position -= row_height * 1.2
        
        # Procesar cada métrica en el grupo
        for metric in metrics:
            # Contar eventos para equipo local
            local_count = 0
            if metric in local_events.columns:
                local_count = pd.to_numeric(local_events[metric], errors='coerce').fillna(0).sum()
            
            # Contar eventos para equipo visitante
            away_count = 0
            if metric in away_events.columns:
                away_count = pd.to_numeric(away_events[metric], errors='coerce').fillna(0).sum()
            
            # Formatear el nombre de la métrica para mostrar
            metric_display = metric.replace('_', ' ').title()
            for prefix in ['Pases Progresivos ', 'Recuperaciones ', 'Duelos Aereos Ganados Zona ', 
                          'Entradas Ganadas Zona ', 'Pases Largos ']:
                metric_display = metric_display.replace(prefix, '')
            
            # Mapear nombres de zonas a formatos más legibles
            zone_mapping = {
                'Area': 'Área',
                'Baja': 'Defensa',
                'Media': 'Medio',
                'Alta': 'Ataque',
                'Inicio': 'Inicio',
                'Creacion': 'Creación',
                'Finalizacion': 'Final',
                'Exitosos': 'Exitosos'
            }
            
            # Aplicar mapeo de zonas
            for old, new in zone_mapping.items():
                metric_display = metric_display.replace(old, new)
            
            all_labels.append(metric_display)
            
            # Asegurarse de que local_count y away_count son números flotantes
            local_count = float(local_count)
            away_count = float(away_count)
            
            # Dibujar barra del equipo local (lado izquierdo, azul)
            ax.barh(y_position, local_count, height=row_height, left=center-local_count, 
                   color=local_color, edgecolor=None, linewidth=0, alpha=1.0)
            
            # Añadir contador para equipo local al final de la barra
            if local_count > 0:
                ax.text(center-local_count-1, y_position, str(int(local_count)), 
                       color=local_color, fontweight='bold', ha='right', va='center')
            
            # Dibujar barra del equipo visitante (lado derecho, rojo)
            ax.barh(y_position, away_count, height=row_height, left=center, 
                   color=away_color, edgecolor=None, linewidth=0, alpha=1.0)
            
            # Añadir contador para equipo visitante al final de la barra
            if away_count > 0:
                ax.text(center+away_count+1, y_position, str(int(away_count)), 
                       color=away_color, fontweight='bold', ha='left', va='center')
            
            # Añadir nombre de la métrica en el centro
            ax.text(center, y_position, metric_display, 
                   color='white', fontweight='bold', ha='center', va='center')
            
            y_position -= row_height * 1.2
        
        # Añadir espacio extra después de cada grupo
        y_position -= row_height * 0.5
    
    # Calcular límites del eje x dinámicamente basados en los valores máximos
    max_local = max([pd.to_numeric(local_events[m], errors='coerce').fillna(0).sum() if m in local_events.columns else 0 
                    for group in metrics_groups.values() for m in group])
    max_away = max([pd.to_numeric(away_events[m], errors='coerce').fillna(0).sum() if m in away_events.columns else 0 
                   for group in metrics_groups.values() for m in group])
    
    # Añadir algo de espacio a los límites del eje x
    padding = max(5, max(max_local, max_away) * 0.15)
    ax.set_xlim(center - max_local - padding, center + max_away + padding)
    
    # Establecer límites del eje y
    ax.set_ylim(y_position, 1)
    
    # Eliminar marcas y etiquetas de los ejes
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Añadir nombres de los equipos en la parte superior
    ax.text(center - max_local/2, 0.5, local_team, color=local_color, 
           fontsize=14, fontweight='bold', ha='center', va='center')
    ax.text(center + max_away/2, 0.5, away_team, color=away_color, 
           fontsize=14, fontweight='bold', ha='center', va='center')
    
    # Añadir una línea vertical en el centro (muy sutil)
    ax.axvline(x=center, color='white', linestyle='-', linewidth=0.5, alpha=0.2)
    
    # Añadir líneas horizontales entre grupos (muy sutiles)
    y_pos = 0
    for group_name, metrics in metrics_groups.items():
        y_pos -= (len(metrics) + 0.5) * row_height * 1.2
        ax.axhline(y=y_pos, color='white', linestyle='-', linewidth=0.3, alpha=0.1)
    
    # Eliminar bordes y líneas de la cuadrícula
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Eliminar líneas de rejilla si están presentes
    ax.grid(False)
    
    plt.tight_layout(pad=2.0)  # Asegurar que todo el contenido sea visible
    return fig

def capturar_visualizaciones_partido(fig, titulo=None, dpi=150):
    """
    Convierte una figura de matplotlib en un objeto Image compatible con ReportLab,
    optimizado para visualizaciones de partidos. Incluye el título de la gráfica.
    
    Args:
        fig: Figura de matplotlib a convertir
        titulo: Título de la gráfica (opcional)
        dpi: Resolución de la imagen en puntos por pulgada
        
    Returns:
        Tuple con (imagen ReportLab, título de la gráfica)
    """
    # Obtener dimensiones originales
    fig_width_inches, fig_height_inches = fig.get_size_inches()
    aspect_ratio = fig_width_inches / fig_height_inches
    
    # Detectar campogramas verticales (más altos que anchos)
    es_campograma_vertical = aspect_ratio < 0.8 and fig_height_inches > fig_width_inches
    
    # Guardar con diferentes configuraciones
    buf = io.BytesIO()
    
    if es_campograma_vertical:
        # Para campogramas verticales, usar configuración especial
        fig.savefig(buf, format='png', dpi=dpi, pad_inches=0.5)
    else:
        # Para otras figuras, usar configuración normal
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    
    buf.seek(0)
    
    # Ajustar tamaños para diferentes tipos de figuras
    if es_campograma_vertical:
        # Para campogramas verticales
        width = 2.5 * inch
        height = width / aspect_ratio
        
        # Limitar la altura máxima
        max_height = 7 * inch
        if height > max_height:
            height = max_height
            width = height * aspect_ratio
    else:
        # Para otras figuras
        max_width = 3.0 * inch  # Más pequeño para permitir 2 en una fila
        max_height = 3.5 * inch
        
        # Calcular dimensiones manteniendo la relación de aspecto
        if aspect_ratio > 1.75:  # Figuras muy anchas
            width = max_width
            height = width / aspect_ratio
        else:
            height = max_height
            width = min(max_width, height * aspect_ratio)
    
    # Crear la imagen ReportLab con dimensiones ajustadas
    return ReportLabImage(buf, width=width, height=height), titulo

def generar_pdf_partido(local_team, away_team, temporada, match_id, figuras_con_titulos, logo_path=None):
    # Inicializar buffer para el PDF
    buffer = io.BytesIO()
    
    # Configurar el documento (A4 vertical)
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,  # Formato vertical
        rightMargin=0.0*cm,
        leftMargin=0.0*cm,
        topMargin=0.0*cm,
        bottomMargin=0.0*cm
    )
    
    # Configurar estilos
    styles = getSampleStyleSheet()
    
    # Estilo para la información del partido
    styles.add(ParagraphStyle(
        name='PartidoInfo',
        parent=styles['Heading1'],
        fontSize=14,
        textColor=colors.white,
        backColor=colors.black,
        alignment=TA_CENTER,
        spaceAfter=6,
        spaceBefore=4
    ))
    
    # Estilo para títulos de gráficas
    styles.add(ParagraphStyle(
        name='TituloGrafica',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.white,
        backColor=colors.black,
        alignment=TA_CENTER,
        spaceAfter=1,
        spaceBefore=1
    ))
    
    # Lista para los elementos del PDF
    elementos = []
    
    # Fondo negro para toda la página
    black_background = Paragraph(
        "<para></para>",
        ParagraphStyle(
            'background',
            parent=styles['Normal'],
            backColor=colors.black,
        )
    )
    elementos.append(black_background)
    
    # Cargar el escudo
    escudo_img = None
    if logo_path:
        try:
            escudo_pil = Image.open(logo_path)
            escudo_pil.thumbnail((60, 60), Image.LANCZOS)
            
            escudo_buffer = io.BytesIO()
            if escudo_pil.mode in ("RGBA", "P"):
                escudo_pil = escudo_pil.convert("RGB")
            escudo_pil.save(escudo_buffer, format="JPEG", quality=85)
            escudo_buffer.seek(0)
            
            escudo_img = ReportLabImage(escudo_buffer, width=0.6*inch, height=0.6*inch)
        except Exception as e:
            print(f"Error al cargar el escudo: {e}")
    
    # Configuración personalizable por tipo de gráfica
    # MODIFICA ESTOS VALORES PARA AJUSTAR LAS DIMENSIONES DE CADA TIPO DE GRÁFICA
    # Formato: [ancho_en_cm, alto_en_cm]
    dimensiones_figuras = {
        # Primera fila (KPIs)
        'kpi_local': [4.0, 3.5],     # Dimensiones KPI equipo local
        'kpi_visitante': [4.0, 3.5], # Dimensiones KPI equipo visitante
        
        # Segunda fila (Redes de pases y pases progresivos)
        'red_pases_local': [4.6, 6.0],     # Red de pases local
        'red_pases_visitante': [4.6, 6.0], # Red de pases visitante
        'pases_prog_local': [4.6, 6.0],    # Pases progresivos local
        'pases_prog_visitante': [4.6, 6.0], # Pases progresivos visitante
        
        # Tercera fila (Mapas defensivos y disparos/centros)
        'defensa_local': [4.6, 6.0],        # Mapa defensivo local
        'defensa_visitante': [4.6, 6.0],    # Mapa defensivo visitante
        'disparos_local': [4.6, 6.0],       # Disparos/centros local
        'disparos_visitante': [4.6, 6.0],   # Disparos/centros visitante
        
        # Cuarta fila (Visualizaciones adicionales)
        'radar': [8.5, 8.0],           # Gráfico radar
        'metricas_zonas': [8.5, 8.0]   # Métricas por zonas
    }
    
    # Convertir cm a puntos (unidad usada por ReportLab)
    for key in dimensiones_figuras:
        dimensiones_figuras[key][0] = dimensiones_figuras[key][0] * cm
        dimensiones_figuras[key][1] = dimensiones_figuras[key][1] * cm
    
    # Clasificar figuras por filas
    fila1_imgs = []  # KPIs (figuras 0-1)
    fila2_imgs = []  # Redes de pases y pases progresivos (figuras 2-5)
    fila3_imgs = []  # Mapas defensivos y disparos/centros (figuras 6-9)
    fila4_imgs = []  # Visualizaciones adicionales (figuras 10-11)
    
    # Tipo de gráfica para cada índice
    tipos_figuras = [
        'kpi_local', 'kpi_visitante',                             # Fila 1 (índices 0-1)
        'red_pases_local', 'red_pases_visitante',                 # Fila 2 (índices 2-3)
        'pases_prog_local', 'pases_prog_visitante',               # Fila 2 (índices 4-5)
        'defensa_local', 'defensa_visitante',                     # Fila 3 (índices 6-7)
        'disparos_local', 'disparos_visitante',                   # Fila 3 (índices 8-9)
        'radar', 'metricas_zonas'                                 # Fila 4 (índices 10-11)
    ]
    
    try:
        # Procesar las figuras con dimensiones personalizadas
        for i, (fig, titulo) in enumerate(figuras_con_titulos):
            if i < len(tipos_figuras):
                tipo = tipos_figuras[i]
                ancho, alto = dimensiones_figuras[tipo]
                
                # Generar imagen para ReportLab
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0.1)
                buf.seek(0)
                img = ReportLabImage(buf, width=ancho, height=alto)
                
                # Agregar a la fila correspondiente
                if i < 2:  # Fila 1 (KPIs)
                    fila1_imgs.append((img, Paragraph(titulo, styles['TituloGrafica'])))
                elif i < 6:  # Fila 2 (Redes de pases y pases progresivos)
                    fila2_imgs.append((img, Paragraph(titulo, styles['TituloGrafica'])))
                elif i < 10:  # Fila 3 (Mapas defensivos y disparos/centros)
                    fila3_imgs.append((img, Paragraph(titulo, styles['TituloGrafica'])))
                else:  # Fila 4 (Visualizaciones adicionales)
                    fila4_imgs.append((img, Paragraph(titulo, styles['TituloGrafica'])))
    
    except Exception as e:
        print(f"Error procesando figuras: {e}")
    
    # FILA 1: KPIs e información del partido
    partido_info = Paragraph(
        f"""
        <font color="#4BB3FD" size="12"><b>Temporada: {temporada}</b></font><br/>
        <font color="white" size="8">Jornada: <b>{partido_seleccionado}</b></font>
        """, 
        styles['PartidoInfo']
    )
    
    # Añadir escudo a la info del partido
    if escudo_img:
        partido_info_container = [
            [partido_info],
            [escudo_img]
        ]
        partido_info_table = Table(
            partido_info_container,
            colWidths=[doc.width/3],
            style=TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('BACKGROUND', (0, 0), (-1, -1), colors.black),
                ('LEFTPADDING', (0, 0), (-1, -1), 0),
                ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                ('TOPPADDING', (0, 0), (-1, -1), 0),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ])
        )
    else:
        partido_info_table = partido_info
    
    # Preparar KPIs para los lados
    kpi_izq = ['', '']
    kpi_der = ['', '']
    
    if len(fila1_imgs) >= 1:
        kpi_izq = fila1_imgs[0]
    if len(fila1_imgs) >= 2:
        kpi_der = fila1_imgs[1]
    
    # Crear la primera fila con KPIs a los lados e info en el centro
    # AJUSTE FILA 1: Puedes modificar las proporciones cambiando estos valores
    col_width_kpi_titulo = 2.0*cm    # Ancho para títulos de KPI
    col_width_centro = doc.width - (2*dimensiones_figuras['kpi_local'][0] + 2*col_width_kpi_titulo)
    
    primera_fila = [
        [kpi_izq[1], kpi_izq[0], partido_info_table, kpi_der[0], kpi_der[1]]
    ]
    
    # AJUSTE ALTURA FILA 1: Modifica este valor para cambiar la altura de la primera fila
    altura_fila1 = 4.0*cm
    
    primera_fila_tabla = Table(
        primera_fila,
        colWidths=[col_width_kpi_titulo, dimensiones_figuras['kpi_local'][0], col_width_centro, 
                  dimensiones_figuras['kpi_visitante'][0], col_width_kpi_titulo],
        rowHeights=[altura_fila1],
        style=TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BACKGROUND', (0, 0), (-1, -1), colors.black),
            ('LEFTPADDING', (0, 0), (-1, -1), 2),
            ('RIGHTPADDING', (0, 0), (-1, -1), 2),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),  # Separación después de la fila 1
        ])
    )
    
    elementos.append(primera_fila_tabla)
    
    # FILA 2: Redes de pases y pases progresivos
    if fila2_imgs:
        titulos = [img[1] for img in fila2_imgs]
        imagenes = [img[0] for img in fila2_imgs]
        
        # AJUSTE ANCHOS FILA 2: Define aquí los anchos específicos para cada columna
        anchos_fila2 = []
        if len(fila2_imgs) >= 4:  # Si tenemos las 4 gráficas esperadas
            anchos_fila2 = [
                dimensiones_figuras['red_pases_local'][0],
                dimensiones_figuras['red_pases_visitante'][0],
                dimensiones_figuras['pases_prog_local'][0],
                dimensiones_figuras['pases_prog_visitante'][0]
            ]
        else:  # Si faltan algunas gráficas, usar anchos iguales
            col_width = doc.width / len(fila2_imgs)
            anchos_fila2 = [col_width] * len(fila2_imgs)
        
        # AJUSTE ALTURA FILA 2: Modifica este valor para cambiar la altura de las imágenes
        altura_imagenes_fila2 = max(dimensiones_figuras['red_pases_local'][1], 
                                   dimensiones_figuras['pases_prog_local'][1])
        
        segunda_fila_tabla = Table(
            [titulos, imagenes],
            colWidths=anchos_fila2,
            rowHeights=[1.2*cm, altura_imagenes_fila2],
            style=TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('BACKGROUND', (0, 0), (-1, -1), colors.black),
                ('LEFTPADDING', (0, 0), (-1, -1), 2),
                ('RIGHTPADDING', (0, 0), (-1, -1), 2),
                ('TOPPADDING', (0, 0), (-1, -1), 1),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),  # Separación después de la fila 2
            ])
        )
        
        elementos.append(segunda_fila_tabla)
    
    # FILA 3: Mapas defensivos y disparos/centros
    if fila3_imgs:
        titulos = [img[1] for img in fila3_imgs]
        imagenes = [img[0] for img in fila3_imgs]
        
        # AJUSTE ANCHOS FILA 3: Define aquí los anchos específicos para cada columna
        anchos_fila3 = []
        if len(fila3_imgs) >= 4:  # Si tenemos las 4 gráficas esperadas
            anchos_fila3 = [
                dimensiones_figuras['defensa_local'][0],
                dimensiones_figuras['defensa_visitante'][0],
                dimensiones_figuras['disparos_local'][0],
                dimensiones_figuras['disparos_visitante'][0]
            ]
        else:  # Si faltan algunas gráficas, usar anchos iguales
            col_width = doc.width / len(fila3_imgs)
            anchos_fila3 = [col_width] * len(fila3_imgs)
        
        # AJUSTE ALTURA FILA 3: Modifica este valor para cambiar la altura de las imágenes
        altura_imagenes_fila3 = max(dimensiones_figuras['defensa_local'][1], 
                                   dimensiones_figuras['disparos_local'][1])
        
        tercera_fila_tabla = Table(
            [titulos, imagenes],
            colWidths=anchos_fila3,
            rowHeights=[1.2*cm, altura_imagenes_fila3],
            style=TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('BACKGROUND', (0, 0), (-1, -1), colors.black),
                ('LEFTPADDING', (0, 0), (-1, -1), 2),
                ('RIGHTPADDING', (0, 0), (-1, -1), 2),
                ('TOPPADDING', (0, 0), (-1, -1), 1),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),  # Separación después de la fila 3
            ])
        )
        
        elementos.append(tercera_fila_tabla)
    
    # FILA 4: Visualizaciones adicionales
    if fila4_imgs:
        titulos = [img[1] for img in fila4_imgs]
        imagenes = [img[0] for img in fila4_imgs]
        
        # AJUSTE ANCHOS FILA 4: Define aquí los anchos específicos para cada columna
        anchos_fila4 = []
        if len(fila4_imgs) >= 2:  # Si tenemos las 2 gráficas esperadas
            anchos_fila4 = [
                dimensiones_figuras['radar'][0],
                dimensiones_figuras['metricas_zonas'][0]
            ]
        else:  # Si faltan algunas gráficas, usar anchos iguales
            col_width = doc.width / len(fila4_imgs)
            anchos_fila4 = [col_width] * len(fila4_imgs)
        
        # AJUSTE ALTURA FILA 4: Modifica este valor para cambiar la altura de las imágenes
        altura_imagenes_fila4 = max(dimensiones_figuras['radar'][1], 
                                   dimensiones_figuras['metricas_zonas'][1])
        
        cuarta_fila_tabla = Table(
            [titulos, imagenes],
            colWidths=anchos_fila4,
            rowHeights=[2.4*cm, altura_imagenes_fila4],
            style=TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('BACKGROUND', (0, 0), (-1, -1), colors.black),
                ('LEFTPADDING', (0, 0), (-1, -1), 2),
                ('RIGHTPADDING', (0, 0), (-1, -1), 2),
                ('TOPPADDING', (0, 0), (-1, -1), 1),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
            ])
        )
        
        elementos.append(cuarta_fila_tabla)
    
    # Para asegurar fondo negro en todas las páginas
    def black_canvas(canvas, doc):
        canvas.setFillColor(colors.black)
        canvas.rect(0, 0, doc.width + 1*cm, doc.height + 1*cm, fill=1)  # Extender más allá de los bordes
        
    # Construir el PDF con fondo negro
    doc.build(elementos, onFirstPage=black_canvas, onLaterPages=black_canvas)
    buffer.seek(0)
    return buffer

def generate_match_viz(df_eventos, df_estadisticas, partido_info):
    """
    Genera las visualizaciones del partido directamente en Streamlit y guarda las figuras para PDF.
    """
    # Declaro aquí la variable global HIGHLIGHT_COLOR para poder modificarla
    global HIGHLIGHT_COLOR
    
    # Obtener información del partido
    local_team = partido_info.get('local_team', 'Local')
    away_team = partido_info.get('away_team', 'Visitante')
    temporada = partido_info.get('temporada', '')
    partido_original = partido_info.get('partido_original', '')
    match_id = partido_info.get('match_id')
    
    # Guardar información para PDF
    st.session_state.partido_info = partido_info
    
    # Lista para guardar figuras y títulos
    figuras_generadas = []
    titulos_figuras = []
    
    # Calcular KPIs
    local_kpi = calculate_team_kpi(df_eventos, df_estadisticas, local_team)
    away_kpi = calculate_team_kpi(df_eventos, df_estadisticas, away_team)
    
    # Extraer información adicional
    if not df_eventos.empty:
        mask_alaves = df_eventos['equipo'].str.contains('Alav', na=False, case=False)
        equipo_alaves = df_eventos[mask_alaves]['equipo'].iloc[0] if mask_alaves.any() else "Equipo no encontrado"
        jornada = f"Jornada {df_eventos['jornada'].iloc[0]}"
        partido = df_eventos['partido'].iloc[0].split(' - ', 1)[1] if ' - ' in df_eventos['partido'].iloc[0] else df_eventos['partido'].iloc[0]
    else:
        equipo_alaves = "Equipo no encontrado"
        jornada = "Jornada desconocida"
        partido = partido_original
    
    # Colores para cada equipo
    local_color = '#3498db'  # Azul para el equipo local
    away_color = '#e74c3c'   # Rojo para el equipo visitante
    
    # Guardar color original
    original_highlight_color = HIGHLIGHT_COLOR
    
    # Mostrar visualizaciones directamente en Streamlit
    viz_container = st.container()
    
    with viz_container:
        # IMPLEMENTACIÓN ORIGINAL DE LAS VISUALIZACIONES
        # En cada sección donde se crea una figura, añadir la figura y su título a las listas
        
        # Primera fila con 3 columnas: KPI Local, Info Partido, KPI Visitante
        st.markdown("---")  # Línea horizontal antes de la primera fila
        col1, col2, col3 = st.columns([1, 2, 1])
        
        # KPI Local
        with col1:
            fig_kpi_local = plt.figure(figsize=(6, 6), facecolor=BACKGROUND_COLOR)
            ax_kpi_local = fig_kpi_local.add_subplot(111)
            draw_single_kpi_square(ax_kpi_local, local_team, local_kpi)
            plt.tight_layout()
            st.pyplot(fig_kpi_local)
            
            # Guardar figura y título
            figuras_generadas.append(fig_kpi_local)
            titulos_figuras.append(f"KPI {local_team}")
        
        # Información del partido en el centro
        with col2:
            st.markdown(f"""
            <div style='
                background-color: {BACKGROUND_COLOR};
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                height: 100%;
                display: flex;
                flex-direction: column;
                justify-content: center;
                color: {TEXT_COLOR};
            '>
                <h2 style='color: {TEXT_COLOR};'>{equipo_alaves}</h2>
                <h3 style='color: {TEXT_COLOR};'>{temporada}</h3>
                <h3 style='color: {TEXT_COLOR};'>{jornada}</h3>
                <h3 style='color: {TEXT_COLOR};'>{partido}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        # KPI Visitante
        with col3:
            fig_kpi_away = plt.figure(figsize=(6, 6), facecolor=BACKGROUND_COLOR)
            ax_kpi_away = fig_kpi_away.add_subplot(111)
            draw_single_kpi_square(ax_kpi_away, away_team, away_kpi)
            plt.tight_layout()
            st.pyplot(fig_kpi_away)
            
            # Guardar figura y título
            figuras_generadas.append(fig_kpi_away)
            titulos_figuras.append(f"KPI {away_team}")
        
        # Línea horizontal que separa las filas
        st.markdown("---")
        
        # Segunda fila con 4 columnas
        cols_fila2 = st.columns(4)
        
        # Columna 1: Red de pases del equipo local (en color azul)
        with cols_fila2[0]:
            st.markdown(f"<h5 style='text-align: center;'>Red de Pases - {local_team}</h5>", unsafe_allow_html=True)
            try:
                # Define colores personalizados
                local_line_color = '#3498db'  # Azul para líneas
                local_node_color = '#3498db'  # Azul para nodos
                
                # Cambiar color global para las líneas
                HIGHLIGHT_COLOR = local_line_color
                
                # Crear la red de pases con color personalizado para nodos
                fig_local_network = create_pass_network(df_eventos, local_team, temporada, custom_node_color=local_node_color)
                
                st.pyplot(fig_local_network)
                
                # Guardar figura y título
                figuras_generadas.append(fig_local_network)
                titulos_figuras.append(f"Red de Pases - {local_team}")
            except Exception as e:
                st.error(f"Error al crear red de pases de {local_team}: {e}")
            finally:
                # Restaurar el color original
                HIGHLIGHT_COLOR = original_highlight_color
            
        # Columna 2: Red de pases del equipo visitante (en color rojo)
        with cols_fila2[1]:
            st.markdown(f"<h5 style='text-align: center;'>Red de Pases - {away_team}</h5>", unsafe_allow_html=True)
            try:
                # Define colores personalizados
                away_line_color = '#e74c3c'  # Rojo para líneas
                away_node_color = '#e74c3c'  # Rojo para nodos
                
                # Cambiar color global para las líneas
                HIGHLIGHT_COLOR = away_line_color
                
                # Crear la red de pases con color personalizado para nodos
                fig_away_network = create_pass_network(df_eventos, away_team, temporada, custom_node_color=away_node_color)
                
                st.pyplot(fig_away_network)
                
                # Guardar figura y título
                figuras_generadas.append(fig_away_network)
                titulos_figuras.append(f"Red de Pases - {away_team}")
            except Exception as e:
                st.error(f"Error al crear red de pases de {away_team}: {e}")
            finally:
                # Restaurar el color original
                HIGHLIGHT_COLOR = original_highlight_color
        
        # Columna 3: Mapa de calor defensivo del equipo local
        with cols_fila2[2]:
            st.markdown(f"<h5 style='text-align: center;'>Defensa - {local_team}</h5>", unsafe_allow_html=True)
            try:
                fig_local_defense = plt.figure(figsize=(8, 10), facecolor=BACKGROUND_COLOR)
                ax_local_defense = fig_local_defense.add_subplot(111)
                draw_defensive_heatmap(ax_local_defense, df_eventos, local_team, match_id, 'Blues')
                plt.tight_layout()
                st.pyplot(fig_local_defense)
                
                # Guardar figura y título
                figuras_generadas.append(fig_local_defense)
                titulos_figuras.append(f"Defensa - {local_team}")
            except Exception as e:
                st.error(f"Error al crear mapa defensivo de {local_team}: {e}")
        
        # Columna 4: Mapa de calor defensivo del equipo visitante
        with cols_fila2[3]:
            st.markdown(f"<h5 style='text-align: center;'>Defensa - {away_team}</h5>", unsafe_allow_html=True)
            try:
                fig_away_defense = plt.figure(figsize=(8, 10), facecolor=BACKGROUND_COLOR)
                ax_away_defense = fig_away_defense.add_subplot(111)
                draw_defensive_heatmap(ax_away_defense, df_eventos, away_team, match_id, 'Reds')
                plt.tight_layout()
                st.pyplot(fig_away_defense)
                
                # Guardar figura y título
                figuras_generadas.append(fig_away_defense)
                titulos_figuras.append(f"Defensa - {away_team}")
            except Exception as e:
                st.error(f"Error al crear mapa defensivo de {away_team}: {e}")
        
        # Línea horizontal que separa las filas
        st.markdown("---")
        
        # Tercera fila con 4 columnas (mapas de pases progresivos)
        cols_fila3 = st.columns(4)
        
        # Columna 1: Mapa de pases progresivos del equipo local
        with cols_fila3[0]:
            st.markdown(f"<h5 style='text-align: center;'>Pases Prog. - {local_team}</h5>", unsafe_allow_html=True)
            try:
                fig_local_passes = plt.figure(figsize=(8, 10), facecolor=BACKGROUND_COLOR)
                ax_local_passes = fig_local_passes.add_subplot(111)
                draw_passing_map(ax_local_passes, df_eventos, local_team, match_id, local_color)
                plt.tight_layout()
                st.pyplot(fig_local_passes)
                
                # Guardar figura y título
                figuras_generadas.append(fig_local_passes)
                titulos_figuras.append(f"Pases Progresivos - {local_team}")
            except Exception as e:
                st.error(f"Error al crear mapa de pases de {local_team}: {e}")
        
        # Columna 2: Mapa de pases progresivos del equipo visitante
        with cols_fila3[1]:
            st.markdown(f"<h5 style='text-align: center;'>Pases Prog. - {away_team}</h5>", unsafe_allow_html=True)
            try:
                fig_away_passes = plt.figure(figsize=(8, 10), facecolor=BACKGROUND_COLOR)
                ax_away_passes = fig_away_passes.add_subplot(111)
                draw_passing_map(ax_away_passes, df_eventos, away_team, match_id, away_color)
                plt.tight_layout()
                st.pyplot(fig_away_passes)
                
                # Guardar figura y título
                figuras_generadas.append(fig_away_passes)
                titulos_figuras.append(f"Pases Progresivos - {away_team}")
            except Exception as e:
                st.error(f"Error al crear mapa de pases de {away_team}: {e}")
        
        # Columna 3: Análisis de disparos del equipo local
        with cols_fila3[2]:
            st.markdown(f"<h5 style='text-align: center;'>Disparos y Centros - {local_team}</h5>", unsafe_allow_html=True)
            try:
                fig_local_shots = plt.figure(figsize=(8, 10), facecolor=BACKGROUND_COLOR)
                ax_local_shots = fig_local_shots.add_subplot(111)
                fig_local_shots = draw_shot_analysis(ax_local_shots, df_eventos, local_team, match_id, 'blue')
                st.pyplot(fig_local_shots)
                
                # Guardar figura y título
                figuras_generadas.append(fig_local_shots)
                titulos_figuras.append(f"Disparos y Centros - {local_team}")
            except Exception as e:
                st.error(f"Error al crear análisis de disparos de {local_team}: {e}")

        # Columna 4: Análisis de disparos del equipo visitante
        with cols_fila3[3]:
            st.markdown(f"<h5 style='text-align: center;'>Disparos y Centros - {away_team}</h5>", unsafe_allow_html=True)
            try:
                fig_away_shots = plt.figure(figsize=(8, 10), facecolor=BACKGROUND_COLOR)
                ax_away_shots = fig_away_shots.add_subplot(111)
                fig_away_shots = draw_shot_analysis(ax_away_shots, df_eventos, away_team, match_id, 'red')
                st.pyplot(fig_away_shots)
                
                # Guardar figura y título
                figuras_generadas.append(fig_away_shots)
                titulos_figuras.append(f"Disparos y Centros - {away_team}")
            except Exception as e:
                st.error(f"Error al crear análisis de disparos de {away_team}: {e}")
        
        # Cuarta fila con gráfico radar centrado y métricas por zonas
        st.markdown("---")
        st.subheader("Estadísticas Adicionales", anchor=False)

        # Crear un contenedor para el radar y placeholders
        radar_container = st.container()

        # División en dos partes: radar (izquierda) y métricas por zonas (derecha)
        with radar_container:
            # Crear dos columnas principales
            col_left, col_right = st.columns(2)
        
        # Columna izquierda para el radar (ocupa 2 columnas originales)
        with col_left:
            st.markdown("<h4 style='text-align: center; color: white; margin-bottom: 20px;'>Comparativa KPI</h4>", unsafe_allow_html=True)
            try:
                # Crear figura para el radar con dimensiones optimizadas
                fig_radar = plt.figure(figsize=(10, 10), facecolor=BACKGROUND_COLOR)
                
                # Añadir subplot con mayor área utilizable
                ax_radar = fig_radar.add_subplot(111, projection='polar')
                
                # Crear el gráfico radar
                create_radar_chart(ax_radar, df_estadisticas, local_team, away_team, match_id)
                
                # Ajustar el layout para maximizar el uso del espacio
                plt.tight_layout(pad=3.0)
                
                # Mostrar el gráfico radar
                st.pyplot(fig_radar)
                
                # Guardar figura y título
                figuras_generadas.append(fig_radar)
                titulos_figuras.append(f"Comparativa KPI - {local_team} vs {away_team}")
                
            except Exception as e:
                st.error(f"Error al crear el gráfico radar: {e}")
        
        # Columna derecha para las métricas por zonas
        with col_right:
            st.markdown(
                "<h4 style='text-align: center; color: white; margin-bottom: 20px;'>Métricas por Zonas</h4>", 
                unsafe_allow_html=True
            )
            try:
                # Crear y mostrar el gráfico de métricas por zonas
                fig_zone_metrics = create_zone_metrics_comparison(
                    df_eventos, 
                    local_team, 
                    away_team, 
                    match_id
                )
                
                # Asegurar que la figura tenga un fondo transparente
                fig_zone_metrics.patch.set_alpha(0.0)  # Fondo de la figura transparente
                for ax in fig_zone_metrics.axes:
                    ax.patch.set_alpha(0.0)  # Fondo de cada eje transparente
                
                # Mostrar la figura en Streamlit con transparencia
                st.pyplot(fig_zone_metrics, transparent=True)
                
                # Guardar figura y título
                figuras_generadas.append(fig_zone_metrics)
                titulos_figuras.append(f"Métricas por Zonas - {local_team} vs {away_team}")
                
            except Exception as e:
                st.error(f"Error al crear métricas por zonas: {e}")

    # Guardar las figuras generadas para exportación a PDF
    st.session_state.figuras_generadas = figuras_generadas
    st.session_state.titulos_figuras = titulos_figuras
    st.session_state.visualizacion_generada = True
    
    # Devolvemos True para indicar que la visualización se generó correctamente
    return True

# Menú de navegación
menu.generarMenu(st.session_state['usuario'])

# Gestión del estado
if 'df_eventos' not in st.session_state:
    st.session_state.df_eventos = None
if 'df_estadisticas' not in st.session_state:
    st.session_state.df_estadisticas = None

# Cargar datos
if st.session_state.df_eventos is None or st.session_state.df_estadisticas is None:
    st.session_state.df_eventos, st.session_state.df_estadisticas = DataManagerMatch.get_match_data()

if st.session_state.df_eventos is not None and st.session_state.df_estadisticas is not None:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        equipos = DataManagerMatch.get_equipos(st.session_state.df_eventos)
        equipo = st.selectbox('Equipo:', equipos)
    
    with col2:
        temporadas = DataManagerMatch.get_temporadas(st.session_state.df_eventos, equipo)
        temporada = st.selectbox('Temporada:', temporadas)
    
    with col3:
        partidos = DataManagerMatch.get_partidos(st.session_state.df_eventos, temporada, equipo)
        partido_seleccionado = st.selectbox('Partido:', partidos)

    if partido_seleccionado:
        partido_original = partido_seleccionado.split(' (')[0]
        match_id = st.session_state.df_eventos[
            st.session_state.df_eventos['partido'] == partido_original
        ]['match_id'].iloc[0]
        
        # Obtener datos específicos del partido
        eventos_partido, stats_partido = DataManagerMatch.get_partido_data(
            match_id,
            st.session_state.df_eventos,
            st.session_state.df_estadisticas
        )
        
        if not eventos_partido.empty:
            local_team, away_team = extract_teams(partido_original)
            equipo_list = st.session_state.df_eventos['equipo'].unique()
            
            local_team = find_similar_team(local_team, equipo_list)
            away_team = find_similar_team(away_team, equipo_list)
            
            if not local_team:
                st.error(f"No se pudo encontrar una coincidencia válida para el equipo local: {extract_teams(partido_original)[0]}")
            if not away_team:
                st.error(f"No se pudo encontrar una coincidencia válida para el equipo visitante: {extract_teams(partido_original)[1]}")
            
            info_partido = eventos_partido.iloc[0]
            
            st.markdown("---")
            cols_info = st.columns(4)
            with cols_info[0]:
                st.metric("Local", local_team)
            with cols_info[1]:
                st.metric("Visitante", away_team)
            with cols_info[2]:
                st.metric("Liga", info_partido.get('liga', 'No disponible'))
            with cols_info[3]:
                st.metric("Temporada", temporada)

            # Primero, modifica la sección de botones (antes del if generar_viz)
            button_cols = st.columns([1, 1, 3])
            with button_cols[0]:
                generar_viz = st.button('Generar Visualización', type='primary')
            with button_cols[1]:
                exportar_pdf = st.button('Exportar a PDF', type='secondary')

            if generar_viz:
                # Mostrar overlay de carga mientras se generan las visualizaciones
                overlay_container = st.empty()
                
                # Mostrar la pantalla de carga
                overlay_container.markdown(
                    f"""
                    <div class="loading-overlay" id="loading-gif">
                        <div class="loading-content">
                            <div class="image-container">
                                <img src="data:image/png;base64,{get_image_base64_optimized(logo_path)}" 
                                    class="loading-logo" alt="Cargando...">
                            </div>
                            <div class="loading-text" id="loading-text">Generando análisis del partido...</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                with st.spinner('Generando visualización...'):
                    try:
                        partido_info = {
                            'local_team': local_team,
                            'away_team': away_team,
                            'temporada': temporada,
                            'partido_original': partido_original,
                            'match_id': match_id
                        }
                        # Llamar a la función que genera y muestra las visualizaciones
                        success = generate_match_viz(eventos_partido, stats_partido, partido_info)
                        
                        # Establecer explícitamente el estado después de generar la visualización
                        if success:
                            st.session_state.visualizacion_generada = True
                            # Código JS para ocultar el loading
                            st.components.v1.html(
                                """
                                <script>
                                    // Cambiar el texto y aplicar el difuminado
                                    const loadingText = document.getElementById("loading-text");
                                    const loadingGif = document.getElementById("loading-gif");

                                    if (loadingText && loadingGif) {
                                        loadingText.textContent = "¡Visualización lista!";
                                        loadingGif.classList.add("fade-out");

                                        // Eliminar el contenedor después de la animación
                                        setTimeout(() => {
                                            if (loadingGif) {
                                                loadingGif.style.display = "none";
                                            }
                                        }, 1000);  // 1000ms = 1 segundo (duración de la animación)
                                    }
                                </script>
                                """,
                                height=0,
                            )
                            
                            # Forzar actualización de la página para que el botón se active
                            st.session_state.visualizacion_generada = True

                        else:
                            st.error("No se pudo generar la visualización")
                            
                        # Limpiar el overlay una vez que todo esté listo
                        overlay_container.empty()
                        
                    except Exception as e:
                        # También limpiar la pantalla de carga en caso de error
                        overlay_container.empty()
                        st.error(f"Error al generar la visualización: {e}")
                        st.exception(e)

            # Mover la lógica de exportación a PDF fuera del bloque generar_viz
            if exportar_pdf:
                if 'visualizacion_generada' in st.session_state and st.session_state.visualizacion_generada:
                    
                    try:
                        # Verificar si existen tanto figuras como títulos en session_state
                        if 'figuras_generadas' in st.session_state and 'titulos_figuras' in st.session_state:
                            # Crear lista de tuplas (figura, título)
                            figuras_con_titulos = list(zip(
                                st.session_state.figuras_generadas,
                                st.session_state.titulos_figuras
                            ))
                            
                            # Obtener información del partido desde session_state
                            partido_info = st.session_state.partido_info
                            
                            # Obtener la ruta del logo para incluirlo en el PDF
                            logo_path_for_pdf = logo_path if os.path.exists(logo_path) else None
                            
                            # Generar el PDF
                            pdf_buffer = generar_pdf_partido(
                                partido_info['local_team'],
                                partido_info['away_team'],
                                partido_info['temporada'],
                                partido_info['match_id'],
                                figuras_con_titulos,
                                logo_path_for_pdf
                            )
                            
                            # Ofrecer la descarga del PDF
                            nombre_archivo = f"Analisis_{partido_info['local_team']}_vs_{partido_info['away_team']}_{partido_info['temporada']}.pdf"
                            
                            st.download_button(
                                label="Descargar PDF",
                                data=pdf_buffer,
                                file_name=nombre_archivo,
                                mime="application/pdf",
                                key='download_pdf'
                            )
                            
                            st.success(f"¡PDF generado! Haz clic en 'Descargar PDF' para guardarlo.")
                            
                        else:
                            st.warning("No hay visualizaciones disponibles para exportar.")
                    except Exception as e:
                        st.error(f"Error al generar el PDF: {str(e)}")
                        import traceback
                        st.text(traceback.format_exc())