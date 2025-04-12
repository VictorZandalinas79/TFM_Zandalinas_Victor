import streamlit as st

# Configuración de la página de Streamlit MUST BE FIRST
st.set_page_config(
    page_title="Rendimiento del Equipo", 
    page_icon="⚽", 
    layout="wide", 
    initial_sidebar_state="collapsed",
    menu_items=None
)

from scipy.spatial import ConvexHull
import common.menu as menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mplsoccer import VerticalPitch
import os
import base64
import io
import gc  # Importar el garbage collector
from pathlib import Path
from PIL import Image
import matplotlib
import matplotlib.patches as patches
matplotlib.use('Agg')  # Usar backend no interactivo para reducir consumo de memoria
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as ReportLabImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_LEFT, TA_CENTER


# Configuración de rutas
current_dir = Path(__file__).parent
BASE_DIR = current_dir.parent

# Intentar importar DataManagerTeams
try:
    from data_manager_teams import DataManagerTeams
except ImportError as e:
    st.error(f"No se pudo importar DataManagerTeams: {e}")
    st.stop()

# Configuración de rutas de assets
icon_path = os.path.join(BASE_DIR, 'assets', 'icono_player.png')
logo_path = os.path.join(BASE_DIR, 'assets', 'escudo_alaves_original.png')
banner_path = os.path.join(BASE_DIR, 'assets', 'bunner_alaves.png')
styles_path = os.path.join(BASE_DIR, 'assets', 'styles.css')

# Cargar estilos CSS personalizados
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css(styles_path)

# Configuración de colores globales
BACKGROUND_COLOR = '#0E1117'
LINE_COLOR = '#FFFFFF'
TEXT_COLOR = '#FFFFFF'
HIGHLIGHT_COLOR = '#4BB3FD'

# Función para limpiar la memoria
def clear_memory():
    # Cerrar todas las figuras de matplotlib para liberar memoria
    plt.close('all')
    
    # Forzar la recolección de basura
    gc.collect()
    
    # Limpiar caché de matplotlib
    matplotlib.pyplot.rcParams.update(matplotlib.rcParamsDefault)
    
    # Limpiar variables de sesión específicas si es necesario
    if 'data_cache' in st.session_state:
        del st.session_state['data_cache']

# Función para obtener imagen base64
def get_image_base64_optimized(path, max_size=(1024, 1024), quality=70):  # Menor tamaño y calidad
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
    
# Función para crear encabezado información equipo - CON COLORES DINÁMICOS  
def create_team_header(equipo, temporada):
    """
    Crea un encabezado con información y KPIs del equipo con colores dinámicos.
    Los colores cambian según el valor:
    - Verde (más oscuro cuanto más alto) para valores >= 6
    - Naranja para valores entre 5 y 6
    - Rojo (más oscuro cuanto más bajo) para valores < 5
    
    Args:
        equipo: Nombre del equipo
        temporada: Temporada seleccionada
        
    Returns:
        None (dibuja directamente en Streamlit)
    """
    try:
        # Cargar datos de eventos acumulados
        eventos_acumulados = DataManagerTeams.get_accumulated_events_data()
        
        if eventos_acumulados is None:
            st.warning("No se pudieron cargar los datos acumulados")
            mostrar_encabezado_predeterminado(equipo, temporada)
            return
        
        # Asegurarse que temporada sea string para la comparación
        temporada_str = str(temporada)
        
        # Filtrar datos - asegurando que los tipos sean compatibles
        if 'temporada' in eventos_acumulados.columns:
            # Convertir temporada a string en el DataFrame si es necesario
            if eventos_acumulados['temporada'].dtype != 'object':
                eventos_acumulados['temporada'] = eventos_acumulados['temporada'].astype(str)
            
            # Filtrar por equipo y temporada
            filtered_data = eventos_acumulados[
                (eventos_acumulados['equipo'] == equipo) &
                (eventos_acumulados['temporada'] == temporada_str)
            ]
        else:
            # Filtrar solo por equipo si no hay temporada
            filtered_data = eventos_acumulados[eventos_acumulados['equipo'] == equipo]
        
        if filtered_data.empty:
            st.warning(f"No hay datos acumulados para {equipo} en {temporada}")
            mostrar_encabezado_predeterminado(equipo, temporada)
            return
        
        # Verificar el nombre exacto de la columna KPI - nombres posibles
        if 'KPI_Rendimiento' in filtered_data.columns:
            kpi_column = 'KPI_Rendimiento'
        elif 'KPI_rendimiento' in filtered_data.columns:
            kpi_column = 'KPI_rendimiento'
        else:
            print("No se encontró columna de KPI. Columnas disponibles:", filtered_data.columns.tolist())
            kpi_column = None
        
        # Calcular promedios de KPI y valoración
        if kpi_column:
            kpi_mean = filtered_data[kpi_column].mean()
        else:
            kpi_mean = 0.0
            
        valoracion_mean = filtered_data['valoracion'].mean() if 'valoracion' in filtered_data.columns else 0.0
        
        # Función para determinar color según el valor
        def get_color(value):
            if value >= 8:
                return '#006400'  # Verde oscuro
            elif value >= 7:
                return '#228B22'  # Verde bosque
            elif value >= 6:
                return '#32CD32'  # Lima verde
            elif value >= 5:
                return '#FFA500'  # Naranja
            elif value >= 4:
                return '#FF4500'  # Rojo-naranja
            else:
                return '#8B0000'  # Rojo oscuro
        
        # Obtener color para KPI y valoración
        kpi_color = get_color(kpi_mean)
        val_color = get_color(valoracion_mean)
        
        # Crear contenedor para el header con estilo personalizado
        st.markdown("""
            <div class="team-header">
                <div class="team-info">
                    <div class="team-name">{}</div>
                    <div class="team-season">{}</div>
                </div>
                <div class="kpi-boxes">
                    <div class="kpi-box" style="background: {};">
                        <div class="kpi-label">KPI RENDIMIENTO</div>
                        <div class="kpi-value">{:.2f}</div>
                        <div class="kpi-category">{}</div>
                    </div>
                    <div class="kpi-box" style="background: {};">
                        <div class="kpi-label">VALORACIÓN BEPRO</div>
                        <div class="kpi-value">{:.2f}</div>
                        <div class="kpi-category">{}</div>
                    </div>
                </div>
            </div>
        """.format(
            equipo, 
            temporada,
            kpi_color, kpi_mean, get_category(kpi_mean),
            val_color, valoracion_mean, get_category(valoracion_mean)
        ), unsafe_allow_html=True)
        
        # Agregar stats adicionales si es necesario
        with st.expander("Ver estadísticas adicionales"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Partidos", filtered_data['match_id'].nunique())
            with col2:
                # Buscar todas las posibles variantes de columnas para goles a favor
                for col_name in ['goles', 'goles_favor', 'goles a favor']:
                    if col_name in filtered_data.columns:
                        goles_favor = filtered_data[col_name].sum()
                        break
                else:
                    goles_favor = 0
                st.metric("Goles a favor", goles_favor)
            with col3:
                # Buscar todas las posibles variantes de columnas para goles en contra
                for col_name in ['goles_concedidos', 'goles_contra', 'goles en contra']:
                    if col_name in filtered_data.columns:
                        goles_contra = filtered_data[col_name].sum()
                        break
                else:
                    goles_contra = 0
                st.metric("Goles en contra", goles_contra)
            with col4:
                diferencia = goles_favor - goles_contra
                st.metric("Diferencia", diferencia, delta=None)
        
    except Exception as e:
        st.error(f"Error al crear el encabezado del equipo: {e}")
        print(f"Error detallado: {str(e)}")
        mostrar_encabezado_predeterminado(equipo, temporada)


def get_category(value):
    """Devuelve la categoría descriptiva según el valor"""
    if value >= 8:
        return "Excelente"
    elif value >= 7:
        return "Muy bueno"
    elif value >= 6:
        return "Bueno"
    elif value >= 5:
        return "Regular"
    elif value >= 4:
        return "Bajo"
    else:
        return "Crítico"


def mostrar_encabezado_predeterminado(equipo, temporada):
    """
    Función auxiliar para mostrar un encabezado con valores predeterminados
    """
    st.markdown("""
        <div class="team-header">
            <div class="team-info">
                <div class="team-name">{}</div>
                <div class="team-season">{}</div>
            </div>
            <div class="kpi-boxes">
                <div class="kpi-box">
                    <div class="kpi-label">KPI RENDIMIENTO</div>
                    <div class="kpi-value">--</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">VALORACIÓN BEPRO</div>
                    <div class="kpi-value">--</div>
                </div>
            </div>
        </div>
    """.format(equipo, temporada), unsafe_allow_html=True)

# Función para crear gráficas específicas
def create_xg_evolution_chart(df_combined, equipo, temporada):
    """
    Crea un gráfico de líneas que muestra la evolución continua de xG a lo largo del tiempo
    para un equipo específico y sus rivales en una temporada.
    """
    # Crear la figura y el eje
    fig, ax = plt.subplots(figsize=(5, 5), facecolor=BACKGROUND_COLOR)

    # Intentar convertir la columna xG a numérico con múltiples estrategias
    def safe_numeric_conversion(x):
        try:
            # Intenta convertir directamente
            return pd.to_numeric(x, errors='coerce')
        except:
            try:
                # Si falla, intenta reemplazar comas por puntos (problema común en algunos locales)
                return pd.to_numeric(x.str.replace(',', '.'), errors='coerce')
            except:
                return pd.Series([np.nan] * len(x))
    
    # Aplicar conversión segura
    df_combined['xg'] = safe_numeric_conversion(df_combined['xg'])
    
    # Filtrar valores no nulos y positivos
    df_combined = df_combined[df_combined['xg'].notna() & (df_combined['xg'] > 0)]
    
    # Filtrar datos por temporada
    df_temporada = df_combined[df_combined['temporada'] == temporada].copy()
    
    # Separar datos del equipo y rivales
    df_team = df_temporada[df_temporada['equipo'] == equipo].copy()
    df_rivals = df_temporada[df_temporada['equipo'] != equipo].copy()
    
    # Función para procesar xG acumulado por minuto
    def process_xg_by_minute(df, label):
        # Verificar si hay datos
        if df.empty:
            print(f"⚠️ No hay datos para {label}")
            return None
        
        # Convertir event_time a minutos
        def convert_to_minutes(row):
            try:
                # Manejar diferentes formatos de tiempo
                time_str = str(row['event_time']).strip()
                
                # Si está en formato HH:MM
                if ':' in time_str:
                    minutes = int(time_str.split(':')[0])
                # Si es un número entero
                elif time_str.isdigit():
                    minutes = int(time_str)
                else:
                    print(f"Formato de tiempo no reconocido: {time_str}")
                    return None
                
                # Ajustar para segunda parte
                if str(row['periodo']) == '2ª_parte':
                    minutes += 45
                
                return minutes
            except Exception as e:
                print(f"Error convirtiendo minutos para {label}: {e}")
                return None
        
        # Añadir columna de minutos
        df.loc[:, 'minutes'] = df.apply(convert_to_minutes, axis=1)
        
        # Filtrar valores válidos
        df = df.dropna(subset=['minutes', 'xg'])
        
        # Verificar si quedan datos después del filtrado
        if df.empty:
            print(f"⚠️ No quedan datos válidos para {label} después del filtrado")
            return None
        
        # Agrupar por partido y minuto, sumando xG
        df_grouped = df.groupby(['match_id', 'minutes'])['xg'].sum().reset_index()
        
        # Calcular xG acumulado por partido
        df_grouped['xg_acum'] = df_grouped.groupby('match_id')['xg'].cumsum()
        
        # Calcular promedio de xG acumulado para cada minuto a través de todos los partidos
        df_avg = df_grouped.groupby('minutes')['xg'].mean().reset_index()
        df_avg['xg_acum'] = df_avg['xg'].cumsum()
        
        # Asegurar valores finitos
        df_avg = df_avg[np.isfinite(df_avg['xg_acum'])]
        
        return df_avg
    
    # Procesar datos para el equipo y rivales
    team_xg = process_xg_by_minute(df_team, equipo)
    rivals_xg = process_xg_by_minute(df_rivals, "Rivales")
    
    # Graficar líneas de xG acumulado
    def plot_xg_line(data, color, label):
        if data is not None and not data.empty:
            try:
                ax.plot(data['minutes'], data['xg_acum'], 
                        color=color, label=label, linewidth=2)
                
                # Añadir valores cada 15 minutos
                for minuto in [0, 15, 30, 45, 60, 75, 90]:
                    cerca = data[data['minutes'] <= minuto]
                    if not cerca.empty:
                        valor = cerca['xg_acum'].iloc[-1]
                        ax.text(minuto, valor, f"{valor:.2f}",
                                color=color, fontsize=10, ha="right", va="bottom")
            except Exception as e:
                print(f"Error graficando línea para {label}: {e}")
    
    # Graficar líneas
    plot_xg_line(team_xg, "white", equipo)
    plot_xg_line(rivals_xg, "red", "Rivales")
    
    # Añadir línea vertical en el medio tiempo
    ax.axvline(x=45, color='white', linestyle='--', alpha=0.3)
    ax.text(45, ax.get_ylim()[1], 'Medio Tiempo',
            color='white', alpha=0.7, ha='center', va='bottom')
    
    # Configuración del gráfico
    ax.set_xlabel("Minuto del Partido", fontsize=12, color="white")
    ax.set_ylabel("xG Acumulado Promedio", fontsize=12, color="white")
    ax.legend(frameon=False, labelcolor='white', fontsize=10)
    
    # Ajustar estilos
    ax.set_facecolor("#1E1E1E")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("white")
    ax.spines["bottom"].set_color("white")
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")
    
    # Asegurar que el eje X vaya de 0 a 90 minutos
    ax.set_xlim(0, 90)
    
    # Añadir grid
    ax.grid(True, alpha=0.2, color='white')
    
    return fig

# Función evolución de las KPI
def create_kpi_evolution_chart(df_combined, equipo, temporada):
    """
    Creates a bar chart for KPI evolution across matchdays with colors based on home/away matches.
    """
    fig, ax = plt.subplots(figsize=(5, 5), facecolor=BACKGROUND_COLOR)

    try:
        # Filtrar datos para el equipo y temporada
        filtered_data = df_combined[
            (df_combined['equipo'] == equipo) &
            (df_combined['temporada'] == temporada)
        ].copy()
        
        if filtered_data.empty:
            ax.text(
                0.5, 0.5, 
                f"No hay datos para\n{equipo}\nen {temporada}", 
                ha='center', 
                va='center', 
                color='white', 
                transform=ax.transAxes,
                fontsize=12
            )
            return fig
        
        # Extraer número de jornada
        filtered_data['numero_jornada'] = filtered_data['jornada'].str.extract(r'(\d+)').astype(int)

        # Convertir KPI_rendimiento a numérico
        filtered_data['KPI_Rendimiento'] = pd.to_numeric(
            filtered_data['KPI_Rendimiento'], 
            errors='coerce'
        )
        
        # Determinar si es local o visitante
        filtered_data['es_local'] = filtered_data['partido'].apply(
            lambda x: equipo in str(x).split('-')[0].strip()
        )
        
        # Eliminar filas con valores nulos
        filtered_data = filtered_data.dropna(subset=['numero_jornada', 'KPI_Rendimiento'])
        
        # Calcular medias de KPI para local y visitante
        media_local = filtered_data[filtered_data['es_local']]['KPI_Rendimiento'].mean()
        media_visitante = filtered_data[~filtered_data['es_local']]['KPI_Rendimiento'].mean()
        
        # Agrupar por número de jornada y calcular la media del KPI de rendimiento
        grouped_data = filtered_data.groupby(['numero_jornada', 'es_local'])['KPI_Rendimiento'].mean().reset_index()
        
        # Ordenar por número de jornada
        grouped_data = grouped_data.sort_values('numero_jornada')
        
        # Crear el gráfico de barras con colores basados en local/visitante
        bars = ax.bar(
            grouped_data['numero_jornada'],
            grouped_data['KPI_Rendimiento'],
            color=[HIGHLIGHT_COLOR if local else '#FF6B6B' for local in grouped_data['es_local']],
            width=0.8,
            alpha=0.7
        )
        
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
        
        # Personalizar el gráfico
        ax.set_facecolor(BACKGROUND_COLOR)
        ax.grid(True, color='white', alpha=0.2, linestyle='--')
        ax.set_xlabel('Jornada', color='white', fontsize=12)
        ax.set_ylabel('Media KPI Rendimiento', color='white', fontsize=12)
        ax.tick_params(colors='white')
        
        # Configurar el eje X para ir siempre de 0 a 38
        ax.set_xlim(0, 38)
        ax.set_xticks(range(0, 39, 2))  # Mostrar números de jornada de 2 en 2
        
        # Configurar el eje Y
        y_max = max(10, grouped_data['KPI_Rendimiento'].max() * 1.1)
        ax.set_ylim(0, y_max)
        
        
        # Personalizar los bordes
        for spine in ax.spines.values():
            spine.set_color('white')
            spine.set_alpha(0.3)
        
        # Añadir leyenda
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=HIGHLIGHT_COLOR, alpha=0.7, label=f'Local (Media: {media_local:.2f})'),
            Patch(facecolor='#FF6B6B', alpha=0.7, label=f'Visitante (Media: {media_visitante:.2f})')
        ]
        ax.legend(handles=legend_elements, 
                 loc='upper right', 
                 facecolor=BACKGROUND_COLOR, 
                 edgecolor='white', 
                 labelcolor='white')
        
        # Añadir padding para las etiquetas
        ax.margins(y=0.1)
        
        return fig
        
    except Exception as e:
        print(f"Error procesando los datos: {e}")
        
        
        # Añadir mensaje de error al gráfico
        ax.text(
            0.5, 0.5, 
            f"Error:\n{str(e)}", 
            ha='center', 
            va='center', 
            color='red', 
            transform=ax.transAxes,
            fontsize=12
        )
        
        return fig

# Agregar la función lineup justo después de las otras funciones de gráficas
def create_team_lineup_heatmap(ax, df_team_lineups, team_name, pitch):
    try:
        if df_team_lineups.empty:
            ax.text(50, 34, f"No hay datos de alineaciones para {team_name}", 
                   ha='center', va='center', color='red', fontsize=8)
            return
        
        # Convertir coordenadas a numéricas y escalar de 0-1 a 0-100
        df_team_lineups.loc[:, 'position_x'] = pd.to_numeric(df_team_lineups['position_x'], errors='coerce') * 68
        df_team_lineups.loc[:, 'position_y'] = pd.to_numeric(df_team_lineups['position_y'], errors='coerce') * 100
        
        # Eliminar filas con coordenadas nulas
        df_team_lineups = df_team_lineups.dropna(subset=['position_x', 'position_y'])
        
        # Obtener las 11 posiciones más repetidas
        top_positions = df_team_lineups['position_name'].value_counts().head(11)
        
        # Preparar datos para visualización
        lineup_summary = []
        for position, total_count in top_positions.items():
            position_data = df_team_lineups[df_team_lineups['position_name'] == position]
            
            if not position_data.empty:
                x_mean = position_data['position_x'].mean()
                y_mean = position_data['position_y'].mean()
                
                # Obtener los jugadores y calcular sus porcentajes
                top_players = (position_data.groupby(['player_name', 'player_last_name'])
                             .size()
                             .reset_index(name='appearances')
                             .sort_values('appearances', ascending=False)
                             .head(3))
                
                total_games = position_data['match_id'].nunique()
                top_players['percentage'] = (top_players['appearances'] / total_games * 100)
                
                lineup_summary.append({
                    'position': position,
                    'x_mean': x_mean,
                    'y_mean': y_mean,
                    'players': top_players
                })
        
        # Dibujar el campo
        pitch.draw(ax=ax)
        
        if lineup_summary:
            # Crear scatter plot para las posiciones con puntos más pequeños
            scatter = ax.scatter(
                [item['x_mean'] for item in lineup_summary],
                [item['y_mean'] for item in lineup_summary],
                color='#4BB3FD',
                s=200,  # Reducido de 600 a 200
                alpha=0.8,
                edgecolors='white'
            )
            
            # Añadir anotaciones
            for item in lineup_summary:
                x = item['x_mean']
                y = item['y_mean']
                
                # Añadir el nombre de la posición dentro del círculo con fuente más pequeña
                ax.text(x, y, item['position'],
                       ha='center', va='center',
                       color='white',
                       fontsize=6,  # Reducido de 12 a 6
                       weight='bold')
                
                # Crear el texto con los jugadores
                player_texts = []
                for _, player in item['players'].iterrows():
                    initial = player['player_name'][0] if player['player_name'] else ''
                    name_text = f"{initial}. {player['player_last_name']}"
                    # Aumentar el tamaño de la fuente de los jugadores
                    font_size = 6 + (player['percentage'] / 100 * 4)  # Aumentado el rango
                    player_texts.append((name_text, font_size))
                
                # Crear un texto para cada jugador con espaciado más pequeño
                for name_text, font_size in player_texts:
                    bbox_props = dict(
                        boxstyle='round,pad=0.2',  # Reducido el padding
                        fc='#4BB3FD',
                        ec='white',
                        alpha=0.8
                    )
                    
                    ax.annotate(
                        name_text,
                        (x, y),
                        xytext=(0, -8 - 10 * player_texts.index((name_text, font_size))),  # Reducido el espaciado
                        textcoords='offset points',
                        ha='center',
                        va='top',
                        weight='bold',
                        color='black',
                        bbox=bbox_props,
                        fontsize=font_size  # Tamaño de fuente aumentado
                    )
                
        # Configurar el fondo de la figura y los ejes
        fig = ax.figure
        fig.patch.set_facecolor('#1E1E1E')  # Fondo de la figura
        ax.set_facecolor('#1E1E1E')  # Fondo de los ejes
        
        # Ajustar márgenes para evitar que el texto se salga
        plt.tight_layout()
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # Ajustar márgenes

    except Exception as e:
        print(f"Error al generar el campograma de alineaciones: {e}")
        print(f"Detalles del error: {str(e)}")
        ax.text(
            50, 34, f"Error: {str(e)}",
            ha='center',
            va='center',
            fontsize=8,
            color='red'
        )

# Grafica para comparación de KPI con rival
def create_team_vs_all_rivals_pizza_chart(ax, df_KPI, equipo, season_id):
    """
    Crea un gráfico de radar comparando los KPIs del equipo con la media de todos sus rivales en la temporada.
    """
    kpi_columns = [
        'Progresion_Ataque',        
        'Verticalidad',             
        'Ataques_Bandas',           
        'Peligro_Generado',         
        'Rendimiento_Finalizacion', 
        'Eficacia_Defensiva',       
        'Estilo_Combinativo_Directo',
        'Zonas_Recuperacion',       
        'Altura_Bloque_Defensivo',  
        'Posesion_Dominante',       
        'KPI_Rendimiento'           
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
        "Posesión\nDominante",
        "Rendimiento"
    ]

    # Datos del equipo principal
    team_data = df_KPI[
        (df_KPI['equipo'] == equipo) &
        (df_KPI['temporada'] == season_id)
    ][kpi_columns].mean()

    # Datos de todos los rivales
    rivals_data = df_KPI[
        (df_KPI['equipo'] != equipo) &
        (df_KPI['temporada'] == season_id)
    ][kpi_columns].mean()

    if team_data.empty:
        print(f"No hay datos para el equipo {equipo}")
        return

    # Calcular valores normalizados para equipo y rivales
    values_team = []
    values_rivals = []
    
    for col in kpi_columns:
        if col in team_data.index:
            team_val = max(0, min(10, team_data[col]))
            rival_val = max(0, min(10, rivals_data[col]))
            values_team.append(team_val)
            values_rivals.append(rival_val)
        else:
            values_team.append(0)
            values_rivals.append(0)

    # Configurar el gráfico
    num_vars = len(params)
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]
    values_team += values_team[:1]
    values_rivals += values_rivals[:1]

    # Limpiar el eje actual
    ax.cla()

    # Dibujar los polígonos para equipo y rivales
    ax.plot(angles, values_team, 'o-', linewidth=2, color='#4BB3FD', label=equipo)
    ax.fill(angles, values_team, alpha=0.25, color='#4BB3FD')
    
    ax.plot(angles, values_rivals, 'o-', linewidth=2, color='#FF6B6B', label='Media Rivales')
    ax.fill(angles, values_rivals, alpha=0.25, color='#FF6B6B')

    # Configurar los ejes
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(params, size=8, fontweight='bold')
    ax.set_ylim(0, 10)

    # Configurar el estilo
    ax.grid(True, color='white', alpha=0.3)
    ax.set_facecolor(BACKGROUND_COLOR)
    
    # Configurar la figura completa
    fig = ax.figure
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    
    # Configurar colores del texto
    for label in ax.get_xticklabels():
        label.set_color('white')
        label.set_fontsize(12)
    for label in ax.get_yticklabels():
        label.set_color('white')
        label.set_fontsize(12)
    
    # Configurar el color de los ejes
    ax.spines['polar'].set_color('white')
    ax.spines['polar'].set_alpha(0.3)
    
    # leyenda con fondo transparente
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.25))
    legend.get_frame().set_facecolor(BACKGROUND_COLOR)
    legend.get_frame().set_alpha(0.8)
    plt.setp(legend.get_texts(), color='white', fontsize=10)

# Función para crear mapa de calor y flujo de pases en campo vertical completo
def create_passes_heatmap(df_combined, equipo, temporada):
    """
    Crea un mapa de calor y flujo de pases para el equipo seleccionado.
    Utiliza un campo vertical completo similar al de las alineaciones.
     
    Args:
        df_combined: DataFrame con los datos combinados
        equipo: Nombre del equipo a analizar
        temporada: Temporada a analizar
         
    Returns:
        Figura de matplotlib con el campograma
    """
    try:
        # Filtrar datos para el equipo y temporada
        df_team = df_combined[
            (df_combined['equipo'] == equipo) & 
            (df_combined['temporada'] == temporada)
        ]
        
        # Filtrar solo eventos de tipo "Pase"
        df_team = df_team[df_team['tipo_evento'] == 'Pase']
        
        # Obtener datos de pases usando las columnas correctas (xstart, ystart, xend, yend)
        df_pass = pd.DataFrame({
            'x': pd.to_numeric(df_team['ystart'], errors='coerce'),
            'y': pd.to_numeric(df_team['xstart'], errors='coerce'),
            'end_x': pd.to_numeric(df_team['yend'], errors='coerce'),
            'end_y': pd.to_numeric(df_team['xend'], errors='coerce')
        }).dropna()
        
        # Si no hay suficientes datos, mostramos un mensaje
        if len(df_pass) < 10:
            fig, ax = plt.subplots(figsize=(5, 7), facecolor='none')
            ax.text(0.5, 0.5, "No hay suficientes datos de pases\npara crear el mapa de calor",
                   ha='center', va='center', color='white', fontsize=12)
            ax.set_facecolor('none')
            return fig
        
        # Configurar el campo vertical tipo wyscout sin ejes y etiquetas
        pitch = VerticalPitch(
            pitch_type='wyscout',
            axis=False, 
            label=False,  
            pitch_color='none',
            line_color='white',
            stripe=False
        )
        
        # Crear figura y ejes
        fig, ax = plt.subplots(figsize=(5, 7), facecolor='none')
        fig.set_facecolor('none')
        
        # Dibujar el campo
        pitch.draw(ax=ax)
        
        # Definir bins para el heatmap (ajustado para campo completo)
        bins = (6, 8)  # Más bins verticales para el campo completo
        
        # Crear mapa de calor
        bs_heatmap = pitch.bin_statistic(df_pass.x, df_pass.y, statistic='count', bins=bins)
        hm = pitch.heatmap(bs_heatmap, ax=ax, cmap='Blues', alpha=0.7)
        
        # Crear mapa de flujo de pases
        fm = pitch.flow(df_pass.x, df_pass.y, df_pass.end_x, df_pass.end_y,
                        color=HIGHLIGHT_COLOR, arrow_type='same',
                        arrow_length=5, bins=bins, ax=ax)
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        # Si hay un error, creamos un gráfico con el mensaje de error
        fig, ax = plt.subplots(figsize=(5, 7), facecolor='none')
        ax.text(0.5, 0.5, f"Error al crear el mapa de pases:\n{str(e)}",
               ha='center', va='center', color='red', fontsize=10)
        print(f"Error detallado: {e}")  # Para depuración
        ax.set_facecolor('none')
        return fig
    
# Función para visualizar acciones defensivas usando VerticalPitch tipo wyscout
def create_defensive_actions_hull(df_combined, equipo, temporada):
    """
    Crea un mapa de acciones defensivas utilizando convex hull
    para mostrar la posición media de los jugadores durante estas acciones.
    Utiliza VerticalPitch de tipo wyscout.
    
    Args:
        df_combined: DataFrame con los datos combinados
        equipo: Nombre del equipo a analizar
        temporada: Temporada a analizar
        
    Returns:
        Figura de matplotlib con el campograma de acciones defensivas
    """
    try:
        # Importar scipy para convex hull
        from scipy.spatial import ConvexHull
        import numpy as np
        
        # Filtrar datos para el equipo y temporada
        df_team = df_combined[
            (df_combined['equipo'] == equipo) & 
            (df_combined['temporada'] == temporada)
        ]
        
        # Filtrar solo eventos defensivos
        eventos_defensivos = ['Interceptación', 'Recuperación', 'Entrada']
        df_defensive = df_team[
            (df_team['tipo_evento'].isin(eventos_defensivos)) & 
            (df_team['demarcacion'] != 'Portero')  # Excluir porteros
        ]
        
        # Contar acciones defensivas por jugador
        acciones_por_jugador = df_defensive['jugador'].value_counts()
        
        # Filtrar jugadores con al menos 50 acciones defensivas
        jugadores_validos = acciones_por_jugador[acciones_por_jugador >= 50].index
        df_defensive = df_defensive[df_defensive['jugador'].isin(jugadores_validos)]
        
        # Si no hay suficientes datos defensivos, mostramos un mensaje
        if len(df_defensive) < 3:  # Necesitamos al menos 3 puntos para un hull
            fig, ax = plt.subplots(figsize=(5, 7), facecolor=BACKGROUND_COLOR)
            ax.text(0.5, 0.5, "No hay suficientes datos de acciones defensivas\npara crear el convex hull",
                   ha='center', va='center', color='white', fontsize=12)
            ax.set_facecolor(BACKGROUND_COLOR)
            return fig
        
        # Convertir coordenadas a numérico (INTERCAMBIADAS)
        for col in ['ystart', 'xstart']:  # Cambiado el orden
            df_defensive[col] = pd.to_numeric(df_defensive[col], errors='coerce')
        
        # Filtrar filas con valores válidos (INTERCAMBIADAS)
        df_defensive = df_defensive[
            df_defensive['ystart'].notna() &  # Cambiado
            df_defensive['xstart'].notna() &  # Cambiado
            (df_defensive['ystart'] != 0) &   # Cambiado
            (df_defensive['xstart'] != 0)     # Cambiado
        ]
        
        # Configurar el campo vertical tipo wyscout con ejes y etiquetas
        pitch = VerticalPitch(
        pitch_type='wyscout',
        axis=False, 
        label=False,  
        pitch_color='none',
        line_color='white',
        stripe=False
    )
        
        # Crear figura y ejes
        fig, ax = plt.subplots(figsize=(5, 7), facecolor=BACKGROUND_COLOR)
        
        # Dibujar el campo
        pitch.draw(ax=ax)
        
        # Obtener posiciones medias por jugador para todos los eventos defensivos juntos
        player_positions = []
        jugadores = df_defensive['jugador'].unique()
        
        for jugador in jugadores:
            df_player = df_defensive[df_defensive['jugador'] == jugador]
            
            if len(df_player) > 0:
                # Obtener posición media para este jugador (INTERCAMBIADAS)
                x_mean = df_player['xstart'].mean()  # Cambiado
                y_mean = df_player['ystart'].mean()  # Cambiado
                
                # Verificar que las medias sean válidas
                if not (pd.isna(x_mean) or pd.isna(y_mean) or x_mean == 0 or y_mean == 0):
                    # Añadir posición media a la lista
                    player_positions.append({
                        'jugador': jugador,
                        'x': x_mean,
                        'y': y_mean
                    })
        
        # Convertir a DataFrame para facilitar el procesamiento
        df_positions = pd.DataFrame(player_positions)
        
        if len(df_positions) < 3:
            ax.text(50, 50, "No hay suficientes posiciones de jugadores\npara crear un convex hull",
                  ha='center', va='center', color='white', fontsize=12)
        else:
            # Dibujar puntos para cada jugador (todos azules y del mismo tamaño)
            ax.scatter(
                df_positions['x'], 
                df_positions['y'],
                s=50,  # Tamaño fijo para todos los puntos
                color='#4BB3FD',  # Azul (usando el color destacado de tu aplicación)
                alpha=0.9,
                edgecolors='white',
                zorder=2  # Para que estén por encima del hull
            )
            
            # Crear el convex hull con todos los puntos
            puntos = df_positions[['x', 'y']].values
            hull = ConvexHull(puntos)
            
            # Colorear el área del hull en blanco con mayor opacidad
            hull_points = puntos[hull.vertices]
            ax.fill(
                hull_points[:, 0], 
                hull_points[:, 1], 
                color='white',
                alpha=0.5,  # Mayor visibilidad
                zorder=1  # Para que esté por debajo de los puntos
            )

            # Dibujar las líneas del hull con mayor grosor
            for simplex in hull.simplices:
                ax.plot(
                    puntos[simplex, 0], 
                    puntos[simplex, 1], 
                    color='white',
                    linestyle='-',
                    linewidth=3,
                    alpha=0.9,
                    zorder=3  # Para que estén por encima de todo
                )
            
            # Ordenar los puntos por su coordenada y
            puntos_ordenados = df_positions.sort_values('y')

            # Calcular la media de los 5 puntos más altos
            top_5_mean = puntos_ordenados['y'].tail(5).mean()

            # Calcular la media de los 5 puntos más bajos
            bottom_5_mean = puntos_ordenados['y'].head(5).mean()

            # Obtener los límites del eje x del campo
            xlim = ax.get_xlim()
            
            # Dibujar líneas horizontales en los valores medios de los 5 puntos más altos y más bajos
            ax.axhline(y=top_5_mean, color='white', linestyle='-', linewidth=2, alpha=0.8, zorder=4)
            ax.axhline(y=bottom_5_mean, color='white', linestyle='-', linewidth=2, alpha=0.8, zorder=4)
            
            # Añadir texto para indicar la altura del bloque
            ax.text(
                xlim[0] + 5, 
                top_5_mean + 3, 
                f"Pos medio más alto: {top_5_mean:.1f}", 
                color='white', 
                fontsize=10, 
                ha='left', 
                va='bottom'
            )
            
            ax.text(
                xlim[0] + 5, 
                bottom_5_mean - 3, 
                f"Pos medio más bajo: {bottom_5_mean:.1f}", 
                color='white', 
                fontsize=10, 
                ha='left', 
                va='top'
            )
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        # Si hay un error, creamos un gráfico con el mensaje de error
        fig, ax = plt.subplots(figsize=(5, 7), facecolor=BACKGROUND_COLOR)
        ax.text(
            0.5, 0.5, 
            f"Error al crear el mapa de acciones defensivas:\n{str(e)}",
            ha='center', 
            va='center', 
            color='red', 
            fontsize=10
        )
        print(f"Error detallado: {e}")  # Para depuración
        ax.set_facecolor(BACKGROUND_COLOR)
        return fig
    
# Función para crear mapa de red de pases entre demarcaciones
def create_pass_network(df_combined, equipo, temporada):
    """
    Versión optimizada del mapa de red de pases entre demarcaciones.
    - Usa abreviaturas en español para las demarcaciones
    - Permite visualizar pases entre jugadores de la misma demarcación
    - Unifica laterales y carrileros
    
    Args:
        df_combined: DataFrame con los datos combinados
        equipo: Nombre del equipo a analizar
        temporada: Temporada a analizar
        
    Returns:
        Figura de matplotlib con el mapa de red de pases
    """
    try:
        # Paso 1: Filtramos solo los datos necesarios de una vez (columnas específicas y filas de pases)
        df_passes = df_combined[
            (df_combined['equipo'] == equipo) & 
            (df_combined['temporada'] == temporada) &
            (df_combined['tipo_evento'] == 'Pase')
        ][['demarcacion', 'xstart', 'ystart', 'xend', 'yend']].copy()
        
        # Paso 2: Aplicar una muestra si hay demasiados datos
        if len(df_passes) > 500:  # Reducido de 1000 a 500
            df_passes = df_passes.sample(n=500, random_state=42)
        
        # Si no hay suficientes pases, mostrar mensaje
        if len(df_passes) < 10:
            fig, ax = plt.subplots(figsize=(5, 7), facecolor=BACKGROUND_COLOR)
            ax.text(0.5, 0.5, f"No hay suficientes pases para crear la red\npara {equipo} en {temporada}",
                   ha='center', va='center', color='white', fontsize=12)
            ax.set_facecolor(BACKGROUND_COLOR)
            return fig
        
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
            alpha = 0.3 + (count / max(connection_counts.values()) * 0.6)
            
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
                    color=HIGHLIGHT_COLOR,
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
                    color=HIGHLIGHT_COLOR,
                    alpha=alpha,
                    zorder=1,
                    ax=ax
                )
        
        # Paso 14: Dibujar nodos con tamaños simplificados
        for _, row in node_df.iterrows():
            # Obtenemos la abreviatura en español
            base_demarcation = row.demarcacion
            abreviatura = abbr_mapping_es.get(base_demarcation, base_demarcation[:2])
            
            # Color azul para los círculos con borde blanco
            circle = plt.Circle(
                (row.xstart, row.ystart),
                row.size,  # Tamaño más razonable
                color='#0066CC',  # Azul
                alpha=0.8,
                zorder=2,
                edgecolor='white',  # Borde blanco
                linewidth=1.5  # Grosor del borde
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
    
def capturar_visualizaciones_equipo(fig, titulo=None, dpi=150):
    """
    Convierte una figura de matplotlib en un objeto Image compatible con ReportLab,
    optimizado para visualizaciones de equipos. Ahora incluye el título de la gráfica.
    
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
        max_width = 2.5 * inch  # Más pequeño para permitir 3 en una fila
        max_height = 3 * inch
        
        # Calcular dimensiones manteniendo la relación de aspecto
        if aspect_ratio > 1.75:  # Figuras muy anchas
            width = max_width
            height = width / aspect_ratio
        else:
            height = max_height
            width = min(max_width, height * aspect_ratio)
    
    # Crear la imagen ReportLab con dimensiones ajustadas
    return ReportLabImage(buf, width=width, height=height), titulo

def generar_pdf_equipo(equipo, temporada, figuras_con_titulos):
    """
    Genera un PDF con las visualizaciones del equipo.
    Todo el contenido debe caber en una sola página con formato horizontal.
    Fondo completamente negro sin márgenes blancos y texto en color blanco.
    Incluye el escudo del Alavés en una esquina del campograma.
    
    Args:
        equipo: Nombre del equipo
        temporada: Temporada seleccionada
        figuras_con_titulos: Lista de tuplas (figura, titulo) o lista de figuras
        
    Returns:
        Buffer del PDF generado
    """
    # Preparar el PDF (formato horizontal)
    buffer = io.BytesIO()
    
    # Sin márgenes, absolutamente cero
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=0,
        leftMargin=0,
        topMargin=0,
        bottomMargin=0
    )
    
    # Obtener estilos
    styles = getSampleStyleSheet()
    
    # Crear estilos personalizados - más compactos para maximizar espacio
    styles.add(ParagraphStyle(
        name='EquipoInfo',
        parent=styles['Heading1'],
        fontSize=14,  # Reducido para ocupar menos espacio
        textColor=colors.white,
        backColor=colors.black,
        alignment=TA_LEFT,
        spaceAfter=6,  # Reducido para ocupar menos espacio
        spaceBefore=6  # Reducido para ocupar menos espacio
    ))
    
    # Estilo para títulos de gráficas
    styles.add(ParagraphStyle(
        name='TituloGrafica',
        parent=styles['Normal'],
        fontSize=9,  # Tamaño pequeño para no ocupar mucho espacio
        textColor=colors.white,
        backColor=colors.black,
        alignment=TA_CENTER,
        spaceAfter=2,
        spaceBefore=2
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
            backColor=colors.black,
        )
    )
    elementos.append(black_background)
    
    # Cargar el escudo del Alavés
    logo_path = os.path.join(BASE_DIR, 'assets', 'escudo_alaves_original.png')
    escudo_img = None
    try:
        # Abrir y optimizar el escudo
        escudo_pil = Image.open(logo_path)
        escudo_pil.thumbnail((70, 70), Image.LANCZOS)  # Tamaño pequeño para la esquina
        
        # Guardar en un buffer temporal
        escudo_buffer = io.BytesIO()
        if escudo_pil.mode in ("RGBA", "P"):
            escudo_pil = escudo_pil.convert("RGB")
        escudo_pil.save(escudo_buffer, format="JPEG", quality=85)
        escudo_buffer.seek(0)
        
        # Crear imagen ReportLab
        escudo_img = ReportLabImage(escudo_buffer, width=0.7*inch, height=0.7*inch)
    except Exception as e:
        print(f"Error al cargar el escudo: {e}")
    
    # Obtener KPIs del equipo - formato compacto
    try:
        eventos_acumulados = DataManagerTeams.get_accumulated_events_data()
        if eventos_acumulados is not None:
            # Filtrar por equipo y temporada
            filtered_data = eventos_acumulados[
                (eventos_acumulados['equipo'] == equipo) &
                (eventos_acumulados['temporada'] == str(temporada))
            ]
            
            if not filtered_data.empty:
                # Obtener KPIs
                kpi_column = 'KPI_Rendimiento' if 'KPI_Rendimiento' in filtered_data.columns else 'KPI_rendimiento'
                kpi_mean = filtered_data[kpi_column].mean() if kpi_column in filtered_data.columns else 0.0
                valoracion_mean = filtered_data['valoracion'].mean() if 'valoracion' in filtered_data.columns else 0.0
                
                # Calcular estadísticas adicionales
                partidos = filtered_data['match_id'].nunique()
                
                # Goles a favor
                goles_favor = 0
                for col_name in ['goles', 'goles_favor', 'goles a favor']:
                    if col_name in filtered_data.columns:
                        goles_favor = filtered_data[col_name].sum()
                        break
                
                # Goles en contra
                goles_contra = 0
                for col_name in ['goles_concedidos', 'goles_contra', 'goles en contra']:
                    if col_name in filtered_data.columns:
                        goles_contra = filtered_data[col_name].sum()
                        break
                
                # Diferencia de goles
                diferencia = goles_favor - goles_contra
                
                # Crear titulo con información del equipo (más compacto, en dos líneas)
                header_text = f"""
                <font color="white" size="14"><b>{equipo}</b></font> - 
                <font color="#4BB3FD" size="12"><b>Temporada: {temporada}</b></font> - 
                <font color="white" size="10">KPI: <b>{kpi_mean:.2f}</b> | Valoración: <b>{valoracion_mean:.2f}</b></font>
                <br/>
                <font color="white" size="9">Partidos: <b>{partidos}</b> | Goles a favor: <b>{goles_favor}</b> | Goles en contra: <b>{goles_contra}</b> | Diferencia: <b>{diferencia}</b></font>
                """
            else:
                header_text = f"""
                <font color="white" size="14"><b>{equipo}</b></font> - 
                <font color="#4BB3FD" size="12"><b>Temporada: {temporada}</b></font>
                """
        else:
            header_text = f"""
            <font color="white" size="14"><b>{equipo}</b></font> - 
            <font color="#4BB3FD" size="12"><b>Temporada: {temporada}</b></font>
            """

    except Exception as e:
        print(f"Error al obtener datos para encabezado: {e}")
        header_text = f"""
        <font color="white" size="14"><b>{equipo}</b></font> - 
        <font color="#4BB3FD" size="12"><b>Temporada: {temporada}</b></font>
        """
    
    # Añadir encabezado (más compacto)
    elementos.append(Paragraph(header_text, styles['EquipoInfo']))
    
    # Crear tabla de visualizaciones
    if len(figuras_con_titulos) >= 7:  # Campograma + 6 figuras
        # Verificar si figuras_con_titulos contiene tuplas o solo figuras
        if isinstance(figuras_con_titulos[0], tuple):
            # Si son tuplas (figura, titulo)
            figuras = [fig for fig, _ in figuras_con_titulos]
            titulos = [titulo for _, titulo in figuras_con_titulos]
        else:
            # Si son solo figuras, usar títulos predeterminados
            figuras = figuras_con_titulos
            titulos = [
                "Alineaciones más utilizadas",
                "Evolución de xG",
                "Evolución de KPI - Temporada",
                "KPI - Equipo vs Rivales",
                "Mapa de Calor y Flujo de Pases",
                "Mapa de Acciones Defensivas",
                "Red de Pases entre Demarcaciones"
            ]
        
        # Extraer campograma y convertirlo para ReportLab con tamaño mucho mayor
        campograma_fig = figuras[0]
        buf = io.BytesIO()
        campograma_fig.savefig(buf, format='png', dpi=200, bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)
        campograma = ReportLabImage(buf, width=4.5*inch, height=7.0*inch)  # Aumentado el ancho a 4.5 pulgadas
        
        # Título del campograma
        titulo_campograma = Paragraph(titulos[0], styles['TituloGrafica'])
        
        # Crear un elemento de tabla para el campograma que incluya el escudo en la esquina
        if escudo_img:
            # Crear tabla para campograma con escudo en la esquina superior derecha
            campograma_con_escudo = Table(
                [[campograma, escudo_img]],
                colWidths=[4.0*inch, 0.15*inch],  # Ajustar el ancho para el escudo
                rowHeights=[6.9*inch]  # Altura para el campograma
            )
            
            # Estilo para la tabla del campograma
            campograma_con_escudo.setStyle(TableStyle([
                ('VALIGN', (0, 0), (0, 0), 'MIDDLE'),  # Alinear campograma al centro vertical
                ('ALIGN', (0, 0), (0, 0), 'CENTER'),   # Alinear campograma al centro horizontal
                ('VALIGN', (1, 0), (1, 0), 'TOP'),     # Escudo arriba
                ('ALIGN', (1, 0), (1, 0), 'RIGHT'),    # Escudo a la derecha
                ('LEFTPADDING', (0, 0), (1, 0), 0),    # Sin padding
                ('RIGHTPADDING', (0, 0), (1, 0), 0),   # Sin padding
                ('TOPPADDING', (0, 0), (1, 0), 0),     # Sin padding
                ('BOTTOMPADDING', (0, 0), (1, 0), 0),  # Sin padding
                ('BACKGROUND', (0, 0), (1, 0), colors.black),  # Fondo negro
            ]))
        else:
            # Si no se pudo cargar el escudo, usar solo el campograma
            campograma_con_escudo = campograma

        # Convertir el resto de figuras para ReportLab con tamaño más pequeño
        images = []
        image_titles = []
        for i in range(1, 4):  # Primeras 3 gráficas (fila superior)
            fig = figuras[i]
            titulo = titulos[i]
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0.05)
            buf.seek(0)
            img = ReportLabImage(buf, width=1.9*inch, height=2.2*inch)  # Reducido ligeramente el ancho
            images.append(img)
            image_titles.append(Paragraph(titulo, styles['TituloGrafica']))

        # Fila inferior con gráficas más altas
        for i in range(4, 7):  # Últimas 3 gráficas (fila inferior)
            fig = figuras[i]
            titulo = titulos[i]
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0.05)
            buf.seek(0)
            img = ReportLabImage(buf, width=1.9*inch, height=3.3*inch)  # Aumentado la altura a 3.3 pulgadas
            images.append(img)
            image_titles.append(Paragraph(titulo, styles['TituloGrafica']))

        # Organizar las figuras y títulos en la estructura solicitada:
        data = [
            [[titulo_campograma, campograma_con_escudo], [image_titles[0], images[0]], [image_titles[1], images[1]], [image_titles[2], images[2]]],
            ['', [image_titles[3], images[3]], [image_titles[4], images[4]], [image_titles[5], images[5]]]
        ]

        # Definir anchos de columna (primera columna más ancha para el campograma)
        col_widths = [4.6*inch, 2.2*inch, 2.2*inch, 2.2*inch]  # Ajustado los anchos de columna

        # Crear tabla con dimensiones calculadas
        table = Table(
            data,
            colWidths=col_widths,
            rowHeights=[2.4*inch, 5.0*inch]  # Ajustado para dar más altura a la fila inferior
        )
        
        # Estilo de tabla sin espaciado y con fondo negro
        table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (0, 0), (0, 1), 'CENTER'),  # Alineación especial para campograma
            ('SPAN', (0, 0), (0, 1)),  # Fusionar celdas para el campograma
            ('LEFTPADDING', (0, 0), (-1, -1), 1),  # Mínimo padding
            ('RIGHTPADDING', (0, 0), (-1, -1), 1), # Mínimo padding
            ('TOPPADDING', (5, 0), (5, 0), 0),   # Mínimo padding
            ('BOTTOMPADDING', (0, 0), (-1, 0), 0),# Mínimo padding
            ('BACKGROUND', (0, 0), (-1, -1), colors.black),
        ]))
        
        elementos.append(table)
    
    # Función para fondo negro que ocupa toda la página sin márgenes
    def black_canvas(canvas, doc):
        canvas.setFillColor(colors.black)
        canvas.rect(0, 0, doc.width, doc.height, fill=1)
        
    # Construir el PDF
    doc.build(elementos, onFirstPage=black_canvas, onLaterPages=black_canvas)
    buffer.seek(0)
    return buffer
    
# Interfaz principal de Streamlit
def main():
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
            <h1 style='color: white; text-align: center;'>Equipos Academia</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
        # Menú de navegación
    with st.container():
        menu.generarMenu(st.session_state['usuario'])

    # Cargar datos básicos para filtros
    df_filtros = DataManagerTeams.get_filter_data()

    if df_filtros is not None:
        # Filtrar solo equipos que contengan 'Alav' para el selectbox
        equipos_alaves = [equipo for equipo in df_filtros['equipo'].unique() 
                        if 'Alav' in equipo]
        
        col1, col2 = st.columns(2)

        with col1:
            equipo = st.selectbox('Equipo:', sorted(equipos_alaves))

        with col2:
            # Filtrar las temporadas disponibles para el equipo seleccionado
            temporadas = df_filtros[df_filtros['equipo'] == equipo]['temporada'].unique()
            temporada = st.selectbox('Temporada:', temporadas)

        st.markdown("---")
        
        # Botones para generar visualización y exportar
        col_buttons = st.columns([1, 1, 4])
                
        with col_buttons[0]:
            generar_viz = st.button('Generar Visualización', type='primary')

        with col_buttons[1]:
            exportar_pdf = st.button('Exportar a PDF', type='secondary')
        
        # Inicializar variables de estado si no existen
        if 'figuras_generadas' not in st.session_state:
            st.session_state.figuras_generadas = None
        if 'visualizacion_generada' not in st.session_state:
            st.session_state.visualizacion_generada = False
        
        ## En _pagina2.py - Modificar la estructura para cargar gráficas progresivamente
        if generar_viz:
            # Cargar datos una sola vez
            df_combined = DataManagerTeams.get_filtered_data_once(equipo, temporada)
                            
            # Limpiar memoria antes de comenzar
            clear_memory()
            
            # Crear overlay para la pantalla de carga
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

            try:
                # Cargar datos detallados
                df_combined = DataManagerTeams.get_detailed_data(equipo, temporada) 

                if df_combined is not None:
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
                    
                    # NUEVO: Agregar el encabezado del equipo con KPIs
                    create_team_header(equipo, temporada)
                    
                    # Crear un diseño de columnas personalizado
                    col1, col2 = st.columns([1, 2])  # Primera columna más estrecha para el lineup
                    
                    with col1:
                        # Campograma vertical que ocupa la primera columna y dos filas
                        st.markdown("<div class='chart-container full-width'>", unsafe_allow_html=True)
                        st.markdown("<div class='plot-title'>Alineaciones más utilizadas</div>", unsafe_allow_html=True)
                        
                        # Obtener datos de alineaciones y filtrar por los match_ids del equipo y temporada seleccionados
                        match_ids = DataManagerTeams.get_match_ids(equipo, temporada)
                        df_lineup = DataManagerTeams.get_lineup_data()
                        
                        if df_lineup is not None:
                            # Filtrar por los match_ids específicos
                            df_lineup = df_lineup[df_lineup['match_id'].isin(match_ids)]
                            
                            # Crear la figura y el campo con dimensiones verticales
                            fig_campo, ax_campo = plt.subplots(figsize=(6, 8))  # Ajustado para formato vertical
                            pitch = VerticalPitch(pitch_type='custom', 
                                            pitch_length=120, 
                                            pitch_width=68,
                                            pitch_color='#1E1E1E', 
                                            line_color='white', 
                                            stripe=False)
                            
                            create_team_lineup_heatmap(ax_campo, df_lineup, equipo, pitch)
                            
                            plt.tight_layout()
                            st.pyplot(fig_campo)
                        else:
                            st.error("No se pudieron cargar los datos de alineaciones")
                            # Crear figura vacía para PDF
                            fig_campo, ax_campo = plt.subplots(figsize=(6, 8))
                            ax_campo.text(0.5, 0.5, "No hay datos de alineaciones", ha='center', va='center')
                        
                        st.markdown("</div>", unsafe_allow_html=True)

                    with col2:
                        # Crear un grid de 2 filas y 3 columnas para las gráficas 1-6
                        rows = [st.columns(3), st.columns(3)]
                        
                        # Gráfica 1: Evolución de xG
                        with rows[0][0]:
                            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                            st.markdown("<div class='plot-title'>Evolución de xG</div>", unsafe_allow_html=True)
                            fig1 = create_xg_evolution_chart(df_combined, equipo, temporada)
                            st.pyplot(fig1)
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Gráfica 2: Evolución de KPI
                        with rows[0][1]:
                            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                            st.markdown("<div class='plot-title'>Evolución de KPI - Temporada</div>", unsafe_allow_html=True)
                            fig2 = create_kpi_evolution_chart(df_combined, equipo, temporada)
                            st.pyplot(fig2)
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Gráfica 3: KPI - Equipo vs Rivales
                        with rows[0][2]:
                            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                            st.markdown("<div class='plot-title'>KPI - Equipo vs Rivales</div>", unsafe_allow_html=True)
                            fig3, ax3 = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
                            create_team_vs_all_rivals_pizza_chart(ax3, df_combined, equipo, temporada)
                            fig3.tight_layout(pad=2.0)
                            st.pyplot(fig3)
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Gráfica 4: Mapa de calor y flujo de pases
                        with rows[1][0]:
                            st.markdown("<div class='chart-container full-width'>", unsafe_allow_html=True)
                            st.markdown("<div class='plot-title'>Mapa de Calor y Flujo de Pases</div>", unsafe_allow_html=True)
                            fig4 = create_passes_heatmap(df_combined, equipo, temporada)
                            st.pyplot(fig4)
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Gráfica 5: Acciones Defensivas (Convex Hull)
                        with rows[1][1]:
                            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                            st.markdown("<div class='plot-title'>Mapa de Acciones Defensivas</div>", unsafe_allow_html=True)
                            fig5 = create_defensive_actions_hull(df_combined, equipo, temporada)
                            st.pyplot(fig5)
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Gráfica 6: Red de Pases
                        with rows[1][2]:
                            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                            st.markdown("<div class='plot-title'>Red de Pases entre Demarcaciones</div>", unsafe_allow_html=True)
                            fig6 = create_pass_network(df_combined, equipo, temporada)
                            st.pyplot(fig6)
                            st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Definir los títulos de las figuras
                    titulos_figuras = [
                        "Alineaciones más utilizadas",
                        "Evolución de xG",
                        "Evolución de KPI - Temporada",
                        "KPI - Equipo vs Rivales",
                        "Mapa de Calor y Flujo de Pases",
                        "Mapa de Acciones Defensivas",
                        "Red de Pases entre Demarcaciones"
                    ]

                    # Guardar las figuras generadas para exportación
                    st.session_state.figuras_generadas = [fig_campo, fig1, fig2, fig3, fig4, fig5, fig6]
                    st.session_state.titulos_figuras = titulos_figuras
                    st.session_state.visualizacion_generada = True
                    
                    # Limpiar memoria después de todas las visualizaciones pero guardar figuras
                    # No cerramos las figuras aquí para mantenerlas para el PDF
                    
                    # Remover la pantalla de carga cuando todo esté listo
                    overlay_container.empty()
                
                else:
                    st.error("No se pudieron cargar los datos detallados")
                    overlay_container.empty()
                
            except Exception as e:
                st.error(f"Error al generar la visualización: {e}")
                overlay_container.empty()
                # Asegurar la limpieza de memoria incluso cuando hay errores
                clear_memory()
        
        # Lógica para exportar a PDF
        if exportar_pdf:
            if 'visualizacion_generada' in st.session_state and st.session_state.visualizacion_generada:
                with st.spinner('Generando PDF...'):
                    try:
                        # Verificar si existen tanto figuras como títulos en session_state
                        if 'figuras_generadas' in st.session_state and 'titulos_figuras' in st.session_state:
                            # Crear lista de tuplas (figura, título)
                            figuras_con_titulos = list(zip(
                                st.session_state.figuras_generadas,
                                st.session_state.titulos_figuras
                            ))
                            
                            # Generar el PDF con figuras y títulos
                            pdf_buffer = generar_pdf_equipo(
                                equipo, 
                                temporada,
                                figuras_con_titulos
                            )
                        else:
                            # Usar solo las figuras si no hay títulos
                            pdf_buffer = generar_pdf_equipo(
                                equipo, 
                                temporada,
                                st.session_state.figuras_generadas
                            )
                        
                        # Ofrecer la descarga del PDF
                        nombre_archivo = f"Informe_{equipo.replace(' ', '_')}_{temporada}.pdf"
                        
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
            else:
                st.warning("Primero debes generar la visualización.")

# Agregar estilos CSS necesarios
def add_custom_styles():
    st.markdown("""
        <style>
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
        
        /* Estilos para el encabezado del equipo */
        .team-header {
            display: flex;
            flex-direction: row;
            justify-content: space-between;
            align-items: center;
            background-color: #1E1E1E;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        }

        .team-info {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
        }

        .team-name {
            font-size: 2.5rem;
            font-weight: bold;
            color: #ffffff;
            margin-bottom: 0.3rem;
        }

        .team-season {
            font-size: 1.7rem;
            font-weight: bold;
            color: #4BB3FD;
        }

        .kpi-boxes {
            display: flex;
            flex-direction: row;
            gap: 1.5rem;
        }

        .kpi-box {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            border-radius: 8px;
            padding: 1rem;
            min-width: 180px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s ease;
        }

        .kpi-box:hover {
            transform: translateY(-5px);
        }

        .kpi-label {
            font-size: 1rem;
            font-weight: bold;
            color: rgba(255, 255, 255, 0.9);
            text-align: center;
            margin-bottom: 0.5rem;
        }

        .kpi-value {
            font-size: 2.5rem;
            font-weight: bold;
            color: white;
            text-align: center;
            margin-bottom: 0.2rem;
        }
        
        .kpi-category {
            font-size: 1rem;
            font-weight: bold;
            color: white;
            text-align: center;
        }

        /* Ajustes responsive */
        @media (max-width: 768px) {
            .team-header {
                flex-direction: column;
                gap: 1rem;
            }
            
            .team-info {
                align-items: center;
            }
            
            .kpi-boxes {
                width: 100%;
                justify-content: center;
            }
            
            .kpi-box {
                min-width: 120px;
            }
            
            .team-name {
                font-size: 2rem;
            }
            
            .team-season {
                font-size: 1.3rem;
            }
            
            .kpi-value {
                font-size: 1.8rem;
            }
            
            .kpi-category {
                font-size: 0.9rem;
            }
        }
        </style>
    """, unsafe_allow_html=True)

# Ejecutar la aplicación principal
if __name__ == "__main__":
    add_custom_styles()
    main()