# Configuración de la página - DEBE IR PRIMERO
import streamlit as st
import common.login as login
import base64
import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from pathlib import Path
from PIL import Image 
import io, base64

# Configuración de la página
st.set_page_config(
    page_title="Academia Deportivo Alavés", 
    page_icon="⚽", 
    layout="wide", 
    initial_sidebar_state="collapsed",
    menu_items=None
)

# Configuración de rutas
ASSETS_PATH = os.path.abspath(os.path.join('assets', 'fondo_login.png'))
banner_path = os.path.abspath(os.path.join('assets', 'bunner_alaves_home.png'))

# Función para cargar imagen con manejo de errores
@st.cache_data
def get_image_base64(path):
    try:
        with open(path, "rb") as f:
            data = f.read()
            return base64.b64encode(data).decode()
    except Exception as e:
        st.error(f"Error cargando imagen: {e}")
        return None

@st.cache_data 
def get_image_base64_optimized(path, max_size=(1024, 1024), quality=70): 
    try: 
        image = Image.open(path) 
        image.thumbnail(max_size, Image.LANCZOS) 
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

banner_base64 = get_image_base64_optimized(banner_path)

# Custom banner mejorado con animación de JS
st.markdown(
    f"""
    <div id="banner" style='
        padding: 2rem; 
        border-radius: 5px; 
        margin-bottom: 2rem; 
        background-image: url(data:image/png;base64,{get_image_base64_optimized(banner_path)}); 
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        height: 150px;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        overflow: hidden;
        transition: all 0.5s ease;
    '>
        <div id="banner-overlay" style="
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, rgba(0,0,0,0.7) 0%, rgba(0,0,0,0) 100%);
            opacity: 0.6;
            transition: opacity 0.5s ease;
        "></div>
    </div>
    """,
    unsafe_allow_html=True
)

# Aplicar JavaScript para animaciones (CORRECTO)
st.markdown(
    """
    <script>
        // Animación del banner al cargar la página
        document.addEventListener('DOMContentLoaded', function() {
            const banner = document.getElementById('banner');
            const overlay = document.getElementById('banner-overlay');
            
            // Efecto de fade-in al cargar
            banner.style.opacity = '0';
            setTimeout(() => {
                banner.style.opacity = '1';
            }, 300);
            
            // Efecto hover
            banner.addEventListener('mouseenter', function() {
                overlay.style.opacity = '0.3';
                banner.style.transform = 'scale(1.02)';
            });
            
            banner.addEventListener('mouseleave', function() {
                overlay.style.opacity = '0.6';
                banner.style.transform = 'scale(1)';
            });
        });
    </script>
    """,
    unsafe_allow_html=True
)

# Módulo de carga de datos
class DataLoader:
    @staticmethod
    @st.cache_data
    def cargar_datos(ruta='data/archivos_parquet/eventos_datos_acumulados.parquet'):
        """
        Carga los datos desde un archivo Parquet con manejo de errores.
        """
        try:
            df = pd.read_parquet(ruta)
            return df
        except Exception as e:
            st.error(f"Error al cargar los datos: {e}")
            return None

# Módulo de análisis de datos
class DataAnalyzer:
    @staticmethod
    def obtener_ranking_por_liga(
        df, 
        liga_seleccionada='Todas', 
        demarcacion_seleccionada='Todas', 
        equipo_seleccionado='Todos'
    ):
        """
        Obtiene el ranking de jugadores basado en filtros.
        """
        # Aplicar filtros
        if liga_seleccionada != 'Todas':
            df = df[df['liga'] == liga_seleccionada]
        
        if demarcacion_seleccionada != 'Todas':
            df = df[df['demarcacion'] == demarcacion_seleccionada]
        
        if equipo_seleccionado != 'Todos':
            df = df[df['equipo'] == equipo_seleccionado]
        
        # Agrupar por liga, demarcación, equipo y jugador
        ranking_por_liga = df.groupby(['liga', 'demarcacion', 'equipo', 'jugador']).agg({
            'KPI_rendimiento': 'mean',
            'KPI_construccion_ataque': 'mean',
            'KPI_progresion': 'mean',
            'KPI_habilidad_individual': 'mean',
            'KPI_peligro_generado': 'mean',
            'KPI_finalizacion': 'mean',
            'KPI_eficacia_defensiva': 'mean',
            'KPI_juego_aereo': 'mean',
            'KPI_capacidad_recuperacion': 'mean',
            'KPI_posicionamiento_tactico': 'mean',
            'xg': 'mean',  # Añadimos xg (en minúscula)
            'POP': 'mean',  # Añadimos POP (en mayúscula)
            'match_id': 'count'  # Contar número de partidos distintos
        }).reset_index()
        
        # Renombrar columnas para mayor claridad
        ranking_por_liga.columns = [
            'Liga', 'Demarcación', 'Equipo', 'Jugador', 
            'KPI Rendimiento', 'KPI Construcción Ataque', 'KPI Progresión', 
            'KPI Habilidad Individual', 'KPI Peligro Generado', 'KPI Finalización', 
            'KPI Eficacia Defensiva', 'KPI Juego Aéreo', 'KPI Capacidad Recuperación', 
            'KPI Posicionamiento Táctico', 'xg', 'POP', 'Número de Partidos'
        ]
        
        return ranking_por_liga

# Módulo de visualización mejorado
class DataVisualizer:
    @staticmethod
    def crear_visualizacion(df_ranking, eje_x, eje_y):
        """
        Crea una visualización de dispersión con los ejes seleccionados.
        """
        # Calcular tamaños de marcadores basados en KPI Rendimiento, manejando valores nulos
        df_ranking_clean = df_ranking.copy()
        df_ranking_clean['KPI Rendimiento'] = df_ranking_clean['KPI Rendimiento'].fillna(df_ranking_clean['KPI Rendimiento'].mean())
        
        min_kpi = df_ranking_clean['KPI Rendimiento'].min()
        max_kpi = df_ranking_clean['KPI Rendimiento'].max()
        
        # Evitar división por cero si min_kpi equals max_kpi
        if min_kpi == max_kpi:
            marker_sizes = [20] * len(df_ranking_clean)
        else:
            marker_sizes = ((df_ranking_clean['KPI Rendimiento'] - min_kpi) / (max_kpi - min_kpi) * 30 + 10).tolist()

        # Crear colores basados en el equipo - Mejorado para usar colores del Alavés
        marker_colors = []
        for equipo in df_ranking_clean['Equipo']:
            if 'Alav' in str(equipo):
                marker_colors.append('#0067B2')  # Azul Alavés
            else:
                marker_colors.append('rgba(200, 200, 200, 0.7)')  # Gris semi-transparente para otros equipos

        # Crear texto para hover - Mejorado
        hover_text = df_ranking_clean.apply(
            lambda row: f"<b>{row['Jugador']}</b><br>" +
                    f"<i>{row['Demarcación']}</i> - {row['Equipo']}<br>" +
                    f"<b>KPI Rendimiento:</b> {row['KPI Rendimiento']:.2f}<br>" + 
                    f"<b>{eje_x}:</b> {row[eje_x]:.2f}<br>" +
                    f"<b>{eje_y}:</b> {row[eje_y]:.2f}<br>" +
                    f"Partidos: {row['Número de Partidos']}",
            axis=1
        )

        # Crear figura de dispersión con mejoras visuales
        fig = go.Figure()

        # Añadir líneas de cuadrantes (medias de los ejes)
        x_mean = df_ranking_clean[eje_x].mean()
        y_mean = df_ranking_clean[eje_y].mean()
        
        # Valores mínimos y máximos para los ejes
        x_min, x_max = df_ranking_clean[eje_x].min() * 0.95, df_ranking_clean[eje_x].max() * 1.05
        y_min, y_max = df_ranking_clean[eje_y].min() * 0.95, df_ranking_clean[eje_y].max() * 1.05
        
        # Añadir líneas de referencia
        fig.add_shape(
            type="line", line=dict(dash="dash", color="rgba(255, 255, 255, 0.5)"),
            x0=x_min, x1=x_max, y0=y_mean, y1=y_mean
        )
        fig.add_shape(
            type="line", line=dict(dash="dash", color="rgba(255, 255, 255, 0.5)"),
            x0=x_mean, x1=x_mean, y0=y_min, y1=y_max
        )

        # Añadir puntos de dispersión
        fig.add_trace(go.Scatter(
            x=df_ranking_clean[eje_x],
            y=df_ranking_clean[eje_y],
            mode='markers',
            marker=dict(
                size=marker_sizes,
                color=marker_colors,
                line=dict(
                    width=1.5,
                    color=['rgba(0, 0, 0, 0.8)' if 'Alav' in str(equipo) else 'rgba(30, 30, 30, 0.6)' for equipo in df_ranking_clean['Equipo']]
                ),
                symbol=['star' if 'Alav' in str(equipo) else 'circle' for equipo in df_ranking_clean['Equipo']]
            ),
            text=hover_text,
            hoverinfo='text',
            hoverlabel=dict(
                bgcolor="rgba(35, 35, 35, 0.95)",
                font_size=14,
                font_family="Arial",
                bordercolor="#0067B2"
            )
        ))

        # Añadir etiquetas para los jugadores destacados (top 5 en rendimiento)
        top_jugadores = df_ranking_clean.nlargest(5, 'KPI Rendimiento')
        for _, jugador in top_jugadores.iterrows():
            fig.add_annotation(
                x=jugador[eje_x],
                y=jugador[eje_y],
                text=jugador['Jugador'],
                showarrow=True,
                arrowhead=3,
                arrowsize=1,
                arrowwidth=1.5,
                arrowcolor="#FFFFFF",
                font=dict(size=10, color="#FFFFFF"),
                bgcolor="#0067B2",
                bordercolor="#FFFFFF",
                borderwidth=1,
                borderpad=4,
                opacity=0.8
            )

        # Layout del gráfico mejorado
        fig.update_layout(
            title={
                'text': f'<b>Análisis de jugadores: {eje_x} vs {eje_y}</b>',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=24, color='white', family="Arial Black")
            },
            xaxis=dict(
                title=dict(text=eje_x, font=dict(size=16, color='white')),
                gridcolor='rgba(255, 255, 255, 0.1)',
                zerolinecolor='rgba(255, 255, 255, 0.2)'
            ),
            yaxis=dict(
                title=dict(text=eje_y, font=dict(size=16, color='white')),
                gridcolor='rgba(255, 255, 255, 0.1)',
                zerolinecolor='rgba(255, 255, 255, 0.2)'
            ),
            height=700,
            width=1200,
            hovermode='closest',
            showlegend=False,
            plot_bgcolor='rgba(25, 25, 35, 0.95)',
            paper_bgcolor='rgba(25, 25, 35, 0.95)',
            font=dict(color='white'),
            margin=dict(l=50, r=50, t=80, b=50),
            shapes=[
                # Añadir líneas de cuadrantes con diferentes colores para identificar zonas
                dict(
                    type="rect",
                    xref="x", yref="y",
                    x0=x_mean, y0=y_mean,
                    x1=x_max, y1=y_max,
                    fillcolor="rgba(0, 153, 0, 0.1)",
                    line_width=0,
                    layer="below"
                ),
                dict(
                    type="rect",
                    xref="x", yref="y",
                    x0=x_min, y0=y_mean,
                    x1=x_mean, y1=y_max,
                    fillcolor="rgba(255, 153, 0, 0.1)",
                    line_width=0,
                    layer="below"
                ),
                dict(
                    type="rect",
                    xref="x", yref="y",
                    x0=x_min, y0=y_min,
                    x1=x_mean, y1=y_mean,
                    fillcolor="rgba(255, 0, 0, 0.1)",
                    line_width=0,
                    layer="below"
                ),
                dict(
                    type="rect",
                    xref="x", yref="y",
                    x0=x_mean, y0=y_min,
                    x1=x_max, y1=y_mean,
                    fillcolor="rgba(0, 0, 255, 0.1)",
                    line_width=0,
                    layer="below"
                )
            ],
            annotations=[
                # Etiquetas para los cuadrantes
                dict(
                    x=(x_mean + x_max) / 2, y=(y_mean + y_max) / 2,
                    text="ALTO RENDIMIENTO",
                    showarrow=False,
                    font=dict(size=12, color="rgba(255, 255, 255, 0.5)")
                ),
                dict(
                    x=(x_min + x_mean) / 2, y=(y_mean + y_max) / 2,
                    text="POTENCIAL",
                    showarrow=False,
                    font=dict(size=12, color="rgba(255, 255, 255, 0.5)")
                ),
                dict(
                    x=(x_min + x_mean) / 2, y=(y_min + y_mean) / 2,
                    text="BAJO RENDIMIENTO",
                    showarrow=False,
                    font=dict(size=12, color="rgba(255, 255, 255, 0.5)")
                ),
                dict(
                    x=(x_mean + x_max) / 2, y=(y_min + y_mean) / 2,
                    text="ESPECIALIZADO",
                    showarrow=False,
                    font=dict(size=12, color="rgba(255, 255, 255, 0.5)")
                )
            ]
        )

        # Mejorar la interactividad con config
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Resaltar Alavés",
                            method="update",
                            args=[{"marker.opacity": [
                                [1 if 'Alav' in str(equipo) else 0.3 for equipo in df_ranking_clean['Equipo']]
                            ]}]
                        ),
                        dict(
                            label="Ver todos",
                            method="update",
                            args=[{"marker.opacity": [[1] * len(df_ranking_clean)]}]
                        )
                    ],
                    direction="down",
                    pad={"r": 10, "t": 10},
                    x=0.05,
                    y=1.15,
                    xanchor="left",
                    yanchor="top",
                    bgcolor="rgba(30, 60, 114, 0.8)",
                    bordercolor="#FFFFFF"
                )
            ]
        )

        return fig

# Integrar JS para mejorar interactividad
st.markdown("""
<script>
    // Función para resaltar filas en la tabla
    function highlightAlavesPlayers() {
        const dataTable = document.querySelector('.stDataFrame');
        if (!dataTable) return;
        
        const rows = dataTable.querySelectorAll('tbody tr');
        rows.forEach(row => {
            const cells = row.querySelectorAll('td');
            if (cells.length > 0) {
                const equipoCell = cells[2]; // Celda con el equipo
                if (equipoCell && equipoCell.textContent.includes('Alav')) {
                    row.style.backgroundColor = 'rgba(0, 103, 178, 0.2)';
                    row.style.fontWeight = 'bold';
                }
            }
        });
    }
    
    // Aplicar destacado después de que la página cargue
    document.addEventListener('DOMContentLoaded', function() {
        setTimeout(highlightAlavesPlayers, 1000);
        
        // También aplicar cuando cambie la tabla
        const observer = new MutationObserver(function(mutations) {
            highlightAlavesPlayers();
        });
        
        // Observar cambios en el DOM
        const targetNode = document.body;
        observer.observe(targetNode, { childList: true, subtree: true });
    });
    
    // Efecto para los botones de filtro
    function enhanceFilterButtons() {
        const buttons = document.querySelectorAll('button[kind="secondary"]');
        buttons.forEach(button => {
            button.addEventListener('mouseenter', function() {
                this.style.transform = 'scale(1.05)';
                this.style.transition = 'transform 0.3s ease';
            });
            
            button.addEventListener('mouseleave', function() {
                this.style.transform = 'scale(1)';
            });
        });
    }
    
    // Aplicar efectos a los botones
    document.addEventListener('DOMContentLoaded', function() {
        setTimeout(enhanceFilterButtons, 1000);
    });
</script>
""", unsafe_allow_html=True)

try:
    # CSS mejorado
    st.markdown(
        f""" 
        <style>
        /* Estilos generales */
        .main {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: white;
        }}
        
        .main-header {{
            color: white;
            font-size: 3em;
            text-align: center;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.6);
            margin-bottom: 0.5em;
            font-family: 'Arial Black', sans-serif;
        }}
        
        /* Estilo del login */
        .login-header {{
            background: linear-gradient(90deg, #0067B2 0%, #004C8A 100%);
            color: white;
            padding: 0.8em;
            text-align: center;
            font-size: 1.2em;
            border-radius: 15px 15px 0 0;
            margin-top: 2em;
            font-weight: bold;
            box-shadow: 0 -4px 10px rgba(0,0,0,0.2);
        }}
        .login-container {{
            background: rgba(35, 35, 45, 0.95);
            padding: 2em;
            border-radius: 0 0 15px 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
            border: 1px solid rgba(255,255,255,0.1);
            animation: fadeIn 0.5s ease-in-out;
        }}
        
        /* Animación de fade in */
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        /* Mejoras para inputs */
        .stTextInput label {{
            color: white !important;
            font-weight: bold !important;
            font-size: 0.9em !important;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .stTextInput input {{
            color: white !important;
            background-color: rgba(30, 30, 45, 0.8) !important;
            border: 1px solid rgba(255,255,255,0.15) !important;
            border-radius: 8px !important;
            padding: 10px 15px !important;
            transition: all 0.3s ease !important;
        }}
        .stTextInput input:focus {{
            border-color: #0067B2 !important;
            box-shadow: 0 0 0 2px rgba(0, 103, 178, 0.25) !important;
        }}
        
        /* Mejoras para botones */
        .stButton button {{
            background: linear-gradient(90deg, #0067B2 0%, #004C8A 100%) !important;
            color: white !important;
            font-weight: bold !important;
            border-radius: 8px !important;
            padding: 0.6em 1.2em !important;
            border: none !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2) !important;
            transition: all 0.3s ease !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
        }}
        .stButton button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 8px rgba(0,0,0,0.3) !important;
        }}
        .stButton button:active {{
            transform: translateY(0) !important;
        }}
        
        /* Textos informativos */
        .info-text {{
            color: white !important;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.6);
            font-family: 'Arial', sans-serif;
            animation: fadeIn 0.5s ease-in-out;
        }}
        
        /* Contenedor del ranking */
        .ranking-container {{
            background: rgba(25, 25, 35, 0.95);
            color: white;
            padding: 1.5em;
            border-radius: 15px;
            margin-top: 1.5em;
            box-shadow: 0 8px 16px rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.05);
            animation: fadeIn 0.8s ease-in-out;
        }}
        
        /* Estilos para selectboxes */
        .stSelectbox label {{
            color: white !important;
            font-weight: bold !important;
            font-size: 0.85em !important;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .stSelectbox > div > div {{
            background-color: rgba(30, 30, 45, 0.8) !important;
            border: 1px solid rgba(255,255,255,0.15) !important;
            border-radius: 8px !important;
            color: white !important;
        }}
        
        /* Estilos para número de partidos */
        .stNumberInput label {{
            color: white !important;
            font-weight: bold !important;
            font-size: 0.85em !important;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .stNumberInput input {{
            background-color: rgba(30, 30, 45, 0.8) !important;
            border: 1px solid rgba(255,255,255,0.15) !important;
            border-radius: 8px !important;
            color: white !important;
        }}
        
        /* Estilos para tablas */
        .dataframe {{
            background-color: rgba(25, 25, 35, 0.9) !important;
            font-family: 'Arial', sans-serif !important;
            border-radius: 10px !important;
            overflow: hidden !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
        }}
        .dataframe th {{
            background-color: rgba(0, 103, 178, 0.8) !important;
            color: white !important;
            font-weight: bold !important;
            text-transform: uppercase !important;
            font-size: 0.9em !important;
            letter-spacing: 1px !important;
            padding: 12px !important;
        }}
        .dataframe td {{
            padding: 10px !important;
            border-bottom: 1px solid rgba(255,255,255,0.05) !important;
        }}
        .dataframe tr:hover {{
            background-color: rgba(0, 103, 178, 0.1) !important;
        }}
        
        /* Encabezados de página */
        h1, h2, h3 {{
            color: white !important;
            font-family: 'Arial Black', sans-serif !important;
            letter-spacing: 1px !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.4) !important;
        }}
        h2 {{
            border-bottom: 2px solid #0067B2 !important;
            padding-bottom: 10px !important;
            margin-bottom: 20px !important;
        }}
        h2::before {{
            content: '⚽ ';
            color: #0067B2;
        }}
        
        /* Estilos para tooltips en gráficos */
        .plotly .hovertext {{
            font-family: 'Arial', sans-serif !important;
        }}
        
        /* Estilos adicionales para mejorar apariencia general */
        div[data-testid="stVerticalBlock"] {{
            animation: fadeIn 0.8s ease-in-out;
        }}
        
        /* Estilo personalizado para el footer */
        footer {{
            visibility: hidden;
        }}
        footer:after {{
            content: 'Academia Deportivo Alavés | Análisis de Rendimiento | 2025';
            visibility: visible;
            display: block;
            position: relative;
            padding: 15px;
            text-align: center;
            color: rgba(255,255,255,0.6);
            font-size: 0.8em;
            border-top: 1px solid rgba(255,255,255,0.1);
            margin-top: 40px;
        }}
        </style>
        """, 
        unsafe_allow_html=True
    )

    # Contenido principal
    with st.container():
        login.generarLogin()
        st.markdown('</div>', unsafe_allow_html=True)

    # Verificar si el usuario está logueado
    if 'usuario' in st.session_state:
        # Agregar un mensaje de bienvenida al usuario
        st.markdown(
            f"""
            <div class="info-text" style="text-align: center; margin-bottom: 30px;">
                <h2>¡Bienvenido/a a la plataforma de análisis de la Academia Deportivo Alavés!</h2>
                <p style="font-size: 1.1em; opacity: 0.9;">
                    Explore los datos de rendimiento de jugadores y compare diferentes métricas para 
                    descubrir talentos y áreas de mejora.
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Añadir un separador con estilo
        st.markdown(
            """
            <div style="
                height: 4px;
                background: linear-gradient(90deg, transparent, #0067B2, transparent);
                margin: 10px 0 30px 0;
            "></div>
            """,
            unsafe_allow_html=True
        )
        
        # Sección de Ranking de Jugadores
        st.markdown(
            """
            <div class="ranking-header" style="
                display: flex;
                align-items: center;
                margin-bottom: 20px;
            ">
                <h2 style="margin: 0; flex-grow: 1;">Ranking de Jugadores</h2>
                <div style="background: rgba(0, 103, 178, 0.2); padding: 10px; border-radius: 8px; display: inline-block;">
                    <span style="font-weight: bold; color: #0067B2;">⭐ Jugadores Academia Alavés</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Cargar datos
        df = DataLoader.cargar_datos()
        
        if df is not None:
            # Lista de KPIs disponibles
            kpis_disponibles = [
                'KPI Rendimiento', 
                'KPI Construcción Ataque', 
                'KPI Progresión', 
                'KPI Habilidad Individual', 
                'KPI Peligro Generado', 
                'KPI Finalización', 
                'KPI Eficacia Defensiva', 
                'KPI Juego Aéreo', 
                'KPI Capacidad Recuperación', 
                'KPI Posicionamiento Táctico',
                'xg',
                'POP'
                    ]

            # Preparar listas únicas con opción "Todas"
            ligas_unicas = ['Todas'] + list(df['liga'].unique())
            demarcaciones_unicas = ['Todas'] + list(df['demarcacion'].unique())
            equipos_unicos = ['Todos'] + list(df['equipo'].unique())
            
            # CSS personalizado para filtros
            st.markdown("""
            <style>
            .filter-container {
                background: rgba(25, 25, 35, 0.8);
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 20px;
                border: 1px solid rgba(255,255,255,0.1);
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Agregar encabezado de filtros
            st.markdown("""
            <div style="margin-bottom: 10px; font-weight: bold; color: #0067B2;">
                🔍 FILTROS DE BÚSQUEDA
            </div>
            """, unsafe_allow_html=True)
            
            # Contenedor para los filtros
            st.markdown('<div class="filter-container">', unsafe_allow_html=True)
            
            # Columnas para filtros
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                # Selector de liga
                liga_seleccionada = st.selectbox('Liga', ligas_unicas)
            
            with col2:
                # Selector de KPI para el eje X
                eje_x = st.selectbox('Eje X', kpis_disponibles, index=0)
            
            with col3:
                # Selector de KPI para el eje Y
                eje_y = st.selectbox('Eje Y', kpis_disponibles, index=1)
            
            with col4:
                # Selector de demarcación
                demarcacion_seleccionada = st.selectbox('Demarcación', demarcaciones_unicas)
            
            with col5:
                # Selector de equipo
                equipo_seleccionado = st.selectbox('Equipo', equipos_unicos)
            
            with col6:
                # Selector de número de partidos mínimos
                min_partidos = st.number_input('Mín. Partidos', min_value=1, value=1)
            
            # Cerrar el contenedor de filtros
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Obtener ranking 
            ranking_por_liga = DataAnalyzer.obtener_ranking_por_liga(
                df, 
                liga_seleccionada=liga_seleccionada, 
                demarcacion_seleccionada=demarcacion_seleccionada, 
                equipo_seleccionado=equipo_seleccionado
            )

            # Filtrar por número mínimo de partidos
            ranking_por_liga = ranking_por_liga[ranking_por_liga['Número de Partidos'] >= min_partidos]

            # Agregar descripción de la tabla
            total_jugadores = len(ranking_por_liga)
            jugadores_alaves = len(ranking_por_liga[ranking_por_liga['Equipo'].str.contains('Alav', na=False)])
            
            st.markdown(f"""
            <div style="margin-bottom: 15px; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-weight: bold; color: white;">Mostrando {total_jugadores} jugadores</span>
                    <span style="color: rgba(255,255,255,0.6); margin-left: 10px; font-size: 0.9em;">
                        ({jugadores_alaves} de Deportivo Alavés)
                    </span>
                </div>
                <div id="table-controls" style="text-align: right;">
                    <button id="btn-highlight-alaves" onclick="highlightAlavesPlayers()" style="
                        background: rgba(0, 103, 178, 0.8);
                        color: white;
                        border: none;
                        padding: 5px 10px;
                        border-radius: 5px;
                        cursor: pointer;
                        font-size: 0.9em;
                    ">
                        Destacar jugadores Alavés
                    </button>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Mostrar tabla de ranking en un contenedor con estilo
            st.markdown('<div class="ranking-container">', unsafe_allow_html=True)
            st.dataframe(
                ranking_por_liga.head(20), 
                use_container_width=True,
                column_config={
                    'Equipo': st.column_config.TextColumn(width="medium"),
                    'Jugador': st.column_config.TextColumn(width="large"),
                    'KPI Rendimiento': st.column_config.NumberColumn(format="%.2f", width="medium"),
                    'Número de Partidos': st.column_config.NumberColumn(format="%d", width="small")
                }
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Añadir un separador con estilo antes de la visualización
            st.markdown(
                """
                <div style="
                    height: 2px;
                    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
                    margin: 30px 0;
                "></div>
                """,
                unsafe_allow_html=True
            )
            
            # Título para la sección de visualización
            st.markdown(
                """
                <div style="text-align: center; margin-bottom: 20px;">
                    <h2 style="font-size: 1.8em;">📊 Visualización de Datos</h2>
                    <p style="color: rgba(255,255,255,0.7); font-size: 1.1em;">
                        Compare los diferentes KPIs para analizar el rendimiento de los jugadores
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Visualización de dispersión
            fig = DataVisualizer.crear_visualizacion(ranking_por_liga, eje_x, eje_y)
            
            # Agregar leyenda explicativa para los cuadrantes
            st.markdown(
                """
                <div style="
                    background-color: rgba(25, 25, 35, 0.9);
                    border-radius: 10px;
                    padding: 15px;
                    margin-top: 20px;
                    border: 1px solid rgba(255,255,255,0.1);
                ">
                    <div style="font-weight: bold; margin-bottom: 10px; color: white;">Interpretación de cuadrantes:</div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                        <div style="background: rgba(0, 153, 0, 0.1); padding: 10px; border-radius: 5px;">
                            <span style="font-weight: bold; color: rgba(0, 153, 0, 0.8);">ALTO RENDIMIENTO:</span>
                            <span style="color: rgba(255,255,255,0.7); font-size: 0.9em;"> Valores altos en ambos KPIs - Jugadores destacados</span>
                        </div>
                        <div style="background: rgba(255, 153, 0, 0.1); padding: 10px; border-radius: 5px;">
                            <span style="font-weight: bold; color: rgba(255, 153, 0, 0.8);">POTENCIAL:</span>
                            <span style="color: rgba(255,255,255,0.7); font-size: 0.9em;"> Alto en Y, bajo en X - Jugadores con potencial</span>
                        </div>
                        <div style="background: rgba(255, 0, 0, 0.1); padding: 10px; border-radius: 5px;">
                            <span style="font-weight: bold; color: rgba(255, 0, 0, 0.8);">BAJO RENDIMIENTO:</span>
                            <span style="color: rgba(255,255,255,0.7); font-size: 0.9em;"> Valores bajos en ambos KPIs - Área de mejora</span>
                        </div>
                        <div style="background: rgba(0, 0, 255, 0.1); padding: 10px; border-radius: 5px;">
                            <span style="font-weight: bold; color: rgba(0, 103, 178, 0.8);">ESPECIALIZADO:</span>
                            <span style="color: rgba(255,255,255,0.7); font-size: 0.9em;"> Alto en X, bajo en Y - Jugadores especializados</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Mostrar gráfico en contenedor
            st.plotly_chart(fig, use_container_width=True)
            
            # Agregar script para controlar los gráficos
            st.markdown("""
            <script>
                // Función para agregar interactividad adicional al gráfico
                document.addEventListener('DOMContentLoaded', function() {
                    // Intentar obtener el contenedor del gráfico
                    setTimeout(function() {
                        // Obtener todos los elementos SVG (donde están los puntos del scatter)
                        const svgElements = document.querySelectorAll('svg');
                        svgElements.forEach(svg => {
                            // Buscar los puntos de datos (circulos o estrellas)
                            const dataPoints = svg.querySelectorAll('path.point');
                            
                            // Añadir efecto hover a cada punto
                            dataPoints.forEach(point => {
                                point.addEventListener('mouseenter', function() {
                                    this.style.transform = 'scale(1.3)';
                                    this.style.transition = 'transform 0.2s ease';
                                    this.style.cursor = 'pointer';
                                });
                                
                                point.addEventListener('mouseleave', function() {
                                    this.style.transform = 'scale(1)';
                                });
                            });
                        });
                    }, 2000); // Esperar a que el gráfico se cargue
                });
            </script>
            """, unsafe_allow_html=True)
            
            # Agregar sección final con información adicional
            st.markdown(
                """
                <div style="
                    background: linear-gradient(90deg, rgba(0, 103, 178, 0.2) 0%, rgba(0, 103, 178, 0.1) 100%);
                    border-left: 4px solid #0067B2;
                    padding: 15px;
                    margin-top: 30px;
                    border-radius: 5px;
                ">
                    <h3 style="margin-top: 0;">💡 Recomendaciones de uso</h3>
                    <ul style="margin-bottom: 0; padding-left: 20px; color: rgba(255,255,255,0.8);">
                        <li>Utilice los filtros para comparar jugadores específicos o posiciones</li>
                        <li>Compare diferentes KPIs para identificar fortalezas y debilidades</li>
                        <li>Los jugadores del Deportivo Alavés aparecen destacados con ⭐</li>
                        <li>El tamaño de los puntos representa el valor general de KPI Rendimiento</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
                
        else:
            # Mensaje de error si no se pueden cargar los datos
            st.markdown(
                """
                <div style="
                    background-color: rgba(220, 53, 69, 0.1);
                    color: #FF6B6B;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    margin: 40px 0;
                    border: 1px solid rgba(220, 53, 69, 0.3);
                ">
                    <h3 style="margin-top: 0;">❌ Error al cargar los datos</h3>
                    <p>No se ha podido acceder a la base de datos. Por favor, inténtelo de nuevo más tarde o contacte con soporte técnico.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Añadir script final para mejorar la experiencia de usuario
        st.markdown("""
        <script>
            // Mejoras generales de UI/UX
            document.addEventListener('DOMContentLoaded', function() {
                // Añadir efecto de hover a elementos específicos
                const addHoverEffect = (selector, scale = '1.03') => {
                    document.querySelectorAll(selector).forEach(el => {
                        el.addEventListener('mouseenter', () => {
                            el.style.transform = `scale(${scale})`;
                            el.style.transition = 'transform 0.3s ease';
                        });
                        el.addEventListener('mouseleave', () => {
                            el.style.transform = 'scale(1)';
                        });
                    });
                };
                
                // Aplicar efectos hover a diferentes elementos
                setTimeout(() => {
                    addHoverEffect('.ranking-container');
                    addHoverEffect('.filter-container', '1.01');
                    addHoverEffect('button');
                }, 1000);
                
                // Notificación de actualización de datos
                setTimeout(() => {
                    const notification = document.createElement('div');
                    notification.innerHTML = `
                        <div style="
                            position: fixed;
                            bottom: 20px;
                            right: 20px;
                            background: rgba(0, 103, 178, 0.9);
                            color: white;
                            padding: 10px 15px;
                            border-radius: 5px;
                            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
                            z-index: 1000;
                            font-size: 0.9em;
                            display: flex;
                            align-items: center;
                            gap: 10px;
                            opacity: 0;
                            transform: translateY(20px);
                            transition: all 0.3s ease;
                        ">
                            <div style="font-size: 1.5em;">✅</div>
                            <div>
                                <div style="font-weight: bold;">Datos actualizados</div>
                                <div style="font-size: 0.8em; opacity: 0.8;">Última actualización: Hoy</div>
                            </div>
                        </div>
                    `;
                    document.body.appendChild(notification);
                    
                    // Mostrar notificación con animación
                    setTimeout(() => {
                        notification.firstElementChild.style.opacity = '1';
                        notification.firstElementChild.style.transform = 'translateY(0)';
                        
                        // Ocultar después de 4 segundos
                        setTimeout(() => {
                            notification.firstElementChild.style.opacity = '0';
                            notification.firstElementChild.style.transform = 'translateY(20px)';
                            
                            // Eliminar del DOM
                            setTimeout(() => {
                                document.body.removeChild(notification);
                            }, 300);
                        }, 4000);
                    }, 100);
                }, 3000);
            });
        </script>
        """, unsafe_allow_html=True)

except Exception as e:
    # Mostrar error con estilo mejorado
    st.markdown(
        f"""
        <div style="
            background-color: rgba(220, 53, 69, 0.1);
            color: #FF6B6B;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin: 40px 0;
            border: 1px solid rgba(220, 53, 69, 0.3);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        ">
            <h3 style="margin-top: 0;">❌ Error en la aplicación</h3>
            <p>Se ha producido un error inesperado:</p>
            <code style="
                background-color: rgba(0,0,0,0.2);
                padding: 10px;
                border-radius: 5px;
                display: block;
                margin: 10px 0;
                text-align: left;
                overflow-x: auto;
            ">{e}</code>
            <p>Por favor, contacte con el administrador del sistema.</p>
            <button onclick="location.reload()" style="
                background-color: #FF6B6B;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 5px;
                cursor: pointer;
                margin-top: 10px;
                font-weight: bold;
            ">Recargar aplicación</button>
        </div>
        """,
        unsafe_allow_html=True
    )

# Módulo de Análisis de POP con Clustering
class POPAnalyzer:
    @staticmethod
    def realizar_clustering_pop(df_ranking, num_clusters=3, min_partidos=5):
        """
        Realiza clustering sobre los datos de jugadores basado en la métrica POP
        y otros KPIs relacionados, devolviendo visualizaciones y análisis.
        
        Args:
            df_ranking (DataFrame): DataFrame con los datos de jugadores
            num_clusters (int): Número de clusters a generar
            min_partidos (int): Número mínimo de partidos para considerar a un jugador
            
        Returns:
            dict: Diccionario con figuras y resultados del clustering
        """
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        
        # Filtrar jugadores con suficientes partidos y que tengan valor de POP
        df_filtrado = df_ranking[
            (df_ranking['Número de Partidos'] >= min_partidos) & 
            (~df_ranking['POP'].isna())
        ].copy()
        
        # Si no hay suficientes jugadores, retornar mensaje
        if len(df_filtrado) < num_clusters * 2:
            return {"error": f"No hay suficientes jugadores con datos de POP (mínimo {num_clusters*2} necesarios)"}
            
        # Seleccionar características para el clustering
        features = ['POP', 'KPI Rendimiento', 'KPI Progresión', 'KPI Peligro Generado']
        
        # Asegurarse de que todas las características existan
        valid_features = [f for f in features if f in df_filtrado.columns]
        
        if len(valid_features) < 2:
            return {"error": "No hay suficientes características válidas para el clustering"}
            
        # Preparar datos para clustering
        X = df_filtrado[valid_features].values
        
        # Normalizar datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Aplicar K-means
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        df_filtrado['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Obtener centroides
        centroids = scaler.inverse_transform(kmeans.cluster_centers_)
        
        # Determinar etiquetas significativas para los clusters
        # Ordenar clusters por valor promedio de POP
        pop_index = valid_features.index('POP')
        cluster_pop_values = {}
        
        for i in range(num_clusters):
            cluster_pop_values[i] = centroids[i][pop_index]
            
        # Ordenar clusters por valor de POP (de mayor a menor)
        sorted_clusters = sorted(cluster_pop_values.items(), key=lambda x: x[1], reverse=True)
        
        # Asignar etiquetas
        cluster_labels = {}
        for i, (cluster_id, _) in enumerate(sorted_clusters):
            if i == 0:
                cluster_labels[cluster_id] = "Alto POP"
            elif i == num_clusters - 1:
                cluster_labels[cluster_id] = "Bajo POP"
            else:
                cluster_labels[cluster_id] = "Medio POP"
        
        # Crear mapa de colores para los clusters
        cluster_colors = {
            "Alto POP": "#0067B2",  # Azul Alavés
            "Medio POP": "#53A2D9",  # Azul medio
            "Bajo POP": "#B0D0E9"   # Azul claro
        }
        
        # Añadir etiquetas de cluster al DataFrame
        df_filtrado['cluster_label'] = df_filtrado['cluster'].map(cluster_labels)
        df_filtrado['cluster_color'] = df_filtrado['cluster_label'].map(cluster_colors)
        
        # Crear visualizaciones
        
        # 1. Gráfico de dispersión 3D con PCA si hay más de 2 características
        if len(valid_features) > 2:
            pca = PCA(n_components=3)
            X_pca = pca.fit_transform(X_scaled)
            
            # Crear un nuevo DataFrame con los componentes principales
            df_pca = pd.DataFrame({
                'PC1': X_pca[:, 0],
                'PC2': X_pca[:, 1],
                'PC3': X_pca[:, 2],
                'Cluster': df_filtrado['cluster_label'],
                'Color': df_filtrado['cluster_color'],
                'Jugador': df_filtrado['Jugador'],
                'Equipo': df_filtrado['Equipo'],
                'POP': df_filtrado['POP'],
                'KPI Rendimiento': df_filtrado['KPI Rendimiento']
            })
            
            # Crear gráfico 3D
            fig_3d = px.scatter_3d(
                df_pca, 
                x='PC1', y='PC2', z='PC3',
                color='Cluster',
                color_discrete_map={label: color for label, color in cluster_colors.items()},
                hover_data=['Jugador', 'Equipo', 'POP', 'KPI Rendimiento'],
                labels={'PC1': 'Componente 1', 'PC2': 'Componente 2', 'PC3': 'Componente 3'},
                title='Agrupación de jugadores por POP (Análisis de Componentes Principales)'
            )
            
            # Personalizar gráfico
            fig_3d.update_layout(
                scene=dict(
                    xaxis=dict(backgroundcolor='rgba(0, 0, 0, 0)', color='white'),
                    yaxis=dict(backgroundcolor='rgba(0, 0, 0, 0)', color='white'),
                    zaxis=dict(backgroundcolor='rgba(0, 0, 0, 0)', color='white')
                ),
                paper_bgcolor='rgba(25, 25, 35, 0.95)',
                plot_bgcolor='rgba(25, 25, 35, 0.95)',
                font=dict(color='white')
            )
        else:
            fig_3d = None
        
        # 2. Gráfico de dispersión POP vs KPI Rendimiento
        fig_scatter = px.scatter(
            df_filtrado, 
            x='POP', 
            y='KPI Rendimiento',
            color='cluster_label',
            color_discrete_map=cluster_colors,
            hover_data=['Jugador', 'Equipo', 'Demarcación', 'Número de Partidos'],
            text='Jugador',
            size='Número de Partidos',
            size_max=20,
            opacity=0.8,
            title='Relación entre POP y KPI Rendimiento por Grupos'
        )
        
        # Personalizar el gráfico
        fig_scatter.update_layout(
            xaxis=dict(title='POP', gridcolor='rgba(255, 255, 255, 0.1)', zerolinecolor='rgba(255, 255, 255, 0.2)'),
            yaxis=dict(title='KPI Rendimiento', gridcolor='rgba(255, 255, 255, 0.1)', zerolinecolor='rgba(255, 255, 255, 0.2)'),
            plot_bgcolor='rgba(25, 25, 35, 0.95)',
            paper_bgcolor='rgba(25, 25, 35, 0.95)',
            font=dict(color='white'),
            legend_title_text='Grupo POP',
            hovermode='closest'
        )
        
        # Añadir texto solo para algunos puntos destacados (top 5 por POP)
        fig_scatter.update_traces(
            textposition='top center',
            textfont=dict(color='rgba(255, 255, 255, 0.8)', size=10)
        )
        
        # Filtrar para mostrar texto solo en los 5 jugadores con mayor POP
        for i, t in enumerate(fig_scatter.data):
            t.textfont.color = cluster_colors[t.name]
            
            # Crear máscara para mostrar solo nombres de los top 5 por POP en cada cluster
            top_indices = df_filtrado[df_filtrado['cluster_label'] == t.name].sort_values('POP', ascending=False).head(3).index
            top_indices_list = top_indices.tolist()
            
            # Crear una nueva lista de textos donde solo los top tienen texto
            text_list = []
            for idx in df_filtrado[df_filtrado['cluster_label'] == t.name].index:
                if idx in top_indices_list:
                    text_list.append(df_filtrado.loc[idx, 'Jugador'])
                else:
                    text_list.append("")
                
            # Asignar los textos actualizados
            t.text = text_list
        
        # 3. Gráfico de radar para comparar características de clusters
        # Preparar datos para el radar chart
        categories = valid_features + [valid_features[0]]  # Repetir el primero para cerrar el polígono
        
        fig_radar = go.Figure()
        
        for cluster_id, label in cluster_labels.items():
            cluster_values = centroids[cluster_id]
            
            # Normalizar valores para el radar chart (de 0 a 1)
            value_min = np.min(centroids, axis=0)
            value_max = np.max(centroids, axis=0)
            normalized_values = (cluster_values - value_min) / (value_max - value_min)
            
            # Añadir el primer valor al final para cerrar el polígono
            radar_values = list(normalized_values) + [normalized_values[0]]
            
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_values,
                theta=categories,
                fill='toself',
                name=label,
                line=dict(color=cluster_colors[label]),
                fillcolor=cluster_colors[label].replace(')', ', 0.2)').replace('rgb', 'rgba')
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    showticklabels=False,
                    gridcolor='rgba(255, 255, 255, 0.15)'
                ),
                angularaxis=dict(
                    color='white',
                    gridcolor='rgba(255, 255, 255, 0.15)'
                ),
                bgcolor='rgba(25, 25, 35, 0.95)'
            ),
            paper_bgcolor='rgba(25, 25, 35, 0.95)',
            plot_bgcolor='rgba(25, 25, 35, 0.95)',
            font=dict(color='white'),
            title={
                'text': 'Perfil de características por grupo de POP',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=20, color='white')
            },
            showlegend=True,
            legend=dict(
                orientation='h',
                y=-0.1,
                x=0.5,
                xanchor='center'
            )
        )
        
        # 4. Estadísticas por cluster
        stats_por_cluster = {}
        for cluster_id, label in cluster_labels.items():
            cluster_df = df_filtrado[df_filtrado['cluster'] == cluster_id]
            
            stats = {
                'Jugadores': len(cluster_df),
                'POP Promedio': cluster_df['POP'].mean(),
                'KPI Rendimiento Promedio': cluster_df['KPI Rendimiento'].mean(),
                'Jugadores Destacados': cluster_df.nlargest(3, 'POP')['Jugador'].tolist(),
                'Equipos': cluster_df['Equipo'].value_counts().nlargest(3).to_dict(),
                'Demarcaciones': cluster_df['Demarcación'].value_counts().to_dict()
            }
            stats_por_cluster[label] = stats
        
        # Crear un DataFrame con los jugadores top de cada cluster
        top_jugadores_por_cluster = pd.DataFrame()
        
        for cluster_id, label in cluster_labels.items():
            cluster_df = df_filtrado[df_filtrado['cluster'] == cluster_id]
            top_jugadores = cluster_df.nlargest(5, 'POP')[['Jugador', 'Equipo', 'Demarcación', 'POP', 'KPI Rendimiento', 'Número de Partidos']]
            top_jugadores['Grupo POP'] = label
            top_jugadores_por_cluster = pd.concat([top_jugadores_por_cluster, top_jugadores])
        
        # Ordenar por POP descendente
        top_jugadores_por_cluster = top_jugadores_por_cluster.sort_values('POP', ascending=False)
        
        # Retornar resultados
        return {
            "fig_3d": fig_3d,
            "fig_scatter": fig_scatter,
            "fig_radar": fig_radar,
            "stats_por_cluster": stats_por_cluster,
            "top_jugadores": top_jugadores_por_cluster,
            "df_clusters": df_filtrado
        }

# Añadir la sección de análisis POP después de las visualizaciones anteriores
st.markdown(
    """
    <div style="
        height: 3px;
        background: linear-gradient(90deg, transparent, #FFFFFF, transparent);
        margin: 50px 0 30px 0;
    "></div>
    """,
    unsafe_allow_html=True
)

# Título de la sección de POP
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 30px;">
        <h2 style="font-size: 2em;">🔍 Análisis Avanzado de Pases de Profundidad (POP)</h2>
        <p style="color: rgba(255,255,255,0.7); font-size: 1.1em;">
            Descubra patrones y agrupaciones de jugadores basados en su capacidad para realizar pases de profundidad
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Mostrar estadísticas de POP
# Usamos el dataframe que ya tenemos cargado (ranking_por_liga)
if 'usuario' in st.session_state and df is not None:
    # Obtener ranking para clustering
    ranking_por_liga = DataAnalyzer.obtener_ranking_por_liga(
        df, 
        liga_seleccionada=liga_seleccionada, 
        demarcacion_seleccionada=demarcacion_seleccionada, 
        equipo_seleccionado=equipo_seleccionado
    )
    
    # Filtrar por número mínimo de partidos
    ranking_por_liga = ranking_por_liga[ranking_por_liga['Número de Partidos'] >= min_partidos]
    
    # Verificar si hay datos de POP
    if 'POP' in ranking_por_liga.columns:
            # Columnas para configuración
            col1, col2, col3 = st.columns(3)
    
    with col1:
        num_clusters = st.slider('Número de grupos', min_value=2, max_value=5, value=3, 
                              help="Seleccione el número de grupos para el algoritmo de clustering")
    
    with col2:
        min_partidos_pop = st.slider('Mínimo de partidos', min_value=1, max_value=10, value=3,
                                 help="Número mínimo de partidos para incluir un jugador en el análisis")
    
    with col3:
        st.markdown(
            """
            <div style="background-color: rgba(0, 103, 178, 0.1); 
                        border-left: 3px solid #0067B2;
                        padding: 10px;
                        border-radius: 3px;
                        margin-top: 25px;">
                <b>POP</b>: Pase de Oportunidad en Profundidad
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Realizar análisis de clustering
    with st.spinner('Realizando análisis avanzado de POP...'):
        resultados_pop = POPAnalyzer.realizar_clustering_pop(
            ranking_por_liga, 
            num_clusters=num_clusters, 
            min_partidos=min_partidos_pop
        )
    
    # Verificar si hay error
    if "error" in resultados_pop:
        st.error(resultados_pop["error"])
    else:
        # Mostrar las visualizaciones
        
        # 1. Visualización principal - Scatter plot
        st.plotly_chart(resultados_pop["fig_scatter"], use_container_width=True)
        
        # 2. Características de los grupos - Radar plot
        st.plotly_chart(resultados_pop["fig_radar"], use_container_width=True)
        
        # 3. Visualización 3D si está disponible
        if resultados_pop["fig_3d"] is not None:
            with st.expander("Ver visualización 3D de los grupos", expanded=False):
                st.plotly_chart(resultados_pop["fig_3d"], use_container_width=True)
        
        # 4. Tabla de jugadores top por grupo
        st.markdown("### 📊 Jugadores destacados por grupo de POP")
        
        st.markdown(
            """
            <div style="margin-bottom: 15px;">
                <p style="color: rgba(255,255,255,0.7);">
                    La tabla muestra los jugadores más destacados de cada grupo según su métrica POP.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Crear columnas para la tabla
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Mostrar tabla sin estilo personalizado para evitar errores
            st.dataframe(
                resultados_pop["top_jugadores"], 
                use_container_width=True,
                column_config={
                    'POP': st.column_config.NumberColumn(format="%.3f"),
                    'KPI Rendimiento': st.column_config.NumberColumn(format="%.2f"),
                    'Grupo POP': st.column_config.TextColumn(width="medium"),
                }
            )
        
        with col2:
            # Añadir estadísticas resumidas
            stats = resultados_pop["stats_por_cluster"]
            
            for grupo, datos in stats.items():
                color = ""
                if grupo == "Alto POP":
                    color = "#0067B2"
                elif grupo == "Medio POP":
                    color = "#53A2D9"
                else:
                    color = "#B0D0E9"
                
                st.markdown(
                    f"""
                    <div style="
                        background-color: rgba(25, 25, 35, 0.8);
                        border-radius: 10px;
                        border-left: 4px solid {color};
                        padding: 10px;
                        margin-bottom: 10px;
                    ">
                        <h4 style="margin: 0 0 8px 0; color: {color};">{grupo}</h4>
                        <p style="margin: 3px 0; font-size: 0.9em;">
                            <b>Jugadores:</b> {datos['Jugadores']}
                        </p>
                        <p style="margin: 3px 0; font-size: 0.9em;">
                            <b>POP promedio:</b> {datos['POP Promedio']:.3f}
                        </p>
                        <p style="margin: 3px 0; font-size: 0.9em;">
                            <b>KPI promedio:</b> {datos['KPI Rendimiento Promedio']:.2f}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        # 5. Análisis y conclusiones
        st.markdown(
            """
            <div style="
                background: linear-gradient(90deg, rgba(0, 103, 178, 0.1) 0%, rgba(0, 103, 178, 0.05) 100%);
                border-radius: 10px;
                padding: 20px;
                margin-top: 30px;
                border: 1px solid rgba(0, 103, 178, 0.2);
            ">
                <h3 style="margin-top: 0; color: white;">💡 Análisis y Conclusiones</h3>
                
                <p style="color: rgba(255,255,255,0.8);">
                    El análisis de <b>Pases de Oportunidad en Profundidad (POP)</b> revela patrones significativos 
                    en la capacidad de los jugadores para realizar pases que generan ventajas ofensivas.
                </p>
                
                <ul style="color: rgba(255,255,255,0.8);">
                    <li><b>Grupo Alto POP:</b> Jugadores con excepcional visión de juego y precisión en pases que rompen líneas defensivas.</li>
                    <li><b>Grupo Medio POP:</b> Jugadores con buena capacidad de pase pero menos consistentes en la creación de peligro.</li>
                    <li><b>Grupo Bajo POP:</b> Jugadores que priorizan la seguridad en el pase o cuya función táctica no les exige pases en profundidad.</li>
                </ul>
                
                <p style="color: rgba(255,255,255,0.8);">
                    La correlación entre POP y otros KPIs de rendimiento sugiere que los jugadores con alta capacidad 
                    de pase en profundidad tienden a destacar también en otras métricas ofensivas.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
else:
    st.warning("No se encontraron datos de POP para realizar el análisis de clustering.")
