import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import os
import polars as pl

class DataManager:
    # Definir BASE_DIR como una variable de clase
    BASE_DIR = Path(__file__).parent

    @staticmethod
    def get_column_dtypes():
        return {
            'match_id': 'int64',
            'team_id': 'string',
            'player_id': 'float64',
            'season_id': 'string',
            'equipo': 'category',
            'demarcacion': 'category',
            'jugador': 'string',
            'temporada': 'string'
        }

    @staticmethod
    def get_required_columns():
        return [
            'match_id', 'team_id', 'player_id', 'season_id', 'equipo', 
            'demarcacion', 'jugador', 'xg', 'pases_progresivos_inicio',
            'pases_progresivos_creacion', 'pases_progresivos_finalizacion',
            'duelos_aereos_ganados_zona_area', 'duelos_aereos_ganados_zona_baja',
            'duelos_aereos_ganados_zona_media', 'duelos_aereos_ganados_zona_alta',
            'recuperaciones_zona_baja', 'recuperaciones_zona_media',
            'recuperaciones_zona_alta', 'entradas_ganadas_zona_baja',
            'entradas_ganadas_zona_media', 'entradas_ganadas_zona_alta',
            'pases_largos_exitosos', 'cambios_orientacion_exitosos'
        ]

    @staticmethod
    @st.cache_data(ttl=3600)
    def load_base_data():
        """Carga los datos base y los cachea para evitar lecturas repetidas."""
        try:
            ruta_archivo = os.path.join(DataManager.BASE_DIR, "data/archivos_parquet/eventos_metricas_alaves.parquet")
            
            # Cargar en modo lazy y seleccionar columnas necesarias
            df = pl.scan_parquet(ruta_archivo).select([
                'equipo', 'temporada', 'match_id', 'season_id', 'player_id', 'jugador', 'demarcacion'
            ]).with_columns([
                pl.col('equipo').cast(pl.Categorical),
                pl.col('temporada').cast(pl.Categorical),
                pl.col('match_id').cast(pl.Utf8),
                pl.col('season_id').cast(pl.Utf8),
                pl.col('player_id').cast(pl.Float64)
            ])
            
            # Ejecutar la consulta y devolver el DataFrame
            return df.collect()
        except Exception as e:
            print(f"Error cargando datos base: {e}")
            return None

    @staticmethod
    @st.cache_data(ttl=3600)
    def get_match_ids(player_id, temporada_seleccionada):
        """Obtiene los IDs de partidos para un jugador y temporada específicos."""
        df_filtros = DataManager.load_base_data()
        
        if df_filtros is None:
            return []
            
        if temporada_seleccionada != 'Todas':
            df_filtros = df_filtros.filter(pl.col('temporada') == temporada_seleccionada)
        
        match_ids = df_filtros.filter(pl.col('player_id') == player_id).get_column('match_id').unique()
        return match_ids

    @staticmethod
    @st.cache_data(ttl=3600)
    def get_filtered_data_for_player(player_id, temporada_seleccionada):
        """Carga y filtra los datos una sola vez para un jugador específico"""
        try:
            match_ids = DataManager.get_match_ids(player_id, temporada_seleccionada)
            
            if not match_ids:
                return None
            
            # Cargar solo las columnas necesarias de jugadores
            columnas_jugadores = [
                'match_id', 'team_id', 'player_id', 'season_id', 'partido', 'equipo', 'jugador', 'demarcacion', 'event_time', 'periodo', 'resultado', 'resultado_partido', 'POP',
                'temporada', 'tipo_evento', 'xstart', 'ystart', 'xend', 'yend', 'xg', 'pases_progresivos_inicio',
                'pases_progresivos_creacion', 'pases_largos_exitosos', 'pases_exitosos_campo_propio', 
                'pases_fallidos_campo_contrario', 'pases_exitosos_campo_contrario', 'pases_fallidos_campo_propio',
                'pases_largos_fallidos', 'cambios_orientacion_exitosos', 'cambios_orientacion_fallidos',
                'pase_clave', 'recuperaciones_zona_baja', 'recuperaciones_zona_media', 'recuperaciones_zona_alta',
                'pases_adelante_inicio', 'pases_adelante_creacion', 'pases_horizontal_inicio', 
                'pases_horizontal_creacion', 'pases_atras_inicio', 'pases_atras_creacion',
                'duelos_aereos_ganados_zona_area', 'duelos_suelo_ganados_zona_area', 'entradas_ganadas_zona_media',
                'entradas_ganadas_zona_alta', 'entradas_ganadas_zona_area', 'entradas_ganadas_zona_baja',
                'duelos_suelo_perdidos_zona_area', 'entradas_perdidas_zona_area', 'duelos_aereos_ganados_zona_baja', 
                'duelos_aereos_ganados_zona_media', 'duelos_aereos_ganados_zona_alta',
                'duelos_aereos_perdidos_zona_area', 'duelos_aereos_perdidos_zona_baja',
                'duelos_aereos_perdidos_zona_media', 'duelos_aereos_perdidos_zona_alta',
                'pases_progresivos_finalizacion'
            ]
            
            # Optimizar tipos de datos desde el inicio
            tipos_datos = {
                'match_id': pl.Utf8,
                'season_id': pl.Utf8,
                'player_id': pl.Float64,
                'equipo': pl.Categorical,
                'temporada': pl.Categorical,
                'demarcacion': pl.Categorical,
                'tipo_evento': pl.Categorical,
                'xstart': pl.Float32,
                'ystart': pl.Float32,
                'xend': pl.Float32,
                'yend': pl.Float32
            }
            
            # Cargar datos de jugadores filtrando por match_ids
            df_jugadores = pl.scan_parquet(
                os.path.join(DataManager.BASE_DIR, "data/archivos_parquet/eventos_metricas_alaves.parquet")
            ).select(columnas_jugadores).filter(
                pl.col('match_id').is_in(match_ids) & 
                pl.col('player_id') == player_id
            ).collect().to_pandas()
            
            # Convertir columnas numéricas
            numeric_columns = ['xg', 'pases_progresivos_inicio', 'pases_progresivos_creacion',
                            'pases_adelante_inicio', 'pases_adelante_creacion',
                            'duelos_aereos_ganados_zona_area', 'duelos_aereos_ganados_zona_baja', 
                            'duelos_aereos_ganados_zona_media', 'duelos_aereos_ganados_zona_alta',
                            'duelos_aereos_perdidos_zona_area', 'duelos_aereos_perdidos_zona_baja',
                            'duelos_aereos_perdidos_zona_media', 'duelos_aereos_perdidos_zona_alta',
                            'recuperaciones_zona_baja', 'recuperaciones_zona_media', 'recuperaciones_zona_alta',
                            'pases_largos_exitosos', 'cambios_orientacion_exitosos']
            
            for col in numeric_columns:
                if col in df_jugadores.columns:
                    df_jugadores[col] = pd.to_numeric(df_jugadores[col], errors='coerce').fillna(0)
            
            return df_jugadores
            
        except Exception as e:
            print(f"Error cargando datos filtrados para jugador: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    @staticmethod
    @st.cache_data(ttl=3600)
    def get_estadisticas_for_player(player_id, temporada_seleccionada):
        """Carga estadísticas específicas para un jugador"""
        try:
            match_ids = DataManager.get_match_ids(player_id, temporada_seleccionada)
            
            if not match_ids:
                return None
            
            # Cargar estadísticas
            columnas_estadisticas = [
                'player_id', 'jugador', 'equipo', 'temporada', 'match_id', 'partido', 'season_id', 'jornada','demarcacion', 'liga', 'POP', 'xg', 'goles', 'liga',
                'KPI_rendimiento', 'KPI_construccion_ataque', 'KPI_progresion', 'KPI_habilidad_individual',
                'KPI_peligro_generado', 'KPI_finalizacion', 'KPI_eficacia_defensiva', 'KPI_juego_aereo', 
                'KPI_capacidad_recuperacion', 'KPI_posicionamiento_tactico', 'goles', 'asistencias',
                'tarjetas_amarillas', 'tarjetas_rojas', 'minutos_jugados'
            ]
            
            df_estadisticas = pl.scan_parquet(
                os.path.join(DataManager.BASE_DIR, "data/archivos_parquet/eventos_datos_acumulados.parquet")
            ).select(columnas_estadisticas).filter(
                pl.col('match_id').is_in(match_ids) & 
                pl.col('player_id') == player_id
            ).collect().to_pandas()
            
            # Convertir KPIs y estadísticas a numérico
            kpi_cols = [col for col in df_estadisticas.columns if col.startswith('KPI_')]
            stat_cols = ['goles', 'asistencias', 'tarjetas_amarillas', 'tarjetas_rojas', 'minutos_jugados']
            
            for col in kpi_cols + stat_cols:
                if col in df_estadisticas.columns:
                    df_estadisticas[col] = pd.to_numeric(df_estadisticas[col], errors='coerce').fillna(0)
            
            return df_estadisticas
            
        except Exception as e:
            print(f"Error cargando estadísticas para jugador: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    @staticmethod
    def filtrar_jugadores(df_jugadores, equipo, temporada, posicion):
        """Filtra el DataFrame de jugadores según los criterios especificados."""
        if df_jugadores is None:
            return pd.DataFrame()
            
        df_filtrado = df_jugadores.copy()
        
        if equipo != 'Toda la Academia':
            df_filtrado = df_filtrado[df_filtrado['equipo'] == equipo]
        else:
            df_filtrado = df_filtrado[df_filtrado['equipo'].str.contains('Alav', case=False, na=False)]
        
        # Usar columna temporada en lugar de season_id
        if temporada != 'Todas':
            df_filtrado = df_filtrado[df_filtrado['temporada'].astype(str) == str(temporada)]
        
        if posicion != 'Todas':
            df_filtrado = df_filtrado[df_filtrado['demarcacion'] == posicion]
        
        return df_filtrado

    @staticmethod
    def get_temporadas(df_jugadores, equipo=None):
        """Obtiene la lista de temporadas disponibles."""
        if df_jugadores is None:
            return []
            
        # Filtrar por equipo si es necesario
        if equipo:
            df_jugadores = df_jugadores[df_jugadores['equipo'] == equipo]
   
        # Ordenar temporadas en formato XX/XX
        temporadas = df_jugadores['temporada'].unique()
        return sorted(temporadas, key=lambda x: int(x.split('/')[0]), reverse=True)

    @staticmethod
    @st.cache_data(ttl=3600)
    def get_filtered_data():
        """Método compatible con el código anterior que utiliza el nuevo enfoque de carga."""
        try:
            # Primero cargamos los datos base
            base_data = DataManager.load_base_data()
            if base_data is None:
                return None, None
            
            # Transformar a pandas para mantener compatibilidad
            df_base = base_data.to_pandas()
            
            # Obtener player_ids únicos para cargar todos los datos una sola vez
            player_ids = df_base['player_id'].unique()
            
            # Lista para almacenar DataFrames de jugadores y estadísticas
            dfs_jugadores = []
            dfs_estadisticas = []
            
            DATA_DIR = DataManager.BASE_DIR / "data" / "archivos_parquet"
            
            # Cargar datos de jugadores y estadísticas para todos los partidos
            columnas_jugadores = [
                'match_id', 'team_id', 'player_id', 'season_id', 'equipo', 'jugador', 'demarcacion', 'event_time', 'periodo', 'partido', 'resultado', 'resultado_partido', 'POP',
                'temporada', 'tipo_evento', 'xstart', 'ystart', 'xend', 'yend', 'xg', 'pases_progresivos_inicio',
                'pases_progresivos_creacion', 'pases_progresivos_finalizacion', 'pases_largos_exitosos', 
                'pases_exitosos_campo_propio', 'pases_fallidos_campo_contrario', 'pases_exitosos_campo_contrario', 
                'pases_fallidos_campo_propio', 'pases_largos_fallidos', 'cambios_orientacion_exitosos', 
                'cambios_orientacion_fallidos', 'pase_clave', 'recuperaciones_zona_baja', 'recuperaciones_zona_media', 
                'recuperaciones_zona_alta', 'pases_adelante_inicio', 'pases_adelante_creacion', 
                'pases_horizontal_inicio', 'pases_horizontal_creacion', 'pases_atras_inicio', 'pases_atras_creacion',
                'duelos_aereos_ganados_zona_area', 'duelos_suelo_ganados_zona_area', 'entradas_ganadas_zona_media',  
                'entradas_ganadas_zona_alta', 'entradas_ganadas_zona_area', 'entradas_ganadas_zona_baja',
                'duelos_suelo_perdidos_zona_area', 'entradas_perdidas_zona_area','duelos_aereos_ganados_zona_baja', 
                'duelos_aereos_ganados_zona_media', 'duelos_aereos_ganados_zona_alta',
                'duelos_aereos_perdidos_zona_area', 'duelos_aereos_perdidos_zona_baja',
                'duelos_aereos_perdidos_zona_media', 'duelos_aereos_perdidos_zona_alta'
            ]
            
            columnas_estadisticas = [
                'player_id', 'jugador', 'equipo', 'temporada', 'match_id', 'season_id', 'jornada', 'demarcacion', 'liga',
                'KPI_rendimiento', 'KPI_construccion_ataque', 'KPI_progresion', 'KPI_habilidad_individual',
                'KPI_peligro_generado', 'KPI_finalizacion', 'KPI_eficacia_defensiva', 'KPI_juego_aereo', 
                'KPI_capacidad_recuperacion', 'KPI_posicionamiento_tactico', 'goles', 'asistencias',
                'tarjetas_amarillas', 'tarjetas_rojas', 'minutos_jugados'
            ]
            
            # Carga optimizada, directamente desde archivos parquet
            df_jugadores = pl.read_parquet(
                os.path.join(DATA_DIR, "eventos_metricas_alaves.parquet"),
                columns=columnas_jugadores
            ).to_pandas()
            
            df_estadisticas = pl.read_parquet(
                os.path.join(DATA_DIR, "eventos_datos_acumulados.parquet"),
                columns=columnas_estadisticas
            ).to_pandas()
            
            # Convertir columnas numéricas
            numeric_columns = ['xg', 'pases_progresivos_inicio', 'pases_progresivos_creacion',
                             'pases_adelante_inicio', 'pases_adelante_creacion',
                             'duelos_aereos_ganados_zona_area', 'duelos_aereos_ganados_zona_baja', 
                             'duelos_aereos_ganados_zona_media', 'duelos_aereos_ganados_zona_alta',
                             'duelos_aereos_perdidos_zona_area', 'duelos_aereos_perdidos_zona_baja',
                             'duelos_aereos_perdidos_zona_media', 'duelos_aereos_perdidos_zona_alta',
                             'recuperaciones_zona_baja', 'recuperaciones_zona_media', 'recuperaciones_zona_alta',
                             'pases_largos_exitosos', 'cambios_orientacion_exitosos']
            
            for col in numeric_columns:
                if col in df_jugadores.columns:
                    df_jugadores[col] = pd.to_numeric(df_jugadores[col], errors='coerce').fillna(0)

            # Convertir KPIs y estadísticas
            kpi_cols = [col for col in df_estadisticas.columns if col.startswith('KPI_')]
            stat_cols = ['goles', 'asistencias', 'tarjetas_amarillas', 'tarjetas_rojas', 'minutos_jugados']
            
            for col in kpi_cols + stat_cols:
                if col in df_estadisticas.columns:
                    df_estadisticas[col] = pd.to_numeric(df_estadisticas[col], errors='coerce').fillna(0)

            # Asegurar que 'temporada' esté presente
            if 'temporada' not in df_jugadores.columns:
                df_jugadores['temporada'] = df_jugadores['season_id'].astype(str)

            if 'temporada' not in df_estadisticas.columns:
                df_estadisticas['temporada'] = df_estadisticas['season_id'].astype(str)

            # Tipos categóricos para mejorar rendimiento
            for col, dtype in DataManager.get_column_dtypes().items():
                if dtype == 'category' and col in df_jugadores.columns:
                    df_jugadores[col] = df_jugadores[col].astype('category')
                if dtype == 'category' and col in df_estadisticas.columns:
                    df_estadisticas[col] = df_estadisticas[col].astype('category')

            return df_jugadores, df_estadisticas

        except Exception as e:
            print(f"Error cargando datos: {e}")
            import traceback
            print(traceback.format_exc())
            return None, None