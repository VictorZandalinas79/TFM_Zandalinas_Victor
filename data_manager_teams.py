import streamlit as st
import polars as pl
import os
from pathlib import Path
import pandas as pd

class DataManagerTeams:
    # Definir BASE_DIR como una variable de clase
    BASE_DIR = Path(__file__).parent

    @staticmethod
    @st.cache_data(ttl=3600)
    def load_base_data():
        """Carga los datos base y los cachea para evitar lecturas repetidas."""
        ruta_archivo = os.path.join(DataManagerTeams.BASE_DIR, "data/archivos_parquet/eventos_metricas_alaves.parquet")
        
        # Cargar en modo lazy y seleccionar columnas necesarias
        df = pl.scan_parquet(ruta_archivo).select([
            'equipo', 'temporada', 'match_id', 'season_id'
        ]).with_columns([
            pl.col('equipo').cast(pl.Categorical),
            pl.col('temporada').cast(pl.Categorical),
            pl.col('match_id').cast(pl.Utf8),
            pl.col('season_id').cast(pl.Utf8)
        ])
        
        # Ejecutar la consulta y devolver el DataFrame
        return df.collect()

    @staticmethod
    @st.cache_data(ttl=3600)
    def get_match_ids(equipo_seleccionado, temporada_seleccionada):
        """Obtiene los IDs de partidos para un equipo y temporada específicos."""
        df_filtros = DataManagerTeams.load_base_data()
        
        if temporada_seleccionada != 'Todas':
            df_filtros = df_filtros.filter(pl.col('temporada') == temporada_seleccionada)
        
        match_ids = df_filtros.filter(pl.col('equipo') == equipo_seleccionado).get_column('match_id').unique()
        return match_ids

    @staticmethod
    @st.cache_data(ttl=3600)
    def get_filter_data():
        """Carga los datos de filtros y los cachea."""
        try:
            df_filtros = DataManagerTeams.load_base_data()
            return df_filtros.to_pandas()
        except Exception as e:
            st.error(f"Error al cargar los datos de filtros: {e}")
            return None
    
    # En data_manager_teams.py - Añadir una nueva función para cargar datos filtrados una sola vez
    # Añadir a DataManagerTeams.py
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_filtered_data_once(equipo_seleccionado, temporada_seleccionada):
        """Carga y filtra los datos una sola vez"""
        try:
            match_ids = DataManagerTeams.get_match_ids(equipo_seleccionado, temporada_seleccionada)
            
            # Cargar solo las columnas necesarias
            columnas_necesarias = [
                'match_id', 'season_id', 'equipo', 'jugador', 'temporada', 
                'tipo_evento', 'demarcacion', 'xstart', 'ystart', 'xend', 'yend',
                'xg', 'periodo', 'event_time', 'jornada', 'partido',
                'KPI_Rendimiento', 'valoracion'
            ]
            
            # Optimizar tipos de datos desde el inicio
            tipos_datos = {
                'match_id': pl.Utf8,
                'season_id': pl.Utf8,
                'equipo': pl.Categorical,
                'temporada': pl.Categorical,
                'tipo_evento': pl.Categorical,
                'demarcacion': pl.Categorical,
                'xstart': pl.Float32,
                'ystart': pl.Float32,
                'xend': pl.Float32,
                'yend': pl.Float32
            }
            
            df_equipos = pl.scan_parquet(
                os.path.join(DataManagerTeams.BASE_DIR, "data/archivos_parquet/eventos_metricas_alaves.parquet")
            ).select(columnas_necesarias).filter(pl.col('match_id').is_in(match_ids)).collect()
            
            # Convertir a pandas manteniendo los tipos de datos
            return df_equipos.to_pandas()
            
        except Exception as e:
            print(f"Error cargando datos filtrados: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    @staticmethod
    @st.cache_data(ttl=3600)
    def get_accumulated_events_data():
        """
        Obtiene los datos acumulados de eventos desde el archivo eventos_datos_acumulados.parquet
        
        Returns:
            DataFrame con los datos acumulados o None si hay un error
        """
        try:
            # Ruta al archivo de datos acumulados
            accumulated_data_path = os.path.join(DataManagerTeams.BASE_DIR, "data/archivos_parquet/eventos_datos_acumulados.parquet")
            
            # Verificar si el archivo existe
            if not os.path.exists(accumulated_data_path):
                print(f"⚠️ No se encontró el archivo de datos acumulados: {accumulated_data_path}")
                return None
                
            
            # Cargar los datos con polars para mejor rendimiento
            df_accumulated = pl.scan_parquet(accumulated_data_path).collect()
            
            # Convertir a pandas para mantener compatibilidad con el resto del código
            return df_accumulated.to_pandas()
            
        except Exception as e:
            print(f"❌ Error al cargar datos acumulados: {e}")
            return None

    @staticmethod
    @st.cache_data(ttl=3600)
    def get_detailed_data(equipo_seleccionado, temporada_seleccionada):
        """Carga los datos detallados para un equipo y temporada específicos."""
        try:
            match_ids = DataManagerTeams.get_match_ids(equipo_seleccionado, temporada_seleccionada)

            # Cargar datos de equipos
            columnas_equipos = [
                'match_id', 'season_id', 'equipo', 'jugador', 'temporada', 
                'tipo_evento','demarcacion', 'xstart', 'ystart', 'xend', 'yend',
                'xg', 'periodo', 'event_time'
            ]
            
            df_equipos = pl.scan_parquet(
                os.path.join(DataManagerTeams.BASE_DIR, "data/archivos_parquet/eventos_metricas_alaves.parquet")
            ).select(columnas_equipos).with_columns(
                pl.col('match_id').cast(pl.Utf8)
            ).filter(pl.col('match_id').is_in(match_ids)).collect()

            # Cargar datos de estadísticas directamente del archivo en vez de usar get_accumulated_events_data
            # para tener más control sobre las columnas que se seleccionan
            columnas_estadisticas = [
                'team_id', 'season_id', 'match_id',
                'goles', 'goles_concedidos',
                'pase_corto', 'pase_corto_exitoso', 
                'pase_medio', 'pase_medio_exitoso', 
                'pase_largo', 'pase_largo_exitoso', 
                'tarjetas_amarillas', 
                'regates',
                'centros_alejados', 'centros_linea_fondo', 
                'duelos_aereos_ganados_zona_area',
                'duelos_aereos_perdidos_zona_area',
                'tiros_a_porteria', 'tiros_totales',
                'KPI_rendimiento', 'valoracion'
            ]
            
            # Verificar si 'equipo' está en el archivo de estadísticas
            stats_path = os.path.join(DataManagerTeams.BASE_DIR, "data/archivos_parquet/eventos_datos_acumulados.parquet")
            schema = pl.scan_parquet(stats_path).schema
            has_equipo = 'equipo' in schema
            
            if has_equipo:
                columnas_estadisticas.append('equipo')
            
            df_estadisticas = pl.scan_parquet(stats_path).select(columnas_estadisticas).with_columns(
                pl.col('match_id').cast(pl.Utf8)
            ).filter(pl.col('match_id').is_in(match_ids)).collect()
            
            # Cargar datos de KPIs
            columnas_KPI = ['match_id', 'season_id','team_id', 'equipo', 'jornada', 'partido'] + [
                'Posesion_Dominante', 
                'Estilo_Combinativo_Directo', 
                'Estilo_Presionante', 
                'Altura_Bloque_Defensivo', 
                'Progresion_Ataque', 
                'Verticalidad', 
                'Ataques_Bandas', 
                'Rendimiento_Finalizacion', 
                'Peligro_Generado', 
                'Solidez_Defensiva', 
                'Eficacia_Defensiva', 
                'Zonas_Recuperacion', 
                'Presion_Alta',
                'KPI_Rendimiento'
            ]

            df_KPI = pl.scan_parquet(
                os.path.join(DataManagerTeams.BASE_DIR, "data/archivos_parquet/KPI_equipos.parquet")
            ).select(columnas_KPI).with_columns(
                pl.col('match_id').cast(pl.Utf8)
            ).filter(pl.col('match_id').is_in(match_ids)).collect()

            # Convertir a pandas con tipos optimizados
            pdf_equipos = df_equipos.to_pandas()
            pdf_estadisticas = df_estadisticas.to_pandas()
            pdf_KPI = df_KPI.to_pandas()

            # Si 'equipo' no está en estadísticas, combinar por match_id y season_id solamente
            if has_equipo:
                df_combinado = pd.merge(pdf_equipos, pdf_estadisticas, on=['match_id', 'season_id', 'equipo'])
            else:
                # Solo unimos por match_id y season_id si no hay 'equipo'
                df_combinado = pd.merge(pdf_equipos, pdf_estadisticas, on=['match_id', 'season_id'])
            
            # Unimos con KPI que sí sabemos que tiene 'equipo'
            df_final = pd.merge(df_combinado, pdf_KPI, on=['match_id', 'season_id', 'equipo'])
            
            return df_final
            
        except Exception as e:
            st.error(f"Error al cargar los datos detallados: {e}")
            return None

    @staticmethod
    @st.cache_data(ttl=3600)
    def get_lineup_data():
        """Carga los datos de alineaciones y los cachea."""
        try:
            # Obtener los IDs de partidos
            df_base = pl.scan_parquet(
                os.path.join(DataManagerTeams.BASE_DIR, "data/archivos_parquet/eventos_metricas_alaves.parquet")
            ).select(['match_id']).with_columns(
                pl.col('match_id').cast(pl.Utf8)
            ).collect()
            match_ids = df_base.get_column('match_id').unique()
            del df_base
            
            # Cargar datos de alineaciones
            columnas_min = [
                'id', 'match_id', 'season_id', 'team_id', 'player_id', 
                'position_name', 'back_number', 'player_name', 'player_last_name', 
                'is_starting_lineup', 'position_x', 'position_y'
            ]
            
            df_alineaciones = pl.scan_parquet(
                os.path.join(DataManagerTeams.BASE_DIR, "data/archivos_parquet/lineups_league_all.parquet")
            ).select(columnas_min).with_columns(
                pl.col('match_id').cast(pl.Utf8)
            ).filter(pl.col('match_id').is_in(match_ids)).collect()
            
            # Convertir a Pandas y optimizar tipos de datos
            df_alineaciones = df_alineaciones.to_pandas()
            
            for col in ['match_id', 'season_id', 'team_id', 'player_id']:
                df_alineaciones[col] = df_alineaciones[col].astype('string')
            
            for col in ['position_x', 'position_y']:
                df_alineaciones[col] = df_alineaciones[col].astype('float32')
            
            return df_alineaciones
            
        except Exception as e:
            st.error(f"Error al cargar los datos de alineaciones: {e}")
            return None

    @staticmethod
    def get_equipos(df_combinado):
        """Obtiene la lista de equipos únicos."""
        return df_combinado['equipo'].unique()

    @staticmethod
    def get_temporadas(df_combinado):
        """Obtiene la lista de temporadas únicas, ordenadas de forma descendente."""
        return sorted(df_combinado['temporada'].unique(), reverse=True)

    @staticmethod
    def filtrar_datos(df_combinado, equipo, temporada):
        """Filtra los datos por equipo y temporada."""
        if temporada != 'Todas':
            df_temporada = df_combinado[df_combinado['temporada'] == temporada]
        else:
            df_temporada = df_combinado.copy()

        match_ids_equipo = df_temporada[df_temporada['equipo'] == equipo]['match_id'].unique()
        df_filtrado = df_temporada[df_temporada['match_id'].isin(match_ids_equipo)].copy()
        
        return df_filtrado