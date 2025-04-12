import streamlit as st 
import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq

class DataManagerMatch:
    @staticmethod
    def get_column_dtypes():
        return {
            'match_id': 'int32',
            'team_id': 'float32',  # Cambiado de int32 a float32
            'jornada': 'int8',
            'tipo_evento': 'category',
            'resultado': 'category',
            'temporada': 'category',
            'liga': 'category',
            'equipo': 'category',
            'demarcacion': 'category',
            'centro': 'category',
            'jugador': 'category',
            'partido': 'string',
            'xstart': 'float32',
            'ystart': 'float32',
            'xend': 'float32',
            'yend': 'float32'
        }
    @staticmethod
    def get_required_columns():
        return [
            'match_id', 'team_id', 'partido', 'liga', 'jornada', 'temporada',  'demarcacion', 'jugador',
            'equipo', 'tipo_evento', 'resultado', 'xstart', 'ystart', 'xend', 'yend', 'centro',
            'player_id', 'dorsal', 'angulo_pase', 'xg',
            'pases_progresivos_inicio', 'pases_progresivos_creacion', 
            'pases_progresivos_finalizacion',
            'duelos_aereos_ganados_zona_area', 'duelos_aereos_ganados_zona_baja',
            'duelos_aereos_ganados_zona_media', 'duelos_aereos_ganados_zona_alta',
            'pases_largos_exitosos', 'recuperaciones_zona_baja',
            'recuperaciones_zona_media', 'recuperaciones_zona_alta',
            'entradas_ganadas_zona_baja', 'entradas_ganadas_zona_media'
        ]

    @staticmethod
    def get_kpi_columns():
        return [
            'KPI_construccion_ataque', 'KPI_progresion', 'KPI_habilidad_individual',
            'KPI_peligro_generado', 'KPI_finalizacion', 'KPI_eficacia_defensiva',
            'KPI_juego_aereo', 'KPI_capacidad_recuperacion', 'KPI_posicionamiento_tactico',
            'KPI_rendimiento'
        ]

    @staticmethod
    @st.cache_data(ttl=1800)
    def get_match_data():
        try:
            BASE_DIR = Path(__file__).parent
            DATA_DIR = BASE_DIR / "data" / "archivos_parquet"
            
            df_eventos = pd.read_parquet(
                DATA_DIR / "eventos_metricas_alaves.parquet",
                columns=DataManagerMatch.get_required_columns()
            )
            
            # Procesar jornada antes de asignar tipos
            df_eventos['jornada'] = df_eventos['jornada'].astype(str).str.extract(r'(\d+)')[0]
            df_eventos['jornada'] = pd.to_numeric(df_eventos['jornada'], errors='coerce').fillna(0).astype('int8')
            
            # Ahora sí asignar tipos de datos
            dtype_dict = DataManagerMatch.get_column_dtypes()
            for col, dtype in dtype_dict.items():
                if col in df_eventos.columns:
                    try:
                        df_eventos[col] = df_eventos[col].astype(dtype)
                    except Exception as e:
                        print(f"Error convirtiendo columna {col} a {dtype}: {e}")
            
            df_eventos['partido'] = df_eventos['jornada'].astype(str) + " - " + df_eventos['partido']
            
            # Cargar estadísticas
            df_estadisticas = pd.read_parquet(
                DATA_DIR / "eventos_datos_acumulados.parquet",
                columns=['match_id', 'team_id', 'equipo'] + DataManagerMatch.get_kpi_columns()
            )
            
            df_eventos.set_index(['match_id', 'team_id'], drop=False, inplace=True)
            df_estadisticas.set_index(['match_id', 'team_id'], drop=False, inplace=True)

            return df_eventos, df_estadisticas

        except Exception as e:
            print(f"Error cargando datos: {e}")
            return None, None

    @staticmethod
    def get_equipos(df_eventos):
        """Obtiene equipos filtrados por Alavés"""
        if df_eventos is None:
            return []
        return df_eventos[df_eventos['equipo'].str.contains('Alav', case=False, na=False)]['equipo'].unique()

    @staticmethod
    def get_temporadas(df_eventos, equipo=None):
        """Obtiene temporadas con filtro opcional por equipo"""
        if df_eventos is None:
            return []
        if equipo:
            df_eventos = df_eventos[df_eventos['equipo'] == equipo]
        return sorted(df_eventos['temporada'].unique(), reverse=True)

    @staticmethod
    def get_partidos(df_eventos, temporada, equipo):
        if df_eventos is None:
            return pd.DataFrame()
            
        df_temp = df_eventos[
            (df_eventos['temporada'] == temporada) & 
            (df_eventos['equipo'] == equipo)
        ].copy()
        
        df_temp['numero'] = df_temp['partido'].str.extract(r'(\d+)').astype(int)  # Escape sequence corregido
        df_temp = df_temp.sort_values('numero')
        df_temp['partido_completo'] = df_temp.apply(
            lambda x: f"{x['partido']} ({x['jornada']})", 
            axis=1
        )
        return df_temp['partido_completo'].unique()

    @staticmethod
    def get_partido_data(match_id, df_eventos, df_estadisticas):
        """Obtiene datos específicos de un partido"""
        if df_eventos is None or df_estadisticas is None:
            return None, None
            
        eventos_partido = df_eventos[df_eventos['match_id'] == match_id].copy()
        stats_partido = df_estadisticas[df_estadisticas['match_id'] == match_id].copy()
        
        return eventos_partido, stats_partido