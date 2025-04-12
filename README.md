# Proyecto de Análisis de Datos - Academia del Deportivo Alavés

Este proyecto contiene un análisis exhaustivo de datos de la Academia del Deportivo Alavés, utilizando la API de BePro para recopilar información desde marzo de 2024 hasta abril de 2025.

## Descripción

El proyecto está enfocado en el análisis de rendimiento de los diferentes equipos de la cantera del Deportivo Alavés, incluyendo:
- División de Honor Juvenil
- Liga Nacional Juvenil
- Liga Vasca Cadete
- Copa Vasca Cadete
- División Honor Araba
- Segunda RFEF
- Tercera RFEF

Mediante el uso de la API de BePro, se han extraído datos para generar métricas de rendimiento (KPIs) tanto a nivel de equipo como de jugador individual.

## Estructura del proyecto

- `/data`: Contiene los archivos de datos en formato parquet
- `/Documentacion`: Notebooks y documentación del proyecto
- `/assets`: Recursos gráficos
- `/common`: Módulos comunes
- `/models`: Modelos de datos
- `/pages`: Páginas de la aplicación Streamlit

## Tecnologías utilizadas

- Python
- Pandas
- Streamlit
- API BePro
- Git

## Puesta en marcha

Para ejecutar la aplicación, sigue estos pasos:

1. Clona el repositorio:
```
git clone https://github.com/VictorZandalinas79/TFM_Zandalinas_Victor.git
```

2. Instala las dependencias:
```
pip install -r requirements.txt
```

3. Ejecuta la aplicación:
```
streamlit run home.py
```

## Autor

Victor Zandalinas

## Licencia

Este proyecto es propiedad del Deportivo Alavés y su uso está restringido.