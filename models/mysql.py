import os
from dotenv import load_dotenv

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Función para obtener una conexión a la base de datos MySQL
def get_connection():
    """Establece y devuelve una conexión a la base de datos MySQL."""
    # Retornar None en lugar de intentar conectarse
    print("Conexión a MySQL deshabilitada")
    return None

# Función para ejecutar una consulta
def execute_query(query, params=()):
    """Ejecuta una consulta y devuelve los resultados si corresponde."""
    # Devuelve una lista vacía en lugar de ejecutar la consulta
    print("Ejecución de consulta deshabilitada")
    return []