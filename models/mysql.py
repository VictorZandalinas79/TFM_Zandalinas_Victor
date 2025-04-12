import mysql.connector
import os
from dotenv import load_dotenv

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Función para obtener una conexión a la base de datos MySQL
def get_connection():
    """Establece y devuelve una conexión a la base de datos MySQL."""
    conn = mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE")
    )
    return conn

# Función para ejecutar una consulta
def execute_query(query, params=()):
    """Ejecuta una consulta y devuelve los resultados si corresponde."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(query, params)
    results = cursor.fetchall()
    conn.commit()
    conn.close()
    return results
