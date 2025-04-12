#!/usr/bin/env python3
import os
import papermill as pm
import pathlib

# Obtener ruta absoluta del directorio con los notebooks
notebooks_dir = os.path.dirname(os.path.abspath(__file__))

# Lista de notebooks a ejecutar
notebooks = [
   "API Bepro_Alaves_Liga_Nacional_Juvenil_G4_parquet.ipynb",
   "API Bepro_Alaves_Segunda_RFEF_parquet.ipynb", 
   "API Bepro_Alaves_Liga_Vasca_Cadete_parquet.ipynb",
   "API Bepro_Alaves_División_Honor_Araba_parquet.ipynb",
   "API Bepro_Alaves_Copa_Vasca_Cadete_parquet.ipynb",
   "API Bepro_Alaves_DHonorJuvenil_parquet.ipynb",
   "API Bepro_Alaves_Tercera_RFEF_parquet.ipynb",
   "all_datos.ipynb",
   "Eventos5_0.ipynb",
   "KPI_jugador.ipynb",
   "KPI_equipos.ipynb",
   "filtrar_temp2425.ipynb"
]

print(f"Directorio de ejecución: {notebooks_dir}")
print("\nArchivos encontrados:")
for file in os.listdir(notebooks_dir):
   if file.endswith('.ipynb'):
       print(f"  - {file}")

print("\nIniciando procesamiento...")

for notebook in notebooks:
   input_path = os.path.join(notebooks_dir, notebook)
   print(f"\nProcesando: {notebook}")
   
   if not os.path.exists(input_path):
       print(f"Error: {notebook} no existe")
       continue
   
   try:
       pm.execute_notebook(
           input_path,
           input_path,
           parameters=dict(
               alpha=0.6,
               ratio=0.1
           )
       )
       print(f"✓ {notebook} ejecutado con éxito")
   except Exception as e:
       print(f"✗ Error en {notebook}: {str(e)}")

print("\nProceso completado")