import streamlit as st
import models.mysql as db
from PIL import Image
import base64 
import os

def get_base64_of_bin_file(bin_file): 
    with open(bin_file, "rb") as f: data = f.read() 
    return base64.b64encode(data).decode()

def generarMenu(usuario):
    """Genera el menú dependiendo del usuario 
    
    Args:
        usuario (str): usuario utilizado para generar el menú
    """
    st.markdown( 
        """ 
        <style> /* Oculta la navegación multipágina automática / [data-testid="stSidebarNav"] { display: none; } / En caso de que la lista esté dentro de un <ul>, también se oculta */ 
        [data-testid="stSidebarNav"] ul { display: none; } 
        </style> 
        """, unsafe_allow_html=True, )


    with st.sidebar:

        sidebar_bg_path = os.path.join("assets", "logo_menuu.png") 
        sidebar_bg = get_base64_of_bin_file(sidebar_bg_path)
        st.markdown( f""" 
                    <style> [data-testid="stSidebar"] {{ background-image: url("data:image/png;base64,{sidebar_bg}"); 
                    background-size: cover; background-position: center; }} 
                    </style> 
                    """, unsafe_allow_html=True )
        # st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True) 
        # logo_path = os.path.join('assets', 'logo_menu.png') 
        # if os.path.exists(logo_path): 
        #     image = Image.open(logo_path) 
        #     st.image(image, width=100,use_container_width=True) 
        #     st.markdown("</div>", unsafe_allow_html=True)

        # Cargar y mostrar el logo
        # logo_path = os.path.join('assets', 'escudo_alaves_original.png')
        # if os.path.exists(logo_path):
        #     image = Image.open(logo_path)
        #     st.image(image, width=100)  # Corregido el parámetro
        
        # st.divider()  # Línea separadora después del logo
        
        # Consultar la base de datos MySQL para obtener el nombre del usuario
        query = "SELECT nombre FROM usuarios WHERE usuario = %s"
        params = (usuario,)
        result = db.execute_query(query, params)
        
        # Verificar si se encontró el usuario
        if result:
            nombre = result[0][0]  # Tomar el nombre del resultado de la consulta
            # Mostrar el nombre del usuario
            st.write(f"Hola **:blue-background[{nombre}]** ")
        else:
            st.error("Usuario no encontrado en la base de datos.")
        
        st.divider()  # Línea separadora después del saludo
        
        # Mostrar los enlaces de páginas
        st.page_link("home.py", label="Home", icon="🏠")
        
        st.subheader("Tableros")
        st.page_link("pages/_pagina1.py", label="Equipos", icon="⚽")
        st.page_link("pages/_pagina2.py", label="Jugadores", icon="👥")
        st.page_link("pages/_pagina3.py", label="Partidos", icon="📊")  # Cambiado el icono
        
        st.divider()  # Línea separadora antes del botón de salir
        
        # Botón para cerrar la sesión
        btnSalir = st.button("Salir", type="primary")
        if btnSalir:
            st.session_state.clear()
            # Luego de borrar el Session State reiniciamos la app para mostrar la opción de usuario y clave
            st.rerun()