import streamlit as st
import pandas as pd
import os
import common.menu as menu

@st.cache_data
def load_users_data():
    """Carga los datos de usuarios desde el CSV
    
    Returns:
        pd.DataFrame: DataFrame con los datos de usuarios
    """
    try:
        df = pd.read_csv('data/usuarios.csv')
        return df
    except Exception as e:
        st.error(f"Error cargando datos de usuarios: {e}")
        return None

def validarUsuario(usuario, clave):
    """Permite la validación de usuario y clave
    
    Args:
        usuario (str): usuario a validar
        clave (str): clave del usuario
    
    Returns:
        bool: True usuario valido, False usuario invalido
    """
    try:
        df = load_users_data()
        if df is None:
            return False
        
        # Validación con tu estructura específica de CSV
        result = df.loc[(df['usuario'] == usuario) & (df['clave'] == clave)]
        return len(result) > 0
    except Exception as e:
        st.error(f"Error en validación: {e}")
        return False

def generarLogin():
    """Genera la ventana de login o muestra el menú si el login es valido"""

    # Si no está autenticado, inyecta CSS para ocultar el sidebar


    if 'usuario' not in st.session_state:
        hide_sidebar_style = """
        <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        button[aria-label="Toggle sidebar"] { display: none; }
        </style>
        """
        st.markdown(hide_sidebar_style, unsafe_allow_html=True)

    if 'usuario' in st.session_state:
        menu.generarMenu(st.session_state['usuario'])
    else:
        st.markdown('<h1 class="main-header">Log In</h1>', unsafe_allow_html=True)
        cols = st.columns(3)
        with cols[1]:
            with st.form('frmLogin'):
                parUsuario = st.text_input('Usuario')
                parPassword = st.text_input('Password', type='password')
                btnLogin = st.form_submit_button('Ingresar', type='primary')
                
                if btnLogin:
                    if validarUsuario(parUsuario, parPassword):
                        st.session_state['usuario'] = parUsuario
                        st.rerun()
                    else:
                        st.error("Usuario o clave inválidos", icon="⚠️")