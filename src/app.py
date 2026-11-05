"""Bib Running Detection — punto de entrada de la aplicación Streamlit.

Ejecutar desde la raíz del proyecto:   streamlit run src/app.py
o desde la carpeta src:                streamlit run app.py
"""
import sys
from pathlib import Path

import streamlit as st

# Garantiza que los módulos de src/ se importen sea cual sea el directorio de trabajo
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core import config  # noqa: E402

_favicon = config.ASSETS_DIR / "images" / "favicon.png"
st.set_page_config(
    page_title="Bib Running Detection",
    page_icon=str(_favicon) if _favicon.exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

from components.sidebar import render_sidebar  # noqa: E402
from views import (dashboard, gallery, image_detection, model_evaluation,  # noqa: E402
                   video_detection, webcam)


def load_css(path: Path) -> None:
    try:
        css = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


ROUTES = {
    "Dashboard": dashboard.render,
    "Detección de imágenes": image_detection.render,
    "Procesamiento de videos": video_detection.render,
    "Cámara en tiempo real": webcam.render,
    "Galería de detecciones": gallery.render,
    "Evaluación del modelo": model_evaluation.render,
}

load_css(config.CSS_PATH)
page = render_sidebar()
ROUTES[page]()
