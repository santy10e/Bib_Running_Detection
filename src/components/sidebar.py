"""Sidebar persistente: identidad, navegación y estado del sistema."""
from __future__ import annotations

import streamlit as st

from components import ui
from components.icons import LOGO_SVG, NAV_ICON, icon, icon_data_uri
from core import config
from core.detection import model_status

PAGES = list(NAV_ICON.keys())
NAV_KEY = "nav"


def go_to(page: str) -> None:
    """Callback para botones de acceso rápido (se ejecuta antes del rerun)."""
    st.session_state[NAV_KEY] = page


def _nav_icon_css() -> str:
    rules = []
    for i, page in enumerate(PAGES, start=1):
        uri = icon_data_uri(NAV_ICON[page])
        rules.append(
            f'[data-testid="stSidebar"] [role="radiogroup"] > label:nth-of-type({i})::before'
            f'{{-webkit-mask-image:url("{uri}");mask-image:url("{uri}")}}')
    return "<style>" + "".join(rules) + "</style>"


def render_sidebar() -> str:
    if st.session_state.get(NAV_KEY) not in PAGES:
        st.session_state[NAV_KEY] = PAGES[0]

    with st.sidebar:
        ui.render(f"""
        <div class="br-brand">
          {LOGO_SVG}
          <div class="br-brand__text">
            <div class="br-brand__name">BIB RUNNING<br>DETECTION</div>
            <div class="br-brand__sub">{config.APP_SUBTITLE}</div>
          </div>
        </div>
        <div class="br-nav-label">Navegación</div>
        """)
        st.markdown(_nav_icon_css(), unsafe_allow_html=True)
        page = st.radio("Navegación", PAGES, key=NAV_KEY, label_visibility="collapsed")

        status = model_status()
        variant = "success" if status["ok"] else "error"
        model_row = ""
        if status["ok"]:
            model_row = f"""
            <div class="br-sys__row">{icon('cpu', 14)}<span>YOLOv4-tiny</span></div>
            <div class="br-sys__row br-sys__row--muted">RBNR (dorsal) + SVHN (dígitos)</div>"""
        ui.render(f"""
        <div class="br-sys">
          <div class="br-sys__title">Estado del sistema</div>
          <div class="br-sys__status">{ui.badge(status['label'], variant)}</div>
          {model_row}
          <div class="br-sys__version">v{config.APP_VERSION}</div>
        </div>
        """)
    return page
