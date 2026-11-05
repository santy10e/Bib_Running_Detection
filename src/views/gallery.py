"""Galería de detecciones: tarjetas, filtros por fecha, búsqueda por dorsal y
vista ampliada. Compatible con el almacenamiento original (usr/validation/Full)."""
from __future__ import annotations

import datetime as dt

import cv2 as cv
import streamlit as st

from components import ui
from components.icons import icon
from core import config, storage

PAGE_SIZE = 12
COLS = 4
PAGE_KEY = "gallery_page"

_dialog = getattr(st, "dialog", None) or getattr(st, "experimental_dialog")


@_dialog("Detalle de la detección", width="large")
def _detail(item: storage.GalleryItem) -> None:
    img = cv.imread(str(item.path))
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        if img is None:
            ui.error_box("No se pudo abrir la imagen", item.name)
        else:
            ui.image(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    with c2:
        ui.render(ui.bib_plate(item.bib, "lg"))
        ui.spacer(12)
        rows = [("Dorsal", str(item.bib)),
                ("Fecha", item.timestamp.strftime("%d/%m/%Y")),
                ("Hora", item.timestamp.strftime("%H:%M:%S")),
                ("Archivo", ui.esc(item.name)),
                ("Tamaño", ui.fmt_bytes(item.path.stat().st_size))]
        if img is not None:
            rows.insert(3, ("Resolución", f"{img.shape[1]}×{img.shape[0]}px"))
        ui.kv_list(rows)
        st.download_button("Descargar imagen", item.path.read_bytes(), file_name=item.name,
                           mime="image/jpeg", use_container_width=True)
        if item.bib == 0:
            ui.render('<div class="br-help" style="margin-top:.5rem">Dorsal 0: se detectó un '
                      'dorsal pero no se pudieron leer sus dígitos.</div>')


def _reset_page() -> None:
    st.session_state[PAGE_KEY] = 0


def _filters(items: list[storage.GalleryItem]) -> list[storage.GalleryItem]:
    dates = [i.timestamp.date() for i in items]
    c1, c2, c3, c4 = st.columns([3, 2, 3, 2], gap="small")
    with c1:
        query = st.text_input("Buscar por número de dorsal", placeholder="Ej. 170",
                              key="gallery_query", on_change=_reset_page).strip()
    with c2:
        mode = st.selectbox("Fecha", ["Todas las fechas", "Día concreto", "Rango de fechas"],
                            key="gallery_date_mode", on_change=_reset_page)
    with c3:
        lo, hi = min(dates), max(dates)
        if mode == "Día concreto":
            day = st.date_input("Día", value=hi, min_value=lo, max_value=hi,
                                format="DD/MM/YYYY", key="gallery_day", on_change=_reset_page)
            date_range = (day, day)
        elif mode == "Rango de fechas":
            rng = st.date_input("Rango", value=(lo, hi), min_value=lo, max_value=hi,
                                format="DD/MM/YYYY", key="gallery_range", on_change=_reset_page)
            date_range = (rng[0], rng[1] if len(rng) > 1 else rng[0]) if rng else None
        else:
            st.text_input("Periodo", value=f"{lo:%d/%m/%Y} — {hi:%d/%m/%Y}", disabled=True,
                          key="gallery_period")
            date_range = None
    with c4:
        order = st.selectbox("Orden", ["Más recientes", "Más antiguas", "Dorsal ↑", "Dorsal ↓"],
                             key="gallery_order")

    out = items
    if query:
        if not query.isdigit():
            ui.alert("Introduce solo dígitos para buscar un dorsal.", "warning")
            return []
        out = [i for i in out if query in str(i.bib)]
    if date_range:
        out = [i for i in out if date_range[0] <= i.timestamp.date() <= date_range[1]]
    key = {"Más recientes": (lambda i: i.timestamp, True),
           "Más antiguas": (lambda i: i.timestamp, False),
           "Dorsal ↑": (lambda i: i.bib, False),
           "Dorsal ↓": (lambda i: i.bib, True)}[order]
    return sorted(out, key=key[0], reverse=key[1])


def _card(item: storage.GalleryItem, idx: int) -> None:
    with st.container(border=True):
        thumb = storage.thumbnail_rgb(str(item.path), storage.mtime(item.path))
        if thumb is not None:
            ui.image(thumb)
        ui.render(f"""{ui.CARD_MARK}
        <div class="br-gcard__meta">
          <span class="br-gcard__bib">#{item.bib}</span>
          <span class="br-gcard__date">{icon('calendar', 12)}{item.timestamp:%d/%m/%y · %H:%M}</span>
        </div>""")
        if st.button("Ver detalle", key=f"g_open_{idx}_{item.name}", use_container_width=True):
            _detail(item)


def render() -> None:
    items = storage.list_gallery()
    ui.page_header(
        "Resultados", "Galería de detecciones",
        "Imágenes procesadas y guardadas por la plataforma. Filtra por fecha o busca un dorsal.",
        "images", ui.badge(f"{len(items)} imágenes", "lime" if items else "neutral", dot=False),
    )
    if not items:
        ui.empty_state(
            "La galería está vacía",
            "Cuando ejecutes una detección con resultados se guardará aquí automáticamente "
            f"(carpeta {config.GALLERY_DIR.relative_to(config.PROJECT_ROOT).as_posix()}).",
            "images")
        return

    filtered = _filters(items)
    if not filtered:
        ui.empty_state("Sin resultados", "Ningún resultado coincide con los filtros aplicados.",
                       "search", compact=True)
        return

    pages = (len(filtered) - 1) // PAGE_SIZE + 1
    page = min(st.session_state.get(PAGE_KEY, 0), pages - 1)
    ui.section_title("Resultados", "images",
                     f"{len(filtered)} de {len(items)} · página {page + 1}/{pages}")
    chunk = filtered[page * PAGE_SIZE:(page + 1) * PAGE_SIZE]
    for row_start in range(0, len(chunk), COLS):
        cols = st.columns(COLS, gap="medium")
        for col, (offset, item) in zip(cols, enumerate(chunk[row_start:row_start + COLS])):
            with col:
                _card(item, page * PAGE_SIZE + row_start + offset)

    if pages > 1:
        ui.spacer(8)
        p1, p2, p3 = st.columns([1, 3, 1])
        with p1:
            if st.button("← Anterior", disabled=page == 0, use_container_width=True):
                st.session_state[PAGE_KEY] = page - 1
                st.rerun()
        with p2:
            ui.render(f'<div class="br-help" style="text-align:center;padding-top:.6rem">'
                      f'Página {page + 1} de {pages}</div>')
        with p3:
            if st.button("Siguiente →", disabled=page >= pages - 1, use_container_width=True):
                st.session_state[PAGE_KEY] = page + 1
                st.rerun()
