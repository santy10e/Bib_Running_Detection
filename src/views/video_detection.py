"""Procesamiento de videos: carga, progreso en vivo, video anotado y resumen."""
from __future__ import annotations

import hashlib
import time

import pandas as pd
import streamlit as st

from components import ui
from components.icons import icon
from core import detection, storage
from core import video as vid

INPUT_KEY = "video_input"
RESULT_KEY = "video_result"
STEP_OPTIONS = [1, 2, 3, 5, 10]


def _prepare_input(uploaded) -> dict | None:
    file_id = f"{uploaded.name}:{uploaded.size}"
    current = st.session_state.get(INPUT_KEY)
    if current and current["file_id"] == file_id and current["path"].exists():
        return current
    digest = hashlib.sha1(file_id.encode()).hexdigest()[:10]
    path = vid.work_path(f"input_{digest}.mp4")
    path.write_bytes(uploaded.getvalue())
    info = vid.probe(path)
    if info is None:
        return None
    current = {"file_id": file_id, "path": path, "info": info, "name": uploaded.name}
    st.session_state[INPUT_KEY] = current
    st.session_state.pop(RESULT_KEY, None)
    return current


def _log_once(state: dict) -> None:
    if state.get("logged"):
        return
    s: vid.VideoSummary = state["summary"]
    storage.log_event("video", file=s.file, frames_total=s.info["frames"],
                      frames_read=s.frames_read, frames_processed=s.frames_analyzed,
                      frames_with_bibs=s.frames_with_bibs, unique_bibs=len(s.tracks),
                      numbers=s.unique_numbers[:50], step=s.step, completed=s.completed,
                      inference_s=round(s.total_time, 2))
    state["logged"] = True


def _process(inp: dict, step: int, stage, progress_slot, status_slot) -> None:
    summary = vid.VideoSummary(file=inp["name"], info=inp["info"], step=step)
    # Se guarda antes de empezar: si el usuario detiene el proceso, el rerun
    # encuentra el resumen parcial (se actualiza en sitio).
    st.session_state[RESULT_KEY] = {"file_id": inp["file_id"], "summary": summary, "logged": False}
    total = inp["info"]["frames"] or 0
    t0 = time.perf_counter()
    last_ui = {"frame": 0.0, "bar": 0.0}
    bar = progress_slot.progress(0.0)

    def on_progress(s: vid.VideoSummary, frame) -> None:
        now = time.perf_counter()
        if now - last_ui["bar"] > 0.15:
            last_ui["bar"] = now
            frac = min(s.frames_read / total, 1.0) if total else 0.0
            fps = s.frames_read / max(now - t0, 1e-6)
            bar.progress(frac)
            status_slot.markdown(
                f'<div class="br-help">Fotograma <b>{s.frames_read}</b> / {total or "?"} · '
                f'<b>{len(s.tracks)}</b> dorsales únicos · {fps:.1f} fps</div>',
                unsafe_allow_html=True)
        if now - last_ui["frame"] > 0.35:
            last_ui["frame"] = now
            ui.image(detection.to_rgb(detection.resize_max(frame, 1280)), target=stage)

    vid.process(inp["path"], summary, on_progress)
    bar.progress(1.0)


def _summary_section(state: dict) -> None:
    s: vid.VideoSummary = state["summary"]
    avg = s.total_time / s.frames_analyzed if s.frames_analyzed else None
    ui.metric_row([
        ui.metric_card("Fotogramas analizados", f"{s.frames_analyzed:,}".replace(",", "."), "layers",
                       f"de {s.frames_read:,} leídos · 1 de cada {s.step}".replace(",", ".")),
        ui.metric_card("Con dorsales", f"{s.frames_with_bibs:,}".replace(",", "."), "scan",
                       f"{(s.frames_with_bibs / s.frames_analyzed * 100) if s.frames_analyzed else 0:.1f}% de los analizados"),
        ui.metric_card("Dorsales únicos", str(len(s.tracks)), "hash",
                       f"{s.unreadable} detecciones sin lectura", accent=bool(s.tracks)),
        ui.metric_card("Inferencia media", f"{avg * 1000:.0f} ms" if avg else None, "timer",
                       "por fotograma analizado"),
    ])
    ui.section_title("Dorsales detectados", "hash",
                     f"{len(s.tracks)} únicos" if s.tracks else "")
    if not s.tracks:
        ui.empty_state("No se reconocieron números de dorsal",
                       "El modelo no leyó ningún número en los fotogramas analizados.",
                       "hash", compact=True)
        return
    items = []
    for n in s.unique_numbers[:48]:
        tr = s.tracks[n]
        items.append({"number": n,
                      "b64": detection.b64_jpg(tr.snapshot) if tr.snapshot is not None else None,
                      "meta": f"{tr.frames} fot. · {ui.fmt_duration(tr.first_t)}–{ui.fmt_duration(tr.last_t)}"})
    ui.track_cards(items)
    ui.render('<div class="br-help" style="margin-top:.6rem">Cada número cuenta los fotogramas '
              'en los que se leyó. Lecturas aisladas (1–2 fotogramas) pueden ser errores de '
              'reconocimiento.</div>')


def render() -> None:
    ui.page_header(
        "Análisis de video", "Procesamiento de videos",
        "Procesa un video MP4 fotograma a fotograma, genera una versión anotada y resume "
        "los dorsales encontrados.", "film",
    )
    uploaded = st.file_uploader("Cargar video", type=["mp4"], key="video_upload",
                                help="Formato admitido: MP4")
    if uploaded is None:
        ui.empty_state("Ningún video cargado",
                       "Arrastra un archivo MP4. Verás la vista previa, el progreso del análisis "
                       "y el video procesado con las detecciones.", "film")
        return

    inp = _prepare_input(uploaded)
    if inp is None:
        ui.error_box("No se pudo abrir el video",
                     "OpenCV no puede leer este archivo. Comprueba que sea un MP4 válido.")
        return
    info = inp["info"]
    state = st.session_state.get(RESULT_KEY)
    if state and state["file_id"] != inp["file_id"]:
        state = None

    left, right = st.columns([7, 4], gap="large")
    with right:
        ui.section_title("Archivo", "file")
        ui.kv_list([
            ("Nombre", ui.esc(inp["name"])), ("Tamaño", ui.fmt_bytes(uploaded.size)),
            ("Duración", ui.fmt_duration(info["duration"])),
            ("Resolución", f"{info['width']}×{info['height']}px"),
            ("Fotogramas", f"{info['frames']:,}".replace(",", ".") + f" · {info['fps']:.1f} fps"),
        ])
        step = st.select_slider(
            "Frecuencia de análisis", STEP_OPTIONS, value=1, key="video_step",
            format_func=lambda v: "Todos los fotogramas" if v == 1 else f"1 de cada {v}",
            help="«Todos» reproduce el comportamiento original. Analizar menos fotogramas acelera "
                 "el proceso; los intermedios reutilizan la última detección.")
        status_ok = detection.model_status()["ok"]
        start = st.button("Iniciar procesamiento" if not state else "Procesar de nuevo",
                          type="primary", use_container_width=True, disabled=not status_ok)
        control_slot = st.empty()
        status_slot = st.empty()

    with left:
        if start:
            ui.render(f"""<div class="br-stage__bar"><span class="br-stage__title">
                {icon('radio', 16)} Procesando en vivo</span>{ui.badge('En curso', 'info')}</div>""")
            progress_slot = st.empty()
            stage = st.empty()
            control_slot.button("Detener procesamiento", use_container_width=True,
                                key="video_stop")
            try:
                _process(inp, step, stage, progress_slot, status_slot)
            except Exception as exc:  # noqa: BLE001
                ui.error_box("El procesamiento falló", exc)
            else:
                _log_once(st.session_state[RESULT_KEY])
                st.rerun()
            return

        if state is None:
            ui.render(f"""<div class="br-stage__bar"><span class="br-stage__title">
                {icon('play', 16)} Vista previa</span>{ui.badge('Sin procesar', 'neutral')}</div>""")
            st.video(uploaded)
        else:
            s: vid.VideoSummary = state["summary"]
            _log_once(state)
            badge = (ui.badge("Completado", "success") if s.completed
                     else ui.badge("Detenido · resultados parciales", "warning"))
            ui.render(f"""<div class="br-stage__bar"><span class="br-stage__title">
                {icon('film', 16)} Resultado</span>{badge}</div>""")
            t1, t2, t3 = st.tabs(["Video procesado", "Detecciones por fotograma", "Video original"])
            with t1:
                out = s.output_path
                if out is not None and out.exists() and out.stat().st_size > 0:
                    data = out.read_bytes()
                    if s.output_playable:
                        st.video(data, format=s.output_mime)
                    else:
                        ui.alert("El códec disponible en esta instalación de OpenCV no se "
                                 "reproduce en el navegador. Descarga el archivo para verlo.",
                                 "warning", "Vista previa no disponible")
                    st.download_button("Descargar video procesado", data,
                                       file_name=out.name, mime=s.output_mime)
                else:
                    ui.alert("OpenCV no pudo crear el video de salida con los códecs disponibles.",
                             "error", "Sin video de salida")
            with t2:
                if s.records:
                    st.dataframe(pd.DataFrame(s.records), use_container_width=True,
                                 hide_index=True, height=380)
                else:
                    ui.empty_state("Ningún fotograma con detecciones", icon_name="layers",
                                   compact=True)
            with t3:
                st.video(uploaded)

    if state is not None and not start:
        ui.spacer(12)
        _summary_section(state)
