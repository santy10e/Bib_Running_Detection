"""Dashboard: centro de operaciones con datos reales del proyecto."""
from __future__ import annotations

import datetime as dt

import cv2 as cv
import streamlit as st

from components import ui
from components.icons import icon
from components.sidebar import go_to
from core import config, storage
from core.detection import THRESHOLDS, model_files, model_status

QUICK_ACTIONS = [
    ("scan", "Detección de imágenes", "Analiza una fotografía y extrae los dorsales."),
    ("film", "Procesamiento de videos", "Procesa un MP4 fotograma a fotograma."),
    ("webcam", "Cámara en tiempo real", "Monitoriza una cámara conectada en vivo."),
    ("images", "Galería de detecciones", "Consulta y filtra los resultados guardados."),
    ("chart", "Evaluación del modelo", "Mide el rendimiento sobre el dataset RBNR."),
]


def _fmt_ts(value: str | dt.datetime) -> str:
    ts = dt.datetime.fromisoformat(value) if isinstance(value, str) else value
    return ts.strftime("%d/%m/%y · %H:%M")


def _event_item(e: dict) -> dict:
    kind = e.get("type")
    when = _fmt_ts(e["ts"])
    if kind == "image":
        n = e.get("bibs", 0)
        nums = ", ".join(str(x) for x in e.get("numbers", [])[:4])
        title = (f"Imagen analizada · <b>{n} dorsal{'es' if n != 1 else ''}</b>"
                 + (f" ({ui.esc(nums)})" if nums else "")) if n else "Imagen analizada · sin dorsales"
        return {"icon": "scan", "title": title, "variant": "lime" if n else "neutral",
                "meta": f"{when} · {e.get('elapsed', 0):.2f} s"}
    if kind == "video":
        u = e.get("unique_bibs", 0)
        return {"icon": "film", "variant": "cyan",
                "title": f"Video procesado · <b>{u} dorsal{'es' if u != 1 else ''} único{'s' if u != 1 else ''}</b>",
                "meta": f"{when} · {e.get('frames_processed', 0)} fotogramas · {ui.esc(e.get('file', ''))}"}
    if kind == "webcam":
        u = e.get("unique_bibs", 0)
        return {"icon": "webcam", "variant": "cyan",
                "title": f"Sesión de cámara · <b>{u} dorsal{'es' if u != 1 else ''}</b>",
                "meta": f"{when} · {e.get('frames', 0)} fotogramas"}
    if kind == "evaluation":
        acc = e.get("accuracy")
        txt = f"exactitud <b>{acc * 100:.1f}%</b>" if acc is not None else "completada"
        return {"icon": "chart", "variant": "warning", "title": f"Evaluación del modelo · {txt}",
                "meta": f"{when} · {e.get('images', 0)} imágenes"}
    return {"icon": "activity", "title": ui.esc(kind or "Evento"), "meta": when}


def _metrics(stats: dict) -> None:
    since = (f"Registro desde {_fmt_ts(stats['since'])[:8]}" if stats["since"]
             else "El registro comienza con el primer análisis")
    last_eval = stats["last_eval"]
    acc_value = acc_foot = None
    if last_eval and last_eval.get("accuracy") is not None:
        acc_value = f"{last_eval['accuracy'] * 100:.1f}%"
        acc_foot = f"Exactitud · evaluación del {_fmt_ts(last_eval['ts'])[:8]} ({last_eval.get('images', 0)} imágenes)"
    ui.metric_row([
        ui.metric_card("Imágenes analizadas", str(stats["images"]), "image",
                       f"{stats['gallery']} con resultado en galería · {since}"),
        ui.metric_card("Dorsales detectados", str(stats["bibs"]), "hash",
                       "Total en imágenes analizadas", accent=stats["bibs"] > 0),
        ui.metric_card("Videos analizados", str(stats["videos"]), "film",
                       "Procesados completos o parciales"),
        ui.metric_card("Precisión del modelo", acc_value, "target", acc_foot or "",
                       accent=True, empty_note="Sin evaluación validada todavía"),
    ])


def _quick_actions() -> None:
    ui.section_title("Accesos rápidos", "zap")
    cols = st.columns(len(QUICK_ACTIONS), gap="small")
    for col, (ic, title, text) in zip(cols, QUICK_ACTIONS):
        with col, st.container(border=True):
            ui.render(f"""{ui.CARD_MARK}
            <div class="br-quick">
              <div class="br-quick__icon">{icon(ic, 20)}</div>
              <div class="br-quick__title">{title}</div>
              <div class="br-quick__text">{text}</div>
            </div>""")
            st.button("Abrir", key=f"qa_{ic}", on_click=go_to, args=(title,),
                      use_container_width=True)


def _latest_detections() -> None:
    items = storage.list_gallery()
    ui.section_title("Últimas detecciones", "images",
                     f"{len(items)} en galería" if items else "")
    if not items:
        ui.empty_state("Todavía no hay detecciones guardadas",
                       "Las imágenes procesadas con dorsales aparecerán aquí.", "images", compact=True)
        return
    cards = []
    for it in items[:4]:
        b64 = storage.thumbnail_b64(str(it.path), storage.mtime(it.path))
        cards.append(f"""
        <div class="br-thumb">
          <div class="br-thumb__img" style="background-image:url('data:image/jpeg;base64,{b64}')"
               role="img" aria-label="Detección dorsal {it.bib}"></div>
          <div class="br-thumb__body">
            <span class="br-thumb__bib">#{it.bib}</span>
            <span class="br-thumb__date">{it.timestamp:%d/%m/%y · %H:%M}</span>
          </div>
        </div>""")
    ui.render(f'<div class="br-thumbs">{"".join(cards)}</div>')
    ui.spacer(10)
    st.button("Ver galería completa", key="dash_gallery", on_click=go_to,
              args=("Galería de detecciones",))


def _recent_activity() -> None:
    events = storage.read_events()
    ui.section_title("Actividad reciente", "activity")
    if events:
        ui.timeline([_event_item(e) for e in reversed(events[-6:])])
        return
    gallery = storage.list_gallery()[:6]
    if gallery:
        ui.timeline([{"icon": "check", "variant": "lime",
                      "title": f"Detección guardada · <b>dorsal {g.bib}</b>",
                      "meta": _fmt_ts(g.timestamp)} for g in gallery])
        ui.render('<div class="br-help" style="margin-top:.5rem">Historial derivado de la galería. '
                  'Los nuevos análisis se registran automáticamente.</div>')
    else:
        ui.empty_state("Sin actividad registrada", "Ejecuta una detección para comenzar.",
                       "activity", compact=True)


def _system_status() -> None:
    ui.section_title("Estado del sistema y modelo", "cpu")
    status = model_status()
    files = model_files()
    ds = storage.validation_dataset()
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        rows = [("Estado", ui.badge(status["label"], "success" if status["ok"] else "error"))]
        rows += [(f"{f['dataset']} · {f['kind']}",
                  (ui.fmt_bytes(f["size"]) if f["exists"] else ui.badge("No encontrado", "error")))
                 for f in files]
        rows.append(("Arquitectura", "YOLOv4-tiny (Darknet) · OpenCV DNN"))
        ui.kv_list(rows)
    with c2:
        ui.kv_list([
            ("Entrada de red", f"{config.MODEL_INPUT_SIZE}×{config.MODEL_INPUT_SIZE}px"),
            ("Umbral detección de dorsal", f"{THRESHOLDS['bib']:.2f}"),
            ("Umbral reconocimiento de dígitos", f"{THRESHOLDS['digit']:.2f}"),
            ("Backend de inferencia", "OpenCV (CPU)"),
            ("OpenCV · Streamlit", f"{cv.__version__} · {st.__version__}"),
            ("Dataset de validación", f"{ds['images']} imágenes · {ds['annotations']} anotaciones"
             if ds["exists"] else "No disponible"),
        ])
    if not status["ok"]:
        ui.alert(status["detail"], "error", "El modelo no está disponible")


def render() -> None:
    hour = dt.datetime.now().hour
    greeting = "Buenos días" if 6 <= hour < 13 else "Buenas tardes" if hour < 20 else "Buenas noches"
    ui.page_header(
        "Centro de operaciones", f"{greeting}, bienvenido",
        "Plataforma de visión artificial para detectar, reconocer y registrar los números "
        "de dorsal de corredores en imágenes, videos y cámaras en directo.",
        "dashboard",
    )
    _metrics(storage.activity_stats())
    _quick_actions()
    ui.spacer(18)
    left, right = st.columns([3, 2], gap="large")
    with left:
        _latest_detections()
    with right:
        _recent_activity()
    ui.spacer(18)
    _system_status()
