"""Cámara en tiempo real: monitor en vivo (OpenCV) y captura desde el navegador."""
from __future__ import annotations

import time

import cv2 as cv
import streamlit as st

from components import ui
from components.icons import icon
from core import detection, storage

RUN_KEY = "cam_running"
SESSION_KEY = "cam_session"
FOUND_KEY = "cam_found"
CONFIRM_FRAMES = 3  # lecturas mínimas para considerar un dorsal "confirmado"

DEFAULT_CAMERAS = {0: "Cámara web integrada · índice 0", 1: "Cámara externa · índice 1"}


def _new_session(cam_id: int) -> dict:
    return {"cam": cam_id, "started": time.time(), "frames": 0, "frames_with_bibs": 0,
            "counts": {}, "unreadable": 0, "fps": 0.0, "last": [], "error": None}


def _start(cam_id_key: str) -> None:
    st.session_state[RUN_KEY] = True
    st.session_state[SESSION_KEY] = _new_session(st.session_state[cam_id_key])


def _stop() -> None:
    was_running = st.session_state.get(RUN_KEY)
    st.session_state[RUN_KEY] = False
    sess = st.session_state.get(SESSION_KEY)
    if was_running and sess and sess["frames"]:
        storage.log_event("webcam", camera=sess["cam"], frames=sess["frames"],
                          frames_with_bibs=sess["frames_with_bibs"],
                          unique_bibs=len(sess["counts"]),
                          numbers=sorted(sess["counts"], key=lambda n: -sess["counts"][n])[:50],
                          duration_s=round(time.time() - sess["started"], 1))


def _scan_cameras(max_index: int = 5) -> dict[int, str]:
    found = {}
    for i in range(max_index):
        cap = cv.VideoCapture(i)
        ok = cap.isOpened() and cap.read()[0]
        if ok:
            w, h = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            found[i] = f"Cámara {i} · {w}×{h}"
        cap.release()
    return found


def _session_panel(sess: dict | None, target) -> None:
    if not sess or not sess["frames"]:
        with target.container():
            ui.section_title("Sesión", "activity")
            ui.empty_state("Sin datos de sesión",
                           "Inicia la cámara para contar los dorsales leídos.", "webcam",
                           compact=True)
        return
    counts = sess["counts"]
    confirmed = [n for n, c in counts.items() if c >= CONFIRM_FRAMES]
    ordered = sorted(counts, key=lambda n: (-counts[n], n))
    chips = "".join(
        f'<span class="br-chip"><span class="br-chip__num">{n}</span>'
        f'<span class="br-chip__meta">×{counts[n]}</span></span>' for n in ordered[:24])
    last = sess["last"]
    last_html = "".join(ui.detection_card(i, n, c, b) for i, (n, c, b) in enumerate(last[:3], 1))
    elapsed = time.time() - sess["started"]
    with target.container():
        ui.section_title("Sesión", "activity", ui.fmt_duration(elapsed))
        ui.render(f"""
        <div class="br-grid br-grid--2" style="margin-bottom:1rem">
          {ui.metric_card("Confirmados", str(len(confirmed)), "check", f"≥{CONFIRM_FRAMES} lecturas", accent=bool(confirmed))}
          {ui.metric_card("Únicos leídos", str(len(counts)), "hash", f"{sess['unreadable']} sin lectura")}
        </div>""")
        ui.kv_list([("Fotogramas", f"{sess['frames']}"),
                    ("Con dorsales", f"{sess['frames_with_bibs']}"),
                    ("Rendimiento", f"{sess['fps']:.1f} fps")])
        if chips:
            ui.section_title("Dorsales leídos", "hash")
            ui.render(f'<div class="br-chips">{chips}</div>')
            ui.spacer(10)
        if last_html:
            ui.section_title("Último fotograma", "scan")
            ui.render(last_html)


def _live_monitor() -> None:
    running = bool(st.session_state.get(RUN_KEY))
    cams = st.session_state.get(FOUND_KEY) or DEFAULT_CAMERAS

    c1, c2, c3, c4 = st.columns([4, 2, 2, 2], gap="small")
    with c1:
        cam_key = "cam_select"
        if st.session_state.get(cam_key) not in cams:
            st.session_state[cam_key] = next(iter(cams))
        st.selectbox("Cámara", list(cams), key=cam_key, format_func=lambda i: cams[i],
                     disabled=running, label_visibility="collapsed")
    with c2:
        if st.button("Buscar cámaras", use_container_width=True, disabled=running):
            with st.spinner("Buscando cámaras…"):
                found = _scan_cameras()
            st.session_state[FOUND_KEY] = found or None
            if not found:
                st.session_state["cam_scan_msg"] = "No se detectó ninguna cámara disponible."
            st.rerun()
    with c3:
        st.button("Iniciar cámara", type="primary", use_container_width=True,
                  disabled=running or not detection.model_status()["ok"],
                  on_click=_start, args=(cam_key,))
    with c4:
        st.button("Detener cámara", use_container_width=True, disabled=not running,
                  on_click=_stop)

    if msg := st.session_state.pop("cam_scan_msg", None):
        ui.alert(msg, "warning")

    left, right = st.columns([3, 1.25], gap="large")
    with left:
        ui.render(f"""<div class="br-stage__bar"><span class="br-stage__title">
            {icon('webcam', 16)} {ui.esc(cams.get(st.session_state['cam_select'], 'Cámara'))}</span>
            {ui.live_badge(running)}</div>""")
        frame_slot = st.empty()
    with right:
        panel = st.empty()

    sess = st.session_state.get(SESSION_KEY)
    if not running:
        if sess and sess.get("error"):
            ui.alert(sess["error"], "error", "Error de cámara", target=frame_slot)
        else:
            ui.render(f"""<div class="br-stage__placeholder">{icon('webcam', 30, 1.6)}
                <span>{'Sesión finalizada' if sess and sess['frames'] else 'La cámara está detenida'}</span>
                </div>""", frame_slot)
        _session_panel(sess, panel)
        return

    cap = cv.VideoCapture(sess["cam"])
    if not cap.isOpened():
        cap.release()
        sess["error"] = (f"No se pudo abrir la cámara {sess['cam']}. Comprueba que esté conectada "
                         "y que ninguna otra aplicación la esté usando.")
        _stop()
        st.rerun()

    last_panel = 0.0
    t_prev = time.perf_counter()
    try:
        while st.session_state.get(RUN_KEY):
            ret, frame = cap.read()
            if not ret:
                sess["error"] = "Se perdió la señal de la cámara."
                _stop()
                break
            result = detection.detect(frame)
            annotated = detection.draw_detections(frame, result)

            sess["frames"] += 1
            if result.bibs:
                sess["frames_with_bibs"] += 1
                sess["last"] = [(b.number, b.confidence, b.box) for b in result.bibs]
            for b in result.bibs:
                if b.readable:
                    sess["counts"][b.number] = sess["counts"].get(b.number, 0) + 1
                else:
                    sess["unreadable"] += 1
            now = time.perf_counter()
            inst = 1.0 / max(now - t_prev, 1e-6)
            sess["fps"] = inst if sess["fps"] == 0 else 0.85 * sess["fps"] + 0.15 * inst
            t_prev = now

            ui.image(detection.resize_max(annotated, 1280), target=frame_slot, channels="BGR")
            if now - last_panel > 0.6:
                last_panel = now
                _session_panel(sess, panel)
    finally:
        cap.release()
    st.rerun()


def _snapshot() -> None:
    ui.alert("Usa la cámara del dispositivo desde el que abres la aplicación. Toma una foto "
             "y se analizará con el mismo modelo.", "info")
    shot = st.camera_input("Tomar foto", label_visibility="collapsed", key="cam_snapshot")
    if shot is None:
        return
    img = detection.decode_image(shot.getvalue())
    if img is None:
        ui.error_box("No se pudo leer la captura", "Vuelve a intentarlo.")
        return
    try:
        with st.spinner("Analizando captura…"):
            result = detection.detect(img)
    except Exception as exc:  # noqa: BLE001
        ui.error_box("La detección falló", exc)
        return
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        ui.image(detection.to_rgb(detection.draw_detections(img, result)))
    with c2:
        ui.section_title("Resultados", "target", f"{result.elapsed:.2f}s")
        if result.bibs:
            ui.render("".join(ui.detection_card(i, b.number, b.confidence, b.box)
                              for i, b in enumerate(result.bibs, 1)))
        else:
            ui.empty_state("Sin dorsales en la captura", icon_name="scan", compact=True)


def render() -> None:
    running = bool(st.session_state.get(RUN_KEY))
    ui.page_header(
        "Monitorización", "Cámara en tiempo real",
        "Detección continua sobre una cámara conectada al equipo que ejecuta la aplicación.",
        "webcam", ui.live_badge(running),
    )
    t_live, t_snap = st.tabs(["Monitor en vivo", "Captura desde el navegador"])
    # La pestaña de captura se ejecuta primero: el monitor en vivo contiene un
    # bucle que mantiene ocupado el script mientras la cámara está activa.
    with t_snap:
        _snapshot()
    with t_live:
        _live_monitor()
