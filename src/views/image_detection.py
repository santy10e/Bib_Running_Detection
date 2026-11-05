"""Detección de imágenes: carga, ejecución, comparación y resultados."""
from __future__ import annotations

import streamlit as st

from components import ui, viewers
from components.icons import icon
from core import detection, storage

STATE_KEY = "img_result"


def _file_id(uploaded) -> str:
    return f"{uploaded.name}:{uploaded.size}"


def _run(img, file_id: str, name: str, save: bool) -> None:
    result = detection.detect(img)
    annotated = detection.draw_detections(img, result)
    saved_path = None
    if result.bibs and save:
        # Mismo criterio que la versión original: se guarda cuando hay
        # detecciones, con el número del primer dorsal en el nombre.
        saved_path = storage.save_detection(annotated, result.bibs[0].number)
    storage.log_event("image", file=name, bibs=len(result.bibs), numbers=result.numbers,
                      elapsed=round(result.elapsed, 3),
                      saved=saved_path.name if saved_path else None)
    st.session_state[STATE_KEY] = {
        "file_id": file_id, "result": result, "annotated": annotated,
        "saved": saved_path.name if saved_path else None,
    }


def _summary(result: detection.DetectionResult, saved: str | None) -> None:
    bibs = result.bibs
    if not bibs:
        ui.render(f"""
        <div class="br-hero br-hero--none">
          {ui.bib_plate(None, "md")}
          <div class="br-hero__stats">
            <div><div class="br-stat__k">Dorsales</div><div class="br-stat__v">0</div></div>
            <div><div class="br-stat__k">Inferencia</div><div class="br-stat__v">{result.elapsed:.2f}s</div></div>
          </div>
        </div>""")
        ui.alert("El modelo no encontró dorsales en esta imagen. Prueba con una foto en la que "
                 "el dorsal sea visible y ocupe una zona razonable del encuadre.", "warning",
                 "Sin detecciones")
        return
    readable = [b for b in bibs if b.readable]
    main = max(bibs, key=lambda b: b.box[2] * b.box[3])
    avg_conf = sum(b.confidence for b in bibs) / len(bibs)
    ui.render(f"""
    <div class="br-hero">
      {ui.bib_plate(main.number, "md")}
      <div class="br-hero__stats">
        <div><div class="br-stat__k">Dorsales</div><div class="br-stat__v br-stat__v--lime">{len(bibs)}</div></div>
        <div><div class="br-stat__k">Leídos</div><div class="br-stat__v">{len(readable)}/{len(bibs)}</div></div>
        <div><div class="br-stat__k">Inferencia</div><div class="br-stat__v">{result.elapsed:.2f}s</div></div>
        <div><div class="br-stat__k">Conf. media</div><div class="br-stat__v">{avg_conf * 100:.0f}%</div></div>
      </div>
    </div>""")
    if saved:
        ui.alert(saved, "success", "Guardado en la galería")


def _results_panel(state: dict | None, name: str) -> None:
    ui.section_title("Resultados", "target")
    if state is None:
        ui.empty_state("Listo para analizar",
                       "Pulsa «Ejecutar detección» para localizar los dorsales y leer sus números.",
                       "scan", compact=True)
        return
    result = state["result"]
    _summary(result, state["saved"])
    if result.bibs:
        ui.section_title("Detecciones", "layers", f"{len(result.bibs)} encontradas")
        ui.render("".join(ui.detection_card(i, b.number, b.confidence, b.box)
                          for i, b in enumerate(result.bibs, start=1)))
        stem = name.rsplit(".", 1)[0]
        st.download_button("Descargar imagen procesada",
                           detection.encode_jpg(state["annotated"]),
                           file_name=f"{stem}_deteccion.jpg", mime="image/jpeg",
                           use_container_width=True)
        ui.render('<div class="br-help" style="margin-top:.6rem"><b>Confianza</b>: probabilidad '
                  'que el detector RBNR asigna a la caja del dorsal. El reconocedor de dígitos no '
                  'expone una confianza por número.</div>')


def render() -> None:
    status = detection.model_status()
    ui.page_header(
        "Análisis de imagen", "Detección de imágenes",
        "Carga una fotografía de una carrera: el sistema localiza cada dorsal y reconoce su número.",
        "scan", ui.badge(status["label"], "success" if status["ok"] else "error"),
    )

    uploaded = st.file_uploader("Cargar imagen", type=["jpg", "jpeg"], key="img_upload",
                                help="Formatos admitidos: JPG y JPEG")
    if uploaded is None:
        st.session_state.pop(STATE_KEY, None)
        ui.render(f"""
        <div class="br-help" style="margin:-.25rem 0 1.25rem;display:flex;gap:6px;align-items:center">
          {icon('info', 14)} Arrastra una imagen JPG o JPEG sobre la zona de carga.
        </div>""")
        ui.empty_state("Ninguna imagen cargada",
                       "La vista previa, la imagen procesada y los dorsales detectados aparecerán aquí.",
                       "image")
        return

    img = detection.decode_image(uploaded.getvalue())
    if img is None:
        ui.error_box("No se pudo leer la imagen",
                     "El archivo parece estar dañado o no es un JPG válido.")
        return

    file_id = _file_id(uploaded)
    state = st.session_state.get(STATE_KEY)
    if state and state["file_id"] != file_id:
        state = None
        st.session_state.pop(STATE_KEY, None)

    h, w = img.shape[:2]
    left, right = st.columns([7, 4], gap="large")

    with right:
        ui.section_title("Archivo", "file")
        ui.kv_list([("Nombre", ui.esc(uploaded.name)), ("Resolución", f"{w}×{h}px"),
                    ("Tamaño", ui.fmt_bytes(uploaded.size))])
        save = st.toggle("Guardar resultado en la galería", value=True, key="img_save",
                         help="Guarda la imagen anotada en usr/validation/Full cuando hay detecciones.")
        run = st.button("Volver a ejecutar" if state else "Ejecutar detección", type="primary",
                        use_container_width=True, disabled=not status["ok"])
        if not status["ok"]:
            ui.alert(status["detail"], "error", status["label"])
        if run:
            try:
                with st.spinner("Analizando imagen…"):
                    _run(img, file_id, uploaded.name, save)
                state = st.session_state[STATE_KEY]
            except Exception as exc:  # noqa: BLE001
                ui.error_box("La detección falló", exc)
        ui.spacer(8)
        _results_panel(state, uploaded.name)

    with left:
        if state is None:
            ui.render(f"""<div class="br-stage__bar"><span class="br-stage__title">
                {icon('image', 16)} Vista previa</span>{ui.badge('Sin procesar', 'neutral')}</div>""")
            ui.image(detection.to_rgb(detection.resize_max(img, 1800)))
        else:
            n = len(state["result"].bibs)
            ui.render(f"""<div class="br-stage__bar"><span class="br-stage__title">
                {icon('scan', 16)} Resultado de visión artificial</span>
                {ui.badge(f'{n} dorsal' + ('es' if n != 1 else ''), 'lime' if n else 'warning')}</div>""")
            t1, t2, t3 = st.tabs(["Comparación", "Lado a lado", "Procesada"])
            with t1:
                viewers.compare_slider(img, state["annotated"])
            with t2:
                viewers.side_by_side(img, state["annotated"])
            with t3:
                ui.image(detection.to_rgb(detection.resize_max(state["annotated"], 1800)))
