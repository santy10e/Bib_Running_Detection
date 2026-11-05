"""Evaluación del modelo sobre el dataset RBNR de validación.

Usa run_all_tests / validate_accuracy de functions.py (lógica original) y deriva
métricas adicionales de los mismos archivos de salida.
"""
from __future__ import annotations

import datetime as dt
import time

import cv2 as cv
import pandas as pd
import streamlit as st

from components import ui
from core import config, storage
from core.detection import model_status
from functions import run_all_tests, validate_accuracy

STATUS_VARIANT = {"Correcta": "success", "Parcial": "warning", "Incorrecta": "error",
                  "Sin detección": "neutral"}


def _pct(v: float | None) -> str | None:
    return None if v is None else f"{v * 100:.1f}%"


def _delta(cur: float | None, prev: float | None) -> tuple[str, str] | None:
    if cur is None or prev is None:
        return None
    d = (cur - prev) * 100
    if abs(d) < 0.05:
        return ("0.0 pp", "flat")
    return (f"{d:+.1f} pp", "up" if d > 0 else "down")


@st.cache_data(show_spinner=False)
def _load_results(pred_mtime: float, true_mtime: float):
    true_df, pred_df, _ = storage.load_evaluation_tables()
    try:
        official = validate_accuracy()
    except Exception:  # noqa: BLE001
        official = None
    return official, storage.recognition_metrics(true_df, pred_df), len(set(true_df["image"]))


def _run_evaluation(total: int) -> None:
    bar = st.progress(0.0)
    txt = st.empty()
    t0 = time.perf_counter()

    def cb(i: int, n: int, name: str) -> None:
        bar.progress(i / n)
        elapsed = time.perf_counter() - t0
        eta = elapsed / i * (n - i)
        txt.markdown(f'<div class="br-help">Imagen <b>{i}</b> / {n} · {ui.esc(name)} · '
                     f'quedan ~{ui.fmt_duration(eta)}</div>', unsafe_allow_html=True)

    run_all_tests(progress_callback=cb)
    duration = time.perf_counter() - t0
    true_df, pred_df, _ = storage.load_evaluation_tables()
    m = storage.recognition_metrics(true_df, pred_df)
    try:
        accuracy = validate_accuracy()
    except Exception:  # noqa: BLE001
        accuracy = None
    storage.log_event("evaluation", accuracy=accuracy, precision=m["precision"],
                      recall=m["recall"], f1=m["f1"], tp=m["tp"], fp=m["fp"], fn=m["fn"],
                      images=total, duration_s=round(duration, 1))


def _dataset_panel(ds: dict, has_results: bool) -> None:
    left, right = st.columns([3, 2], gap="large")
    with left:
        ui.section_title("Dataset de validación", "database")
        sets = " · ".join(f"{k}: {v}" for k, v in ds["sets"].items()) or "—"
        ui.kv_list([
            ("Origen", "RBNR · Racing Bib Number Recognition"),
            ("Imágenes", str(ds["images"])),
            ("Anotaciones (.mat)", str(ds["annotations"])),
            ("Subconjuntos", sets),
            ("Ruta", config.VALIDATION_DIR.relative_to(config.PROJECT_ROOT).as_posix()),
        ])
    with right:
        ui.section_title("Ejecución de pruebas", "flask")
        ui.render('<div class="br-help" style="margin-bottom:.75rem">Ejecuta la cascada de '
                  'detección sobre todas las imágenes de validación y compara el número leído '
                  'con la anotación real. Sobrescribe <b>bib_numbers.txt</b> y <b>preds.txt</b> '
                  'y guarda las imágenes anotadas (<b>*_annot.JPG</b>).</div>')
        status = model_status()
        run = st.button("Ejecutar evaluación" if not has_results else "Ejecutar nueva evaluación",
                        type="primary", use_container_width=True,
                        disabled=not (status["ok"] and ds["images"]))
        if not ds["images"]:
            ui.alert("No se encontró el dataset de validación.", "error")
        if run:
            try:
                _run_evaluation(ds["images"])
            except Exception as exc:  # noqa: BLE001
                ui.error_box("La evaluación falló", exc)
            else:
                st.rerun()


def _metrics_section(official, m: dict, n_images: int, results_ts: float) -> None:
    evals = storage.read_events("evaluation")
    prev = evals[-2] if len(evals) >= 2 else None
    ui.section_title("Métricas de rendimiento", "gauge",
                     f"Resultados del {dt.datetime.fromtimestamp(results_ts):%d/%m/%y %H:%M} · "
                     f"{n_images} imágenes")
    ui.metric_row([
        ui.metric_card("Exactitud", _pct(official), "target",
                       "validate_accuracy(): coincidencias / dorsales anotados",
                       accent=True, tag=ui.badge("Oficial", "lime", dot=False),
                       delta=_delta(official, prev and prev.get("accuracy"))),
        ui.metric_card("Precision", _pct(m["precision"]), "check",
                       f"TP / (TP + FP) · {m['tp']} de {m['tp'] + m['fp']} predicciones",
                       tag=ui.badge("Derivada", "info", dot=False),
                       delta=_delta(m["precision"], prev and prev.get("precision"))),
        ui.metric_card("Recall", _pct(m["recall"]), "search",
                       f"TP / (TP + FN) · {m['tp']} de {m['tp'] + m['fn']} dorsales",
                       tag=ui.badge("Derivada", "info", dot=False),
                       delta=_delta(m["recall"], prev and prev.get("recall"))),
        ui.metric_card("F1-score", _pct(m["f1"]), "activity",
                       "Media armónica de precision y recall",
                       tag=ui.badge("Derivada", "info", dot=False),
                       delta=_delta(m["f1"], prev and prev.get("f1"))),
    ])
    if prev:
        ui.render(f'<div class="br-help" style="margin:-.75rem 0 1rem">Variaciones en puntos '
                  f'porcentuales frente a la evaluación del '
                  f'{dt.datetime.fromisoformat(prev["ts"]):%d/%m/%y %H:%M}.</div>')

    per = m["per_image"]
    counts = per["Estado"].value_counts().to_dict() if not per.empty else {}
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        ui.render(ui.segmented_bar(
            [("Aciertos (TP)", m["tp"], "success"), ("Falsos positivos (FP)", m["fp"], "warning"),
             ("No detectados (FN)", m["fn"], "error")],
            "Desglose de lecturas"))
    with c2:
        ui.render(ui.segmented_bar(
            [(k, counts.get(k, 0), v) for k, v in STATUS_VARIANT.items()],
            "Resultado por imagen"))

    # Comparación entre subconjuntos del dataset
    if not per.empty:
        per = per.assign(Subconjunto=per["Imagen"].str.split("_").str[0])
        g = per.groupby("Subconjunto").agg(Imágenes=("Imagen", "count"), TP=("Aciertos", "sum"),
                                           FP=("FP", "sum"), FN=("FN", "sum")).reset_index()
        g["Precision"] = g["TP"] / (g["TP"] + g["FP"]).where(lambda s: s > 0)
        g["Recall"] = g["TP"] / (g["TP"] + g["FN"]).where(lambda s: s > 0)
        ui.section_title("Comparación por subconjunto", "layers")
        st.dataframe(
            g, use_container_width=True, hide_index=True,
            column_config={
                "Precision": st.column_config.ProgressColumn("Precision", min_value=0, max_value=1, format="%.3f"),
                "Recall": st.column_config.ProgressColumn("Recall", min_value=0, max_value=1, format="%.3f"),
            })


def _unavailable_section() -> None:
    ui.section_title("Matriz de confusión", "box")
    ui.empty_state(
        "No disponible",
        "El reconocimiento produce números de dorsal de longitud variable (cientos de clases "
        "posibles), y el proyecto no guarda la verdad por dígito. Por eso no hay datos para "
        "una matriz de confusión con sentido. Usa el desglose TP/FP/FN.",
        "info", compact=True)


def _per_image_section(m: dict) -> None:
    per = m["per_image"]
    if per.empty:
        return
    ui.section_title("Resultados por imagen", "images", f"{len(per)} imágenes")
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        sel = st.multiselect("Filtrar por estado", list(STATUS_VARIANT), key="eval_status",
                             placeholder="Todos los estados")
        view = per[per["Estado"].isin(sel)] if sel else per
        st.dataframe(view, use_container_width=True, hide_index=True, height=420)
    with c2:
        options = list(view["Imagen"])
        if not options:
            ui.empty_state("Sin imágenes para este filtro", compact=True)
            return
        name = st.selectbox("Ver imagen anotada", options, key="eval_img")
        row = view[view["Imagen"] == name].iloc[0]
        annot = config.EVAL_OUTPUT_DIR / f"{name[:-4]}_annot.JPG"
        orig = config.VALIDATION_DIR / name
        path = annot if annot.exists() else orig
        img = cv.imread(str(path))
        if img is not None:
            ui.image(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        ui.kv_list([("Estado", ui.badge(row["Estado"], STATUS_VARIANT[row["Estado"]])),
                    ("Real", ui.esc(row["Dorsales reales"])),
                    ("Predicción", ui.esc(row["Predicción"]))])
        ui.render('<div class="br-help">' + (
            "Imagen anotada por run_all_tests. Las cajas reales y las predichas usan el mismo "
            "color en el pipeline original." if annot.exists() else
            "Sin imagen anotada: run_all_tests solo la guarda cuando hay predicciones.") + "</div>")


def _history_section() -> None:
    evals = storage.read_events("evaluation")
    ui.section_title("Historial de evaluaciones", "clock", f"{len(evals)} registradas" if evals else "")
    if not evals:
        ui.empty_state("Sin historial",
                       "Las evaluaciones ejecutadas desde esta versión se registran aquí.",
                       "clock", compact=True)
        return
    rows = [{
        "Fecha": dt.datetime.fromisoformat(e["ts"]),
        "Exactitud": e.get("accuracy"), "Precision": e.get("precision"),
        "Recall": e.get("recall"), "F1": e.get("f1"),
        "Imágenes": e.get("images"), "Duración (s)": e.get("duration_s"),
    } for e in reversed(evals)]
    pct = {"format": "%.3f"}
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                 column_config={
                     "Fecha": st.column_config.DatetimeColumn("Fecha", format="DD/MM/YY HH:mm"),
                     "Exactitud": st.column_config.NumberColumn("Exactitud", **pct),
                     "Precision": st.column_config.NumberColumn("Precision", **pct),
                     "Recall": st.column_config.NumberColumn("Recall", **pct),
                     "F1": st.column_config.NumberColumn("F1", **pct),
                 })


def render() -> None:
    ds = storage.validation_dataset()
    has_results = storage.evaluation_files_exist()
    last = storage.read_events("evaluation")
    badge = (ui.badge(f"Última evaluación · {dt.datetime.fromisoformat(last[-1]['ts']):%d/%m/%y}",
                      "success") if last else
             ui.badge("Resultados disponibles", "info") if has_results else
             ui.badge("Sin evaluar", "neutral"))
    ui.page_header(
        "Machine Learning", "Evaluación del modelo",
        "Mide el rendimiento de la cascada YOLOv4-tiny sobre el dataset RBNR anotado y "
        "compara ejecuciones.", "chart", badge,
    )
    _dataset_panel(ds, has_results)
    ui.spacer(8)

    if not has_results:
        ui.empty_state("Todavía no hay resultados de evaluación",
                       "Ejecuta la evaluación para obtener la exactitud, precision, recall y "
                       "F1-score reales del modelo.", "chart")
        _history_section()
        return

    pred_p = config.EVAL_OUTPUT_DIR / "preds.txt"
    true_p = config.EVAL_OUTPUT_DIR / "bib_numbers.txt"
    official, m, n_images = _load_results(storage.mtime(pred_p), storage.mtime(true_p))
    ui.alert("run_all_tests dibuja las cajas reales sobre cada imagen antes de predecir, y eso "
             "puede influir en el detector. Usa estas cifras para comparar ejecuciones, no como "
             "medida definitiva del modelo en producción.", "warning", "Resultados experimentales")
    ui.render(f"""<div class="br-help" style="display:flex;gap:1rem;flex-wrap:wrap;margin:-.25rem 0 1.25rem">
      <span>{ui.badge('Oficial', 'lime', dot=False)} calculada por validate_accuracy() del proyecto</span>
      <span>{ui.badge('Derivada', 'info', dot=False)} calculada a partir de los mismos archivos</span>
      <span>{ui.badge('No disponible', 'neutral', dot=False)} sin datos suficientes</span></div>""")
    _metrics_section(official, m, n_images, storage.mtime(pred_p))
    _unavailable_section()
    ui.spacer(12)
    _per_image_section(m)
    ui.spacer(12)
    _history_section()
