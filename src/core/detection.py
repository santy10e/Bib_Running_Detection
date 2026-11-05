"""Capa de servicio de detección.

Envuelve el pipeline existente de ``utils`` (Detector + get_rbns_detailed) sin
modificar su lógica: solo cachea los modelos, serializa la inferencia y ofrece
utilidades de presentación (dibujo de anotaciones, codificación de imágenes).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import cv2 as cv
import numpy as np
import streamlit as st

from core import config
from utils import (BIB_CONF_THRESHOLD, DIGIT_CONF_THRESHOLD, Detector,
                   get_rbns_detailed)

# cv.dnn no es seguro para llamadas concurrentes sobre la misma red; Streamlit
# ejecuta cada sesión en un hilo, así que serializamos la inferencia.
_INFERENCE_LOCK = threading.Lock()


@dataclass
class Bib:
    number: int
    box: tuple[int, int, int, int]  # x, y, w, h
    confidence: float               # confianza del detector de dorsales

    @property
    def readable(self) -> bool:
        # get_rbns devuelve 0 cuando detecta el dorsal pero no lee dígitos
        return self.number != 0


@dataclass
class DetectionResult:
    bibs: list[Bib] = field(default_factory=list)
    elapsed: float = 0.0

    @property
    def numbers(self) -> list[int]:
        return [b.number for b in self.bibs if b.readable]

    @property
    def raw_output(self):
        """Formato original de get_rbns: [[número, [x, y, w, h]], ...] o None."""
        if not self.bibs:
            return None
        return [[b.number, list(b.box)] for b in self.bibs]


def model_files() -> list[dict]:
    files = [
        ("Detector de dorsales", "RBNR", "cfg", config.BD_CONFIG_PATH),
        ("Detector de dorsales", "RBNR", "weights", config.BD_WEIGHTS_PATH),
        ("Reconocedor de dígitos", "SVHN", "cfg", config.NR_CONFIG_PATH),
        ("Reconocedor de dígitos", "SVHN", "weights", config.NR_WEIGHTS_PATH),
    ]
    out = []
    for role, dataset, kind, path in files:
        exists = path.exists()
        out.append({
            "role": role, "dataset": dataset, "kind": kind, "path": path,
            "exists": exists, "size": path.stat().st_size if exists else 0,
        })
    return out


def models_available() -> bool:
    return all(f["exists"] for f in model_files())


@st.cache_resource(show_spinner=False)
def load_detectors():
    """Carga las dos redes YOLOv4-tiny una sola vez por proceso."""
    bd = Detector(str(config.BD_CONFIG_PATH), str(config.BD_WEIGHTS_PATH), config.BD_CLASSES)
    nr = Detector(str(config.NR_CONFIG_PATH), str(config.NR_WEIGHTS_PATH), config.NR_CLASSES)
    return bd, nr


def model_status() -> dict:
    """Estado real del modelo: archivos presentes y redes cargables."""
    if not models_available():
        missing = [f["path"].name for f in model_files() if not f["exists"]]
        return {"ok": False, "label": "Modelo no disponible",
                "detail": "Faltan archivos: " + ", ".join(sorted(set(missing)))}
    try:
        load_detectors()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "label": "Error al cargar el modelo", "detail": str(exc)}
    return {"ok": True, "label": "Modelo operativo",
            "detail": "YOLOv4-tiny · RBNR + SVHN"}


THRESHOLDS = {"bib": BIB_CONF_THRESHOLD, "digit": DIGIT_CONF_THRESHOLD}


def detect(img_bgr: np.ndarray) -> DetectionResult:
    """Ejecuta la cascada de detección existente sobre una imagen BGR."""
    bd, nr = load_detectors()
    start = time.perf_counter()
    with _INFERENCE_LOCK:
        output = get_rbns_detailed(img_bgr, bd, nr)
    elapsed = time.perf_counter() - start
    bibs = []
    for number, box, conf in output or []:
        bibs.append(Bib(int(number), tuple(int(v) for v in box), float(conf)))
    return DetectionResult(bibs=bibs, elapsed=elapsed)


# --- Presentación ------------------------------------------------------------

def draw_detections(img_bgr: np.ndarray, result: DetectionResult,
                    show_conf: bool = True) -> np.ndarray:
    """Devuelve una copia con cajas y etiquetas con el estilo de la plataforma."""
    out = img_bgr.copy()
    h_img, w_img = out.shape[:2]
    scale = max(min(w_img, h_img) / 900.0, 0.45)
    thickness = max(int(round(2 * scale)), 2)
    font = cv.FONT_HERSHEY_DUPLEX
    font_scale = 0.75 * scale
    font_th = max(int(round(1.4 * scale)), 1)
    pad = max(int(6 * scale), 4)

    for bib in result.bibs:
        x, y, w, h = bib.box
        color = config.ANNOT_BGR if bib.readable else config.ANNOT_WARN_BGR
        cv.rectangle(out, (x, y), (x + w, y + h), color, thickness, cv.LINE_AA)

        # esquinas reforzadas (estética de "escáner")
        c = max(int(min(w, h) * 0.25), 6)
        t2 = thickness + 2
        for (px, py, dx, dy) in ((x, y, 1, 1), (x + w, y, -1, 1),
                                 (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
            cv.line(out, (px, py), (px + dx * c, py), color, t2, cv.LINE_AA)
            cv.line(out, (px, py), (px, py + dy * c), color, t2, cv.LINE_AA)

        label = f"#{bib.number}" if bib.readable else "Sin lectura"
        if show_conf:
            label += f"  {bib.confidence * 100:.0f}%"
        (tw, th), base = cv.getTextSize(label, font, font_scale, font_th)
        lx = max(min(x, w_img - tw - 2 * pad), 0)
        ly_top = y - th - base - 2 * pad - thickness
        if ly_top < 0:  # sin espacio arriba: etiqueta debajo de la caja
            ly_top = min(y + h + thickness, h_img - th - base - 2 * pad)
        cv.rectangle(out, (lx, ly_top), (lx + tw + 2 * pad, ly_top + th + base + 2 * pad),
                     color, -1)
        cv.putText(out, label, (lx + pad, ly_top + pad + th), font, font_scale,
                   config.ANNOT_TEXT_BGR, font_th, cv.LINE_AA)
    return out


def decode_image(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv.imdecode(arr, cv.IMREAD_COLOR)


def encode_jpg(img_bgr: np.ndarray, quality: int = 92) -> bytes:
    ok, buf = cv.imencode(".jpg", img_bgr, [cv.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def b64_jpg(img_bgr: np.ndarray, quality: int = 85) -> str:
    import base64
    return base64.b64encode(encode_jpg(img_bgr, quality)).decode()


def resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img
    return cv.resize(img, (int(w * s), int(h * s)), interpolation=cv.INTER_AREA)


def to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
