"""Almacenamiento: galería de detecciones, registro de actividad y resultados
de evaluación.

La galería conserva el formato original: ``usr/validation/Full`` con archivos
``{dorsal}_{dd-mm-yy_HH_MM_SS}.jpg``.
"""
from __future__ import annotations

import base64
import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import pandas as pd
import streamlit as st

from core import config

GALLERY_PATTERN = re.compile(
    r"^(?P<bib>\d+)_(?P<date>\d{2}-\d{2}-\d{2})_(?P<h>\d{2})_(?P<m>\d{2})_(?P<s>\d{2})\.jpe?g$",
    re.IGNORECASE,
)


@dataclass
class GalleryItem:
    path: Path
    bib: int
    timestamp: dt.datetime

    @property
    def name(self) -> str:
        return self.path.name


# --- Galería -----------------------------------------------------------------

def _parse_gallery_name(path: Path) -> GalleryItem | None:
    m = GALLERY_PATTERN.match(path.name)
    if not m:
        return None
    try:
        ts = dt.datetime.strptime(
            f"{m['date']} {m['h']}:{m['m']}:{m['s']}", "%d-%m-%y %H:%M:%S")
    except ValueError:
        return None
    return GalleryItem(path=path, bib=int(m["bib"]), timestamp=ts)


def list_gallery() -> list[GalleryItem]:
    """Imágenes guardadas por la detección (más recientes primero).

    Ignora los archivos de evaluación (``*_annot.JPG``, ``*.txt``) que
    ``run_all_tests`` escribe en la misma carpeta.
    """
    if not config.GALLERY_DIR.exists():
        return []
    items = [it for p in config.GALLERY_DIR.iterdir()
             if p.is_file() and (it := _parse_gallery_name(p))]
    return sorted(items, key=lambda i: i.timestamp, reverse=True)


def save_detection(annotated_bgr, bib_number: int) -> Path:
    """Guarda la imagen procesada con la convención de nombres original."""
    config.GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%d-%m-%y_%H_%M_%S")
    path = config.GALLERY_DIR / f"{bib_number}_{stamp}.jpg"
    cv.imwrite(str(path), annotated_bgr)
    return path


@st.cache_data(show_spinner=False, max_entries=512)
def thumbnail_rgb(path: str, mtime: float, max_side: int = 480, aspect: float | None = 4 / 3):
    """Miniatura RGB; con ``aspect`` se recorta al centro para tarjetas uniformes."""
    img = cv.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    if aspect:
        if w / h > aspect:          # más ancha: recorta laterales
            nw = int(h * aspect)
            img = img[:, (w - nw) // 2:(w - nw) // 2 + nw]
        else:                       # más alta: recorta arriba/abajo (sesgo superior)
            nh = int(w / aspect)
            top = max(int((h - nh) * 0.35), 0)
            img = img[top:top + nh]
        h, w = img.shape[:2]
    s = min(max_side / max(h, w), 1.0)
    if s < 1:
        img = cv.resize(img, (int(w * s), int(h * s)), interpolation=cv.INTER_AREA)
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)


@st.cache_data(show_spinner=False, max_entries=256)
def thumbnail_b64(path: str, mtime: float, max_side: int = 360) -> str:
    img = cv.imread(path)
    if img is None:
        return ""
    h, w = img.shape[:2]
    s = min(max_side / max(h, w), 1.0)
    if s < 1:
        img = cv.resize(img, (int(w * s), int(h * s)), interpolation=cv.INTER_AREA)
    ok, buf = cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


# --- Registro de actividad ---------------------------------------------------
# Archivo JSONL con un evento por línea. Solo registra operaciones reales
# realizadas desde la app (no hay datos sembrados).

def log_event(kind: str, **data) -> None:
    config.USR_DIR.mkdir(parents=True, exist_ok=True)
    event = {"ts": dt.datetime.now().isoformat(timespec="seconds"), "type": kind, **data}
    with open(config.ACTIVITY_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


@st.cache_data(show_spinner=False)
def _read_log(path: str, mtime_: float) -> list[dict]:
    events = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return []
    return events


def read_events(kind: str | None = None) -> list[dict]:
    events = _read_log(str(config.ACTIVITY_LOG_PATH), mtime(config.ACTIVITY_LOG_PATH))
    if kind:
        events = [e for e in events if e.get("type") == kind]
    return events


def activity_stats() -> dict:
    events = read_events()
    images = [e for e in events if e["type"] == "image"]
    videos = [e for e in events if e["type"] == "video"]
    evals = [e for e in events if e["type"] == "evaluation"]
    return {
        "images": len(images),
        "bibs": sum(e.get("bibs", 0) for e in images),
        "videos": len(videos),
        "last_eval": evals[-1] if evals else None,
        "since": events[0]["ts"] if events else None,
        "gallery": len(list_gallery()),
    }


# --- Evaluación --------------------------------------------------------------

def validation_dataset() -> dict:
    d = config.VALIDATION_DIR
    if not d.exists():
        return {"exists": False, "images": 0, "annotations": 0, "sets": {}}
    images = [p.name for p in d.iterdir() if p.name.endswith(".JPG")]
    annots = [p for p in d.iterdir() if p.name.endswith(".JPG.mat")]
    sets: dict[str, int] = {}
    for name in images:
        sets[name.split("_")[0]] = sets.get(name.split("_")[0], 0) + 1
    return {"exists": True, "images": len(images), "annotations": len(annots),
            "sets": dict(sorted(sets.items()))}


def evaluation_files_exist() -> bool:
    return ((config.EVAL_OUTPUT_DIR / "bib_numbers.txt").exists()
            and (config.EVAL_OUTPUT_DIR / "preds.txt").exists())


def load_evaluation_tables():
    """Lee bib_numbers.txt (verdad) y preds.txt (predicciones) de run_all_tests."""
    true_path = config.EVAL_OUTPUT_DIR / "bib_numbers.txt"
    pred_path = config.EVAL_OUTPUT_DIR / "preds.txt"
    true_df = pd.read_csv(true_path, names=["image", "rbn"])
    pred_df = (pd.read_csv(pred_path, names=["image", "pred_rbn"])
               if pred_path.exists() and pred_path.stat().st_size else
               pd.DataFrame(columns=["image", "pred_rbn"]))
    return true_df, pred_df, mtime(pred_path)


def recognition_metrics(true_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
    """Métricas extremo a extremo calculadas a partir de los mismos archivos.

    Por imagen se emparejan los números reales y predichos como multiconjuntos:
      TP = números predichos que coinciden con un número real de la imagen
      FP = predicciones sin correspondencia (incluye dorsales sin lectura, 0)
      FN = números reales que no fueron predichos
    """
    per_image = []
    tp = fp = fn = 0
    images = sorted(set(true_df["image"]) | set(pred_df["image"]))
    true_g = true_df.groupby("image")["rbn"].apply(list).to_dict()
    pred_g = pred_df.groupby("image")["pred_rbn"].apply(list).to_dict()
    for img in images:
        t = [int(v) for v in true_g.get(img, [])]
        p = [int(v) for v in pred_g.get(img, [])]
        remaining = list(t)
        hits = 0
        for v in p:
            if v in remaining:
                remaining.remove(v)
                hits += 1
        i_fp, i_fn = len(p) - hits, len(remaining)
        tp, fp, fn = tp + hits, fp + i_fp, fn + i_fn
        if not p:
            status = "Sin detección"
        elif hits == len(t) and i_fp == 0:
            status = "Correcta"
        elif hits > 0:
            status = "Parcial"
        else:
            status = "Incorrecta"
        per_image.append({
            "Imagen": img,
            "Dorsales reales": ", ".join(map(str, t)) or "—",
            "Predicción": ", ".join(map(str, p)) or "—",
            "Aciertos": hits, "FP": i_fp, "FN": i_fn, "Estado": status,
        })
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": f1, "per_image": pd.DataFrame(per_image)}
