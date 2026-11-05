"""Procesamiento de video fotograma a fotograma con la cascada existente."""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2 as cv

from core import detection

WORK_DIR = Path(tempfile.gettempdir()) / "bib_running_detection"

# Códecs en orden de preferencia. H.264 y VP8 se reproducen en el navegador;
# mp4v es el último recurso (solo descarga).
_CODECS = [("avc1", ".mp4", "video/mp4", True),
           ("VP80", ".webm", "video/webm", True),
           ("mp4v", ".mp4", "video/mp4", False)]


def work_path(name: str) -> Path:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    return WORK_DIR / name


def probe(path: str | Path) -> dict | None:
    cap = cv.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv.CAP_PROP_FPS) or 0) or 25.0
        w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return {"frames": frames, "fps": fps, "width": w, "height": h,
            "duration": frames / fps if fps else 0}


def open_writer(base: Path, fps: float, size: tuple[int, int]):
    for fourcc, ext, mime, playable in _CODECS:
        path = base.with_suffix(ext)
        writer = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer, path, mime, playable
        writer.release()
    return None, None, None, False


@dataclass
class BibTrack:
    number: int
    frames: int = 0
    first_t: float = 0.0
    last_t: float = 0.0
    best_conf: float = 0.0
    snapshot: object = None  # recorte BGR del dorsal con mayor confianza


@dataclass
class VideoSummary:
    file: str
    info: dict
    step: int
    frames_read: int = 0
    frames_analyzed: int = 0
    frames_with_bibs: int = 0
    unreadable: int = 0
    total_time: float = 0.0
    completed: bool = False
    tracks: dict[int, BibTrack] = field(default_factory=dict)
    records: list[dict] = field(default_factory=list)
    output_path: Path | None = None
    output_mime: str | None = None
    output_playable: bool = False

    @property
    def unique_numbers(self) -> list[int]:
        return sorted(self.tracks, key=lambda n: (-self.tracks[n].frames, n))


def _crop(frame, box, pad=0.25, max_side=180):
    x, y, w, h = box
    H, W = frame.shape[:2]
    px, py = int(w * pad), int(h * pad)
    x0, y0 = max(x - px, 0), max(y - py, 0)
    x1, y1 = min(x + w + px, W), min(y + h + py, H)
    if x1 <= x0 or y1 <= y0:
        return None
    return detection.resize_max(frame[y0:y1, x0:x1], max_side).copy()


def process(path: str | Path, summary: VideoSummary,
            on_progress: Callable[[VideoSummary, object], None] | None = None) -> VideoSummary:
    """Recorre el video, detecta dorsales y escribe un video anotado.

    ``summary`` se actualiza en sitio, de modo que si la ejecución se
    interrumpe (p. ej. el usuario pulsa «Detener») conserva los resultados
    parciales.
    """
    info = summary.info
    cap = cv.VideoCapture(str(path))
    base = work_path(Path(summary.file).stem + "_procesado")
    writer, out_path, mime, playable = open_writer(
        base, info["fps"], (info["width"], info["height"]))
    summary.output_path, summary.output_mime, summary.output_playable = out_path, mime, playable

    last = detection.DetectionResult()
    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t = idx / info["fps"]
            analyzed = idx % summary.step == 0
            if analyzed:
                last = detection.detect(frame)
                summary.frames_analyzed += 1
                summary.total_time += last.elapsed
                if last.bibs:
                    summary.frames_with_bibs += 1
                    summary.records.append({
                        "Fotograma": idx, "Tiempo (s)": round(t, 2),
                        "Dorsales": ", ".join(str(b.number) if b.readable else "?"
                                              for b in last.bibs),
                        "Detecciones": len(last.bibs),
                        "Conf. máx.": round(max(b.confidence for b in last.bibs), 3),
                    })
                for b in last.bibs:
                    if not b.readable:
                        summary.unreadable += 1
                        continue
                    tr = summary.tracks.get(b.number)
                    if tr is None:
                        tr = summary.tracks[b.number] = BibTrack(b.number, first_t=t)
                    tr.frames += 1
                    tr.last_t = t
                    if b.confidence > tr.best_conf:
                        tr.best_conf = b.confidence
                        tr.snapshot = _crop(frame, b.box)
            annotated = detection.draw_detections(frame, last) if last.bibs else frame
            if writer is not None:
                writer.write(annotated)
            idx += 1
            summary.frames_read = idx
            if on_progress is not None and analyzed:
                on_progress(summary, annotated)
        summary.completed = True
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    return summary
