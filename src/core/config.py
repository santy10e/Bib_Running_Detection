"""Configuración central de la aplicación.

Todas las rutas se resuelven a partir de la ubicación del proyecto, de modo que
la app funciona tanto con ``streamlit run src/app.py`` (desde la raíz) como con
``streamlit run app.py`` (desde ``src/``).
"""
from pathlib import Path

APP_NAME = "BIB RUNNING DETECTION"
APP_SUBTITLE = "Computer Vision Platform"
APP_VERSION = "2.0.0"

SRC_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_DIR.parent

ASSETS_DIR = PROJECT_ROOT / "assets"
CSS_PATH = ASSETS_DIR / "styles.css"

# --- Modelos (no se modifican: solo se referencian) -------------------------
MODELS_DIR = PROJECT_ROOT / "content" / "bib-project"

BD_CONFIG_PATH = MODELS_DIR / "RBNR" / "custom-yolov4-tiny-detector.cfg"
BD_WEIGHTS_PATH = MODELS_DIR / "RBNR" / "custom-yolov4-tiny-detector_best.weights"
BD_CLASSES = ["bib"]

NR_CONFIG_PATH = MODELS_DIR / "SVHN" / "custom-yolov4-tiny-detector.cfg"
NR_WEIGHTS_PATH = MODELS_DIR / "SVHN" / "custom-yolov4-tiny-detector_best.weights"
NR_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

MODEL_INPUT_SIZE = 416  # tamaño del blob usado por Detector.detect

# --- Almacenamiento ----------------------------------------------------------
USR_DIR = PROJECT_ROOT / "usr"
GALLERY_DIR = USR_DIR / "validation" / "Full"      # imágenes procesadas (galería)
VALIDATION_DIR = USR_DIR / "validation" / "test"   # dataset RBNR de validación
EVAL_OUTPUT_DIR = GALLERY_DIR                      # salida de run_all_tests (sin cambios)
ACTIVITY_LOG_PATH = USR_DIR / "activity_log.jsonl"

# --- Colores de anotación (BGR para OpenCV) ----------------------------------
ANNOT_BGR = (61, 255, 199)        # #C7FF3D lima
ANNOT_WARN_BGR = (11, 158, 245)   # #F59E0B ámbar (dorsal sin número legible)
ANNOT_TEXT_BGR = (20, 15, 11)     # #0B0F14 texto sobre etiqueta
