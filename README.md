# Bib Running Detection

**Computer Vision Platform** para detectar, reconocer y registrar los números de dorsal de corredores en imágenes, videos y cámaras en directo.

Construida con **Streamlit** y **OpenCV**. Usa una cascada de dos redes **YOLOv4-tiny** (Darknet):

1. **RBNR**: localiza el dorsal en la imagen.
2. **SVHN**: reconoce los dígitos dentro del dorsal, que se ordenan de izquierda a derecha para formar el número.

## Requisitos

- Python 3.11
- Dependencias en `requirements.txt` (Streamlit 1.34, OpenCV 4.9, …)

## Instalación

```bash
git clone https://github.com/santy10e/Bib_Running_Detection
cd Bib_Running_Detection
python -m venv venv
venv\Scripts\activate          # Windows  ·  source venv/bin/activate en Linux/macOS
pip install -r requirements.txt
```

Los modelos (`.cfg` y `.weights`) ya están incluidos en `content/bib-project/`.

## Uso

Desde la raíz del proyecto:

```bash
streamlit run src/app.py
```

También funciona desde `src/` con `streamlit run app.py`: todas las rutas se resuelven a partir de la ubicación del proyecto.

## Funcionalidades

| Página | Qué hace |
|---|---|
| **Dashboard** | Imágenes analizadas, dorsales detectados, videos analizados y precisión de la última evaluación. Incluye accesos rápidos, últimas detecciones, actividad reciente y estado del modelo. Solo muestra datos reales. |
| **Detección de imágenes** | Carga JPG/JPEG, ejecución de la detección y comparador original/detección (deslizador, lado a lado). Muestra cada dorsal con su número, la confianza del detector y su caja. Permite descargar la imagen procesada y la guarda en la galería. |
| **Procesamiento de videos** | Carga MP4 y procesa fotograma a fotograma con progreso en vivo y botón para detener. Genera un video anotado reproducible y descargable, una tabla de detecciones por fotograma y un resumen de dorsales únicos con recortes. |
| **Cámara en tiempo real** | Selector y búsqueda de cámaras, iniciar/detener, indicador LIVE y contador de dorsales de la sesión. Incluye un modo de captura desde el navegador (`st.camera_input`). |
| **Galería de detecciones** | Tarjetas con miniatura, dorsal y fecha. Búsqueda por dorsal, filtro por día o rango de fechas, orden, paginación y vista ampliada con descarga. |
| **Evaluación del modelo** | Ejecuta `run_all_tests()` sobre el dataset RBNR (217 imágenes). Muestra la exactitud oficial (`validate_accuracy()`) y precision, recall y F1 derivadas, el desglose TP/FP/FN, la comparación por subconjunto, los resultados por imagen y el historial. |

## Estructura

```text
src/
├── app.py                  # Punto de entrada: tema, sidebar y enrutado
├── utils.py                # Pipeline original (Detector, get_rbns, …)
├── functions.py            # Evaluación original (run_all_tests, validate_accuracy)
├── core/                   # Lógica sin UI
│   ├── config.py           # Rutas, modelos, versión, colores
│   ├── detection.py        # Modelos en caché, inferencia, dibujo de anotaciones
│   ├── video.py            # Procesamiento de video y escritura del resultado
│   └── storage.py          # Galería, registro de actividad, métricas de evaluación
├── components/             # Sistema de diseño
│   ├── ui.py               # Header, metric cards, badges, empty states, alertas…
│   ├── sidebar.py          # Navegación persistente y estado del sistema
│   ├── viewers.py          # Comparador antes/después y vista lado a lado
│   └── icons.py            # Iconos Lucide (SVG) e isotipo
└── views/                  # Una vista por página
    ├── dashboard.py · image_detection.py · video_detection.py
    └── webcam.py · gallery.py · model_evaluation.py
assets/
├── styles.css              # Design system (tokens + componentes br-*)
└── images/favicon.png
.streamlit/config.toml      # Tema oscuro base (copia en src/.streamlit)
```

Las vistas están en `views/` y no en `pages/` porque Streamlit convierte automáticamente una carpeta `pages/` en un menú propio.

## Almacenamiento

- **Galería**: `usr/validation/Full/{dorsal}_{dd-mm-yy_HH_MM_SS}.jpg`. Es el mismo formato de la versión anterior, así que las imágenes antiguas siguen apareciendo.
- **Actividad**: `usr/activity_log.jsonl`, un evento por análisis real (imagen, video, sesión de cámara o evaluación). Alimenta el dashboard y el historial de evaluaciones.
- **Evaluación**: `bib_numbers.txt`, `preds.txt` y `*_annot.JPG` en `usr/validation/Full/` (comportamiento original de `run_all_tests`).

## Limitaciones conocidas

- La **cámara en vivo** se abre en el equipo donde corre el servidor (OpenCV). Si la app se despliega en un servidor remoto, usa la pestaña "Captura desde el navegador".
- Durante el procesamiento de un video o la cámara en vivo, cualquier interacción con otro control reinicia la ejecución de Streamlit. El video conserva los resultados parciales.
- La **evaluación** dibuja las cajas reales sobre la imagen antes de predecir (lógica original de `functions.py`), lo que puede influir en el detector. Las métricas se marcan como experimentales.
- No hay **matriz de confusión**: el proyecto no guarda la verdad por dígito, y los números de dorsal tienen cientos de clases posibles.
- La confianza mostrada es la del **detector de dorsales**. El reconocedor de dígitos no expone una confianza por número.

## Dataset

[RBNR: Racing Bib Number Recognition](https://people.csail.mit.edu/talidekel/RBNR.html)
