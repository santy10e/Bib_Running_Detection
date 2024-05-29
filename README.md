# Detección de Dorsales de Corredores

Este proyecto utiliza Streamlit y OpenCV para detectar dorsales de corredores en imágenes y videos. La aplicación permite cargar imágenes, videos y también usar la cámara web para realizar detecciones en tiempo real.

## Requisitos

- Python 3.x
- Streamlit
- OpenCV
- Otros paquetes necesarios listados en `requirements.txt`

## Instalación

1. Clona este repositorio:

    ```bash
    git clone https://github.com/tu_usuario/deteccion_dorsales.git
    cd deteccion_dorsales
    ```

2. Instala los requisitos:

    ```bash
    pip install -r requirements.txt
    ```

3. Asegúrate de tener los archivos de configuración y pesos de los modelos en las rutas correctas especificadas en el código.

## Estructura del Proyecto

- `app.py`: Archivo principal de la aplicación.
- `utils.py`: Contiene las funciones auxiliares para anotación y obtención de resultados.
- `functions.py`: Contiene las funciones para ejecutar pruebas y validar la precisión.
- `assets/styles.css`: Archivo de estilos CSS para la aplicación.
- `requirements.txt`: Lista de dependencias del proyecto.

## Uso

Para ejecutar la aplicación, usa el siguiente comando:

```bash
streamlit run app.py
```

Esto abrirá una nueva pestaña en tu navegador con la interfaz de la aplicación.

### Funcionalidades

- **Inicio**: Permite cargar una imagen y ejecutar la detección de dorsales.
- **Galería de Imágenes**: Muestra las imágenes guardadas y permite filtrar por fecha o número de dorsal.
- **Actualizar Modelo**: Ejecuta pruebas y valida la precisión del modelo.
- **Cámara Web**: Permite capturar video en tiempo real desde la cámara web y realizar detecciones.
- **Cargar Video**: Permite cargar un video y ejecutar la detección de dorsales en cada fotograma.

### Cargar Imagen

1. Selecciona la pestaña "Inicio".
2. Carga una imagen en formato JPG o JPEG.
3. Haz clic en "Ejecutar detección" para detectar dorsales en la imagen cargada.

### Ver Imágenes Guardadas

1. Selecciona la pestaña "Galería de Imágenes".
2. Filtra las imágenes guardadas por fecha o número de dorsal.

### Actualizar Modelo

1. Selecciona la pestaña "Actualizar Modelo".
2. Haz clic en "Ejecutar modelo" para correr todas las pruebas y validar la precisión del modelo.

### Usar Cámara Web

1. Selecciona la pestaña "Cámara Web".
2. Selecciona la cámara (integrada o externa) y haz clic en "Iniciar Camara".

### Cargar Video

1. Selecciona la pestaña "Cargar Video".
2. Carga un video en formato MP4.
3. La detección se realizará en cada fotograma del video cargado.

---

## Dataset
[Dataset de los bib números de los corredores](https://people.csail.mit.edu/talidekel/RBNR.html)

Si tienes alguna pregunta o sugerencia, no dudes en contactarme. ¡Gracias por usar esta aplicación de detección de dorsales!