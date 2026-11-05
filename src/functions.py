from datetime import datetime
import numpy as np
import os
import cv2 as cv
import pandas as pd

# Importar funciones del archivo src.utils
from utils import get_true_annot, get_rbns, annotate
from core import config

# Configuración de detección de dorsales
# (rutas resueltas desde la raíz del proyecto; antes eran absolutas '/content/...'
#  y no se encontraban al ejecutar en local)
bd_configPath = str(config.BD_CONFIG_PATH)
bd_weightsPath = str(config.BD_WEIGHTS_PATH)
bd_classes = config.BD_CLASSES

# Configuración de detección de números
nr_configPath = str(config.NR_CONFIG_PATH)
nr_weightsPath = str(config.NR_WEIGHTS_PATH)
nr_classes = config.NR_CLASSES

# Colores para anotaciones
true_color = [66, 245, 108]
color = [66, 245, 108]

# Rutas de entrada y salida (deben terminar en separador: utils concatena rutas)
images_path = str(config.VALIDATION_DIR) + os.sep
output_path = str(config.EVAL_OUTPUT_DIR) + os.sep


def check_existing_files():
    # Eliminar archivos de salida existentes si los hay
    for file_name in ['bib_numbers.txt', 'preds.txt']:
        file_path = os.path.join(output_path, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)


def list_test_images():
    # Lista de imágenes en el directorio de entrada
    return [file for file in os.listdir(images_path) if file.endswith('.JPG')]


def run_all_tests(progress_callback=None):
    """Ejecuta el modelo sobre el dataset de validación.

    progress_callback (opcional): función (índice, total, nombre_imagen) que se
    invoca tras procesar cada imagen. No altera el cálculo.
    """
    os.makedirs(output_path, exist_ok=True)
    check_existing_files()

    images = list_test_images()

    for idx, image in enumerate(images):
        img = cv.imread(os.path.join(images_path, image))

        # Agregar anotaciones verdaderas
        true_values = get_true_annot(image, images_path, output_path)
        for value in true_values:
            img = annotate(img, value, true_color)

        # Realizar predicciones
        output = get_rbns(img, bd_configPath, bd_weightsPath,
                          bd_classes, nr_configPath, nr_weightsPath, nr_classes)

        # Agregar anotaciones predichas y guardar predicción en archivo
        if output:
            preds_file_path = os.path.join(output_path, 'preds.txt')
            with open(preds_file_path, 'a') as rbn_file:
                for detection in output:
                    img = annotate(img, detection, color)
                    rbn_file.write(f"{image},{detection[0]}\n")

            # Guardar imagen anotada
            cv.imwrite(os.path.join(
                output_path, f"{image[:-4]}_annot.JPG"), img)

        if progress_callback is not None:
            progress_callback(idx + 1, len(images), image)


def validate_accuracy():
    true_df = pd.read_csv(os.path.join(output_path, 'bib_numbers.txt'), delimiter=',',
                          index_col=0, names=['image', 'rbn'])

    pred_df = pd.read_csv(os.path.join(output_path, 'preds.txt'), delimiter=',',
                          index_col=0, names=['image', 'pred_rbn'])

    all_df = pd.merge(true_df, pred_df, on='image', how='left')

    true_positives = len(all_df.loc[all_df['rbn'] == all_df['pred_rbn']])
    total = len(true_df)

    return true_positives / total
