from datetime import datetime
import numpy as np
import os
import cv2 as cv
import pandas as pd

# Importar funciones del archivo src.utils
from utils import get_true_annot, get_rbns, annotate

# Configuración de detección de dorsales
bd_configPath = '/content/bib-project/RBNR/custom-yolov4-tiny-detector.cfg'
bd_weightsPath = '/content/bib-project/RBNR/custom-yolov4-tiny-detector_best.weights'
bd_classes = ['bib']

# Configuración de detección de números
nr_configPath = '/content/bib-project/SVHN/custom-yolov4-tiny-detector.cfg'
nr_weightsPath = '/content/bib-project/SVHN/custom-yolov4-tiny-detector_best.weights'
nr_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Colores para anotaciones
true_color = [66, 245, 108]
color = [66, 245, 108]

# Rutas de entrada y salida
images_path = '/usr/validation/test/'
output_path = '/usr/validation/Full/'


def check_existing_files():
    # Eliminar archivos de salida existentes si los hay
    for file_name in ['bib_numbers.txt', 'preds.txt']:
        file_path = os.path.join(output_path, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)


def run_all_tests():
    check_existing_files()

    # Lista de imágenes en el directorio de entrada
    images = [file for file in os.listdir(
        images_path) if file.endswith('.JPG')]

    for image in images:
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


def validate_accuracy():
    true_df = pd.read_csv(os.path.join(output_path, 'bib_numbers.txt'), delimiter=',',
                          index_col=0, names=['image', 'rbn'])

    pred_df = pd.read_csv(os.path.join(output_path, 'preds.txt'), delimiter=',',
                          index_col=0, names=['image', 'pred_rbn'])

    all_df = pd.merge(true_df, pred_df, on='image', how='left')

    true_positives = len(all_df.loc[all_df['rbn'] == all_df['pred_rbn']])
    total = len(true_df)

    return true_positives / total

