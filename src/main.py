# -*- coding: utf-8 -*-
from __future__ import print_function
from image_processor import ImageProcessor
from matplotlib import pyplot as plt
import cv2
import numpy as np 
import argparse
import os

"""
  Mostramos las imagenes en una grilla 2x2
  - Imagen Original
  - Imagen Binarizada segun
"""
def show_images(image_processor, k_threshold, median_window_size):
  original_image = image_processor.get_original_image()

  plt.subplot(2, 2, 1), plt.imshow(original_image, cmap='gray', vmin=0, vmax=255), plt.title('Imagen Original')
  plt.show()

def show_errors(image_processor, k_threshold):
  """
    Establecer los rangos de ruido (0.2, 0.5, 0.8, 0.11 … , 0.3) de al menos 5 valores.
    Establecer una grilla 2x2 de la siguiente manera para el calculo de error de la siguiente manera
    (2, 2, 1) MSE: Error cuadrático medio
    (2, 2, 2) PSNR: Proporción Máxima de Señal a Ruido
    (2, 2, 3) MAE: Error absoluto medio
    Usar diferentes colores y símbolos a criterio del autor.
    Establecer propiedades de los gráficos a criterio del autor (subtítulos, lineas, etiquetas, etc).

    TODO: Implementar solucion aqui
  """
  t_range = np.arange(1., 10., 2)
  plt.title("Medicion de ejemplo".format(k_threshold))
  plt.subplot(2, 2, 1), plt.plot(t_range, t_range, '^-k', t_range, t_range ** 2, '-or'), plt.title('Prueba 1')
  plt.subplot(2, 2, 2), plt.plot(t_range, t_range ** 2, '-or'), plt.title('Prueba 2')
  plt.subplot(2, 2, 3), plt.plot(t_range, t_range ** 2.2, '--sb'), plt.title('Prueba 3')

  plt.show()

if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('-i', '--image', required=True, help='Ubicación de la imagen')
  ap.add_argument('-g', '--show-image-grid', required=False, help='Mostramos una cuadricula', action='store_true')
  ap.add_argument('-k', '--k-threshold', required=False, help='Valor Umbral K para filtro')
  ap.add_argument('-e', '--show-error', required=False, help='Valor de ventana de la media', action='store_true')

  ap.set_defaults(show_image_grid=False)
  ap.set_defaults(show_error=False)
  ap.set_defaults(k_threshold=100)
  ap.set_defaults(media_size_window=3)

  args = vars(ap.parse_args())

  image_url = args['image']
  print(image_url)
  k_threshold = args['k_threshold']
  median_window_size = args['media_size_window']

  image_processor = ImageProcessor(image_url)

  if (args['show_image_grid']):
    show_images(image_processor, k_threshold, median_window_size)

  if (args['show_error']):
    show_errors(image_processor, k_threshold)
