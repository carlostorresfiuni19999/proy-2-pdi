# -*- coding: utf-8 -*-
from __future__ import print_function
from image_processor import ImageProcessor
import matplotlib
matplotlib.use('TkAgg')
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
  binarized_image = image_processor.get_binarized_image()

  plt.subplot(2, 2, 1), plt.imshow(original_image, cmap='gray', vmin=0, vmax=255), plt.title('Imagen Original')
  plt.subplot(2, 2, 2), plt.imshow(binarized_image, cmap='gray', vmin=0, vmax=255), plt.title('Imagen Binarizada')
  plt.show(block=True)
  
def errors(image_processor):
  y = ImageProcessor.calc_mse(image_processor.get_original_image(), image_processor.get_binarized_image())
  print(y)
  plt.title("Medicion de ejemplo".format(k_threshold))
  plt.plot([0, 1], [y, y], 'r--', label="MSE")
  plt.xlabel('Eje x')
  plt.ylabel('Eje y')
  plt.title('Graficos')
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
  
  values = [ 0.2, 0.5, 0.8, 1.1 ]
  results_mse = {}
  results_psnr = {}
  results_mae = {}
  img_original = image_processor.get_original_image()
  img_binarized = image_processor.get_binarized_image()
  
  for v in values:
    img_threshold = image_processor.threshold(img_original, v)
    mse = ImageProcessor.calc_mse(img_original, img_threshold)
    psnr = ImageProcessor.calc_psnr(img_original, img_threshold)
    mae = ImageProcessor.calc_mae(img_original, img_threshold)
    results_mse[v] = mse
    results_psnr[v] = psnr
    results_mae[v] = mae
    
  
  th = image_processor.get_otzu_thresh()
  mse = ImageProcessor.calc_mse(img_original, img_binarized)
  mae = ImageProcessor.calc_mae(img_original, img_binarized)
  psnr = ImageProcessor.calc_psnr(img_original, img_binarized)
  results_mse[th] = mse
  results_psnr[th] = psnr
  results_mae[th] = mae
  

  results_mse = dict(sorted(results_mse.items()))
  results_mae =dict(sorted(results_mse.items()))
  results_psnr =dict(sorted(results_psnr.items()))
  
  values_mse = np.array(list(results_mse.keys())) 
  values_psnr = np.array(list(results_psnr.keys()))
  values_mae = np.array(list(results_mae.keys()))
  
  t_range_mse = np.array(list(results_mse.values()))
  t_range_psnr = np.array(list(results_psnr.values()))
  t_range_mae = np.array(list(results_mae.values()))
  
  
  print(t_range_mse)
  print(t_range_mae)
  

    
  plt.title("Medicion de ejemplo".format(k_threshold))
  plt.subplot(2, 2, 1), plt.plot(values_mse, t_range_mse, 'bo-', [0,1], [mse, mse], 'y--'), plt.xlabel("thresh"), plt.ylabel("mse")
  plt.subplot(2, 2, 2), plt.plot(values_psnr, t_range_psnr, '-or', [0,1], [psnr, psnr],'y--'), plt.xlabel("thresh"), plt.ylabel("psnr")
  plt.subplot(2, 2, 3), plt.plot(values_mae, t_range_mae, '--sb',[0,1], [mse, mse], 'y--'), plt.xlabel("thresh"), plt.ylabel("mae")
  plt.show()

if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('-i', '--image', required=True, help='Ubicación de la imagen')
  ap.add_argument('-g', '--show-image-grid', required=False, help='Mostramos una cuadricula', action='store_true')
  ap.add_argument('-k', '--k-threshold', required=False, help='Valor Umbral K para filtro')
  ap.add_argument('-e', '--show-error', required=False, help='Valor de ventana de la media', action='store_true')
  ap.add_argument('-er', '--error', required=False, help='Calculo de desviaciones', action='store_true')

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
    
  if(args['error']):
    errors(image_processor)
