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
  binarized_image = image_processor.get_binarized_image()
  normal_thresh = image_processor.threshold(original_image)
  _, otsu_and_gaussian = image_processor.otzu_and_median_blur()

  plt.subplot(2, 2, 1), plt.imshow(original_image, cmap='gray', vmin=0, vmax=255), plt.title('Original')
  plt.subplot(2, 2, 2), plt.imshow(binarized_image, cmap='gray', vmin=0, vmax=255), plt.title('Otzu Threshold')
  plt.subplot(2, 2, 3), plt.imshow(normal_thresh, cmap='gray', vmin=0, vmax=255), plt.title('Thresh 0.5')
  plt.subplot(2, 2, 4), plt.imshow(otsu_and_gaussian, cmap='gray', vmin=0, vmax=255), plt.title('Otzu + Median')
  
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
  
  img_original = image_processor.get_original_image()
  img_binarized = image_processor.get_binarized_image()
  th_o = image_processor.get_otzu_thresh()
  th, img_otzu_and_median = image_processor.otzu_and_median_blur()
  th = th/255
  
  th_s = []
  mse_s = []
  psnr_s = []
  mae_s =[]
  
  for v in values :
    threshold_img = image_processor.threshold(img_original, v)
    mse = ImageProcessor.calc_mse(img_original, threshold_img)
    mae = ImageProcessor.calc_mae(img_original, threshold_img)
    psnr = ImageProcessor.calc_psnr(img_original, threshold_img)
    th_s.append(v)
    mse_s.append(mse)
    mae_s.append(mae)
    psnr_s.append(psnr)
  
  mse_o = ImageProcessor.calc_mse(img_original, img_binarized)
  mae_o = ImageProcessor.calc_mae(img_original, img_binarized)
  psnr_o = ImageProcessor.calc_psnr(img_original, img_binarized)
  
  th_s.append(th_o)
  th_s.append(th)
  
  mse_s.append(mse_o)
  mae_s.append(mae_o)
  psnr_s.append(psnr_o)
  
  mse_o_m = ImageProcessor.calc_mse(img_original, img_otzu_and_median)
  mae_o_m = ImageProcessor.calc_mae(img_original, img_otzu_and_median)
  psnr_o_m = ImageProcessor.calc_psnr(img_original, img_otzu_and_median)
  
  mse_s.append(mse_o_m)
  mae_s.append(mae_o_m)
  psnr_s.append(psnr_o_m)
  
  th_s = np.array(th_s)
  mse_s = np.array(mse_s)
  mae_s = np.array(mae_s)
  psnr_s = np.array(psnr_s)
  
    
  plt.title("Medicion de ejemplo".format(k_threshold))
  plt.subplot(2, 2, 1), plt.plot(th_s, mse_s, 'or', [0,1], [mse_o, mse_o], 'y--',[0, 1], [mse_o_m, mse_o_m], 'b--'), plt.xlabel("thresh"), plt.ylabel("mse")
  plt.subplot(2, 2, 2), plt.plot(th_s, psnr_s, 'or', [0,1], [psnr_o, psnr_o],'y--', [0, 1], [psnr_o_m, psnr_o_m], 'b--'), plt.xlabel("thresh"), plt.ylabel("psnr")
  plt.subplot(2, 2, 3), plt.plot(th_s, mae_s, 'or',[0,1], [mae_o, mae_o], 'y--', [0, 1], [mae_o_m, mae_o_m], 'b--'), plt.xlabel("thresh"), plt.ylabel("mae")
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
