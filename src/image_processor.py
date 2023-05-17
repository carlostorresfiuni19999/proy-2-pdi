# -*- coding: utf-8 -*-
import cv2
import numpy as np
from math import log10, sqrt  

class ImageProcessor(object):
  """docstring for ImageProcessor"""
  def __init__(self, image_url):
    super(ImageProcessor, self).__init__()
    self.init(image_url)

  def init(self, image_url):
    image = cv2.imread(image_url, 0)
    self.original_image = image

  def threshold(image, thresh = 0.5):
    image[image >= thresh] = 255
    image[image < thresh] = 0
    return image

  def get_original_image(self):
    """
      Retornamos la imagen original
    """
    return self.original_image

  def get_binarized_image(self):
    """
      Retornamos la imagen binarizada
    """

    #Guardamos la copia de la imagen original
    img_copy = np.copy(self.get_original_image)
    binarized_img = self.threshold(img_copy)
    self.get_binarized_image = binarized_img
    return self.binarized_image

  @staticmethod
  def calc_mse(original_image, stimated_image):
    """
      Calculamos el Error Cuadratico Medio de Acuerdo entre dos matrices.
    """
    mse = 0
    return mse

  @staticmethod
  def calc_psnr(original_image, stimated_image):
    """
      Calculamos el Proporción Máxima de Señal a Ruido dado un mse
    """
    psnr = 0
    return psnr

  @staticmethod
  def calc_mae(original_image, stimated_image):
    """
      Calculamos el Error absoluto medio de Acuerdo entre dos matrices.
    """
    mae = 0
    return mae
