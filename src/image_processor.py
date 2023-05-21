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
    self.original_image = cv2.imread(image_url, 0)
  

  def otzu_threshold(self, image):
    result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return result
  
  def threshold(self, img, thresh = 0.5):
    image = np.copy(img)
    image[image >= (thresh*255)] = 255
    image[image < (thresh*255)] = 0
    return image
  
  def get_otzu_thresh(self):
    return self.otzu_thresh
  
 
    
  def get_original_image(self):
    """
      Retornamos la imagen original
    """
    return self.original_image

  def get_binarized_image(self):
    """
      Retornamos la imagen binarizada
    """
    _, binarized_img = self.otzu_threshold(self.original_image)
    self.binarized_image = binarized_img
    self.otzu_thresh = _/255
    return self.binarized_image

  @staticmethod
  def calc_mse(original_image, stimated_image):
    """
      Calculamos el Error Cuadratico Medio de Acuerdo entre dos matrices.
    """
    
    diff = original_image - stimated_image
    
    mse = np.mean(diff**2)
    return mse

  @staticmethod
  def calc_psnr(original_image, stimated_image):
    """
      Calculamos el Proporción Máxima de Señal a Ruido dado un mse
    """
    
    mse = ImageProcessor.calc_mse(original_image, stimated_image)
    psnr = sqrt(mse)
    return psnr

  @staticmethod
  def calc_mae(original_image, stimated_image):
    """
      Calculamos el Error absoluto medio de Acuerdo entre dos matrices.
    """
    mae = np.mean(np.abs(original_image - stimated_image))
    return mae
