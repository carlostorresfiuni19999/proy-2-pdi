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
    self.original_image = cv2.imread(image_url)

  

  def otzu_threshold(self, image):
    result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return result
  
  def threshold(self, img, thresh = 0.5):
    image = np.copy(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image[image >= (thresh*255)] = 255
    image[image < (thresh*255)] = 0
    return image
  
  def get_otzu_thresh(self):
    return self.otzu_thresh
  
  def otzu_and_median_blur(self):
    img_copy = np.copy(self.original_image)
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    img_copy = cv2.medianBlur(img_copy, 7)
    return self.otzu_threshold(img_copy)
    
 
    
  def get_original_image(self):
    """
      Retornamos la imagen original
    """
    return self.original_image

  def get_binarized_image(self):
    """
      Retornamos la imagen binarizada
    """
    
    gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
    _, binarized_img = self.otzu_threshold(gray)
    self.binarized_image = binarized_img
    self.otzu_thresh = _/255
    return self.binarized_image

  @staticmethod
  def calc_mse(original_image, stimated_image):
    """
      Calculamos el Error Cuadratico Medio de Acuerdo entre dos matrices.
    """
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    diff = gray - stimated_image
    
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
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    diff = gray - stimated_image
    mae = np.mean(np.abs(diff))
    return mae
