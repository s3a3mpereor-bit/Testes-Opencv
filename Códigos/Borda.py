import cv2
import os
import numpy as np
# Ler imagem
image_path = os.path.join('.', 'imagens', 'Falcon.jpg')
img = cv2.imread(image_path)
imgredimencionada = cv2.resize(img, (640,480,))
#detector de borda
img_edge = cv2.Canny(imgredimencionada, 120, 150)
#dilatar bordas
img_edge_dilate = cv2.dilate(img_edge, np.ones((2,2), dtype=np.int8))
img_edge_e = cv2.erode(img_edge, np.ones((2,2), dtype=np.int8))
cv2.imshow('img', imgredimencionada)
cv2.imshow('img_edge', img_edge)
cv2.imshow('img_edge_dilate', img_edge_dilate)
cv2.imshow('img_edge_e', img_edge_e)
cv2.waitKey(0)
cv2.destroyAllWindows()