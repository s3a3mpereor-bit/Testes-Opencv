import cv2
import os
# Ler imagem
image_path = os.path.join('.', 'imagens', 'Anomalocaris.jpg')
img = cv2.imread(image_path)
# Ver tamanho
print(img.shape)
#converter espaço de cor de BGR para outro espaço de cor
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# Ver imagem, posicionando as janelas separadas na tela
cv2.namedWindow('img')
cv2.moveWindow('img', 100, 100)
cv2.imshow('img', img)
cv2.namedWindow('img_rgb')
cv2.moveWindow('img_rgb', 700, 100)
cv2.imshow('img_rgb', img_rgb)
cv2.imshow('img_gray', img_gray)
cv2.imshow('img_hsv', img_hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()