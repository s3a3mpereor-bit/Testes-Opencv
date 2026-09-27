import cv2
import os
# Ler imagem
image_path = os.path.join('.', 'imagens', 'Falcon.jpg')
img = cv2.imread(image_path)
imgredimencionada = cv2.resize(img, (640,480,))
#Ver tamanho
print(img.shape)
print(imgredimencionada.shape)
#cortar a imagem
cropped_img = imgredimencionada[320:640, 420:840]
#Ver imagem
cv2.imshow('image', img)
cv2.imshow('imgredimencionada', imgredimencionada)
cv2.imshow('cropped_img', cropped_img)
cv2.waitKey(0)