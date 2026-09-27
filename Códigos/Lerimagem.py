import cv2
import os
# Ler imagem
image_path = os.path.join('.', 'imagens', 'Falcon.jpg')
img = cv2.imread(image_path)
#escrever imagem
cv2.imwrite(os.path.join('.', 'imagens', 'Falconon.jpg'), img)
#Ver imagem
cv2.imshow('image', img)
cv2.waitKey(0)