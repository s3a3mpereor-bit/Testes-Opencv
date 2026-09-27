import cv2
import os
# Ler imagem
image_path = os.path.join('.', 'imagens', 'Anomalocaris.jpg')
img = cv2.imread(image_path)
# Ver tamanho
print(img.shape)
#Blur 1
k_size = 7
img_blur = cv2.blur(img, (k_size,k_size))
#Blur 2
img_blur2 = cv2.GaussianBlur(img, (k_size,k_size),3)
#blur 3
img_blur3 = cv2.medianBlur(img, k_size)
# Ver imagem
cv2.imshow('img', img)
cv2.imshow('img_blur', img_blur)
cv2.imshow('img_blur2', img_blur2)
cv2.imshow('img_blur3', img_blur3)
cv2.waitKey(0)
cv2.destroyAllWindows()