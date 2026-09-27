import cv2
import os
# Ler imagem
image_path = os.path.join('.', 'imagens', 'Anomalocaris.jpg')
img = cv2.imread(image_path)
Gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Ver tamanho
print(img.shape)
#Threshold 1
ret, thresh = cv2.threshold(Gray,80,255,cv2.THRESH_BINARY)
thresh = cv2.blur(thresh,(2,2))
ret, thresh = cv2.threshold(thresh,80,255,cv2.THRESH_BINARY)
#Threshold 2
thres = cv2.adaptiveThreshold(Gray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 30 )
cv2.imshow('img', img)
cv2.imshow('thresh', thresh)
cv2.imshow('thres', thres)
cv2.waitKey(0)
cv2.destroyAllWindows()