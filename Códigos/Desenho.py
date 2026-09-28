import cv2
import os

# Ler imagem
image_path = os.path.join('.', 'imagens', 'Falcon.jpg')
img = cv2.imread(image_path)
imgredimencionada = cv2.resize(img, (800,600,))
print(imgredimencionada.shape)
# Linha: (ponto 1) (ponto 2) (cor BGR) (espessura)
cv2.line(imgredimencionada, (100, 150), (300, 450), (0, 255, 0), 3)
# Retângulo: (canto superior esquerdo) (canto inferior direito) (cor BGR) (espessura)
cv2.rectangle(imgredimencionada, (100, 300), (350, 400), (255, 0, 0), 6)
# Círculo: (centro) (raio) (cor BGR) (espessura, -1 = preenchido)
cv2.circle(imgredimencionada, (200, 200), 50, (0, 0, 255), -1)
# Texto: (texto) (posição do canto inferior esquerdo) (fonte) (escala) (cor BGR) (espessura)
cv2.putText(imgredimencionada, 'Hello World', (100, 100), cv2.FONT_ITALIC, 1, (0, 255, 255), 1)
# Ver imagem
cv2.imshow('img', imgredimencionada)
cv2.waitKey(0)
cv2.destroyAllWindows()