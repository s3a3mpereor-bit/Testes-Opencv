import cv2

#Escolha da Webcam de acordo com o numero
webcam = cv2.VideoCapture(2)

while True:
    #ler o video da webcam
    ret, frame = webcam.read()
    if not ret:
        print("Falha ao capturar frame")
        break
    #Usado para rotacionar a camera frontal do celular
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow('frame', frame)
    #fecha a webcam quando pressionar q
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()