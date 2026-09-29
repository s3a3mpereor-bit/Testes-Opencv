import cv2
import numpy as np
#Qual camera usar
CAMERA_INDEX = 2  # índice em que a câmera virtual do OBS apareceu no DSHOW

# Valores padrão dos sliders
CANNY_MIN = 3
CANNY_MAX = 80
FECHAR = 3
COMP_MIN = 0
BLUR = 7

def nada(x):
    pass

webcam = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not webcam.isOpened():
    print(f"Não consegui abrir a câmera no índice {CAMERA_INDEX}")
    exit()

# aqui não pedimos MJPG: a câmera virtual do OBS entrega a resolução configurada no OBS
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print(webcam.get(cv2.CAP_PROP_FRAME_WIDTH), webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
#Create trackbar(nome,janela,valor inicial,valor maximo.função ao mexer)
cv2.namedWindow('controles')
cv2.createTrackbar('canny_min', 'controles', CANNY_MIN, 255, nada)
cv2.createTrackbar('canny_max', 'controles', CANNY_MAX, 255, nada)
cv2.createTrackbar('fechar', 'controles', FECHAR, 25, nada)
cv2.createTrackbar('comp_min', 'controles', COMP_MIN, 1000, nada)
cv2.createTrackbar('blur', 'controles', BLUR, 25, nada)

while True:
    #captura de frame
    ret, frame = webcam.read()
    if not ret:
        print("Falha ao capturar frame")
        break
    #Rotação da camera
    img = frame
    # img = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    #valor inicial sliders
    c_min = cv2.getTrackbarPos('canny_min', 'controles')
    c_max = cv2.getTrackbarPos('canny_max', 'controles')
    k = max(1, cv2.getTrackbarPos('fechar', 'controles'))
    comp_min = cv2.getTrackbarPos('comp_min', 'controles')
    blur = cv2.getTrackbarPos('blur', 'controles')
    if blur % 2 == 0:      # força ímpar (0 vira 1, 2 vira 3, etc.)
        blur += 1
    #conversão de cor
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #tirarruido
    gray = cv2.GaussianBlur(gray, (blur, blur), 0)

    edges = cv2.Canny(gray, c_min, c_max)

    kernel_morf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_morf)
    #função de achar contorno
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        if cv2.arcLength(cnt, False) > comp_min:
            cv2.drawContours(img, [cnt], -1, (0, 0, 255), 2)
    #ver webcam
    cv2.imshow('img', img)
    cv2.imshow('edges', edges)
    #sair da janela
    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord('q') or tecla == 27:
        break
    if (cv2.getWindowProperty('img', cv2.WND_PROP_VISIBLE) < 1 or
            cv2.getWindowProperty('edges', cv2.WND_PROP_VISIBLE) < 1):
        break

webcam.release()
cv2.destroyAllWindows()
cv2.waitKey(1)