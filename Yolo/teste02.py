from ultralytics import YOLO
import cv2
import time

model = YOLO('yolo26n.pt')
webcam = cv2.VideoCapture(2)

tempo_anterior = time.time()

while True:
    ret, frame = webcam.read()
    if not ret:
        print("Falha ao capturar frame")
        break

    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    results = model.track(frame, persist=True, verbose=False, imgsz=320)
    frame = results[0].plot()

    # calcula o FPS
    tempo_atual = time.time()
    fps = 1 / (tempo_atual - tempo_anterior)
    tempo_anterior = tempo_atual

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()