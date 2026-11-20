from ultralytics import YOLO
import cv2
import time

N = 2   # detecta a cada N quadros (1 = todos)

model = YOLO('yolo26n.pt')
webcam = cv2.VideoCapture(2)

contador = 0
n_frames = 0                 # NOVO
n_deteccoes = 0              # NOVO
t_janela = time.time()       # NOVO
fps = 0.0                    # NOVO
det_fps = 0.0                # NOVO

while True:
    ret, frame = webcam.read()
    if not ret:
        print("Falha ao capturar frame")
        break

    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    if contador % N == 0:
        results = model.track(frame, persist=True, verbose=False, device="cpu",
                              tracker="bytetrack.yaml", imgsz=384)
        ultimo = results[0]
        n_deteccoes += 1     # NOVO
    contador += 1
    n_frames += 1            # NOVO

    frame = ultimo.plot(img=frame)

    agora = time.time()                                   # NOVO
    if agora - t_janela >= 1.0:                           # NOVO
        fps = n_frames / (agora - t_janela)               # NOVO
        det_fps = n_deteccoes / (agora - t_janela)        # NOVO
        n_frames = 0                                      # NOVO
        n_deteccoes = 0                                   # NOVO
        t_janela = agora                                  # NOVO

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"DET/s: {det_fps:.1f}", (10, 65),   # NOVO
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()