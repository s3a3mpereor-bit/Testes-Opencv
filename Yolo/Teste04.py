from collections import defaultdict, deque
from ultralytics import YOLO
from ultralytics.utils.plotting import colorize_depth
import cv2
import numpy as np
import time

N = 2               # detecta a cada N quadros (1 = todos)
N_DEPTH = 6         # estima profundidade a cada N_DEPTH quadros (pesado na CPU)
IMGSZ_DEPTH = 768   # MUDOU: tamanho em que os pesos foram treinados (mais preciso)
DEPTH_MAX = 5.0     # metros, só para colorir o mapa
MOSTRAR_MAPA = True # janela extra com o mapa de profundidade colorido
JANELA = 7          # NOVO: nº de medidas por objeto usadas na mediana temporal

model = YOLO('yolo26n.pt')
model_depth = YOLO('yolo26n-depth.pt')
webcam = cv2.VideoCapture(2)

escala = 1.0                                   # NOVO: fator de calibração (real / estimado)
historico = defaultdict(lambda: deque(maxlen=JANELA))   # NOVO: histórico por track id


def distancia_caixa(depth, xyxy):
    """Mediana da profundidade (m) na metade central da caixa."""
    h, w = depth.shape
    x1, y1, x2, y2 = xyxy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    mw, mh = (x2 - x1) / 4, (y2 - y1) / 4   # metade central = 50% da caixa
    xa, xb = int(max(cx - mw, 0)), int(min(cx + mw, w))
    ya, yb = int(max(cy - mh, 0)), int(min(cy + mh, h))
    if xb <= xa or yb <= ya:
        return None
    regiao = depth[ya:yb, xa:xb]
    validos = regiao[regiao > 0]
    return float(np.median(validos)) if validos.size else None


contador = 0
n_frames = 0
n_deteccoes = 0
n_depth = 0
t_janela = time.time()
fps = 0.0
det_fps = 0.0
depth_fps = 0.0
ultimo = None
depth = None
mapa_cor = None
brutas = {}   # NOVO: última distância bruta (sem escala) por objeto, usada na calibração

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
        n_deteccoes += 1

    nova_profundidade = False
    if contador % N_DEPTH == 0:
        r = model_depth(frame, imgsz=IMGSZ_DEPTH, verbose=False, device="cpu")[0]
        depth = r.depth.data.cpu().numpy()   # (H, W) em metros
        nova_profundidade = True
        if MOSTRAR_MAPA:
            mapa_cor = colorize_depth(depth, vmin=0.0, vmax=DEPTH_MAX,
                                      cmap="inferno", mode="metric")
        n_depth += 1

    contador += 1
    n_frames += 1

    frame = ultimo.plot(img=frame)

    # distância de cada objeto rastreado (mediana temporal + escala calibrada)
    brutas = {}
    if depth is not None and ultimo.boxes is not None:
        xyxys = ultimo.boxes.xyxy.cpu().numpy()
        ids = (ultimo.boxes.id.int().cpu().tolist()
               if ultimo.boxes.id is not None else [None] * len(xyxys))
        for xyxy, tid in zip(xyxys, ids):
            d = distancia_caixa(depth, xyxy)
            if d is None:
                continue
            if tid is not None:
                if nova_profundidade:                 # só acumula quando há medida nova
                    historico[tid].append(d)
                d_filtrada = float(np.median(historico[tid])) if historico[tid] else d
                brutas[tid] = d_filtrada
            else:
                d_filtrada = d
            d_final = d_filtrada * escala
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.putText(frame, f"{d_final:.2f} m", (x1, min(y2 + 22, frame.shape[0] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    agora = time.time()
    if agora - t_janela >= 1.0:
        fps = n_frames / (agora - t_janela)
        det_fps = n_deteccoes / (agora - t_janela)
        depth_fps = n_depth / (agora - t_janela)
        n_frames = 0
        n_deteccoes = 0
        n_depth = 0
        t_janela = agora

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"DET/s: {det_fps:.1f}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"DEPTH/s: {depth_fps:.1f}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2)
    cv2.putText(frame, f"escala: x{escala:.2f}  [c]=calibrar  [r]=resetar", (10, 135),  # NOVO
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('frame', frame)
    if MOSTRAR_MAPA and mapa_cor is not None:
        cv2.imshow('profundidade', mapa_cor)

    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord('q'):
        break
    elif tecla == ord('r'):                                    # NOVO
        escala = 1.0
        print("Escala resetada para x1.00")
    elif tecla == ord('c'):                                    # NOVO
        # Calibração de 1 ponto: deixe UM objeto parado a uma distância conhecida
        if len(brutas) != 1:
            print(f"Calibração: precisa de exatamente 1 objeto rastreado (agora: {len(brutas)}).")
        else:
            bruta = next(iter(brutas.values()))
            try:
                real = float(input(f"Estimado (sem escala): {bruta:.2f} m. "
                                   f"Distância real em metros: ").replace(",", "."))
                if real > 0:
                    escala = real / bruta
                    print(f"Nova escala: x{escala:.3f}")
            except ValueError:
                print("Valor inválido, calibração ignorada.")

webcam.release()
cv2.destroyAllWindows()