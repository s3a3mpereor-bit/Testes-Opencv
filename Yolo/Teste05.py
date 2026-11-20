from collections import defaultdict, deque
from itertools import combinations
from ultralytics import YOLO
from ultralytics.utils.plotting import colorize_depth
import csv
import math
import cv2
import numpy as np
import time

N = 2               # detecta a cada N quadros (1 = todos)
N_DEPTH = 6         
IMGSZ_DEPTH = 768
DEPTH_MAX = 5.0     
MOSTRAR_MAPA = True
JANELA = 7          # nº de medidas por objeto usadas na mediana temporal

FOV_H_GRAUS = 60.0
FX = None
FY = None
CX = None
CY = None
DIST_ALERTA = 0.50      # metros: pares mais próximos que isso ficam vermelhos
JANELA_PAR = 7
MAX_LINHAS_PAINEL = 6
EXPORTAR_CSV = True
ARQUIVO_CSV = "distancias.csv"

model = YOLO('yolo26n.pt')
model_depth = YOLO('yolo26n-depth.pt')
webcam = cv2.VideoCapture(2)

escala = 1.0    # fator de calibração (real / estimado)
historico = defaultdict(lambda: deque(maxlen=JANELA))
historico_par = defaultdict(lambda: deque(maxlen=JANELA_PAR))


def distancia_caixa(depth, xyxy):
    """Mediana da profundidade (m) na metade central da caixa."""
    h, w = depth.shape
    x1, y1, x2, y2 = xyxy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    mw, mh = (x2 - x1) / 4, (y2 - y1) / 4
    xa, xb = int(max(cx - mw, 0)), int(min(cx + mw, w))
    ya, yb = int(max(cy - mh, 0)), int(min(cy + mh, h))
    if xb <= xa or yb <= ya:
        return None
    regiao = depth[ya:yb, xa:xb]
    validos = regiao[regiao > 0]
    return float(np.median(validos)) if validos.size else None


def parametros_camera(h, w):
    if FX is not None and FY is not None:
        fx, fy = FX, FY
    else:
        f = (max(h, w) / 2) / math.tan(math.radians(FOV_H_GRAUS) / 2)
        fx = fy = f
    cx = CX if CX is not None else w / 2
    cy = CY if CY is not None else h / 2
    return fx, fy, cx, cy


def ponto_3d(u, v, z, cam):
    fx, fy, cx, cy = cam
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])


contador = 0
n_frames = 0
n_deteccoes = 0
n_depth = 0
t_inicio = time.time()
t_janela = t_inicio
fps = 0.0
det_fps = 0.0
depth_fps = 0.0
ultimo = None
depth = None
mapa_cor = None
brutas = {}

arq_csv = None
escritor_csv = None
if EXPORTAR_CSV:
    arq_csv = open(ARQUIVO_CSV, "w", newline="", encoding="utf-8")
    escritor_csv = csv.writer(arq_csv)
    escritor_csv.writerow(["quadro", "tempo_s", "obj_a", "obj_b",
                           "dist_3d_m", "dist_px", "delta_z_m"])

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
        depth = r.depth.data.cpu().numpy()
        nova_profundidade = True
        if MOSTRAR_MAPA:
            mapa_cor = colorize_depth(depth, vmin=0.0, vmax=DEPTH_MAX,
                                      cmap="inferno", mode="metric")
        n_depth += 1

    contador += 1
    n_frames += 1

    frame = ultimo.plot(img=frame)

    brutas = {}
    objetos = {}
    if depth is not None and ultimo.boxes is not None:
        xyxys = ultimo.boxes.xyxy.cpu().numpy()
        ids = (ultimo.boxes.id.int().cpu().tolist()
               if ultimo.boxes.id is not None else [None] * len(xyxys))
        clss = ultimo.boxes.cls.int().cpu().tolist()
        for xyxy, tid, cls in zip(xyxys, ids, clss):
            d = distancia_caixa(depth, xyxy)
            if d is None:
                continue
            if tid is not None:
                if nova_profundidade:
                    historico[tid].append(d)
                d_filtrada = float(np.median(historico[tid])) if historico[tid] else d
                brutas[tid] = d_filtrada
                objetos[tid] = (xyxy, d_filtrada, ultimo.names[cls])
            else:
                d_filtrada = d
            d_final = d_filtrada * escala
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.putText(frame, f"{d_final:.2f} m", (x1, min(y2 + 22, frame.shape[0] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    pares_info = []
    if objetos:
        cam = parametros_camera(*frame.shape[:2])
        pontos = {}
        for tid, (xyxy, d_bruta, nome) in objetos.items():
            x1, y1, x2, y2 = xyxy
            u, v = (x1 + x2) / 2, (y1 + y2) / 2
            pontos[tid] = (u, v, ponto_3d(u, v, d_bruta * escala, cam), nome)

        for a, b in combinations(sorted(pontos), 2):
            ua, va, Pa, na = pontos[a]
            ub, vb, Pb, nb = pontos[b]
            d3 = float(np.linalg.norm(Pa - Pb))
            dpx = math.hypot(ua - ub, va - vb)
            dz = abs(float(Pa[2] - Pb[2]))

            hist = historico_par[(a, b)]
            if nova_profundidade:
                hist.append(d3)
            d_est = float(np.median(hist)) if hist else d3

            cor = (0, 0, 255) if d_est < DIST_ALERTA else (0, 255, 0)
            pa, pb = (int(ua), int(va)), (int(ub), int(vb))
            cv2.line(frame, pa, pb, cor, 2)
            meio = ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2)
            cv2.putText(frame, f"{d_est:.2f} m", meio,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
            cv2.putText(frame, f"{dpx:.0f}px  dZ {dz:.2f}", (meio[0], meio[1] + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, cor, 1)

            pares_info.append((d_est, na, a, nb, b))
            if escritor_csv is not None and nova_profundidade:
                escritor_csv.writerow([contador, round(time.time() - t_inicio, 3),
                                       f"{na}#{a}", f"{nb}#{b}",
                                       round(d_est, 3), round(dpx, 1), round(dz, 3)])
        if arq_csv is not None and nova_profundidade:
            arq_csv.flush()

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
    cv2.putText(frame, f"escala: x{escala:.2f}  [c]=calibrar  [r]=resetar", (10, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    pares_info.sort(key=lambda p: p[0])
    for i, (d_est, na, a, nb, b) in enumerate(pares_info[:MAX_LINHAS_PAINEL]):
        cor = (0, 0, 255) if d_est < DIST_ALERTA else (0, 255, 0)
        cv2.putText(frame, f"{na}#{a} <-> {nb}#{b}: {d_est:.2f} m", (10, 165 + 25 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

    cv2.imshow('frame', frame)
    if MOSTRAR_MAPA and mapa_cor is not None:
        cv2.imshow('profundidade', mapa_cor)

    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord('q'):
        break
    elif tecla == ord('r'):
        escala = 1.0
        print("Escala resetada para x1.00")
    elif tecla == ord('c'):
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
if arq_csv is not None:
    arq_csv.close()
cv2.destroyAllWindows()