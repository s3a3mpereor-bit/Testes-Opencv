"""
Detecção + rastreamento com YOLO para drone (versão de teste no PC).

Para levar para a Raspberry Pi 5, altere só o bloco de CONFIGURAÇÕES:
  USAR_NCNN = True, MOSTRAR_JANELA = False, ROTACAO = None,
  INDICE_USB = 0 (ou 1), ou FONTE_CAMERA = "csi".
"""

import os
import signal
import threading
import time

import cv2
import numpy as np
from ultralytics import YOLO

# ======================= CONFIGURAÇÕES (PC) =======================
FONTE_CAMERA = "usb"                # "usb" (webcam) ou "csi" (câmera da Pi, via picamera2)
INDICE_USB = 2                      # mesmo índice que você já usava
LARGURA, ALTURA = 640, 480
FPS_CAMERA = 30
ROTACAO = cv2.ROTATE_90_CLOCKWISE   # câmera do celular; use None para webcam normal

MODELO_PT = "yolo26n.pt"
MODELO_NCNN = "yolo26n_ncnn_model"  # pasta criada pela exportação
USAR_NCNN = False                   # False no PC; True na Raspberry Pi
IMGSZ = 320
CONF = 0.4
CLASSES = None                      # None = todas; [0] = só pessoas
RASTREADOR = "bytetrack.yaml"
DETECTAR_A_CADA = 1                 # no PC, 1 = detecta em todos os frames

MOSTRAR_JANELA = True               # mostra o vídeo com caixas e FPS (aperte q para sair)
TIMEOUT_CAMERA = 1.0                # segundos sem frame novo = falha de câmera
# ==================================================================


class Camera:
    """Lê a câmera em uma thread própria e guarda só o frame mais recente."""

    def __init__(self):
        self.frame = None
        self.frame_id = 0
        self.ultimo_t = 0.0
        self.rodando = True
        self.lock = threading.Lock()

        if FONTE_CAMERA == "csi":
            from picamera2 import Picamera2

            self.cam = Picamera2()
            cfg = self.cam.create_video_configuration(
                # no picamera2, "RGB888" entrega os pixels em ordem BGR, que é o que o OpenCV usa
                main={"size": (LARGURA, ALTURA), "format": "RGB888"},
                controls={"FrameRate": FPS_CAMERA},
            )
            self.cam.configure(cfg)
            self.cam.start()
            alvo = self._loop_csi
        else:
            self.cap = cv2.VideoCapture(INDICE_USB)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, LARGURA)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTURA)
            self.cap.set(cv2.CAP_PROP_FPS, FPS_CAMERA)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self.cap.isOpened():
                raise RuntimeError(f"Não foi possível abrir a câmera USB {INDICE_USB}")
            alvo = self._loop_usb

        self.thread = threading.Thread(target=alvo, daemon=True)
        self.thread.start()

    def _guardar(self, frame):
        if ROTACAO is not None:
            frame = cv2.rotate(frame, ROTACAO)
        with self.lock:
            self.frame = frame
            self.frame_id += 1
            self.ultimo_t = time.time()

    def _loop_usb(self):
        while self.rodando:
            ret, frame = self.cap.read()
            if ret:
                self._guardar(frame)
            else:
                time.sleep(0.01)

    def _loop_csi(self):
        while self.rodando:
            self._guardar(self.cam.capture_array())

    def ler(self):
        with self.lock:
            return self.frame_id, self.frame, self.ultimo_t

    def parar(self):
        self.rodando = False
        self.thread.join(timeout=1.0)
        if FONTE_CAMERA == "csi":
            self.cam.stop()
        else:
            self.cap.release()


def carregar_modelo():
    """PC: usa o .pt direto. Pi: usa NCNN (exporta na primeira vez; cai para .pt se falhar)."""
    if not USAR_NCNN:
        modelo = YOLO(MODELO_PT)
    else:
        try:
            if not os.path.isdir(MODELO_NCNN):
                print("Exportando modelo para NCNN (só na primeira vez)...")
                YOLO(MODELO_PT).export(format="ncnn", imgsz=IMGSZ)
            modelo = YOLO(MODELO_NCNN, task="detect")
        except Exception as e:
            print(f"Falha com NCNN ({e}). Usando {MODELO_PT} (mais lento).")
            modelo = YOLO(MODELO_PT)

    # aquecimento: a primeira inferência é sempre lenta
    modelo(np.zeros((ALTURA, LARGURA, 3), dtype=np.uint8), imgsz=IMGSZ, verbose=False)
    return modelo


def extrair_deteccoes(results, frame):
    """Converte o resultado do YOLO em dados simples para o controle.

    erro_x: -1.0 (borda esquerda) a +1.0 (borda direita), 0 = centro da imagem.
    area:   fração da imagem ocupada pela caixa (maior = mais perto).
    """
    alt, larg = frame.shape[:2]
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xyxy)

    saida = []
    for (x1, y1, x2, y2), p, c, i in zip(xyxy, conf, cls, ids):
        cx = (x1 + x2) / 2
        saida.append(
            {
                "id": None if i is None else int(i),
                "classe": results[0].names[c],
                "conf": float(p),
                "erro_x": float((cx - larg / 2) / (larg / 2)),
                "area": float((x2 - x1) * (y2 - y1) / (larg * alt)),
            }
        )
    return saida


def decidir(deteccoes):
    """PONTO DE INTEGRAÇÃO: aqui entra o envio de comandos ao controlador de voo
    (MAVLink, etc.). Por enquanto só imprime o que foi detectado."""
    for d in deteccoes:
        print(
            f"id={d['id']} {d['classe']} conf={d['conf']:.2f} "
            f"erro_x={d['erro_x']:+.2f} area={d['area']:.3f}"
        )


def main():
    rodando = True

    def parar(*_):
        nonlocal rodando
        rodando = False

    signal.signal(signal.SIGINT, parar)
    signal.signal(signal.SIGTERM, parar)

    modelo = carregar_modelo()
    camera = Camera()

    ultimo_id = -1
    contador = 0
    fps = 0.0
    t_det_anterior = time.time()
    t_print = time.time()
    camera_ok = True

    while rodando:
        frame_id, frame, t_frame = camera.ler()

        # ---- falha de câmera ----
        if frame is None or time.time() - t_frame > TIMEOUT_CAMERA:
            if camera_ok:
                print("ALERTA: sem frames da câmera! Acione o failsafe do drone.")
                camera_ok = False
                # aqui: avisar o controlador de voo (ex.: parar de enviar comandos de desvio)
            time.sleep(0.05)
            continue
        camera_ok = True

        # ---- só processa frame novo ----
        if frame_id == ultimo_id:
            time.sleep(0.001)
            continue
        ultimo_id = frame_id

        # ---- detectar a cada N frames ----
        contador += 1
        if contador % DETECTAR_A_CADA != 0:
            continue

        results = modelo.track(
            frame,
            persist=True,
            tracker=RASTREADOR,
            imgsz=IMGSZ,
            conf=CONF,
            classes=CLASSES,
            verbose=False,
        )

        decidir(extrair_deteccoes(results, frame))

        # ---- FPS de detecção (média suavizada) ----
        agora = time.time()
        inst = 1.0 / max(agora - t_det_anterior, 1e-6)
        t_det_anterior = agora
        fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
        if agora - t_print >= 2.0:
            print(f"[FPS de detecção: {fps:.1f}]")
            t_print = agora

        if MOSTRAR_JANELA:
            anotado = results[0].plot()
            cv2.putText(anotado, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("drone", anotado)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    camera.parar()
    cv2.destroyAllWindows()
    print("Encerrado.")


if __name__ == "__main__":
    main()