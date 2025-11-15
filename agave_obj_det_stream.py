"""
agave_obj_det_stream.py
───────────────────────────────────────────────
Autores:    - Arian Yolanda Reyes Aguilar - A01795124
            - Oscar Nava Jiménez - A01745524
            - Bruno Sánchez García - A01378960
Descripción:
    Script para detección en tiempo real de agaves mediante un
    modelo TransformerObjectDetection. Captura video desde una
    cámara local, realiza inferencia cuadro por cuadro y transmite
    los resultados anotados hacia un servidor remoto por socket TCP.

Dependencias:
    - OpenCV (cv2)
    - PyTorch
    - Pillow
    - Numpy
    - Módulos locales: model.transformer, utils.counting

Uso:
    Ejecutar este script en el dispositivo que posee la cámara.
    Asegurarse de que el servidor remoto esté ejecutando el script
    receptor antes de iniciar la transmisión.
"""

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
import socket
import struct
import pickle

import model.transformer
import utils.counting


# ---------------------------------------------------------------------
# Configuración del dispositivo y modelo
# ---------------------------------------------------------------------
DEVICE = torch.device("cpu")  # Se puede cambiar a "cuda" si hay GPU disponible

# Ruta al modelo entrenado
MODEL_PATH = (
    "/home/bsgrp5/Documents/bitsted/bit-STED/Models AI Master Final/medfv1.pt"

)

# Parámetros de la arquitectura
N_MODEL = 512
NUM_BLOCKS = 2


# ---------------------------------------------------------------------
# Carga del modelo
# ---------------------------------------------------------------------
model = model.transformer.TransformerObjectDetection(
    224,
    N_channels=3,
    n_model=N_MODEL,
    num_blks=NUM_BLOCKS,
    obj="cbbox",
    device=DEVICE,
    bitNet=True,
).to(DEVICE)

# Carga del estado entrenado desde el archivo .pt
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
state_dict = checkpoint["model"] if isinstance(checkpoint, dict) else checkpoint
model.load_state_dict(state_dict)
model.eval()
print(f"Modelo cargado en {DEVICE}")

# Transformación básica a tensor
TRANSFORM = transforms.ToTensor()


# ---------------------------------------------------------------------
# Función de inferencia
# ---------------------------------------------------------------------
def infer_frame(frame):
    """
    Ejecuta la inferencia sobre un solo cuadro (frame) de video
    y dibuja las detecciones de agaves sobre la imagen.

    Args:
        frame (np.ndarray): Frame en formato BGR proveniente de OpenCV.

    Returns:
        np.ndarray: Frame anotado con detecciones.
    """
    # Obtiene dimensiones originales
    orig_h, orig_w = frame.shape[:2]

    # Conversión a RGB y normalización de valores
    img_np = frame[..., ::-1]
    img_tensor = (
        torch.from_numpy(img_np.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )

    # Redimensiona la imagen para adaptarla al tamaño de entrada del modelo
    target_size = 224
    img_tensor = torch.nn.functional.interpolate(
        img_tensor,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    ).to(DEVICE)

    # Ejecución de la inferencia mediante la función auxiliar del módulo utils
    boxes, scores, classes, _ = utils.counting.inference(
        model, img_tensor, "cbbox", conf_thr=0.95, diou_thr=0.001
    )

    boxes, scores = boxes[0], scores[0]

    # Conversión a formato PIL para dibujar anotaciones
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    if boxes is not None:
        # Cálculo de factores de escala entre la imagen original y la de entrada
        scale_x = orig_w / target_size
        scale_y = orig_h / target_size

        # Itera sobre las detecciones válidas
        for box, score in zip(boxes, scores):
            xc, yc, r = box

            # Reescalado de coordenadas al tamaño original
            xc *= scale_x
            yc *= scale_y
            r *= (scale_x + scale_y) / 2.0  # Escalado promedio

            # Dibuja un círculo representando la detección
            color = (0, 255, 0)
            draw.ellipse(
                (xc - r, yc - r, xc + r, yc + r),
                outline=color,
                width=2,
            )
            draw.text((xc, yc - 10), f"{score * 100:.1f}%", fill=color)

    # Convierte de nuevo a formato BGR para visualización con OpenCV
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------
# Bucle principal de captura y transmisión
# ---------------------------------------------------------------------
def main():
    """
    Captura video en tiempo real desde la cámara local,
    realiza detección de agaves y envía los cuadros procesados
    a un servidor remoto usando sockets TCP.
    """
    remote_host = "192.168.0.23"  # Dirección IP del receptor
    remote_port = 9999            # Puerto TCP remoto

    # Inicializa la cámara
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Error: no se puede acceder a la cámara.")
        return

    print("Transmisión iniciada. Presiona 'q' para salir.")

    # Configura el socket cliente
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((remote_host, remote_port))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Inferencia y anotación del cuadro
            result_frame = infer_frame(frame)

            # Codificación a formato JPEG para reducir tamaño
            encoded, buffer = cv2.imencode(".jpg", result_frame)
            data = pickle.dumps(buffer)

            # Empaqueta el mensaje con su longitud
            msg = struct.pack(">L", len(data)) + data
            client.sendall(msg)

            # Salida si se presiona 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Liberación ordenada de recursos
        cap.release()
        client.close()
        cv2.destroyAllWindows()
        print("Transmisión finalizada correctamente.")


# ---------------------------------------------------------------------
# Punto de entrada del script
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
