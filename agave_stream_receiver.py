"""
agave_stream_receiver.py
───────────────────────────────────────────────
Autores:    - Arian Yolanda Reyes Aguilar - A01795124
            - Oscar Nava Jiménez - A01745524
            - Bruno Sánchez García - A01378960
Descripción:
    Script servidor para recibir un flujo de video (streaming)
    enviado desde un cliente remoto mediante sockets TCP.

    El servidor escucha en una dirección IP y puerto específicos,
    recibe tramas serializadas con pickle y las decodifica
    con OpenCV para mostrarlas en una ventana en tiempo real.

Uso:
    Ejecutar en la máquina receptora antes de iniciar el cliente emisor.
    Presionar 'q' para cerrar la ventana y terminar el programa.
"""

import socket   # Comunicación entre cliente y servidor
import struct   # Empaquetado/desempaquetado de datos binarios
import pickle   # Serialización de objetos Python (en este caso, imágenes codificadas)
import cv2      # Procesamiento y visualización de imágenes

# ------------------------------------------------------------
# Configuración del servidor
# ------------------------------------------------------------
server_ip = "0.0.0.0"      # Acepta conexiones desde cualquier interfaz de red
server_port = 9999         # Puerto de escucha del servidor

# ------------------------------------------------------------
# Creación y configuración del socket TCP
# ------------------------------------------------------------
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((server_ip, server_port))
server.listen(1)  # Se permite solo una conexión simultánea
print(f"Escuchando en {server_ip}:{server_port}")

# Espera una conexión entrante
conn, _ = server.accept()
print("Cliente conectado exitosamente")

# ------------------------------------------------------------
# Recepción y reconstrucción de los frames enviados
# ------------------------------------------------------------
data = b""  # Buffer para almacenar los bytes recibidos
payload_size = struct.calcsize(">L")  # Tamaño del encabezado (4 bytes, formato big-endian)

# Bucle principal de recepción
while True:
    # Asegura que haya suficientes bytes para leer el tamaño del mensaje
    while len(data) < payload_size:
        packet = conn.recv(4096)  # Recibe hasta 4096 bytes
        if not packet:
            break  # Si el cliente se desconecta, termina el bucle
        data += packet

    # Extrae y desempaqueta el tamaño del mensaje (en bytes)
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]

    # Espera hasta recibir el frame completo
    while len(data) < msg_size:
        data += conn.recv(4096)

    # Extrae los datos del frame y limpia el buffer
    frame_data = data[:msg_size]
    data = data[msg_size:]

    # Deserializa y decodifica la imagen
    frame = pickle.loads(frame_data)
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

    # --------------------------------------------------------
    # Visualización del frame recibido
    # --------------------------------------------------------
    cv2.imshow("Remote Stream", frame)

    # Sale del bucle si el usuario presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ------------------------------------------------------------
# Cierre ordenado de conexiones y ventanas
# ------------------------------------------------------------
conn.close()
server.close()
cv2.destroyAllWindows()
print("Conexión cerrada correctamente.")
