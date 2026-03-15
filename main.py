import cv2
import os
import time
from cv2_enumerate_cameras import enumerate_cameras

# Configuración de carpetas
folder = 'dataset_stereo'
if not os.path.exists(folder):
    os.makedirs(folder)

def buscar_indice_camara(nombre_objetivo):
    for camera_info in enumerate_cameras():
        if nombre_objetivo in camera_info.name:
            return camera_info.index
    return None

indice = buscar_indice_camara("3D USB Camera")
cap = cv2.VideoCapture(indice)

# Resolución nativa para capturar ambos lentes sin pérdida
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Captura preparada. Teclas: [S] Guardar | [Q] Salir")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Mostramos una guía visual (línea central) para verificar alineación
    preview = frame.copy()
    cv2.line(preview, (1280, 0), (1280, 720), (0, 255, 0), 2)
    cv2.imshow('Captura para Entrenamiento 3D', preview)

    key = cv2.waitKey(1)
    if key == ord('s'):
        # Usamos timestamp para evitar cualquier duplicado
        ts = int(time.time())
        img_path = os.path.join(folder, f"stereo_frame_{ts}.jpg")
        
        # Guardamos el frame original (sin la línea de guía)
        cv2.imwrite(img_path, frame)
        print(f"Frame estéreo guardado: {img_path}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()