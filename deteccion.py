import cv2
import numpy as np
from ultralytics import YOLO
from cv2_enumerate_cameras import enumerate_cameras

# 1. Cargar Parámetros de Calibración
try:
    data = np.load('calibracion_estereo.npz')
    mtx_l, dist_l = data['mtx_l'], data['dist_l']
    mtx_r, dist_r = data['mtx_r'], data['dist_r']
    R1, R2, P1, P2 = data['R1'], data['R2'], data['P1'], data['P2']
    Q = data['Q'].astype(np.float32)
    print("Matrices de calibración cargadas.")
except Exception as e:
    print(f"Error al cargar .npz: {e}")
    exit()

# 2. Configuración de Stereo SGBM (Más robusto que BM)
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,  # Debe ser divisible por 16
    blockSize=5,
    P1=8 * 3 * 5**2,    # Parámetros de suavizado
    P2=32 * 3 * 5**2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=100,
    speckleRange=32
)

model = YOLO('yolov8n.pt') 

def buscar_indice():
    for cam in enumerate_cameras():
        if "3D USB Camera" in cam.name: return cam.index
    return 0

cap = cv2.VideoCapture(buscar_indice())
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

img_size = (1280, 720)
map1x, map1y = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, img_size, cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, img_size, cv2.CV_32FC1)

while True:
    ret, frame = cap.read()
    if not ret: break

    # 3. Separar y Rectificar
    imgL = frame[:, :1280]
    imgR = frame[:, 1280:]
    rectL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
    rectR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

    # 4. Mapa de Disparidad denso
    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

    # 5. Detección con YOLO
    results = model(rectL, conf=0.5, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]

            # --- ORIENTACIÓN REAL (minAreaRect) ---
            # Extraemos la máscara o contorno simple del ROI para el ángulo
            roi_gray = grayL[y1:y2, x1:x2]
            _, thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            angulo = 0
            if contours:
                c = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(c)
                angulo = rect[2] # Ángulo real

            # --- DISTANCIA ROBUSTA (Mediana de Disparidad) ---
            # En lugar de un pixel, usamos el centro del bounding box con un margen
            margin = 0.2
            bx1, by1 = int(x1 + (x2-x1)*margin), int(y1 + (y2-y1)*margin)
            bx2, by2 = int(x2 - (x2-x1)*margin), int(y2 - (y2-y1)*margin)
            
            disp_roi = disparity[by1:by2, bx1:bx2]
            # Filtramos valores de disparidad inválidos (<= 0)
            valid_disp = disp_roi[disp_roi > 0]
            
            distancia_m = 0
            if valid_disp.size > 0:
                med_disp = np.median(valid_disp)
                
                # CORRECCIÓN: Cálculo manual usando la matriz Q para evitar el error de aserción
                # La fórmula de distancia en estéreo rectificado es: Z = (f * baseline) / disparidad
                # En la matriz Q: Q[2,3] es f y Q[3,2] es -1/baseline
                
                f = Q[2, 3]
                inv_baseline = abs(Q[3, 2])
                
                if med_disp > 0:
                    # Z = f / (disparidad * (1/baseline))
                    distancia_z = f / (med_disp * inv_baseline)
                    distancia_m = distancia_z / 100.0  # Conversión a metros
                else:
                    distancia_m = 0

            # Visualización
            color = (0, 255, 0)
            cv2.rectangle(rectL, (x1, y1), (x2, y2), color, 2)
            cv2.putText(rectL, f"{label} | Ang:{angulo:.1f} | Dist:{distancia_m:.2f}m", 
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Sistema Estéreo Evolucionado', rectL)
    # Opcional: ver el mapa de disparidad
    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imshow('Mapa de Disparidad', disp_vis)

    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()