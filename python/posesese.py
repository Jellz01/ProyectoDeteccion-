import os
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"
import cv2
import numpy as np
import joblib
from skimage.feature import local_binary_pattern
from imutils.object_detection import non_max_suppression

# --- Rutas ---
model_path = "/home/jellz/Documents/VisionPorComputador/ProyectoDeteccion/imagenesPEntrenar/person_detector_svm.pkl"
video_path = "/home/jellz/Documents/VisionPorComputador/ProyectoDeteccion/python/videoppp.mp4"

clf = joblib.load(model_path)
hog = cv2.HOGDescriptor()

def extract_features(img):
    img_res = cv2.resize(img, (64, 128))
    h_feat = hog.compute(img_res).ravel()
    gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59))
    hist = hist.astype("float") / (hist.sum() + 1e-7)
    return np.concatenate([h_feat, hist])

cap = cv2.VideoCapture(video_path)

# --- CONFIGURACIÓN AGRESIVA PARA QUE DETECTE ---
step_size = 16        # Escaneo ultra fino (lento pero no se salta nada)
umbral_svm = -0.5     # UMBRAL NEGATIVO: Forzamos a que acepte "dudas"
skip_frames = 4       # Subimos para compensar la lentitud del step_size

cv2.namedWindow("FORZANDO DETECCION", cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # IMPORTANTE: En tu foto te ves pequeño. Bajamos la resolución de escaneo 
    # para que tú ocupes más espacio relativo en la imagen.
    frame_scan = cv2.resize(frame, (320, 240))
    h, w = frame_scan.shape[:2]
    
    rects, scores = [], []
    
    # Escaneamos solo donde estás tú (mitad derecha y abajo) para ganar velocidad
    # Si quieres toda la pantalla, quita los rangos en y/x
    for y in range(0, h - 128, step_size):
        for x in range(0, w - 64, step_size):
            window = frame_scan[y:y+128, x:x+64]
            feat = extract_features(window)
            score = clf.decision_function([feat])[0]
            
            if score > umbral_svm:
                scale_x = frame.shape[1] / w
                scale_y = frame.shape[0] / h
                rects.append([int(x*scale_x), int(y*scale_y), int((x+64)*scale_x), int((y+128)*scale_y)])
                scores.append(score)

    if len(rects) > 0:
        pick = non_max_suppression(np.array(rects), probs=np.array(scores), overlapThresh=0.3)
        for (x1, y1, x2, y2) in pick:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("FORZANDO DETECCION", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()