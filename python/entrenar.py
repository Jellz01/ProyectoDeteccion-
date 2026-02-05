import cv2
import os
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# -----------------------
# Paths completos del dataset
# -----------------------
positivos_path = "/home/jellz/Documents/VisionPorComputador/ProyectoDeteccion/imagenesPEntrenar"
negativos_path = "/home/jellz/Documents/VisionPorComputador/ProyectoDeteccion/imagenesPEntrenar/negativas"

# -----------------------
# Función para extraer features HOG + LBP
# -----------------------
def extract_features(img):
    img = cv2.resize(img, (64, 128))
    
    # --- HOG ---
    hog = cv2.HOGDescriptor()
    hog_feat = hog.compute(img)

    # --- LBP ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0,59))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    return np.concatenate([hog_feat.ravel(), lbp_hist])

# -----------------------
# Función para cargar imágenes de un folder y sus subcarpetas con progreso
# -----------------------
def load_images_recursive(folder_path, label):
    X_local = []
    y_local = []
    valid_exts = ('.jpg', '.jpeg', '.png')
    
    # Recorrer subcarpetas
    files_list = []
    for root, dirs, files in os.walk(folder_path):
        files_list += [os.path.join(root, f) for f in files if f.lower().endswith(valid_exts)]
    
    total = len(files_list)
    print(f"Cargando {total} imágenes de {folder_path} (incluyendo subcarpetas)...")
    
    for idx, img_path in enumerate(files_list, 1):
        img = cv2.imread(img_path)
        if img is None:
            print("No se pudo leer:", img_path)
            continue
        X_local.append(extract_features(img))
        y_local.append(label)
        
        # Mostrar progreso
        if idx % 50 == 0 or idx == total:
            print(f"  {idx}/{total} imágenes procesadas ({idx*100/total:.1f}%)")
    
    return X_local, y_local

# -----------------------
# Cargar dataset completo
# -----------------------
X_pos, y_pos = load_images_recursive(positivos_path, 1)
X_neg, y_neg = load_images_recursive(negativos_path, 0)

X = np.array(X_pos + X_neg)
y = np.array(y_pos + y_neg)

print("Total imágenes cargadas:", len(X))

# -----------------------
# Entrenamiento SVM
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LinearSVC(max_iter=5000)
print("Entrenando SVM...")
clf.fit(X_train, y_train)

# Evaluación
y_pred = clf.predict(X_test)
print("Accuracy en test:", accuracy_score(y_test, y_pred))

# Guardar modelo entrenado
model_path = "/home/jellz/Documents/VisionPorComputador/ProyectoDeteccion/imagenesPEntrenar/person_detector_svm.pkl"
joblib.dump(clf, model_path)
print("Modelo guardado como", model_path)
