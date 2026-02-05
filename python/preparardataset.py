import scipy.io
import cv2
import os
import numpy as np
import shutil

# --- RUTAS ---
base = os.path.expanduser("~/Documents/VisionPorComputador/ProyectoDeteccion/imagenesPEntrenar")
mat_path = os.path.join(base, "mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat")
neg_src = os.path.join(base, "negativas/images.cv_76qe20679lbnvenn96u04/data/train")

# Carpetas de salida limpias
dataset_limpio = os.path.join(base, "dataset_limpio")
pos_out = os.path.join(dataset_limpio, "pos")
neg_out = os.path.join(dataset_limpio, "neg")

# Limpiar directorio si ya existe para evitar mezclar datos viejos
if os.path.exists(dataset_limpio):
    print(f"🧹 Limpiando carpeta antigua en {dataset_limpio}...")
    shutil.rmtree(dataset_limpio)

os.makedirs(pos_out)
os.makedirs(neg_out)

# 1. PROCESAR POSITIVAS (Personas de MPII)
print("⏳ Extrayendo personas de MPII (esto puede tardar un poco)...")
mat = scipy.io.loadmat(mat_path)
annolist = mat['RELEASE']['annolist'][0,0][0]
count_pos = 0

for i in range(len(annolist)):
    img_info = annolist[i]
    img_name = img_info['image'][0,0]['name'][0]
    img_path = os.path.join(base, img_name)
    
    if not os.path.exists(img_path):
        continue

    # Verificar si hay anotaciones de rectángulos
    if 'annorect' in img_info.dtype.names and img_info['annorect'].size > 0:
        rects = img_info['annorect'][0]
        img = cv2.imread(img_path)
        if img is None: continue
        
        for j in range(len(rects)):
            try:
                obj = rects[j]
                
                # VALIDACIÓN CRÍTICA: Verificar si tiene los campos necesarios antes de acceder
                if 'objpos' not in obj.dtype.names or 'scale' not in obj.dtype.names:
                    continue
                
                # Si el campo existe pero está vacío, saltar
                if obj['objpos'].size == 0 or obj['scale'].size == 0:
                    continue

                c_x = obj['objpos'][0,0]['x'][0,0]
                c_y = obj['objpos'][0,0]['y'][0,0]
                sc = obj['scale'][0,0]
                
                # Ajuste de ventana HOG (proporción 1:2)
                h = int(sc * 200 * 1.3)
                w = int(h * 0.5)
                
                y1, y2 = max(0, int(c_y - h/2)), min(img.shape[0], int(c_y + h/2))
                x1, x2 = max(0, int(c_x - w/2)), min(img.shape[1], int(c_x + w/2))
                
                crop = img[y1:y2, x1:x2]
                # Filtro de calidad: ignorar recortes demasiado pequeños o vacíos
                if crop.size > 0 and crop.shape[0] > 40 and crop.shape[1] > 20:
                    resized = cv2.resize(crop, (64, 128))
                    cv2.imwrite(f"{pos_out}/p_{count_pos}.jpg", resized)
                    count_pos += 1
            except Exception:
                # Cualquier otro error estructural lo ignoramos para seguir procesando
                continue
    
    if count_pos % 500 == 0 and count_pos > 0:
        print(f"   ...llevamos {count_pos} personas extraídas")
        
    if count_pos >= 4000: 
        break

# 2. PROCESAR NEGATIVAS (Fondos)
print("\n⏳ Preparando recortes negativos...")
count_neg = 0

for root, dirs, files in os.walk(neg_src):
    for f in files:
        if not f.lower().endswith(('.jpg', '.png', '.jpeg')): 
            continue
            
        img_path = os.path.join(root, f)
        img = cv2.imread(img_path)
        if img is None: 
            continue
        
        h, w = img.shape[:2]
        
        # VALIDACIÓN: Solo si la imagen es suficientemente grande para el recorte
        if h > 128 and w > 64:
            for _ in range(3):
                y_max = h - 128
                x_max = w - 64
                
                y = np.random.randint(0, y_max)
                x = np.random.randint(0, x_max)
                
                roi = img[y:y+128, x:x+64]
                cv2.imwrite(f"{neg_out}/n_{count_neg}.jpg", roi)
                count_neg += 1
                
                if count_neg >= 6000: break
        else:
            # Si es pequeña, redimensionamos
            resized = cv2.resize(img, (64, 128))
            cv2.imwrite(f"{neg_out}/n_{count_neg}.jpg", resized)
            count_neg += 1
            
        if count_neg >= 6000: 
            break
    if count_neg >= 6000: 
        break

print("-" * 30)
print(f"✅ ¡PROCESO COMPLETADO!")
print(f"📁 Positivos generados (poses): {count_pos}")
print(f"📁 Negativos generados (fondo): {count_neg}")
print(f"📍 Ubicación: {dataset_limpio}")
print("-" * 30)