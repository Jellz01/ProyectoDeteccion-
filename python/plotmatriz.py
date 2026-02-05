import requests
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import pandas as pd
import time

# Configuración
API_URL = "http://localhost:8000/api/confusion"

def get_data():
    try:
        response = requests.get(API_URL)
        if response.status_code == 200:
            return response.json()
    except:
        return {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

fig, ax = plt.subplots(figsize=(6, 5))

def animate(i):
    data = get_data()
    
    # Estructuramos la matriz
    # Filas: HOG (Predicción), Columnas: YOLO (Realidad)
    matrix = [
        [data["TP"], data["FP"]],
        [data["FN"], data["TN"]]
    ]
    
    df_cm = pd.DataFrame(matrix, 
                         index=['HOG Persona', 'HOG Nada'],
                         columns=['YOLO Persona', 'YOLO Nada'])
    
    ax.clear()
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    
    ax.set_title(f"Matriz de Confusión en Tiempo Real\nTotal HOG Detections: {data['TP'] + data['FP']}")
    ax.set_xlabel("Verdad (YOLOv8)")
    ax.set_ylabel("Predicción (HOG)")

# Animación cada 1 segundo (1000ms)
ani = animation.FuncAnimation(fig, animate, interval=1000)

plt.show()