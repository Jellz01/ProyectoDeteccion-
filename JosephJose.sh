#!/bin/bash
echo "=============================================="
echo "🚀 DEMO: SISTEMA INTEGRADO + MATRIZ (RTX 3050 Ti)"
echo "=============================================="

export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
PROJECT_DIR="$HOME/Documents/VisionPorComputador/ProyectoDeteccion"
PY_DIR="$PROJECT_DIR/python"

# 1. Limpieza
pkill -f python3
sleep 1

# 2. Compilación
cd "$PROJECT_DIR/build" || exit
make -j$(nproc)

# 3. Lanzar API e IA
source "$PROJECT_DIR/venv/bin/activate"

echo "🧠 Iniciando API de IA (pose.py)..."
python3 -u "$PY_DIR/pose.py" & 
POSE_PID=$!

echo "⏳ Esperando carga de YOLO..."
sleep 8

echo "📊 Lanzando Matriz de Confusión..."
python3 -u "$PY_DIR/plotmatriz.py" &
PLOT_PID=$!

echo "🤖 Iniciando Bot de Telegram..."
python3 -u "$PY_DIR/Bot/Bot.py" & 
BOT_PID=$!

sleep 2

# 4. Ejecutar C++
echo "📷 Iniciando Detector HOG..."
./detect_pedestrians

# Al cerrar (Ctrl+C en la terminal)
kill $POSE_PID $BOT_PID $PLOT_PID