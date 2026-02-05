import cv2
import numpy as np
import torch
import io
import uvicorn
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from ultralytics import YOLO
from telegram import Bot
from datetime import datetime

app = FastAPI()

# --- ESTADO Y ESTADÍSTICAS ---
config = {
    "active": True,
    "send_telegram": True,
    "visual_mode": "yolo",
}
stats = {"total_detections": 0, "last_alert": "Ninguna"}
last_processed_base64 = ""

TOKEN = "8509450066:AAHYzJcxdRjIS7XtR58I-f5Y_xKz6MXmIO8"
CHAT_ID = "7619160347"
bot = Bot(token=TOKEN)

device = 'cpu'
model = YOLO('yolov8n-pose.pt').to(device)

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    # Note: Double {{ }} used for CSS to prevent f-string conflicts
    html_content = f"""
    <html>
        <head>
            <title>macOS Security Monitor</title>
            <script>
                async function toggle(feature) {{ await fetch('/api/toggle/' + feature); }}
                setInterval(async () => {{
                    try {{
                        const res = await fetch('/api/data');
                        const data = await res.json();
                        if (data.image) document.getElementById('monitor-img').src = "data:image/jpeg;base64," + data.image;
                        document.getElementById('count').innerText = data.count;
                        document.getElementById('last').innerText = data.last;
                    }} catch(e) {{}}
                }}, 1000);
            </script>
            <style>
                body {{ margin: 0; height: 100vh; display: flex; align-items: center; justify-content: center; background: url('https://images.unsplash.com/photo-1477346611705-65d1883cee1e?auto=format&fit=crop&w=1920&q=80') center/cover; font-family: -apple-system, sans-serif; }}
                .glass {{ width: 900px; height: 600px; display: flex; background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(30px); border-radius: 30px; border: 1px solid rgba(255, 255, 255, 0.2); box-shadow: 0 25px 50px rgba(0,0,0,0.5); overflow: hidden; }}
                .sidebar {{ width: 320px; padding: 40px; color: white; border-right: 1px solid rgba(255,255,255,0.1); }}
                .monitor {{ flex-grow: 1; padding: 20px; display: flex; flex-direction: column; align-items: center; background: rgba(0,0,0,0.2); }}
                h1 {{ font-size: 22px; margin-bottom: 40px; }}
                .row {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px; }}
                .switch {{ position: relative; width: 44px; height: 24px; display: inline-block; }}
                .switch input {{ opacity: 0; width: 0; height: 0; }}
                .slider {{ position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: rgba(255,255,255,0.2); transition: .4s; border-radius: 24px; }}
                .slider:before {{ position: absolute; content: ""; height: 18px; width: 18px; left: 3px; bottom: 3px; background-color: white; transition: .4s; border-radius: 50%; }}
                input:checked + .slider {{ background-color: #34C759; }}
                input:checked + .slider:before {{ transform: translateX(20px); }}
                .screen {{ width: 100%; height: 350px; background: #000; border-radius: 20px; overflow: hidden; border: 1px solid rgba(255,255,255,0.1); display: flex; align-items: center; justify-content: center; }}
                img {{ width: 100%; height: 100%; object-fit: contain; }}
                .stats-panel {{ margin-top: 20px; width: 100%; color: white; font-size: 12px; opacity: 0.8; }}
            </style>
        </head>
        <body>
            <div class="glass">
                <div class="sidebar">
                    <h1>Security Control</h1>
                    <div class="row"><span>Sistema Activo</span><label class="switch"><input type="checkbox" {'checked' if config['active'] else ''} onchange="toggle('active')"><span class="slider"></span></label></div>
                    <div class="row"><span>Bot Telegram</span><label class="switch"><input type="checkbox" {'checked' if config['send_telegram'] else ''} onchange="toggle('send_telegram')"><span class="slider"></span></label></div>
                    <div class="row"><span>YOLO Pose Mode</span><label class="switch"><input type="checkbox" {'checked' if config['visual_mode'] == 'yolo' else ''} onchange="toggle('visual_mode')"><span class="slider"></span></label></div>
                    <div class="stats-panel">
                        <hr style="opacity: 0.2">
                        <p>Detecciones totales: <span id="count">0</span></p>
                        <p>Última alerta: <span id="last">Ninguna</span></p>
                    </div>
                </div>
                <div class="monitor">
                    <p style="color: white; font-size: 14px; margin-bottom: 10px;">Monitor de IA en Tiempo Real</p>
                    <div class="screen"><img id="monitor-img" src=""></div>
                    <p style="color: gray; font-size: 10px; margin-top: 20px;">STATUS: OK</p>
                </div>
            </div>
        </body>
    </html>
    """
    return html_content

@app.get("/api/toggle/{{feature}}")
async def api_toggle(feature: str):
    if feature == "active":
        config["active"] = not config["active"]
    elif feature == "visual_mode":
        config["visual_mode"] = "original" if config["visual_mode"] == "yolo" else "yolo"
    elif feature in config:
        config[feature] = not config[feature]
    return JSONResponse(content={{"status": "success", "state": config.get(feature)}})

@app.get("/api/data")
async def get_data():
    return JSONResponse(content={
        "image": last_processed_base64,
        "count": stats["total_detections"],
        "last": stats["last_alert"]
    })

@app.post("/detect")
async def detect_api(file: UploadFile = File(...)):
    global last_processed_base64
    if not config["active"]:
        return {"status": "paused"}

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    results = model.predict(img, conf=0.6, device=device, verbose=False)
    result = results[0]

    # Process Annotation
    annotated = result.plot()
    _, buffer = cv2.imencode(".jpg", annotated)
    last_processed_base64 = base64.b64encode(buffer).decode('utf-8')

    # Logical check for person pose detection
    if len(result.boxes) > 0 and result.keypoints is not None:
        # Check if confidence of keypoints exists and meets criteria
        confidences = result.keypoints.conf[0]
        if torch.sum(confidences > 0.5).item() > 7:
            stats["total_detections"] += 1
            stats["last_alert"] = datetime.now().strftime("%H:%M:%S")

            if config["send_telegram"]:
                tg_img = annotated if config["visual_mode"] == "yolo" else img
                _, buffer_tg = cv2.imencode(".jpg", tg_img)
                # Note: Bot usage might require an initialized session in some environments
                await bot.send_photo(chat_id=CHAT_ID, photo=buffer_tg.tobytes(), caption=f"🚨 Alerta #{stats['total_detections']}")
                return {"status": "ok", "action": "sent"}
            
    return {"status": "ok", "detected": False}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)