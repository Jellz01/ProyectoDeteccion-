import cv2
import easyocr
import time
import threading
import numpy as np

# --- HILO DE CAPTURA DE VIDEO ---
class VideoStream:
    def __init__(self, src=1):
        # Intentamos índice 1, si falla vamos al 0
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened(): 
            self.cap = cv2.VideoCapture(0)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        t = threading.Thread(target=self.update, args=(), daemon=True)
        t.start()
        return self

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()

    # ESTE ES EL MÉTODO QUE FALTABA
    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

# --- HILO DE PROCESAMIENTO OCR ---
class OCRProcessor:
    def __init__(self):
        print("🚀 Cargando EasyOCR en CPU...")
        self.reader = easyocr.Reader(['es', 'en'], gpu=False)
        self.frame = None
        self.results = []
        self.running = True
        self.new_frame = False

    def start(self):
        t = threading.Thread(target=self.process, args=(), daemon=True)
        t.start()
        return self

    def process(self):
        while self.running:
            if self.new_frame and self.frame is not None:
                try:
                    h, w = self.frame.shape[:2]
                    # ROI: Solo el centro de la pantalla
                    roi = self.frame[h//4:3*h//4, w//4:3*w//4] 
                    
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    # Reducimos tamaño para que el CPU no muera
                    small = cv2.resize(gray, (0,0), fx=0.8, fy=0.8) 
                    
                    # Ejecutar OCR
                    temp_results = self.reader.readtext(small, paragraph=True)
                    
                    final = []
                    for res in temp_results:
                        if len(res) >= 2:
                            bbox, text = res[0], res[1]
                            prob = res[2] if len(res) == 3 else 1.0
                            
                            # Ajuste de coordenadas: escala (1/0.8) + offset del ROI
                            offset_x, offset_y = w//4, h//4
                            adj_bbox = [[int(c[0]*1.25) + offset_x, int(c[1]*1.25) + offset_y] for c in bbox]
                            final.append((adj_bbox, text, prob))
                    
                    self.results = final
                except Exception as e:
                    print(f"⚠️ Error OCR: {e}")
                
                self.new_frame = False
            else:
                time.sleep(0.01)

def main():
    vs = VideoStream(src=1).start()
    ocr = OCRProcessor().start()
    
    time.sleep(2.0) # Esperar a que la cámara inicie
    print("✅ Sistema a 60 FPS iniciado. Coloca el texto en el recuadro central.")

    while True:
        frame = vs.read()
        if frame is None: continue

        # Visualización
        display_frame = frame.copy()

        # Mandar a procesar si el hilo de OCR terminó el anterior
        if not ocr.new_frame:
            ocr.frame = frame.copy()
            ocr.new_frame = True

        # Dibujar resultados (últimos conocidos)
        for (bbox, text, prob) in ocr.results:
            if prob > 0.2:
                tl, br = tuple(bbox[0]), tuple(bbox[2])
                cv2.rectangle(display_frame, tl, br, (0, 255, 0), 2)
                cv2.putText(display_frame, text, (tl[0], tl[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Guía visual del ROI
        h, w = display_frame.shape[:2]
        cv2.rectangle(display_frame, (w//4, h//4), (3*w//4, 3*h//4), (255, 255, 255), 1)
        
        cv2.imshow("60 FPS OCR - Zona Central", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    vs.stop()
    ocr.running = False
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()