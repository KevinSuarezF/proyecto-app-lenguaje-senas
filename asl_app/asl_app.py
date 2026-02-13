"""
ASL Recognition App - Modulo principal de la aplicacion Reflex.

Este modulo define y exporta la aplicacion Reflex que reconoce
senas del lenguaje de senas americano (ASL) en tiempo real.
"""
import reflex as rx
from asl_app.state import ASLState
from asl_app.pages.index import index
import base64
import json
import subprocess
import os
import sys
import socket
from pathlib import Path
import mediapipe as mp
import requests


def download_mediapipe_model():
    model_path = "hand_landmarker.task"
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    
    if not os.path.exists(model_path):
        print("Descargando modelo de MediaPipe...")
        response = requests.get(url)
        with open(model_path, "wb") as f:
            f.write(response.content)
        print("✓ Modelo descargado.")

# Llamar a la función antes de inicializar el detector
download_mediapipe_model()

# ============================================================================
# SERVIDOR FASTAPI (se ejecuta en subprocess separado)
# ============================================================================

_server_process = None

def start_fastapi_server():
    """Inicia processor_server.py en un subprocess separado."""
    global _server_process
    
    # Verificar si el puerto ya está en uso
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', 8002))
        sock.close()
        
        if result == 0:
            # Puerto ya está en uso, servidor ya está corriendo
            return
    except:
        pass
    
    try:
        # Verificar si el archivo existe
        server_file = Path(__file__).parent.parent / "processor_server.py"
        
        if not server_file.exists():
            print(f"⚠ processor_server.py no encontrado en {server_file}")
            return
        
        # Iniciar en subprocess
        _server_process = subprocess.Popen(
            [sys.executable, str(server_file)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        )
        print(f"✓ Servidor FastAPI iniciado en background (PID: {_server_process.pid})")
        
    except Exception as e:
        print(f"Error iniciando servidor: {e}")

# Iniciar servidor FastAPI antes de crear Reflex
start_fastapi_server()

# Crear la aplicacion
app = rx.App()

# Agregar pagina principal
app.add_page(index, route="/", title="ASL Recognizer")

# Variables globales para mantener el modelo en memoria
_model_instance = None
_model_loaded = False

def load_model_once():
    """Cargar modelo una sola vez."""
    global _model_instance, _model_loaded
    if _model_loaded:
        return _model_instance, True
    
    try:
        from tensorflow import keras
        from pathlib import Path
        
        # Usar path absoluto resuelto desde el archivo actual
        file_path = Path(__file__).resolve()
        project_root = file_path.parent.parent
        model_path = project_root / "models" / "DeepCNN.keras"
        
        print(f"[DEBUG] __file__: {file_path}")
        print(f"[DEBUG] project_root: {project_root}")
        print(f"[DEBUG] model_path: {model_path}")
        print(f"[DEBUG] model_path.exists(): {model_path.exists()}")
        
        if model_path.exists():
            print(f"[LOAD] Cargando modelo desde {model_path}")
            _model_instance = keras.models.load_model(str(model_path))
            _model_loaded = True
            print(f"[OK] Modelo cargado exitosamente")
            return _model_instance, True
        else:
            print(f"[ERROR] Modelo no encontrado en {model_path}")
    except Exception as e:
        print(f"[ERROR] Error cargando modelo: {e}")
        import traceback
        traceback.print_exc()
    
    return None, False


# Inicializar MediaPipe globalmente para no recargarlo en cada frame
# --- 1. CONFIGURACIÓN GLOBAL (Fuera de la función para mayor velocidad) ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Inicializamos el detector una sola vez globalmente
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1 # Cambiado a 1 para optimizar la clasificación de señas
)
detector = HandLandmarker.create_from_options(options)

def process_frame_data(frame_data: str):
    """Procesar frame usando la nueva API de MediaPipe Tasks."""
    try:
        # 1. Decodificación de la imagen
        if "," in frame_data:
            frame_data = frame_data.split(",")[1]
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"success": False, "error": "No se pudo decodificar el frame"}

        h, w, _ = frame.shape
        # Convertir a RGB para MediaPipe
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. DETECCIÓN con MediaPipe Tasks
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        detection_result = detector.detect(mp_image)

        # 3. PROCESAMIENTO de los resultados
        if detection_result.hand_landmarks:
            # Tomamos la primera mano detectada
            hand_landmarks = detection_result.hand_landmarks[0]
            
            # Extraer coordenadas para el recorte (Landmarks son objetos con x, y, z)
            x_coords = [lm.x for lm in hand_landmarks]
            y_coords = [lm.y for lm in hand_landmarks]
            
            x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
            y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)

            # Recortar con margen (padding)
            offset = 30
            y1, y2 = max(0, y_min - offset), min(h, y_max + offset)
            x1, x2 = max(0, x_min - offset), min(w, x_max + offset)
            hand_crop = frame[y1:y2, x1:x2]

            # 4. PREPARACIÓN PARA LA CNN (Notebook 03)
            gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
            normalized = resized.astype('float32') / 255.0
            input_data = np.expand_dims(normalized, axis=(0, -1))

            # 5. PREDICCIÓN
            model, loaded = load_model_once()
            if not loaded: 
                return {"success": False, "error": "Modelo no cargado"}
            
            predictions = model.predict(input_data, verbose=0)
            predicted_class = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0]))

            # Mapeo de tu diccionario label_to_letter
            label_to_letter = {
                0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
                10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
                18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
            }

            return {
                "success": True,
                "letter": label_to_letter.get(predicted_class, "?"),
                "confidence": f"{confidence*100:.1f}%"
            }
        
        return {"success": False, "error": "Mano no detectada"}

    except Exception as e:
        return {"success": False, "error": str(e)}


# Exportar la app para que Reflex la encuentre
__all__ = ["app"]
