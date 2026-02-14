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
import numpy as np
import os
import warnings
import cv2
# 1. Silenciar logs de TensorFlow (0 = todos, 1 = info, 2 = warnings, 3 = solo errores)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 2. Desactivar el aviso de optimizaciones oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 3. Silenciar los avisos de Protobuf y otros UserWarnings de librerías
warnings.filterwarnings("ignore", category=UserWarning)
# Específicamente para el ruido de protobuf que mostraste:
warnings.filterwarnings("ignore", module="google.protobuf.runtime_version")


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "hand_landmarker.task"

def download_mediapipe_model():
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    
    # Verificamos si el archivo NO existe en la ruta absoluta
    if not MODEL_PATH.exists():
        print(f"Descargando modelo de MediaPipe en: {MODEL_PATH}...")
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status() # Lanza error si falla la descarga
            
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            print("✓ Modelo descargado y guardado correctamente.")
        except Exception as e:
            print(f"X Error al descargar el modelo: {e}")
    else:
        print(f"✓ El modelo ya existe en: {MODEL_PATH}")

# Llamar a la función
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
    """Cargar modelo una sola vez y retornarlo."""
    global _model_instance, _model_loaded
    if _model_loaded:
        return _model_instance, True
    
    try:
        from tensorflow import keras
        file_path = Path(__file__).resolve()
        project_root = file_path.parent.parent
        # Ajusta esta ruta si tu modelo está en otra carpeta
        model_path = project_root / "models" / "DeepCNN.keras"
        
        if model_path.exists():
            _model_instance = keras.models.load_model(str(model_path))
            _model_loaded = True
            print(f"[OK] Modelo CNN cargado desde {model_path}")
            return _model_instance, True
    except Exception as e:
        print(f"[ERROR] Error cargando modelo: {e}")
    
    return None, False

# Cargamos el modelo CNN globalmente para que 'model' esté definido
model, loaded = load_model_once()

# Inicializamos el detector de MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_task_path = Path(__file__).parent / "hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(model_task_path)),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.6
)

detector = HandLandmarker.create_from_options(options)

# Mapeo de etiquetas
label_to_letter = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
    18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
}

# ============================================================================
# PROCESAMIENTO
# ============================================================================

def process_frame_data(frame_data: str):
    """Procesar frame usando MediaPipe Tasks + CNN con recorte cuadrado."""
    # Verificación de seguridad: si el modelo no cargó, avisar
    if model is None:
        return {"success": False, "error": "Modelo CNN no cargado en el servidor"}

    try:
        # 1. Decodificación
        if "," in frame_data:
            frame_data = frame_data.split(",")[1]
        
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"success": False, "error": "Error decodificando frame"}

        h, w, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. DETECCIÓN
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        detection_result = detector.detect(mp_image)

        # 3. PROCESAMIENTO DE RESULTADOS
        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            
            x_coords = [lm.x * w for lm in hand_landmarks]
            y_coords = [lm.y * h for lm in hand_landmarks]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # --- LÓGICA DE RECORTE CUADRADO ---
            box_w, box_h = x_max - x_min, y_max - y_min
            center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
            side = max(box_w, box_h) + 60
            
            x1, y1 = max(0, int(center_x - side / 2)), max(0, int(center_y - side / 2))
            x2, y2 = min(w, int(center_x + side / 2)), min(h, int(center_y + side / 2))

            hand_crop = img_rgb[y1:y2, x1:x2]
            
            if hand_crop.size == 0:
                return {"success": False, "error": "Recorte inválido"}

            # 4. PREPARACIÓN PARA LA CNN
            gray = cv2.cvtColor(hand_crop, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
            normalized = resized.astype('float32') / 255.0
            input_data = normalized.reshape(1, 28, 28, 1)

            # 5. PREDICCIÓN (Aquí ya existe 'model' porque lo cargamos arriba)
            predictions = model.predict(input_data, verbose=0)
            predicted_class = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0]))

            if confidence < 0.45:
                return {"success": False, "error": "Baja confianza"}

            return {
                "success": True,
                "letter": label_to_letter.get(predicted_class, "?"),
                "confidence": f"{confidence*100:.1f}%"
            }
        
        return {"success": False, "error": "No hand detected"}

    except Exception as e:
        return {"success": False, "error": f"Error: {str(e)}"}

# Exportar la app para que Reflex la encuentre
__all__ = ["app"]
