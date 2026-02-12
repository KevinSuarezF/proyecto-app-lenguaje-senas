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

def process_frame_data(frame_data: str):
    """Procesar frame de video y retornar predicción."""
    try:
        import cv2
        import numpy as np
        
        # Decodificar imagen base64
        if not frame_data or not isinstance(frame_data, str):
            return {"success": False, "error": "Invalid frame data"}
        
        if "," in frame_data:
            frame_data = frame_data.split(",")[1]
        
        try:
            frame_bytes = base64.b64decode(frame_data)
        except Exception:
            return {"success": False, "error": "Failed to decode base64"}
        
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"success": False, "error": "Failed to decode frame"}
        
        # Procesar imagen
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28, 28))
        normalized = resized.astype('float32') / 255.0
        input_data = np.expand_dims(normalized, axis=-1)
        input_data = np.expand_dims(input_data, axis=0)
        
        # Cargar modelo
        model, loaded = load_model_once()
        if not loaded or model is None:
            return {"success": False, "error": "Model not loaded"}
        
        # Hacer predicción
        predictions = model.predict(input_data, verbose=0)
        predicted_class = int(np.argmax(predictions[0]))
        conf_value = float(np.max(predictions[0]))
        
        # Mapeo de clases
        label_to_letter = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
            10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
            18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
        }
        
        predicted_letter = label_to_letter.get(predicted_class, "?")
        confidence_str = f"{conf_value*100:.1f}%"
        
        print(f"[OK] Prediccion: {predicted_letter} ({confidence_str})")
        
        return {
            "success": True,
            "letter": predicted_letter,
            "confidence": confidence_str
        }
    
    except Exception as e:
        print(f"Error procesando frame: {e}")
        return {"success": False, "error": str(e)[:100]}


# Exportar la app para que Reflex la encuentre
__all__ = ["app"]