import reflex as rx
import base64
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import os
import warnings

# 1. Silenciar logs de TensorFlow (0 = todos, 1 = info, 2 = warnings, 3 = solo errores)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 2. Desactivar el aviso de optimizaciones oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 3. Silenciar los avisos de Protobuf y otros UserWarnings de librerías
warnings.filterwarnings("ignore", category=UserWarning)
# Específicamente para el ruido de protobuf que mostraste:
warnings.filterwarnings("ignore", module="google.protobuf.runtime_version")
# --- INICIALIZACIÓN GLOBAL (Persistente en RAM) ---
_CNN_MODEL = None  # Aquí guardaremos tu DeepCNN.keras

# Configuración de MediaPipe Tasks
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Ruta al modelo de MediaPipe
MP_MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

# Solo inicializamos si el archivo existe para evitar errores al arrancar
detector_mp = None
if MP_MODEL_PATH.exists():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MP_MODEL_PATH)),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.6
    )
    detector_mp = HandLandmarker.create_from_options(options)

class ASLState(rx.State):
    """Estado global optimizado de la app ASL."""
    
    is_running: bool = False
    predicted_letter: str = "?"
    confidence: str = "0%"
    model_loaded: bool = False
    error_message: str = ""

    label_to_letter = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
        10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
        18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
    }

    def load_model(self):
        """Carga el modelo CNN en la variable global persistente."""
        global _CNN_MODEL
        if _CNN_MODEL is not None:
            self.model_loaded = True
            return

        try:
            from tensorflow import keras
            model_path = Path(__file__).parent.parent / "models" / "DeepCNN.keras"
            
            if not model_path.exists():
                self.error_message = f"No se encontró el modelo .keras en {model_path}"
                return

            _CNN_MODEL = keras.models.load_model(str(model_path))
            self.model_loaded = True
            print("✓ DeepCNN cargada en memoria.")
        except Exception as e:
            self.error_message = f"Error carga CNN: {str(e)[:50]}"

    def process_captured_frame(self, frame_data: str):
        """Función única de procesamiento: MediaPipe + DeepCNN."""
        global _CNN_MODEL
        
        # 1. Validaciones previas
        if not frame_data: return
        if not self.model_loaded: self.load_model()
        if detector_mp is None:
            self.error_message = "Detector MediaPipe no inicializado"
            return

        try:
            # 2. Decodificación de Base64 a OpenCV
            if "," in frame_data:
                frame_data = frame_data.split(",")[1]
            
            img_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None: return

            # 3. Paso 1: MediaPipe (Encontrar la mano)
            h, w, _ = frame.shape
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            
            result = detector_mp.detect(mp_image)

            if result.hand_landmarks:
                # 4. Paso 2: Recortar la mano
                landmarks = result.hand_landmarks[0]
                x_px = [lm.x * w for lm in landmarks]
                y_px = [lm.y * h for lm in landmarks]
                
                # Definir cuadro con margen
                padding = 20
                x1, x2 = int(min(x_px) - padding), int(max(x_px) + padding)
                y1, y2 = int(min(y_px) - padding), int(max(y_px) + padding)
                
                hand_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

                # 5. Paso 3: Preparar para la CNN (28x28 gris)
                gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
                final_input = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
                final_input = final_input.reshape(1, 28, 28, 1).astype('float32') / 255.0

                # 6. Paso 4: Predicción
                preds = _CNN_MODEL.predict(final_input, verbose=0)
                idx = np.argmax(preds[0])
                conf = float(np.max(preds[0]))

                self.predicted_letter = self.label_to_letter.get(idx, "?")
                self.confidence = f"{conf*100:.1f}%"
                self.error_message = ""
            else:
                self.error_message = "Pon tu mano más cerca de la cámara"
                self.predicted_letter = "?"

        except Exception as e:
            self.error_message = f"Error: {str(e)[:50]}"

    def toggle_camera(self):
        """Activa/Desactiva y asegura que el modelo se cargue."""
        if not self.is_running:
            self.load_model()
        self.is_running = not self.is_running
