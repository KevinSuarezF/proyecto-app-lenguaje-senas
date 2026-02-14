"""Estado de la aplicación ASL con Reflex - Optimizado 2026."""
import os
import warnings

# --- 1. SILENCIAR AVISOS (Debe ir antes de importar tensorflow/mediapipe) ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore", category=UserWarning)

import reflex as rx
import base64
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

# --- 2. INICIALIZACIÓN GLOBAL (Persistente en RAM) ---
_CNN_MODEL = None  # Modelo DeepCNN.keras

# Configuración MediaPipe Tasks
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Ruta absoluta al modelo de MediaPipe
MP_MODEL_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"

# Inicialización única del detector MediaPipe
detector_mp = None
if MP_MODEL_PATH.exists():
    options_mp = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MP_MODEL_PATH)),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.6
    )
    detector_mp = HandLandmarker.create_from_options(options_mp)

class ASLState(rx.State):
    """Estado global de la app ASL con localización y predicción."""
    
    # Variables de estado
    is_running: bool = False
    predicted_letter: str = "?"
    confidence: str = "0%"
    model_loaded: bool = False
    error_message: str = ""
    
    # Mapeo de clases (A-I, K-Y)
    label_to_letter = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
        10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
        18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
    }

    def load_model(self):
        """Cargar el modelo CNN en la variable global para evitar recargas lentas."""
        global _CNN_MODEL
        if _CNN_MODEL is not None:
            self.model_loaded = True
            return

        try:
            from tensorflow import keras
            model_path = Path(__file__).resolve().parent.parent / "models" / "DeepCNN.keras"
            
            if not model_path.exists():
                self.error_message = f"Archivo no encontrado: {model_path.name}"
                self.model_loaded = False
                return

            _CNN_MODEL = keras.models.load_model(str(model_path))
            self.model_loaded = True
            self.error_message = ""
            print("✓ DeepCNN cargada exitosamente en memoria RAM.")
        except Exception as e:
            self.error_message = f"Error al cargar CNN: {str(e)[:50]}"
            self.model_loaded = False

    def start_camera(self):
        """Iniciar cámara y actualizar estado is_running."""
        self.is_running = True
        if not self.model_loaded:
            self.load_model()
        return rx.call_script("window.startCamera()")
    
    def stop_camera(self):
        """Detener cámara y actualizar estado is_running."""
        self.is_running = False
        return rx.call_script("window.stopCamera()")

    def toggle_camera(self):
        """Iniciar/detener cámara e invocar scripts de JavaScript."""
        self.is_running = not self.is_running
        if self.is_running:
            if not self.model_loaded:
                self.load_model()
            # Llamada al script de JS en assets/camera.js
            return rx.call_script("window.startCamera()")
        else:
            return rx.call_script("window.stopCamera()")

    def process_captured_frame(self, frame_data: str):
        """
        Punto de entrada principal para la captura.
        Realiza: Localización (MediaPipe) -> Recorte -> Clasificación (CNN).
        """
        global _CNN_MODEL
        
        if not frame_data:
            self.error_message = "No se recibieron datos de imagen."
            return

        if detector_mp is None:
            self.error_message = "Error: El archivo hand_landmarker.task no está en asl_app/"
            return

        try:
            # 1. Decodificar Base64
            if "," in frame_data:
                frame_data = frame_data.split(",")[1]
            
            img_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                self.error_message = "Error decodificando la imagen capturada."
                return

            # 2. Localizar Mano con MediaPipe Tasks
            h, w, _ = frame.shape
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            
            result = detector_mp.detect(mp_image)

            if result.hand_landmarks:
                # 3. Obtener Coordenadas y Recortar
                landmarks = result.hand_landmarks[0]
                x_px = [lm.x * w for lm in landmarks]
                y_px = [lm.y * h for lm in landmarks]
                
                # Definir Bounding Box con padding de 30px
                padding = 30
                x1, x2 = int(min(x_px) - padding), int(max(x_px) + padding)
                y1, y2 = int(min(y_px) - padding), int(max(y_px) + padding)
                
                # Recorte con validación de límites
                hand_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

                # 4. Preprocesar para CNN (Imagen de $28 \times 28$ en escala de grises)
                gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
                normalized = resized.astype('float32') / 255.0
                input_data = normalized.reshape(1, 28, 28, 1)

                # 5. Predicción con DeepCNN
                if _CNN_MODEL is None:
                    self.load_model()
                
                preds = _CNN_MODEL.predict(input_data, verbose=0)
                idx = np.argmax(preds[0])
                conf = float(np.max(preds[0]))

                # 6. Actualizar Estado
                self.predicted_letter = self.label_to_letter.get(idx, "?")
                self.confidence = f"{conf*100:.1f}%"
                self.error_message = ""
                print(f"✓ {self.predicted_letter} detectada con {self.confidence}")
            else:
                self.error_message = "MediaPipe no detectó ninguna mano. Intenta acercarla."
                self.predicted_letter = "?"
                self.confidence = "0%"

        except Exception as e:
            self.error_message = f"Error en procesamiento: {str(e)[:60]}"
            print(f"Error procesando frame: {e}")

    # --- Mantenemos tus otros métodos por compatibilidad ---
    def update_prediction(self, letter: str, confidence: str):
        self.predicted_letter = letter
        self.confidence = confidence

    def check_prediction_update(self):
        pass

    def process_from_localStorage(self):
        pass
