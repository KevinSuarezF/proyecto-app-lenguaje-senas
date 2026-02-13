"""Estado de la aplicación ASL con Reflex."""
import reflex as rx
import base64
from pathlib import Path
import mediapipe as mp

# --- INICIALIZACIÓN GLOBAL (Fuera de la clase para velocidad) ---
# MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path_mp = Path(__file__).parent.parent / "asl_app" / "hand_landmarker.task"
options_mp = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(model_path_mp)),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)
# Solo creamos el detector si el archivo existe
detector_mp = HandLandmarker.create_from_options(options_mp) if model_path_mp.exists() else None


class ASLState(rx.State):
    """Estado global de la app ASL."""
    
    # Variables de estado
    is_running: bool = False
    predicted_letter: str = "?"
    confidence: str = "0%"
    model_loaded: bool = False
    error_message: str = ""
    current_frame: str = ""  # Para guardar el frame base64
    
    # Mapeo de clases (24 clases: A-I, K-Y)
    # Nota: No hay J (label 9) ni Z (label 25) porque requieren movimiento
    # Dataset format MNIST: labels 0-8, 10-24 (sin 9 y 25)
    label_to_letter = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
        10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
        18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
    }
    
    def load_model(self):
        """Cargar el modelo al iniciar."""
        try:
            try:
                from tensorflow import keras
            except ImportError:
                self.error_message = "TensorFlow no instalado. Ejecuta: pip install tensorflow"
                self.model_loaded = False
                return
            
            try:
                import cv2
            except ImportError:
                self.error_message = "OpenCV no instalado. Ejecuta: pip install opencv-python"
                self.model_loaded = False
                return
            
            try:
                import numpy
            except ImportError:
                self.error_message = "NumPy no instalado. Ejecuta: pip install numpy"
                self.model_loaded = False
                return
            
            model_path = Path(__file__).parent.parent / "models" / "DeepCNN.keras"
            if not model_path.exists():
                self.error_message = f"Modelo no encontrado en {model_path}"
                self.model_loaded = False
                return
            
            try:
                _ = keras.models.load_model(str(model_path))
                self.model_loaded = True
                self.error_message = ""
            except Exception as e:
                self.error_message = f"Error cargando modelo: {str(e)[:100]}"
                self.model_loaded = False
                
        except Exception as e:
            self.error_message = f"Error inesperado: {str(e)[:100]}"
            self.model_loaded = False
    
    def process_captured_frame(self, frame_data: str):
        """Procesamiento optimizado con MediaPipe."""
        if not frame_data:
            self.error_message = "No hay datos de imagen"
            return

        try:
            # 1. Decodificar
            if "," in frame_data:
                frame_data = frame_data.split(",")[1]
            frame_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                self.error_message = "Error decodificando frame"
                return

            # 2. MediaPipe: Localizar la mano
            h, w, _ = frame.shape
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            
            # Usar el detector global
            detection_result = detector_mp.detect(mp_image)

            if detection_result.hand_landmarks:
                # 3. Recortar (Crop)
                landmarks = detection_result.hand_landmarks[0]
                x_coords = [lm.x * w for lm in landmarks]
                y_coords = [lm.y * h for lm in landmarks]
                
                x1, x2 = int(min(x_coords) - 20), int(max(x_coords) + 20)
                y1, y2 = int(min(y_coords) - 20), int(max(y_coords) + 20)
                
                # Validar límites y recortar
                hand_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                
                # 4. Preprocesar para CNN (28x28 gray)
                gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
                normalized = resized.astype('float32') / 255.0
                input_data = normalized.reshape(1, 28, 28, 1)

                # 5. Predicción (Usando el modelo ya cargado en memoria)
                # NOTA: Asegúrate de que load_model() guarde el modelo en una variable global
                from tensorflow import keras
                model_path_cnn = Path(__file__).parent.parent / "models" / "DeepCNN.keras"
                
                # Optimización: En producción, no cargues el modelo aquí. 
                # Cárgalo en __init__ o load_model una sola vez.
                model = keras.models.load_model(str(model_path_cnn)) 
                
                res = model.predict(input_data, verbose=0)
                idx = np.argmax(res[0])
                conf = float(np.max(res[0]))

                self.predicted_letter = self.label_to_letter.get(idx, "?")
                self.confidence = f"{conf*100:.1f}%"
                self.error_message = ""
            else:
                self.error_message = "Mano no detectada por MediaPipe"
                self.predicted_letter = "?"
                self.confidence = "0%"

        except Exception as e:
            self.error_message = f"Error: {str(e)}"
    
    def toggle_camera(self):
        """Iniciar/detener captura de cámara."""
        if not self.is_running and not self.model_loaded:
            self.load_model()
        self.is_running = not self.is_running
    
    def update_prediction(self, letter: str, confidence: str):
        """Actualizar predicción desde el API endpoint."""
        self.predicted_letter = letter
        self.confidence = confidence
    
    def check_prediction_update(self):
        """Verificar localStorage para actualizaciones de predicción."""
        # Este método se llama regularmente para sincronizar con localStorage
        # Reflex ejecutará este método desde el frontend
        pass
    
    
    def process_captured_frame(self, frame_data: str):
        """Procesar frame capturado (llamado desde JavaScript)."""
        try:
            from tensorflow import keras
            import cv2
            import numpy as np
            
            if not frame_data or not isinstance(frame_data, str):
                self.error_message = "Frame inválido"
                return
            
            # Limpiar data URI si existe
            if "," in frame_data:
                frame_data = frame_data.split(",")[1]
            
            # Decodificar base64
            try:
                frame_bytes = base64.b64decode(frame_data)
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    self.error_message = "No se pudo decodificar la imagen"
                    return
                
                # Procesar imagen
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (28, 28))
                normalized = resized.astype('float32') / 255.0
                input_data = np.expand_dims(normalized, axis=-1)
                input_data = np.expand_dims(input_data, axis=0)
                
                # Cargar modelo y predecir
                model_path = Path(__file__).parent.parent / "models" / "DeepCNN.keras"
                model = keras.models.load_model(str(model_path))
                predictions = model.predict(input_data, verbose=0)
                predicted_class = int(np.argmax(predictions[0]))
                conf_value = float(np.max(predictions[0]))
                
                # Actualizar estado
                self.predicted_letter = self.label_to_letter.get(predicted_class, "?")
                self.confidence = f"{conf_value*100:.1f}%"
                self.error_message = ""
                
                print(f"✓ Predicción: {self.predicted_letter} ({self.confidence})")
                
            except Exception as e:
                self.error_message = f"Error procesando: {str(e)[:80]}"
                print(f"Error: {e}")
        
        except Exception as e:
            self.error_message = f"Error: {str(e)[:80]}"
            print(f"Error general: {e}")
    
    def process_from_localStorage(self):
        """Leer frame de localStorage en JavaScript y procesarlo."""
        # Este método será llamado desde un botón
        # Ejecutará JavaScript para leer localStorage y llamar process_captured_frame
        pass
