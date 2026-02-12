"""Estado de la aplicación ASL con Reflex."""
import reflex as rx
import base64
from pathlib import Path


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
    
    def process_frame(self, frame_data: str):
        """Procesar frame de video y hacer predicción."""
        if not self.is_running or not self.model_loaded:
            return
        
        try:
            # Lazy imports - solo se cargan cuando se necesitan
            from tensorflow import keras
            import cv2
            import numpy as np
            
            # Decodificar imagen base64
            if not frame_data or not isinstance(frame_data, str):
                return
                
            if "," in frame_data:
                frame_data = frame_data.split(",")[1]
            
            try:
                frame_bytes = base64.b64decode(frame_data)
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    self.error_message = "No se pudo decodificar la imagen"
                    return
                
                # Procesar imagen para el modelo
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (28, 28))
                normalized = resized.astype('float32') / 255.0
                input_data = np.expand_dims(normalized, axis=-1)
                input_data = np.expand_dims(input_data, axis=0)
                
                # Cargar modelo y hacer predicción
                model_path = Path(__file__).parent.parent / "models" / "DeepCNN.keras"
                model = keras.models.load_model(str(model_path))
                predictions = model.predict(input_data, verbose=0)
                predicted_class = int(np.argmax(predictions[0]))
                conf_value = float(np.max(predictions[0]))
                
                # Actualizar estado
                self.predicted_letter = self.label_to_letter.get(predicted_class, "?")
                self.confidence = f"{conf_value*100:.1f}%"
                self.error_message = ""
                
            except Exception as e:
                self.error_message = f"Error procesando frame: {str(e)[:80]}"
        
        except Exception as e:
            self.error_message = f"Error: {str(e)[:80]}"
    
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
