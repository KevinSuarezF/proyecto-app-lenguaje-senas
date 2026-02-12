"""
Módulo de preprocesamiento de imágenes para detección en tiempo real.

Incluye funciones para procesar frames de webcam y prepararlos para inferencia.
"""

import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array


class ImagePreprocessor:
    """
    Preprocesador de imágenes para detección de señas en tiempo real.
    """
    
    def __init__(self, target_size=(28, 28), normalize=True, channels=1):
        """
        Inicializa el preprocesador.
        
        Args:
            target_size: Tamaño objetivo (ancho, alto)
            normalize: Si True, normaliza a [0, 1]
            channels: Número de canales (1=grayscale, 3=RGB)
        """
        self.target_size = target_size
        self.normalize = normalize
        self.channels = channels
        
    def preprocess_frame(self, frame):
        """
        Preprocesa un frame de webcam para inferencia.
        
        Args:
            frame: Frame de OpenCV (numpy array BGR)
        
        Returns:
            Frame preprocesado listo para el modelo
        """
        # Convertir a grayscale si es necesario
        if self.channels == 1 and len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif self.channels == 3 and len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # Redimensionar
        frame_resized = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
        
        # Normalizar si es necesario
        if self.normalize:
            frame_resized = frame_resized.astype(np.float32) / 255.0
        
        # Ajustar dimensiones
        if self.channels == 1:
            frame_resized = np.expand_dims(frame_resized, axis=-1)
        
        # Añadir dimensión de batch
        frame_resized = np.expand_dims(frame_resized, axis=0)
        
        return frame_resized
    
    def extract_hand_region(self, frame, margin=20):
        """
        Extrae región de la mano del frame usando detección de contornos.
        
        Args:
            frame: Frame de OpenCV
            margin: Margen adicional alrededor de la mano
        
        Returns:
            Región de la mano recortada, o frame original si no se detecta
        """
        # Convertir a grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Aplicar blur para reducir ruido
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Umbralización
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return frame
        
        # Encontrar el contorno más grande (asumiendo que es la mano)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Obtener bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Añadir margen
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(frame.shape[1] - x, w + 2 * margin)
        h = min(frame.shape[0] - y, h + 2 * margin)
        
        # Recortar región
        hand_region = frame[y:y+h, x:x+w]
        
        return hand_region, (x, y, w, h)
    
    def apply_advanced_preprocessing(self, frame):
        """
        Aplica preprocesamiento avanzado para mejorar detección.
        
        Args:
            frame: Frame de OpenCV
        
        Returns:
            Frame preprocesado
        """
        # Convertir a grayscale si es necesario
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Ecualización de histograma adaptativo (mejor para diferentes iluminaciones)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Reducción de ruido
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        return denoised


def create_roi(frame, x_center=0.5, y_center=0.5, size=200):
    """
    Crea una región de interés (ROI) en el frame.
    
    Args:
        frame: Frame de OpenCV
        x_center: Posición X del centro (0-1)
        y_center: Posición Y del centro (0-1)
        size: Tamaño del ROI en píxeles
    
    Returns:
        ROI recortada, coordenadas (x1, y1, x2, y2)
    """
    h, w = frame.shape[:2]
    
    # Calcular coordenadas del ROI
    x_center_px = int(w * x_center)
    y_center_px = int(h * y_center)
    
    half_size = size // 2
    
    x1 = max(0, x_center_px - half_size)
    y1 = max(0, y_center_px - half_size)
    x2 = min(w, x_center_px + half_size)
    y2 = min(h, y_center_px + half_size)
    
    roi = frame[y1:y2, x1:x2]
    
    return roi, (x1, y1, x2, y2)


def draw_prediction_overlay(frame, prediction, confidence, label_map, roi_coords=None):
    """
    Dibuja información de predicción sobre el frame.
    
    Args:
        frame: Frame de OpenCV
        prediction: Clase predicha (índice)
        confidence: Confianza de la predicción
        label_map: Diccionario {índice: letra}
        roi_coords: Coordenadas del ROI (x1, y1, x2, y2)
    
    Returns:
        Frame con overlay
    """
    frame_overlay = frame.copy()
    
    # Dibujar ROI si se proporciona
    if roi_coords:
        x1, y1, x2, y2 = roi_coords
        cv2.rectangle(frame_overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_overlay, "ROI", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Preparar texto de predicción
    letter = label_map.get(prediction, '?')
    text = f"Letra: {letter}"
    conf_text = f"Confianza: {confidence*100:.1f}%"
    
    # Configuración de texto
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3
    
    # Fondo para texto
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    (conf_w, conf_h), _ = cv2.getTextSize(conf_text, font, 0.8, 2)
    
    # Dibujar fondo semitransparente
    overlay = frame_overlay.copy()
    cv2.rectangle(overlay, (10, 10), (max(text_w, conf_w) + 30, text_h + conf_h + 40),
                 (0, 0, 0), -1)
    frame_overlay = cv2.addWeighted(overlay, 0.7, frame_overlay, 0.3, 0)
    
    # Color basado en confianza
    if confidence > 0.8:
        color = (0, 255, 0)  # Verde - alta confianza
    elif confidence > 0.6:
        color = (0, 255, 255)  # Amarillo - media confianza
    else:
        color = (0, 0, 255)  # Rojo - baja confianza
    
    # Dibujar texto
    cv2.putText(frame_overlay, text, (20, 40),
               font, font_scale, color, thickness)
    cv2.putText(frame_overlay, conf_text, (20, 75),
               font, 0.8, (255, 255, 255), 2)
    
    return frame_overlay


def augment_for_inference(frame, num_augmentations=5):
    """
    Crea variaciones aumentadas de un frame para inferencia más robusta.
    
    Args:
        frame: Frame preprocesado
        num_augmentations: Número de variaciones
    
    Returns:
        Lista de frames aumentados
    """
    augmented_frames = [frame]
    
    for _ in range(num_augmentations - 1):
        aug = frame.copy()
        
        # Rotación ligera
        angle = np.random.uniform(-10, 10)
        h, w = aug.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        aug = cv2.warpAffine(aug, M, (w, h))
        
        # Ajuste de brillo
        brightness = np.random.uniform(0.8, 1.2)
        aug = np.clip(aug * brightness, 0, 255).astype(np.uint8)
        
        augmented_frames.append(aug)
    
    return augmented_frames
