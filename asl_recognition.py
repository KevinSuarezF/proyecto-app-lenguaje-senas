"""
Aplicación Reflex para Reconocimiento de Lenguaje de Señas en Tiempo Real

Esta aplicación usa Reflex (100% Python) para crear una interfaz web
que detecta señas del lenguaje de señas americano (ASL) usando la webcam.

Autor: Kevin Suarez
Fecha: Febrero 2026
"""

import reflex as rx
import cv2
import numpy as np
import tensorflow as tf
from typing import List, Dict, Optional
import os
import base64
from pathlib import Path
import sys

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

# Mapeo de etiquetas a letras
LABEL_TO_LETTER = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R',
    17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}


class State(rx.State):
    """Estado de la aplicación de reconocimiento de señas."""
    
    # Modelo
    model_loaded: bool = False
    available_models: List[str] = []
    selected_model: str = ""
    model = None
    
    # Predicción
    predicted_letter: str = "-"
    confidence: float = 0.0
    top3_predictions: List[Dict[str, float]] = []
    
    # Configuración
    confidence_threshold: float = 0.5
    roi_size: int = 250
    
    # Estadísticas
    total_predictions: int = 0
    letter_counts: Dict[str, int] = {}
    
    # UI
    camera_active: bool = False
    show_help: bool = True
    status_message: str = "Selecciona un modelo para comenzar"
    
    # Imagen
    current_frame_data: str = ""
    preprocessed_frame_data: str = ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.letter_counts = {letter: 0 for letter in LABEL_TO_LETTER.values()}
        self.load_available_models()
    
    def load_available_models(self):
        """Carga la lista de modelos disponibles."""
        models_dir = Path("models")
        if models_dir.exists():
            models = [f.name for f in models_dir.iterdir() 
                     if f.suffix in ['.h5', '.keras']]
            self.available_models = models if models else ["No hay modelos disponibles"]
        else:
            self.available_models = ["Carpeta 'models' no encontrada"]
    
    def set_selected_model(self, model_name: str):
        """Establece el modelo seleccionado."""
        self.selected_model = model_name
        self.model_loaded = False
        self.status_message = f"Modelo seleccionado: {model_name}"
    
    def load_model(self):
        """Carga el modelo TensorFlow."""
        if not self.selected_model or self.selected_model == "No hay modelos disponibles":
            self.status_message = "Error: No hay modelo seleccionado"
            return
        
        try:
            model_path = Path("models") / self.selected_model
            if not model_path.exists():
                self.status_message = f"Error: Modelo no encontrado en {model_path}"
                return
            
            self.status_message = "Cargando modelo..."
            self.model = tf.keras.models.load_model(str(model_path))
            self.model_loaded = True
            self.status_message = f"Modelo cargado exitosamente: {self.selected_model}"
        except Exception as e:
            self.status_message = f"Error al cargar modelo: {str(e)}"
            self.model_loaded = False
    
    def set_confidence_threshold(self, values: List[float]):
        """Establece el umbral de confianza."""
        self.confidence_threshold = values[0] / 100.0
    
    def set_roi_size(self, values: List[float]):
        """Establece el tamaño del ROI."""
        self.roi_size = int(values[0])
    
    def toggle_camera(self):
        """Activa/desactiva la cámara."""
        if not self.model_loaded:
            self.status_message = "Error: Primero carga un modelo"
            return
        
        self.camera_active = not self.camera_active
        if self.camera_active:
            self.status_message = "Cámara activada - Coloca tu mano en el ROI"
        else:
            self.status_message = "Cámara desactivada"
    
    def toggle_help(self):
        """Muestra/oculta la ayuda."""
        self.show_help = not self.show_help
    
    def reset_stats(self):
        """Reinicia las estadísticas."""
        self.total_predictions = 0
        self.letter_counts = {letter: 0 for letter in LABEL_TO_LETTER.values()}
        self.status_message = "Estadísticas reiniciadas"
    
    def preprocess_frame(self, frame_data: str) -> Optional[np.ndarray]:
        """
        Preprocesa un frame codificado en base64.
        
        Args:
            frame_data: Frame en formato base64
            
        Returns:
            Frame preprocesado para el modelo
        """
        try:
            # Decodificar base64
            img_bytes = base64.b64decode(frame_data.split(',')[1])
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return None
            
            # Convertir a grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Ecualización adaptativa de histograma
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
            # Redimensionar a 28x28
            resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
            
            # Normalizar
            normalized = resized.astype(np.float32) / 255.0
            
            # Reshape para el modelo
            preprocessed = normalized.reshape(1, 28, 28, 1)
            
            return preprocessed
            
        except Exception as e:
            print(f"Error en preprocesamiento: {e}")
            return None
    
    def process_frame(self, frame_data: str):
        """
        Procesa un frame de la webcam y hace predicción.
        
        Args:
            frame_data: Frame codificado en base64
        """
        if not self.model_loaded or not self.camera_active:
            return
        
        # Preprocesar frame
        preprocessed = self.preprocess_frame(frame_data)
        
        if preprocessed is None:
            return
        
        try:
            # Predicción
            predictions = self.model.predict(preprocessed, verbose=0)
            predicted_class = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0]))
            
            # Actualizar predicción si supera el umbral
            if confidence >= self.confidence_threshold:
                self.predicted_letter = LABEL_TO_LETTER[predicted_class]
                self.confidence = confidence
                
                # Actualizar estadísticas
                self.total_predictions += 1
                self.letter_counts[self.predicted_letter] += 1
                
                # Top 3 predicciones
                top3_indices = np.argsort(predictions[0])[-3:][::-1]
                self.top3_predictions = [
                    {
                        "letter": LABEL_TO_LETTER[int(i)],
                        "confidence": float(predictions[0][i])
                    }
                    for i in top3_indices
                ]
            else:
                self.predicted_letter = "-"
                self.confidence = confidence
                self.top3_predictions = []
            
        except Exception as e:
            print(f"Error en predicción: {e}")
            self.status_message = f"Error en predicción: {str(e)}"


def navbar() -> rx.Component:
    """Barra de navegación superior."""
    return rx.box(
        rx.hstack(
            rx.heading("🤟 Reconocimiento de Señas ASL", size="lg"),
            rx.spacer(),
            rx.text(
                "Deep Learning 2 - Proyecto Final",
                size="sm",
                color="gray.600"
            ),
            width="100%",
            padding="1rem",
            align_items="center",
        ),
        bg="blue.500",
        color="white",
        width="100%",
        box_shadow="lg",
    )


def model_selector() -> rx.Component:
    """Selector de modelo."""
    return rx.vstack(
        rx.heading("Configuración del Modelo", size="md"),
        rx.divider(),
        
        rx.vstack(
            rx.text("Selecciona el modelo:", font_weight="bold"),
            rx.select(
                State.available_models,
                placeholder="Selecciona un modelo...",
                on_change=State.set_selected_model,
                width="100%",
            ),
            rx.cond(
                State.selected_model != "",
                rx.button(
                    "Cargar Modelo",
                    on_click=State.load_model,
                    color_scheme="blue",
                    width="100%",
                    is_disabled=State.model_loaded,
                ),
            ),
            rx.cond(
                State.model_loaded,
                rx.badge("Modelo cargado", color_scheme="green", variant="solid"),
            ),
            width="100%",
            spacing="0.5rem",
        ),
        
        spacing="1rem",
        width="100%",
        padding="1rem",
        border_radius="lg",
        border="1px solid",
        border_color="gray.200",
    )


def configuration_panel() -> rx.Component:
    """Panel de configuración."""
    return rx.vstack(
        rx.heading("Parámetros de Detección", size="md"),
        rx.divider(),
        
        rx.vstack(
            rx.text("Umbral de confianza:", font_weight="bold"),
            rx.slider(
                default_value=50,
                min_=0,
                max_=100,
                on_change_end=State.set_confidence_threshold,
                width="100%",
            ),
            rx.text(
                f"Actual: {State.confidence_threshold * 100:.0f}%",
                size="sm",
                color="gray.600",
            ),
            width="100%",
        ),
        
        rx.vstack(
            rx.text("Tamaño de ROI:", font_weight="bold"),
            rx.slider(
                default_value=250,
                min_=150,
                max_=400,
                step=50,
                on_change_end=State.set_roi_size,
                width="100%",
            ),
            rx.text(
                f"Actual: {State.roi_size} px",
                size="sm",
                color="gray.600",
            ),
            width="100%",
        ),
        
        spacing="1rem",
        width="100%",
        padding="1rem",
        border_radius="lg",
        border="1px solid",
        border_color="gray.200",
    )


def camera_view() -> rx.Component:
    """Vista de la cámara."""
    return rx.vstack(
        rx.heading("Vista de Cámara", size="md"),
        rx.divider(),
        
        # Aquí iría el componente de cámara
        # Reflex requiere integración con JavaScript para webcam
        rx.box(
            rx.center(
                rx.vstack(
                    rx.icon(tag="video", size="3em", color="gray.400"),
                    rx.text(
                        "Vista de cámara en vivo",
                        font_weight="bold",
                        color="gray.600",
                    ),
                    rx.text(
                        "Nota: Integración de webcam requiere componente custom",
                        size="sm",
                        color="gray.500",
                    ),
                    rx.text(
                        "Ver documentación de Reflex para implementación completa",
                        size="xs",
                        color="gray.400",
                    ),
                ),
            ),
            width="100%",
            height="400px",
            bg="gray.100",
            border_radius="lg",
            border="2px dashed",
            border_color="gray.300",
        ),
        
        rx.button(
            rx.cond(
                State.camera_active,
                "⏹️ Detener Cámara",
                "▶️ Iniciar Cámara",
            ),
            on_click=State.toggle_camera,
            color_scheme=rx.cond(State.camera_active, "red", "green"),
            size="lg",
            width="100%",
            is_disabled=~State.model_loaded,
        ),
        
        spacing="1rem",
        width="100%",
    )


def prediction_display() -> rx.Component:
    """Muestra la predicción actual."""
    return rx.vstack(
        rx.heading("Predicción Actual", size="md"),
        rx.divider(),
        
        rx.center(
            rx.vstack(
                rx.heading(
                    State.predicted_letter,
                    size="3xl",
                    color=rx.cond(
                        State.confidence > 0.8,
                        "green.500",
                        rx.cond(
                            State.confidence > 0.6,
                            "yellow.500",
                            "red.500",
                        ),
                    ),
                ),
                rx.text(
                    f"{State.confidence * 100:.1f}%",
                    font_size="2xl",
                    font_weight="bold",
                    color=rx.cond(
                        State.confidence > 0.8,
                        "green.500",
                        rx.cond(
                            State.confidence > 0.6,
                            "yellow.500",
                            "red.500",
                        ),
                    ),
                ),
                spacing="0.5rem",
            ),
            padding="2rem",
            bg="gray.50",
            border_radius="lg",
            width="100%",
            min_height="150px",
        ),
        
        spacing="1rem",
        width="100%",
    )


def top3_predictions() -> rx.Component:
    """Muestra las top 3 predicciones."""
    return rx.vstack(
        rx.heading("Top 3 Predicciones", size="md"),
        rx.divider(),
        
        rx.vstack(
            rx.foreach(
                State.top3_predictions,
                lambda pred: rx.hstack(
                    rx.text(
                        pred["letter"],
                        font_weight="bold",
                        font_size="xl",
                    ),
                    rx.spacer(),
                    rx.text(
                        f"{pred['confidence'] * 100:.1f}%",
                        color="gray.600",
                    ),
                    width="100%",
                    padding="0.5rem",
                    bg="gray.50",
                    border_radius="md",
                ),
            ),
            width="100%",
            spacing="0.5rem",
        ),
        
        spacing="1rem",
        width="100%",
    )


def statistics_panel() -> rx.Component:
    """Panel de estadísticas."""
    return rx.vstack(
        rx.heading("Estadísticas", size="md"),
        rx.divider(),
        
        rx.vstack(
            rx.stat(
                rx.stat_label("Total Predicciones"),
                rx.stat_number(State.total_predictions),
                rx.stat_help_text("Predicciones válidas realizadas"),
            ),
            
            rx.cond(
                State.total_predictions > 0,
                rx.vstack(
                    rx.text("Letra más detectada:", font_weight="bold", size="sm"),
                    rx.text(
                        f"{max(State.letter_counts.items(), key=lambda x: x[1])[0] if State.letter_counts else '-'}",
                        font_size="xl",
                        color="blue.500",
                    ),
                    width="100%",
                ),
            ),
            
            rx.button(
                "Reiniciar Estadísticas",
                on_click=State.reset_stats,
                color_scheme="orange",
                size="sm",
                width="100%",
            ),
            
            width="100%",
            spacing="1rem",
        ),
        
        spacing="1rem",
        width="100%",
        padding="1rem",
        border_radius="lg",
        border="1px solid",
        border_color="gray.200",
    )


def help_section() -> rx.Component:
    """Sección de ayuda."""
    return rx.cond(
        State.show_help,
        rx.vstack(
            rx.heading("Instrucciones de Uso", size="md"),
            rx.divider(),
            
            rx.ordered_list(
                rx.list_item("Selecciona un modelo entrenado"),
                rx.list_item("Haz clic en 'Cargar Modelo'"),
                rx.list_item("Haz clic en 'Iniciar Cámara'"),
                rx.list_item("Coloca tu mano en el ROI (región de interés)"),
                rx.list_item("Realiza la seña claramente"),
                rx.list_item("La predicción aparecerá en tiempo real"),
                spacing="0.5rem",
            ),
            
            rx.divider(),
            
            rx.vstack(
                rx.text("Consejos:", font_weight="bold"),
                rx.unordered_list(
                    rx.list_item("Usa buena iluminación"),
                    rx.list_item("Fondo simple y contrastante"),
                    rx.list_item("Mantén la mano estable"),
                    rx.list_item("Centra la seña en el ROI"),
                    spacing="0.3rem",
                    font_size="sm",
                ),
                width="100%",
                align_items="start",
            ),
            
            spacing="1rem",
            width="100%",
            padding="1rem",
            border_radius="lg",
            bg="blue.50",
            border="1px solid",
            border_color="blue.200",
        ),
    )


def status_bar() -> rx.Component:
    """Barra de estado."""
    return rx.box(
        rx.hstack(
            rx.icon(tag="info", size="1em"),
            rx.text(State.status_message),
            spacing="0.5rem",
            align_items="center",
        ),
        width="100%",
        padding="0.75rem",
        bg="gray.100",
        border_radius="md",
    )


def index() -> rx.Component:
    """Página principal de la aplicación."""
    return rx.vstack(
        navbar(),
        
        rx.container(
            rx.vstack(
                # Barra de estado
                status_bar(),
                
                # Contenido principal
                rx.grid(
                    # Columna izquierda - Configuración
                    rx.vstack(
                        model_selector(),
                        configuration_panel(),
                        statistics_panel(),
                        rx.button(
                            rx.cond(
                                State.show_help,
                                "Ocultar Ayuda",
                                "Mostrar Ayuda",
                            ),
                            on_click=State.toggle_help,
                            color_scheme="gray",
                            size="sm",
                            width="100%",
                        ),
                        spacing="1.5rem",
                        width="100%",
                    ),
                    
                    # Columna central - Cámara
                    rx.vstack(
                        camera_view(),
                        help_section(),
                        spacing="1.5rem",
                        width="100%",
                    ),
                    
                    # Columna derecha - Predicciones
                    rx.vstack(
                        prediction_display(),
                        top3_predictions(),
                        spacing="1.5rem",
                        width="100%",
                    ),
                    
                    template_columns="1fr 2fr 1fr",
                    gap="1.5rem",
                    width="100%",
                ),
                
                # Footer
                rx.divider(),
                rx.center(
                    rx.vstack(
                        rx.text(
                            "Proyecto de Reconocimiento de Lenguaje de Señas ASL",
                            size="sm",
                            color="gray.600",
                        ),
                        rx.text(
                            "Deep Learning 2 - Maestría en Estadística Aplicada y Ciencia de Datos",
                            size="xs",
                            color="gray.500",
                        ),
                        spacing="0.25rem",
                    ),
                    padding="2rem",
                ),
                
                spacing="2rem",
                width="100%",
                padding_y="2rem",
            ),
            max_width="1400px",
        ),
        
        width="100%",
        min_height="100vh",
        bg="gray.50",
        spacing="0",
    )


# Crear la aplicación
app = rx.App()
app.add_page(index, route="/", title="Reconocimiento de Señas ASL")
