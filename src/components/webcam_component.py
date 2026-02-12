"""
Componente de upload de imágenes para Reflex.

Este es un enfoque alternativo y más simple que usar webcam en tiempo real.
El usuario puede subir imágenes de señas y obtener predicciones.
"""

import reflex as rx
from pathlib import Path


def create_upload_component():
    """
    Crea un componente de upload de imágenes.
    
    Esta es una alternativa más simple a la webcam en tiempo real.
    Es más fácil de implementar y no requiere componentes JavaScript custom.
    """
    return rx.vstack(
        rx.heading("Subir Imagen de Seña", size="md"),
        rx.divider(),
        
        rx.upload(
            rx.vstack(
                rx.button(
                    "Seleccionar Imagen",
                    color_scheme="blue",
                    size="lg",
                ),
                rx.text(
                    "Arrastra una imagen aquí o haz clic para seleccionar",
                    size="sm",
                    color="gray.600",
                ),
            ),
            id="image_upload",
            accept={
                "image/png": [".png"],
                "image/jpeg": [".jpg", ".jpeg"],
            },
            max_files=1,
            disabled=False,
            on_keyboard=True,
            border="2px dashed",
            border_color="gray.300",
            border_radius="lg",
            padding="2rem",
        ),
        
        rx.hstack(
            rx.foreach(
                rx.selected_files("image_upload"),
                lambda file: rx.text(file),
            ),
        ),
        
        rx.button(
            "Procesar Imagen",
            on_click=lambda: rx.upload_files(
                upload_id="image_upload",
            ),
            color_scheme="green",
            size="lg",
            width="100%",
        ),
        
        spacing="1rem",
        width="100%",
    )


# INSTRUCCIONES PARA IMPLEMENTAR WEBCAM REAL-TIME EN REFLEX
"""
Para implementar webcam en tiempo real en Reflex, necesitas crear un componente 
custom de React. Aquí está el proceso:

1. OPCIÓN A: Usar react-webcam (Recomendado)
   ================================================
   
   a) Instalar react-webcam:
      ```bash
      npm install react-webcam
      ```
   
   b) Crear el componente wrapper en Python:
      ```python
      from reflex.components.component import Component
      from typing import Any, Dict
      
      class Webcam(Component):
          library = "react-webcam"
          tag = "Webcam"
          
          # Props
          audio: rx.Var[bool] = False
          height: rx.Var[int] = 480
          width: rx.Var[int] = 640
          screenshotFormat: rx.Var[str] = "image/jpeg"
          screenshotQuality: rx.Var[float] = 0.92
          
          def get_event_triggers(self) -> Dict[str, Any]:
              return {
                  "on_user_media": lambda: [],
                  "on_user_media_error": lambda e: [e],
              }
      
      webcam = Webcam.create
      ```
   
   c) Usar en tu aplicación:
      ```python
      def camera_view():
          return rx.vstack(
              webcam(
                  height=480,
                  width=640,
                  audio=False,
                  screenshotFormat="image/jpeg",
                  id="webcam",
              ),
              rx.button(
                  "Capturar",
                  on_click=State.capture_screenshot,
              ),
          )
      ```
   
   d) Capturar screenshots:
      ```python
      class State(rx.State):
          def capture_screenshot(self):
              # Usar JavaScript para capturar
              return rx.call_script(
                  "const webcam = document.getElementById('webcam');"
                  "const imageSrc = webcam.getScreenshot();"
                  "return imageSrc;"
              )
      ```

2. OPCIÓN B: HTML5 getUserMedia (Control total)
   ================================================
   
   a) Crear componente HTML5:
      ```python
      def webcam_html5():
          return rx.html(
              '''
              <video id="video" autoplay></video>
              <canvas id="canvas" style="display:none;"></canvas>
              <script>
                  const video = document.getElementById('video');
                  const canvas = document.getElementById('canvas');
                  const context = canvas.getContext('2d');
                  
                  navigator.mediaDevices.getUserMedia({ video: true })
                      .then(stream => {
                          video.srcObject = stream;
                      })
                      .catch(err => {
                          console.error("Error accessing webcam:", err);
                      });
                  
                  function captureFrame() {
                      canvas.width = video.videoWidth;
                      canvas.height = video.videoHeight;
                      context.drawImage(video, 0, 0);
                      return canvas.toDataURL('image/jpeg');
                  }
              </script>
              '''
          )
      ```
   
   b) Capturar frames:
      ```python
      class State(rx.State):
          current_frame: str = ""
          
          def capture_frame(self):
              # Llamar a la función JavaScript
              return rx.call_script("captureFrame()")
          
          def on_frame_captured(self, frame_data: str):
              self.current_frame = frame_data
              self.process_frame(frame_data)
      ```

3. OPCIÓN C: Intervalos automáticos
   ================================================
   
   Para capturar frames automáticamente (como video continuo):
   
   ```python
   class State(rx.State):
       capturing: bool = False
       
       def start_capture(self):
           self.capturing = True
           # Capturar cada 100ms
           return rx.window_alert("Iniciando captura continua")
       
       def stop_capture(self):
           self.capturing = False
   
   def camera_view():
       return rx.vstack(
           webcam(id="webcam"),
           rx.cond(
               State.capturing,
               rx.text("Capturando..."),
               rx.text("Detenido"),
           ),
           rx.button_group(
               rx.button("Iniciar", on_click=State.start_capture),
               rx.button("Detener", on_click=State.stop_capture),
           ),
       )
   ```

4. ALTERNATIVA: Usar VideoStream del navegador
   ================================================
   
   Si Reflex soporta web components:
   
   ```python
   def video_stream():
       return rx.html.video(
           id="video",
           autoplay=True,
           playsinline=True,
           width="640",
           height="480",
       )
   ```

5. RECOMENDACIÓN FINAL
   ================================================
   
   Para este proyecto, recomiendo:
   
   1. PARA DESARROLLO RÁPIDO:
      - Usar el componente de upload de imágenes (más simple)
      - Funciona inmediatamente sin configuración adicional
      - Suficiente para demostración del modelo
   
   2. PARA PRODUCCIÓN:
      - Implementar react-webcam (Opción A)
      - Mejor experiencia de usuario
      - Soporte completo del navegador
      - Documentación abundante

RECURSOS:
=========
- React Webcam: https://www.npmjs.com/package/react-webcam
- Reflex Custom Components: https://reflex.dev/docs/custom-components/overview
- MDN getUserMedia: https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
- Reflex JavaScript Interop: https://reflex.dev/docs/api-reference/browser
"""
