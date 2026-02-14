"""Página principal de la app ASL con Reflex."""
import reflex as rx
from asl_app.state import ASLState


def index() -> rx.Component:
    """Página principal."""
    return rx.container(
        # Script global para captura de cámara - Definen funciones en window
        rx.script(
            """
            // Limpiar localStorage al iniciar la app
            localStorage.removeItem('current_frame');
            localStorage.removeItem('asl_hand_crop');
            localStorage.removeItem('asl_processed_image');
            localStorage.removeItem('asl_original_frame');
            localStorage.removeItem('asl_error');
            localStorage.removeItem('asl_prediction');
            
            window.streaming = false;
            window.current_frame = null;
            
            window.initCamera = async function() {
                const video = document.getElementById('asl-video');
                if (!video) {
                    console.log('Video element not found');
                    return;
                }
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({
                        video: { facingMode: 'user' }
                    });
                    video.srcObject = stream;
                    video.onloadedmetadata = () => {
                        window.streaming = true;
                        console.log('Camera initialized');
                    };
                } catch (err) {
                    console.error('Error accediendo cámara:', err);
                    alert('Permiso de cámara denegado. Por favor, permite el acceso a la cámara.');
                }
            };
            
            window.captureFrame = function() {
                if (!window.streaming) {
                    alert('La cámara no está activa. Haz clic en Iniciar primero.');
                    return;
                }
                
                const video = document.getElementById('asl-video');
                const canvas = document.getElementById('asl-canvas');
                
                if (!video || !canvas) return;
                
                canvas.width = 640;
                canvas.height = 480;
                
                const ctx = canvas.getContext('2d');
                
                // Aplicar espejo (scaleX(-1)) como en el CSS del video
                ctx.scale(-1, 1);
                ctx.drawImage(video, -640, 0, 640, 480);
                ctx.scale(-1, 1);  // Restaurar
                
                const imageData = canvas.toDataURL('image/jpeg', 0.8);
                
                // Guardar en localStorage
                localStorage.setItem('current_frame', imageData);
                
                console.log('Foto capturada. Procesando...');
                
                // Procesar automáticamente
                (async () => {
                    try {
                        const res = await fetch('http://localhost:8002/_process_frame', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({frame_data: imageData}),
                            mode: 'cors',
                            credentials: 'omit'
                        });
                        
                        if (!res.ok) {
                            console.error('Error HTTP:', res.status);
                            return;
                        }
                        
                        const text = await res.text();
                        const data = JSON.parse(text);
                        
                        if (data && data.success) {
                            console.log('✓ Predicción:', data.letter, data.confidence);
                            
                            // Limpiar error cuando hay predicción exitosa
                            localStorage.removeItem('asl_error');
                            
                            // Guardar imágenes en localStorage si vienen en la respuesta
                            if (data.hand_crop) {
                                localStorage.setItem('asl_hand_crop', data.hand_crop);
                                console.log('✓ Mano recortada guardada en localStorage');
                                // Actualizar directamente la imagen si existe
                                const handImg = document.getElementById('asl-hand-crop-img');
                                if (handImg) {
                                    handImg.src = data.hand_crop;
                                    handImg.style.display = 'block';
                                    const placeholder = document.getElementById('hand-crop-placeholder');
                                    if (placeholder) placeholder.style.display = 'none';
                                    console.log('✓ Imagen recortada actualizada');
                                }
                            }
                            if (data.processed_image) {
                                localStorage.setItem('asl_processed_image', data.processed_image);
                                console.log('✓ Imagen procesada guardada en localStorage');
                                // Actualizar directamente la imagen si existe
                                const procImg = document.getElementById('asl-processed-img');
                                if (procImg) {
                                    procImg.src = data.processed_image;
                                    procImg.style.display = 'block';
                                    const placeholder = document.getElementById('processed-img-placeholder');
                                    if (placeholder) placeholder.style.display = 'none';
                                    console.log('✓ Imagen procesada actualizada');
                                }
                            }
                            
                            // Guardar en localStorage para Reflex
                            localStorage.setItem('asl_prediction', JSON.stringify({
                                letter: data.letter || '?',
                                confidence: data.confidence || '0%'
                            }));
                            
                            // Actualizar visualización del DOM directamente
                            // Buscar todos los textos que son solo la letra predicha
                            const allElements = document.querySelectorAll('*');
                            for (let el of allElements) {
                                if (el.childNodes.length === 1 && el.childNodes[0].nodeType === 3) {
                                    const text = el.textContent.trim();
                                    if (text === '?' || text.match(/^[A-Y]$/)) {
                                        el.textContent = data.letter;
                                        console.log('Actualizado:', el.tagName, 'con', data.letter);
                                    }
                                    if (text === '0%' || text.match(/^\\d+(\\.\\d)?%$/)) {
                                        el.textContent = data.confidence;
                                        console.log('Actualizado:', el.tagName, 'con', data.confidence);
                                    }
                                }
                            }
                        } else {
                            console.error('Error:', data.error);
                            // Remover predicción anterior para que polling muestre error
                            localStorage.removeItem('asl_prediction');
                            // Guardar el error en localStorage para mostrar en la UI
                            localStorage.setItem('asl_error', data.error || 'Error desconocido');
                        }
                    } catch(e) {
                        console.error('Error de conexión:', e.message);
                    }
                })();
            };
            
            window.processFrame = function() {
                // Leer frame de localStorage
                const frameData = localStorage.getItem('current_frame');
                if (!frameData) {
                    alert('Primero debes capturar una foto');
                    return;
                }
                // El procesamiento se hace en Python via Reflex
                // Este JS solo obtiene el frame del localStorage
            };
            
            window.startCamera = function() {
                console.log('startCamera called, streaming:', window.streaming);
                if (!window.streaming) window.initCamera();
            };
            
            window.stopCamera = function() {
                const video = document.getElementById('asl-video');
                if (video && video.srcObject) {
                    video.srcObject.getTracks().forEach(track => track.stop());
                    window.streaming = false;
                    console.log('Camera stopped');
                }
            };
            """
        ),
        
        # Script para sincronizar localStorage con la UI
        rx.script(
            """
            // Mostrar errores desde localStorage por más tiempo
            setInterval(function() {
                const errorData = localStorage.getItem('asl_error');
                const predData = localStorage.getItem('asl_prediction');
                const errorContainer = document.getElementById('asl-error-container');
                const errorText = document.getElementById('asl-error-text');
                
                if (errorContainer && errorText) {
                    // Si hay predicción exitosa, limpiar error
                    if (predData) {
                        errorContainer.style.display = 'none';
                        localStorage.removeItem('asl_error');
                        sessionStorage.removeItem('errorTimestamp');
                    } 
                    // Si hay error, mostrar
                    else if (errorData) {
                        errorText.textContent = errorData;
                        errorContainer.style.display = 'block';
                        console.log('Error visible:', errorData);
                        sessionStorage.setItem('errorTimestamp', Date.now());
                    } 
                    // Si no hay error, mantener oculto después de 2 segundos
                    else {
                        const errorTimestamp = sessionStorage.getItem('errorTimestamp');
                        if (errorTimestamp && (Date.now() - parseInt(errorTimestamp)) > 2000) {
                            errorContainer.style.display = 'none';
                            sessionStorage.removeItem('errorTimestamp');
                        }
                    }
                }
            }, 50);
            
            // Actualizar imágenes procesadas desde localStorage
            setInterval(function() {
                const handCrop = localStorage.getItem('asl_hand_crop');
                const processedImg = localStorage.getItem('asl_processed_image');
                
                if (handCrop) {
                    const imgElement = document.getElementById('asl-hand-crop-img');
                    const placeholder = document.getElementById('hand-crop-placeholder');
                    if (imgElement) {
                        imgElement.src = handCrop;
                        imgElement.style.display = 'block';
                        if (placeholder) placeholder.style.display = 'none';
                    }
                }
                
                if (processedImg) {
                    const imgElement = document.getElementById('asl-processed-img');
                    const placeholder = document.getElementById('processed-img-placeholder');
                    if (imgElement) {
                        imgElement.src = processedImg;
                        imgElement.style.display = 'block';
                        if (placeholder) placeholder.style.display = 'none';
                    }
                }
            }, 100);
            """
        ),
        
        rx.vstack(
            # Encabezado
            rx.vstack(
                rx.hstack(
                    rx.vstack(
                        rx.heading("ASL Recognizer", size="8", background="linear-gradient(135deg, #3182ce 0%, #2d5aa3 100%)", background_clip="text", color="transparent"),
                        rx.text(
                            "Reconocimiento de Señas del Alfabeto Americano",
                            text_align="center",
                            font_size="md",
                            color=rx.color_mode_cond("#666", "#aaa")
                        ),
                        width="100%",
                        spacing="2",
                        align_items="center"
                    ),
                    rx.spacer(),
                    width="100%"
                ),
                width="100%",
                padding="2em 0"
            ),
            
            rx.divider(margin="1em 0"),
            
            # Contenedor principal
            rx.hstack(
                # Columna izquierda: Cámara
                rx.vstack(
                    rx.hstack(
                        rx.box(
                            width="8px",
                            height="24px",
                            border_radius="4px",
                            background="linear-gradient(180deg, #3182ce, #2d5aa3)"
                        ),
                        rx.heading("Cámara Web", size="6", font_weight="700"),
                        align_items="center",
                        spacing="2"
                    ),
                    
                    rx.html("""
                    <div style="position: relative; width: 100%; max-width: 500px; margin: 0 auto;">
                        <video 
                            id="asl-video" 
                            autoplay 
                            playsinline
                            style="width: 100%; border: 3px solid #3182ce; border-radius: 12px; transform: scaleX(-1); background: #1a1a1a;"
                        ></video>
                        <canvas id="asl-canvas" style="display: none;"></canvas>
                    </div>
                    """),
                    
                    rx.vstack(
                        rx.hstack(
                            rx.button(
                                rx.hstack(
                                    rx.icon(tag="play", width="16px", height="16px"),
                                    rx.text("Iniciar"),
                                    align_items="center",
                                    spacing="1"
                                ),
                                on_click=ASLState.start_camera,
                                color_scheme="blue",
                                size="2",
                                font_weight="bold",
                                flex="1",
                                _hover={"transform": "scale(1.05)", "transition": "0.2s"}
                            ),
                            rx.button(
                                rx.hstack(
                                    rx.icon(tag="camera", width="16px", height="16px"),
                                    rx.text("Capturar"),
                                    align_items="center",
                                    spacing="1"
                                ),
                                on_click=rx.call_script("window.captureFrame()"),
                                color_scheme="green",
                                size="2",
                                font_weight="bold",
                                flex="1",
                                _hover={"transform": "scale(1.05)", "transition": "0.2s"}
                            ),
                            width="100%",
                            spacing="2"
                        ),
                        
                        rx.button(
                            rx.hstack(
                                rx.icon(tag="square", width="16px", height="16px"),
                                rx.text("Detener"),
                                align_items="center",
                                spacing="1"
                            ),
                            on_click=ASLState.stop_camera,
                            color_scheme="red",
                            size="2",
                            font_weight="bold",
                            width="100%",
                            _hover={"transform": "scale(1.05)", "transition": "0.2s"}
                        ),
                        width="100%",
                        spacing="3"
                    ),
                    
                    width="100%",
                    flex="1",
                    spacing="4"
                ),
                
                # Columna derecha: Predicción e Instrucciones
                rx.vstack(
                    # Predicción
                    rx.vstack(
                        rx.hstack(
                            rx.box(
                                width="8px",
                                height="24px",
                                border_radius="4px",
                                background="linear-gradient(180deg, #9f7aea, #7c3aed)"
                            ),
                            rx.heading("Predicción", size="6", font_weight="700"),
                            align_items="center",
                            spacing="2"
                        ),
                        
                        rx.box(
                            rx.vstack(
                                rx.heading(
                                    ASLState.predicted_letter,
                                    size="9",
                                    text_align="center",
                                    font_weight="900",
                                    color="#3182ce"
                                ),
                                rx.text(
                                    ASLState.confidence,
                                    text_align="center",
                                    font_size="xl",
                                    color="#7c3aed",
                                    font_weight="bold"
                                ),
                                spacing="2"
                            ),
                            width="100%",
                            padding="3em 2em",
                            border="3px solid #7c3aed",
                            border_radius="16px",
                            background=rx.color_mode_cond("#faf5ff", "#2a1a40"),
                            box_shadow="0 8px 16px rgba(124, 58, 237, 0.2)"
                        ),
                        
                        # Mensaje de error - Actualizado por JavaScript desde localStorage
                        rx.html("""
                        <div id="asl-error-container" style="width: 100%; display: none;">
                            <div style="padding: 1em; background: #fed7d7; border: 1px solid #fc8181; border-radius: 8px; display: flex; gap: 0.5em; align-items: center;">
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#f56565" stroke-width="2">
                                    <circle cx="12" cy="12" r="10"></circle>
                                    <line x1="12" y1="8" x2="12" y2="12"></line>
                                    <line x1="12" y1="16" x2="12.01" y2="16"></line>
                                </svg>
                                <span id="asl-error-text" style="color: #f56565; font-weight: 500; font-size: 0.9em;"></span>
                            </div>
                        </div>
                        """),
                        
                        # Imágenes procesadas
                        rx.vstack(
                            rx.heading("Procesamiento", size="5", font_weight="600", color="#48bb78"),
                            rx.hstack(
                                rx.vstack(
                                    rx.text("Mano Recortada", font_size="sm", font_weight="600", color="#3182ce"),
                                    rx.html("""
                                    <div id="hand-crop-container" style="position: relative; width: 100%; aspect-ratio: 1;">
                                        <img id="asl-hand-crop-img" src="" style="width: 100%; height: 100%; border: 2px solid #3182ce; border-radius: 6px; object-fit: contain; display: none;" />
                                        <div id="hand-crop-placeholder" style="width: 100%; height: 100%; background: #1a1a1a; border: 2px dashed #3182ce; border-radius: 6px; display: flex; align-items: center; justify-content: center; color: #666; font-size: 11px;">
                                            Esperando...
                                        </div>
                                    </div>
                                    """),
                                    flex="1"
                                ),
                                rx.vstack(
                                    rx.text("Imagen CNN (28x28)", font_size="sm", font_weight="600", color="#9f7aea"),
                                    rx.html("""
                                    <div id="processed-img-container" style="position: relative; width: 100%; aspect-ratio: 1;">
                                        <img id="asl-processed-img" src="" style="width: 100%; height: 100%; border: 2px solid #9f7aea; border-radius: 6px; object-fit: contain; image-rendering: pixelated; display: none;" />
                                        <div id="processed-img-placeholder" style="width: 100%; height: 100%; background: #1a1a1a; border: 2px dashed #9f7aea; border-radius: 6px; display: flex; align-items: center; justify-content: center; color: #666; font-size: 11px;">
                                            Entrada...
                                        </div>
                                    </div>
                                    """),
                                    flex="1"
                                ),
                                width="100%",
                                spacing="2"
                            ),
                            width="100%",
                            spacing="2",
                            border="1px solid #e2e8f0",
                            padding="1em",
                            border_radius="8px",
                            background=rx.color_mode_cond("#fafafa", "#111111")
                        ),
                        
                        # Estado de grabación
                        rx.cond(
                            ASLState.is_running,
                            rx.badge(
                                rx.hstack(
                                    rx.icon(tag="dot", width="10px", height="10px", color="red"),
                                    rx.text("GRABANDO"),
                                    spacing="1"
                                ),
                                color_scheme="red",
                                padding="0.5em 1em",
                                variant="surface"
                            ),
                            rx.badge(
                                rx.hstack(
                                    rx.icon(tag="dot", width="10px", height="10px"),
                                    rx.text("PAUSADO"),
                                    spacing="1"
                                ),
                                color_scheme="gray",
                                padding="0.5em 1em",
                                variant="surface"
                            )
                        ),
                        
                        width="100%",
                        spacing="3",
                        align_items="center"
                    ),
                    
                    # Instrucciones actualizadas
                    rx.box(
                        rx.vstack(
                            rx.hstack(
                                rx.icon(tag="info", width="20px", height="20px", color="#9f7aea"),
                                rx.heading("¿Cómo funciona?", size="5", font_weight="700"),
                                spacing="2",
                                align_items="center"
                            ),
                            rx.unordered_list(
                                rx.list_item("Haz clic en 'Iniciar' para abrir la cámara"),
                                rx.list_item("Forma una letra ASL frente a la cámara"),
                                rx.list_item("Haz clic en 'Capturar' para analizar"),
                                rx.list_item("La predicción aparecerá arriba"),
                                rx.list_item("Repite con diferentes letras"),
                                spacing="2"
                            ),
                            width="100%",
                            spacing="3"
                        ),
                        padding="1.5em",
                        background=rx.color_mode_cond("#faf5ff", "#2a1a40"),
                        border="1px solid #9f7aea",
                        border_radius="12px",
                        width="100%"
                    ),
                    
                    width="100%",
                    flex="1",
                    spacing="4"
                ),
                
                width="100%",
                spacing="4",
                align_items="stretch"
            ),
            
            # Footer
            rx.vstack(
                rx.divider(),
                rx.text(
                    "DeepCNN • Accuracy: 99.89% • 24 Clases (A-Y sin J, Z)",
                    text_align="center",
                    color=rx.color_mode_cond("#999", "#777"),
                    font_size="sm"
                ),
                width="100%",
                padding_y="2em"
            ),
            
            width="100%",
            max_width="1400px",
            spacing="5",
            padding="2em"
        ),
        size="3"
    )
