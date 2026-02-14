# Aplicación de Reconocimiento de Lenguaje de Señas ASL

Sistema completo de ciencia de datos para reconocimiento de lenguaje de señas americano (ASL) usando Deep Learning y una interfaz web interactiva con Reflex.

## Descripción del Proyecto

Este proyecto implementa un sistema **end-to-end** para el reconocimiento de señas del alfabeto americano (ASL) usando redes neuronales convolucionales (CNN) y Transfer Learning. Incluye:

- Análisis exploratorio de datos (EDA) exhaustivo con visualizaciones
- Preprocesamiento avanzado de imágenes con técnicas de data augmentation
- Entrenamiento y comparación de múltiples arquitecturas de modelos
- Fine-tuning con transfer learning de modelos pre-entrenados
- Aplicación web interactiva para detección en tiempo real con Reflex
- Servidor FastAPI para procesamiento de frames en backend dedicado

## Estructura del Proyecto

```
proyecto-app-lenguaje-senas/
├── asl_app/                        # Aplicacion Reflex principal
│   ├── __init__.py                # Exporta la app de Reflex
│   ├── asl_app.py                 # Modulo principal Reflex
│   ├── state.py                   # Estado reactivo de la aplicacion
│   └── pages/
│       ├── __init__.py
│       └── index.py               # Interfaz de usuario
│
├── notebooks/                      # Analisis y entrenamientos
│   ├── 01_descarga_datos.ipynb    # Descarga del dataset
│   ├── 02_EDA.ipynb               # Analisis exploratorio
│   ├── 03_preprocesamiento.ipynb  # Procesamiento de datos
│   └── 04_entrenamiento_modelos.ipynb  # Entrenamiento y comparación de modelos
│
├── data/                           # Datasets
│   ├── raw/                        # Datos crudos
│   └── processed/                  # Datos procesados
│
├── models/                         # Modelos entrenados
│   └── DeepCNN.keras              # Modelo principal
│
├── results/                        # Resultados y reportes
│   ├── figures/                    # Graficos
│   └── reports/                    # Reportes
│
├── processor_server.py             # Servidor FastAPI (puerto 8002)
├── rxconfig.py                     # Configuracion de Reflex
├── requirements.txt                # Dependencias
├── README.md                       # Este archivo
├── LICENSE                         # Licencia MIT
└── .gitignore
```

## Inicio Rápido

### Requisitos Previos

- Python 3.8+
- pip o conda
- Cámara web (para usar la aplicación)

### Instalación

1. **Clonar el repositorio**
   ```bash
   git clone <URL_REPOSITORIO>
   cd proyecto-app-lenguaje-senas
   ```

2. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

### Ejecutar la Aplicación Web

La aplicación requiere 2 terminales separadas:

**Terminal 1 - Servidor de procesamiento (FastAPI):**
```bash
python processor_server.py
```
Debera mostrar:
```
[OK] SERVIDOR FASTAPI INICIADO CORRECTAMENTE
[INFO] URL: http://127.0.0.1:8002
[INFO] Endpoint: POST http://127.0.0.1:8002/_process_frame
[OK] CORS habilitado para http://localhost:3000
[OK] Procesamiento de frames disponible
```

**Terminal 2 - Aplicacion Reflex:**
```bash
reflex run
```

Accede en tu navegador a: `http://localhost:3000`

### Flujo de la Aplicacion

1. El frontend captura frames de la camara web
2. Envia el frame en Base64 al servidor FastAPI (puerto 8002)
3. FastAPI procesa la imagen y carga el modelo Keras
4. Retorna la prediccion (letra + confianza)
5. El frontend actualiza la interfaz con el resultado

### Ejecutar Notebooks

```bash
jupyter notebook
```

Navega a la carpeta `notebooks/` para abrir los análisis.

## Características Principales

### Reconocimiento en Tiempo Real
- Captura de video en tiempo real desde la cámara web
- Procesamiento de frames con Deep Learning
- Predicción instantánea de letras ASL (A-Y, sin J ni Z)
- Confianza de predicción mostrada en porcentaje
- **Precisión: 99.89%**

### Análisis y Entrenamiento
- Dataset: Sign Language MNIST (34,627 imágenes, 24 clases)
- Múltiples arquitecturas: CNN personalizado, Transfer Learning, etc.
- Técnicas de regularización: Dropout, BatchNormalization
- Data augmentation para mejorar la generalización
- Validación cruzada y métricas comprehensivas

### Interfaz Web
- Diseño responsivo y moderno con Reflex
- Tema claro/oscuro automático
- Instrucciones integradas
- Visualización instantánea de predicciones
- Controles intuitivos para cámara

## Componentes Técnicos

### Backend
- **Reflex**: Framework web full-stack en Python
- **FastAPI**: Servidor dedicado para procesamiento de frames
- **TensorFlow/Keras**: Deep Learning
- **OpenCV**: Procesamiento de imágenes

### Frontend
- **Reflex Components**: UI moderna y responsiva
- **JavaScript**: Captura de cámara del navegador
- **Canvas API**: Procesamiento de frames

### Data Science
- **NumPy/Pandas**: Manipulación de datos
- **Matplotlib/Seaborn**: Visualizaciones
- **Scikit-learn**: ML utilities
- **Jupyter**: Análisis interactivo

## Resultados

### Métricas de Entrenamiento

| Métrica | Valor |
|---------|-------|
| Precisión (Accuracy) | 99.89% |
| Pérdida (Loss) | 0.0031 |
| Clases | 24 (A-I, K-Y) |
| Imágenes Entrenadas | 27,455 |
| Imágenes de Prueba | 7,172 |

### Arquitectura del Modelo

Modelo CNN personalizado (DeepCNN) con:
- 3 capas convolucionales
- 2 capas de pooling
- Capas densas con dropout
- Activación ReLU/Softmax

## Documentacion

Para mas detalles sobre cada componente:

- **[Descarga de Datos](notebooks/01_descarga_datos.ipynb)** - Obtención del dataset
- **[Análisis EDA](notebooks/02_EDA.ipynb)** - Exploración y visualización del dataset
- **[Preprocesamiento](notebooks/03_preprocesamiento.ipynb)** - Transformaciones y data augmentation
- **[Entrenamiento](notebooks/04_entrenamiento_modelos.ipynb)** - Entrenamiento y comparación de modelos
- **[Código fuente - asl_app](asl_app/)** - Implementación de la aplicación web

## Autores

- **Kevin Suarez**
- **David Zabala**

Maestría en Estadística Aplicada y Ciencia de Datos  
Curso: Deep Learning 2

## Fuentes y Referencias

- **Dataset**: [Sign Language MNIST - Kaggle](https://www.kaggle.com/datamunge/sign-language-mnist)
- **ASL Alphabet**: [Lifeprint ASL Alphabet](https://www.lifeprint.com/asl101/fingerspelling/abc.htm)
- **Transfer Learning**: [Keras Applications](https://keras.io/api/applications/)
- **Reflex Framework**: [Reflex Documentation](https://reflex.dev/)

## Licencia

Este proyecto es de código abierto bajo licencia **MIT**. Ver archivo [LICENSE](LICENSE) para más detalles.

---

**Última actualización**: Febrero 2026