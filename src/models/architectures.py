"""
Módulo de Modelos para Reconocimiento de Lenguaje de Señas.

Este módulo contiene arquitecturas de modelos CNN personalizadas y
wrappers para Transfer Learning.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    MobileNetV2, EfficientNetB0, ResNet50, VGG16, InceptionV3
)


def create_simple_cnn(input_shape=(28, 28, 1), num_classes=24, name='SimpleCNN'):
    """
    Crea una CNN simple como baseline.
    
    Args:
        input_shape: Forma de entrada (altura, ancho, canales)
        num_classes: Número de clases de salida
        name: Nombre del modelo
    
    Returns:
        modelo Keras compilado
    """
    model = models.Sequential(name=name)
    
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model


def create_improved_cnn(input_shape=(28, 28, 1), num_classes=24, name='ImprovedCNN'):
    """
    Crea una CNN mejorada con BatchNormalization y más capas.
    
    Args:
        input_shape: Forma de entrada
        num_classes: Número de clases
        name: Nombre del modelo
    
    Returns:
        modelo Keras compilado
    """
    model = models.Sequential(name=name)
    
    # Bloque 1
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Bloque 2
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Bloque 3
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Clasificador
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model


def create_deep_cnn(input_shape=(28, 28, 1), num_classes=24, name='DeepCNN'):
    """
    Crea una CNN profunda con arquitectura residual simplificada.
    
    Args:
        input_shape: Forma de entrada
        num_classes: Número de clases
        name: Nombre del modelo
    
    Returns:
        modelo Keras
    """
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Bloque inicial
    x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Bloques residuales simplificados
    for filters in [64, 128, 256]:
        # Shortcut
        shortcut = x
        
        # Convoluciones
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Ajustar shortcut si es necesario
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        # Sumar shortcut
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)
    
    # Clasificador
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name=name)
    return model


def create_transfer_learning_model(
    base_model_name='EfficientNetB0',
    input_shape=(224, 224, 3),
    num_classes=24,
    trainable_base=False,
    name=None
):
    """
    Crea un modelo de Transfer Learning usando modelos pre-entrenados.
    
    Args:
        base_model_name: Nombre del modelo base ('EfficientNetB0', 'MobileNetV2', 
                         'ResNet50', 'VGG16', 'InceptionV3')
        input_shape: Forma de entrada (debe ser compatible con el modelo base)
        num_classes: Número de clases de salida
        trainable_base: Si True, el modelo base es entrenable
        name: Nombre del modelo
    
    Returns:
        modelo Keras
    """
    # Seleccionar modelo base
    base_models = {
        'EfficientNetB0': EfficientNetB0,
        'MobileNetV2': MobileNetV2,
        'ResNet50': ResNet50,
        'VGG16': VGG16,
        'InceptionV3': InceptionV3
    }
    
    if base_model_name not in base_models:
        raise ValueError(f"Modelo base no soportado: {base_model_name}")
    
    model_name = name or f'TL_{base_model_name}'
    
    # Cargar modelo base pre-entrenado
    base_model = base_models[base_model_name](
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Congelar o descongelar modelo base
    base_model.trainable = trainable_base
    
    # Construir modelo completo
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Base pre-entrenada
    x = base_model(inputs, training=False)
    
    # Cabeza de clasificación personalizada
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)
    
    return model


def compile_model(model, learning_rate=0.001, metrics=['accuracy']):
    """
    Compila un modelo con configuración estándar.
    
    Args:
        model: Modelo Keras a compilar
        learning_rate: Tasa de aprendizaje
        metrics: Lista de métricas
    
    Returns:
        modelo compilado
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=metrics
    )
    
    return model


def get_model_summary(model):
    """
    Obtiene un resumen del modelo en formato string.
    
    Args:
        model: Modelo Keras
    
    Returns:
        string con resumen del modelo
    """
    import io
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\\n'))
    return stream.getvalue()


# Función auxiliar para crear todos los modelos
def create_all_models(input_shape=(28, 28, 1), num_classes=24):
    """
    Crea todos los modelos disponibles para comparación.
    
    Args:
        input_shape: Forma de entrada
        num_classes: Número de clases
    
    Returns:
        diccionario con todos los modelos
    """
    models_dict = {
        'SimpleCNN': create_simple_cnn(input_shape, num_classes),
        'ImprovedCNN': create_improved_cnn(input_shape, num_classes),
        'DeepCNN': create_deep_cnn(input_shape, num_classes)
    }
    
    return models_dict
