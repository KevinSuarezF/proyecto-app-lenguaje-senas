"""
Módulo de utilidades para entrenamiento de modelos.

Incluye callbacks, funciones de logging y helpers para entrenar modelos.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger, Callback
)


def create_callbacks(model_name, output_dir='models', patience=10, min_delta=0.001):
    """
    Crea callbacks estándar para entrenamiento.
    
    Args:
        model_name: Nombre del modelo
        output_dir: Directorio para guardar modelos
        patience: Paciencia para early stopping
        min_delta: Cambio mínimo para considerar mejora
    
    Returns:
        Lista de callbacks
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks = []
    
    # ModelCheckpoint - Guardar mejor modelo
    checkpoint_path = os.path.join(output_dir, f'{model_name}_{timestamp}_best.h5')
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # EarlyStopping - Detener si no mejora
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=patience,
        restore_best_weights=True,
        mode='max',
        verbose=1,
        min_delta=min_delta
    )
    callbacks.append(early_stop)
    
    # ReduceLROnPlateau - Reducir learning rate
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=max(3, patience // 3),
        min_lr=1e-7,
        mode='min',
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # TensorBoard - Visualización
    tensorboard_dir = os.path.join('logs', f'{model_name}_{timestamp}')
    tensorboard = TensorBoard(
        log_dir=tensorboard_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=False
    )
    callbacks.append(tensorboard)
    
    # CSVLogger - Log de métricas
    csv_path = os.path.join('logs', f'{model_name}_{timestamp}.csv')
    csv_logger = CSVLogger(csv_path, append=False)
    callbacks.append(csv_logger)
    
    return callbacks


class MetricsLogger(Callback):
    """Callback personalizado para logging detallado de métricas."""
    
    def __init__(self, log_file='training_log.txt'):
        super().__init__()
        self.log_file = log_file
        self.epoch_metrics = []
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Guardar métricas
        metrics = {
            'epoch': epoch + 1,
            'loss': logs.get('loss'),
            'accuracy': logs.get('accuracy'),
            'val_loss': logs.get('val_loss'),
            'val_accuracy': logs.get('val_accuracy'),
            'lr': float(tf.keras.backend.get_value(self.model.optimizer.lr))
        }
        self.epoch_metrics.append(metrics)
        
        # Escribir a archivo
        with open(self.log_file, 'a') as f:
            f.write(f"Epoch {metrics['epoch']}: ")
            f.write(f"loss={metrics['loss']:.4f}, ")
            f.write(f"acc={metrics['accuracy']:.4f}, ")
            f.write(f"val_loss={metrics['val_loss']:.4f}, ")
            f.write(f"val_acc={metrics['val_accuracy']:.4f}, ")
            f.write(f"lr={metrics['lr']:.2e}\n")


def plot_training_history(history, save_path=None, model_name='Model'):
    """
    Visualiza el historial de entrenamiento.
    
    Args:
        history: History object de Keras
        save_path: Ruta para guardar la figura
        model_name: Nombre del modelo para el título
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_title(f'{model_name} - Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    plt.show()


def save_training_results(model, history, model_name, output_dir='results/reports'):
    """
    Guarda resultados completos del entrenamiento.
    
    Args:
        model: Modelo entrenado
        history: History object
        model_name: Nombre del modelo
        output_dir: Directorio de salida
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Guardar historial como JSON
    history_dict = {
        'model_name': model_name,
        'timestamp': timestamp,
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'epochs_trained': len(history.history['loss']),
        'history': {
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        }
    }
    
    json_path = os.path.join(output_dir, f'{model_name}_{timestamp}_history.json')
    with open(json_path, 'w') as f:
        json.dump(history_dict, f, indent=4)
    
    print(f"Historial guardado en: {json_path}")
    
    # Guardar resumen del modelo
    summary_path = os.path.join(output_dir, f'{model_name}_{timestamp}_summary.txt')
    with open(summary_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    print(f"Resumen del modelo guardado en: {summary_path}")


def compare_models(results_dict, save_path='results/figures/models_comparison.png'):
    """
    Compara múltiples modelos visualizando sus métricas.
    
    Args:
        results_dict: Diccionario {nombre_modelo: history}
        save_path: Ruta para guardar la figura
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Training Accuracy
    for name, history in results_dict.items():
        axes[0, 0].plot(history.history['accuracy'], label=name, linewidth=2)
    axes[0, 0].set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Validation Accuracy
    for name, history in results_dict.items():
        axes[0, 1].plot(history.history['val_accuracy'], label=name, linewidth=2)
    axes[0, 1].set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Training Loss
    for name, history in results_dict.items():
        axes[1, 0].plot(history.history['loss'], label=name, linewidth=2)
    axes[1, 0].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Validation Loss
    for name, history in results_dict.items():
        axes[1, 1].plot(history.history['val_loss'], label=name, linewidth=2)
    axes[1, 1].set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    
    # Crear tabla de comparación
    comparison_data = []
    for name, history in results_dict.items():
        comparison_data.append({
            'Model': name,
            'Final Train Acc': f"{history.history['accuracy'][-1]:.4f}",
            'Final Val Acc': f"{history.history['val_accuracy'][-1]:.4f}",
            'Best Val Acc': f"{max(history.history['val_accuracy']):.4f}",
            'Final Train Loss': f"{history.history['loss'][-1]:.4f}",
            'Final Val Loss': f"{history.history['val_loss'][-1]:.4f}",
            'Epochs': len(history.history['loss'])
        })
    
    import pandas as pd
    df = pd.DataFrame(comparison_data)
    print("\n" + "="*80)
    print("COMPARACIÓN DE MODELOS")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    return df
