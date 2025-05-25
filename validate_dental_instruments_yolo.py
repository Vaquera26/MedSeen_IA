"""
Validación de Modelo YOLO Entrenado para Instrumentos Dentales

Descripción:
Este script valida y evalúa el rendimiento de un modelo YOLO v8 previamente 
entrenado para la detección de instrumentos dentales. Carga el mejor modelo 
generado durante el entrenamiento y ejecuta métricas de validación completas 
sobre el conjunto de datos de prueba.

Las métricas calculadas incluyen:
- mAP (mean Average Precision) @ IoU 0.5
- mAP @ IoU 0.5:0.95
- Precisión y Recall por clase
- Matrices de confusión
- Tiempos de inferencia

Autor(es):
- Juan Fernando Vaquera Sanchez (21130869)
- Miriam Alicia Sanchez Cervantes (21130882)  
- Diego Muñoz Rede (21130893)

Fecha de creación: Mayo 2025
Archivo: validate_dental_instruments_yolo.py

Requisitos:
- ultralytics
- PyTorch
- Modelo entrenado (best.pt) en la ruta especificada
- Dataset de validación estructurado

Estructura esperada:
runs/detect/instrumentos_dentales_yolo_model5/weights/best.pt
datasets/data.yaml
"""

from ultralytics import YOLO

def main():
    """
    Función principal para validar el modelo YOLO entrenado.
    
    Carga el mejor modelo entrenado y ejecuta una evaluación completa
    sobre el conjunto de datos de validación, mostrando métricas detalladas
    de rendimiento del modelo.
    
    Returns:
        None: Imprime las métricas de validación en consola
    """
    
    # Cargar el modelo entrenado desde el archivo de pesos
    # Ruta al mejor modelo generado durante el entrenamiento (epoch con menor loss)
    # El archivo best.pt contiene los pesos optimizados del modelo
    model = YOLO("runs/detect/instrumentos_dentales_yolo_model5/weights/best.pt")
    
    # Ejecutar validación del modelo sobre el conjunto de datos
    # data: Archivo YAML que contiene la configuración del dataset
    # Incluye rutas a imágenes de validación, clases y estructura del dataset
    metrics = model.val(data="datasets/data.yaml")
    
    # Mostrar métricas de evaluación en consola
    # Las métricas incluyen:
    # - mAP50: mean Average Precision con IoU threshold de 0.5
    # - mAP50-95: mean Average Precision promediado desde IoU 0.5 hasta 0.95
    # - Precision: Precisión por clase y promedio
    # - Recall: Recall por clase y promedio
    # - F1-score: Puntuación F1 por clase
    print(metrics)

if __name__ == "__main__":
    # Configuración específica para compatibilidad con Windows
    # freeze_support() evita problemas de multiprocessing en sistemas Windows
    # cuando el script se ejecuta como ejecutable empaquetado (.exe)
    import multiprocessing
    multiprocessing.freeze_support()  # ✅ Recomendado para Windows
    
    # Ejecutar función principal de validación
    main()