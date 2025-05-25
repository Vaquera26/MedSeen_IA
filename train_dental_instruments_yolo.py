"""
Entrenamiento de Modelo YOLO para Reconocimiento de Instrumentos Dentales

Descripción:
Este script entrena un modelo de detección de objetos YOLO (You Only Look Once) 
versión 8 para reconocer y clasificar instrumentos dentales. Utiliza el framework 
Ultralytics YOLO con configuración optimizada para GPU NVIDIA RTX 3050.

El modelo entrenado será capaz de detectar y clasificar diferentes tipos de 
instrumentos dentales en imágenes, proporcionando coordenadas de bounding boxes 
y etiquetas de clasificación.

Autor(es):
- Juan Fernando Vaquera Sanchez (21130869)
- Miriam Alicia Sanchez Cervantes (21130882)  
- Diego Muñoz Rede (21130893)

Archivo: train_dental_instruments_yolo.py

Requisitos:
- ultralytics
- PyTorch con soporte CUDA
- GPU NVIDIA RTX 3050 
- Dataset estructurado en formato YOLO
"""

from ultralytics import YOLO

def main():
    """
    Función principal para entrenar el modelo YOLO de instrumentos dentales.
    
    Configura y ejecuta el entrenamiento del modelo utilizando los parámetros
    optimizados para detección de instrumentos dentales con hardware RTX 3050.
    """
    
    # Inicialización del modelo YOLO pre-entrenado
    # yolov8n.pt: Modelo nano (más rápido, menos preciso)
    # Alternativa: yolov8s.pt para mayor precisión con mayor tiempo de procesamiento
    model = YOLO("yolov8n.pt")
    
    # Configuración y ejecución del entrenamiento
    model.train(
        data="datasets/data.yaml",                    # Ruta al archivo de configuración del dataset
        epochs=100,                                   # Número de épocas de entrenamiento
        imgsz=640,                                   # Tamaño de imagen de entrada (640x640 píxeles)
        batch=8,                                     # Tamaño del batch (optimizado para RTX 3050)
        name="instrumentos_dentales_yolo_model",     # Nombre del experimento para guardar resultados
        device="cuda"                                # Utilizar GPU CUDA para acelerar entrenamiento
    )

if __name__ == "__main__":
    # Punto de entrada del programa
    # Ejecuta la función main() solo si el script se ejecuta directamente
    main()