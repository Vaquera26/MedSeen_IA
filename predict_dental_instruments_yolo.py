"""
Predicción e Inferencia con Modelo YOLO para Instrumentos Dentales

Descripción:
Este script realiza inferencia utilizando un modelo YOLO v8 previamente entrenado
para detectar instrumentos dentales en imágenes individuales. El programa carga
una imagen específica, aplica el modelo de detección y muestra los resultados
tanto guardando la imagen con las detecciones como visualizándola en tiempo real.

Funcionalidades:
- Carga de modelo entrenado desde archivo de pesos
- Predicción sobre imagen específica
- Guardado automático de resultados con bounding boxes
- Visualización interactiva usando OpenCV
- Detección de múltiples instrumentos dentales en una sola imagen

Autor(es):
- Juan Fernando Vaquera Sanchez (21130869)
- Miriam Alicia Sanchez Cervantes (21130882)  
- Diego Muñoz Rede (21130893)

Fecha de creación: Mayo 2025
Archivo: predict_dental_instruments_yolo.py

Requisitos:
- ultralytics
- opencv-python (cv2)
- PyTorch
- Modelo entrenado (best.pt)
- Imagen de prueba (forceps.jpg)

Estructura esperada:
runs/detect/instrumentos_dentales_yolo_model5/weights/best.pt
forceps.jpg (imagen de entrada)

Salida:
- Imagen guardada con detecciones en runs/detect/predict/
- Ventana de visualización interactiva
"""

from ultralytics import YOLO
import cv2

def main():
    """
    Función principal para realizar predicciones con el modelo YOLO entrenado.
    
    Carga el modelo entrenado, ejecuta la predicción sobre una imagen específica,
    guarda los resultados automáticamente y muestra la visualización interactiva
    de las detecciones encontradas.
    
    Returns:
        None: Muestra resultados en ventana de OpenCV y guarda archivos
    """
    
    # Cargar modelo YOLO entrenado desde archivo de pesos
    # best.pt contiene los pesos del modelo con mejor rendimiento durante entrenamiento
    # El modelo ya está entrenado para reconocer instrumentos dentales específicos
    model = YOLO("runs/detect/instrumentos_dentales_yolo_model5/weights/best.pt")
    
    # Ejecutar predicción sobre imagen específica
    # source: Ruta a la imagen de entrada (cambiar según imagen deseada)
    # save=True: Guarda automáticamente la imagen con detecciones en runs/detect/predict/
    # La predicción retorna objetos Results con información de detecciones
    results = model.predict(source="forceps.jpg", save=True)
    
    # Procesamiento y visualización de resultados
    # Itera sobre todos los resultados de predicción (normalmente uno por imagen)
    for r in results:
        # Generar imagen con bounding boxes y etiquetas dibujadas
        # plot() renderiza las detecciones sobre la imagen original
        # Incluye: cajas delimitadoras, etiquetas de clase, scores de confianza
        im_bgr = r.plot()  # Obtiene imagen con los boxes dibujados
        
        # Mostrar resultado usando OpenCV
        # Crea ventana interactiva para visualizar las detecciones
        cv2.imshow("Resultado", im_bgr)
        
        # Esperar interacción del usuario
        # waitKey(0): Pausa ejecución hasta que se presione cualquier tecla
        # Permite al usuario examinar detalladamente los resultados
        cv2.waitKey(0)  # Espera hasta que presiones una tecla
        
        # Limpiar recursos de OpenCV
        # Cierra todas las ventanas abiertas y libera memoria
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Punto de entrada del programa
    # Ejecuta la función main() solo cuando el script se ejecuta directamente
    # No se ejecuta si el archivo es importado como módulo
    main()