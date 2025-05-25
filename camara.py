from ultralytics import YOLO
import cv2
import time

def main():
    model = YOLO("runs/detect/instrumentos_dentales_yolo_model5/weights/best.pt")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ No se pudo acceder a la cámara.")
        return

    print("✅ Cámara abierta. Presiona 'q' para salir.")

    clase_en_proceso = ""
    tiempo_detectando = 0
    UMBRAL_TIEMPO = 3

    while True:
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("⚠️ No se pudo leer el frame.")
                continue

            results = model.predict(source=frame, conf=0.5, verbose=False)

            if results and hasattr(results[0], "boxes") and results[0].boxes is not None:
                boxes = results[0].boxes
                clases = results[0].names

                if boxes.conf is not None and len(boxes.conf) > 0:
                    scores = boxes.conf.tolist()
                    class_ids = boxes.cls.tolist()

                    max_idx = scores.index(max(scores))
                    nombre = clases[int(class_ids[max_idx])]

                    if nombre == clase_en_proceso:
                        tiempo_detectando += 1
                    else:
                        clase_en_proceso = nombre
                        tiempo_detectando = 1

                    if tiempo_detectando >= UMBRAL_TIEMPO:
                        print(f"✅ Confirmado: {nombre}")
                        tiempo_detectando = 0  # Reinicia para no repetir

            annotated = results[0].plot()
            cv2.imshow("Detección en tiempo real", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("👋 Saliendo por tecla 'q'")
                break

        except Exception as e:
            print(f"❌ Error inesperado: {e}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()