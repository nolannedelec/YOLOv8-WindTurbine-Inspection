import cv2
import numpy as np
import time
import argparse
from ultralytics import YOLO

def main(engine_path, source):
    # Chargement du modèle TensorRT (déjà compilé en FP16)
    model = YOLO(engine_path, task='detect')

    # Initialisation de la capture vidéo
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Format standard supporté par les capteurs

    print("Démarrage de l'inférence temps réel...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        start_time = time.perf_counter()

        # Inférence avec le moteur TensorRT
        results = model.predict(
            source=frame,
            conf=0.35,
            device=0,
            verbose=False
            # half=True retiré : déjà encodé dans le .engine
        )

        end_time = time.perf_counter()
        latency = (end_time - start_time) * 1000
        fps = 1 / (end_time - start_time)

        # Visualisation des détections
        annotated_frame = results[0].plot()

        # Post-traitement vectorisé des boîtes (NumPy)
        boxes = results[0].boxes
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy().astype(int)   # coords en int vectorisé
            confs = boxes.conf.cpu().numpy()               # confiances
            cls   = boxes.cls.cpu().numpy().astype(int)    # classes

        # Affichage des métriques
        info_text = f"{latency:.1f} ms | {fps:.1f} FPS"
        cv2.putText(annotated_frame, info_text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("Inspection Eolienne - Nordtank", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='modeles/best.engine')
    parser.add_argument('--source', type=int, default=0)
    args = parser.parse_args()
    main(args.engine, args.source)