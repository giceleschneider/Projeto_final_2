from pathlib import Path
import shutil

from ultralytics import YOLO


DATASET_YAML = "data.yaml"
RUN_DIR = Path("runs/detect/ppe_detector")
BEST_MODEL_PATH = RUN_DIR / "weights" / "best.pt"
LAST_MODEL_PATH = RUN_DIR / "weights" / "last.pt"


def main():
    if LAST_MODEL_PATH.exists():
        print("Retomando treinamento a partir do último checkpoint...")
        model = YOLO(str(LAST_MODEL_PATH))
        resume = True
    else:
        print("Carregando YOLOv8n pré-treinado...")
        model = YOLO("yolov8n.pt")
        resume = False

    print("Iniciando treinamento...")
    model.train(
        data=DATASET_YAML,
        epochs=30,
        imgsz=640,
        batch=8,
        patience=10,
        project="runs/detect",
        name="ppe_detector",
        exist_ok=True,
        resume=resume,
    )

    destination = Path("models/best.pt")
    destination.parent.mkdir(exist_ok=True)

    if BEST_MODEL_PATH.exists():
        shutil.copy2(BEST_MODEL_PATH, destination)
        print(f"\nModelo salvo em: {destination}")
    else:
        print("\nNão foi possível encontrar best.pt.")
        print(f"Verifique a pasta {RUN_DIR / 'weights'}/")


if __name__ == "__main__":
    main()
