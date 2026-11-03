from ultralytics import YOLO

MODEL_PATH = "models/best.pt"
DATASET_YAML = "data.yaml"


def main():
    model = YOLO(MODEL_PATH)

    metrics = model.val(
        data=DATASET_YAML,
        imgsz=640,
    )

    print("\nValidação concluída.")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
