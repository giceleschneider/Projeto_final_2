from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from ultralytics import YOLO

app = FastAPI(
    title="PPE Detector API",
    description="API simples de detecção de objetos usando YOLOv8.",
    version="1.0.0",
)

MODEL_CANDIDATES = [
    Path("models/best.pt"),
    Path("runs/detect/ppe_detector/weights/best.pt"),
    Path("runs/detect/runs/ppe_detector/weights/best.pt"),
]

MODEL_PATH = next((p for p in MODEL_CANDIDATES if p.exists()), None)

# O modelo é carregado uma vez quando a API inicia.
if MODEL_PATH is None:
    model = None
else:
    model = YOLO(str(MODEL_PATH))


@app.get("/")
def root():
    return {
        "message": "PPE Detector API está funcionando.",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    global model

    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Modelo não encontrado. Coloque o arquivo best.pt em models/best.pt.",
        )

    # Verifica se o arquivo enviado é uma imagem.
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Envie um arquivo de imagem.",
        )

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Não foi possível abrir a imagem.",
        )

    # Faz a previsão.
    results = model.predict(
        source=image,
        conf=0.25,
        verbose=False,
    )

    detections = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()

            detections.append(
                {
                    "class_name": result.names[class_id],
                    "confidence": round(confidence, 2),
                    "box": {
                        "x_min": round(x_min),
                        "y_min": round(y_min),
                        "x_max": round(x_max),
                        "y_max": round(y_max),
                    },
                }
            )

    return {
        "status": "success",
        "detections": detections,
    }
