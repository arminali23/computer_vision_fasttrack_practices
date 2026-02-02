import time
import cv2
import torch
from ultralytics import YOLO

# cihaz secimi (mac icin mps)
device = "mps" if torch.backends.mps.is_available() else "cpu"

# model
model = YOLO("yolov8n.pt")

# kamera
cap = cv2.VideoCapture(0)

# ayarlar
imgsz = 320
infer_every_n = 2
alpha = 0.2

# ema degerleri
fps_ema = None
conf_ema = 0.0

frame_index = 0
last_boxes = []

prev_time = time.time()

while True:
    start_loop = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    # her n frame'de bir inference
    if frame_index % infer_every_n == 0:
        results = model.predict(
            source=frame,
            imgsz=imgsz,
            device=device,
            verbose=False
        )

        boxes = results[0].boxes
        last_boxes = []

        if boxes is not None:
            for box in boxes:
                cls = int(box.cls.item())
                conf = float(box.conf.item())

                # sadece person (coco id = 0)
                if cls == 0:
                    last_boxes.append((box.xyxy[0], conf))

            # frame confidence = en guclu person
            if last_boxes:
                frame_conf = max(c for _, c in last_boxes)
            else:
                frame_conf = 0.0

            # confidence ema
            conf_ema = (1 - alpha) * conf_ema + alpha * frame_conf

    # kutulari ciz
    for (xyxy, conf) in last_boxes:
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"person {conf:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # fps hesapla
    now = time.time()
    fps = 1.0 / max(now - prev_time, 1e-6)
    prev_time = now

    if fps_ema is None:
        fps_ema = fps
    else:
        fps_ema = (1 - alpha) * fps_ema + alpha * fps

    # bilgi yaz
    cv2.putText(frame, f"fps: {fps_ema:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"person_conf_ema: {conf_ema:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("yolo simple", frame)

    if cv2.waitKey(1) & 0xff == ord("q"):
        break

    frame_index += 1

cap.release()
cv2.destroyAllWindows()