import cv2
import time
from ultralytics import YOLO
import torch

device = "mps" if torch.mps.is_available() else "cpu"

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("camera could not be opened")

prev_t = time.time()
fps_ema = None
alpha = 0.2

print("Controls:")
print("  Q: quit")

while True:
    t_loop0 = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    
    t0 = time.time()
    results = model.predict(
        source = frame,
        imgsz = 320,
        iou = 0.45,
        conf = 0.25,
        device = device,
        verbose = False,
        
    )
    infer_ms = (time.time() - t0) * 1000
    annotated = results[0].plot()
    
    loop_ms = (time.time() - t_loop0) * 1000
    fps = 1000 / max(loop_ms, 1e-6)
    
    if fps_ema is None:
        fps_ema = fps
    else:
        fps_ema = (1 - alpha) * fps_ema + alpha * fps

    cv2.putText(annotated, f"FPS: {fps_ema:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(annotated, f"infer: {infer_ms:.1f} ms | loop: {loop_ms:.1f} ms | imgsz=320", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("YOLOv8 Webcam", annotated)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        
    
cap.release()
cv2.destroyAllWindows()

