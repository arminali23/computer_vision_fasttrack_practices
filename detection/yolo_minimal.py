import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO

model_name = "yolov8n.pt"
imgsize=320 
infer_every_n =2
alpha = 0.2
present_thresh = 0.35

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = YOLO(model_name)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("cap not opened")

conf_ema = 0.0
fps_ema = None

frame_i = 0
prev_t = time.time()

last_person_boxes = []

while True:
    ret, frame = cap.read()
    if not ret :
        break
    if frame_i % infer_every_n == 0:
        results = model.predict(source=frame, imgsize=imgsize, device=device, verbose=False)
        r0 = results[0]
        last_person_boxes = []
        frame_conf = 0.0
        if r0.boxes is not None and len(r0.boxes) > 0:
            xyxy = r0.boxes.xyxy.detach().cpu().numpy()
            cls = r0.boxes.cls.detach().cpu().numpy().astype(int)
            conf = r0.boxes.conf.detach().cpu().numpy()
            
            for (x1, y1, x2, y2), c, cf in zip(xyxy, cls, conf):
                if c == 0:
                    last_person_boxes.append((x1, y1, x2, y2, cf))
            
            if last_person_boxes:
                frame_conf = max(cf for _, _, _, _, cf in last_person_boxes)
        
        conf_ema = (1 - alpha) * conf_ema + alpha * frame_conf
    
    present = conf_ema >= present_thresh
    
    for x1, y1, x2, y2, cf in last_person_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"person {cf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    now = time.time()
    fps = 1.0 / max(now - prev_t, 1e-6)
    prev_t = now
    fps_ema = fps if fps_ema is None else (1 - alpha) * fps_ema + alpha * fps
    
    cv2.putText(frame, f"fps: {fps_ema:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"present: {'yes' if present else 'no'}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow("yolo minimal", frame)
    
    if cv2.waitKey(1) & 0xFF in (ord("Q"), ord("q")):
        break
        
    frame_i += 1
    
cap.release()
cv2.destroyAllWindows()