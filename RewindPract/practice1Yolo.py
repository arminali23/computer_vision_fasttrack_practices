import cv2
import numpy as np
import os 
from ultralytics import YOLO 
import torch

imgsz = 320
conf_thres = 0.35
infer_every_n = 3
frame_i = 0
last_boxes = []

alpha = 0.2
present_on = 0.45
present_off = 0.25

conf_ema = 0.0
present = False

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("cam not opened")

while True: 
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_i % infer_every_n == 0:
        results = model.predict(source=frame, imgsz=imgsz, device=device, verbose=False)
        r0 = results[0]
        last_boxes = []
        
        
        
        if r0.boxes is not None and len(r0.boxes) > 0 :
            cls = r0.boxes.cls.detach().cpu().numpy().astype(int)
            conf = r0.boxes.conf.detach().cpu().numpy()
            xyxy = r0.boxes.xyxy.detach().cpu().numpy()
            
            frame_conf = 0.0
            person_confs = []
            
            for c, cf, box in zip(cls, conf, xyxy):
                if c == 0:
                    person_confs.append(float(cf))
                if c == 0 and cf>= conf_thres:
                    last_boxes.append((cf, box))
            
            frame_conf = max(person_confs) if person_confs else 0.0
            
            conf_ema = (1-alpha) * conf_ema + alpha * frame_conf
            
            if present :
                present = conf_ema >= present_off
            else: 
                present = conf_ema >= present_on
            
            for cf, box in last_boxes:
                x1,y1,x2,y2 = box.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, 
                    f"person {cf:.2f}",
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0), 2)
                
        
    cv2.putText(frame, f"conf_ema: {conf_ema:.2f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"present: {present}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("yolo person only (q to quit)", frame)
    if cv2.waitKey(1) & 0xff in (ord("q"), ord("Q")):
        break
    frame_i += 1

cap.release()
cv2.destroyAllWindows()