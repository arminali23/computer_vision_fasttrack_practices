import cv2
from numba import uint
import numpy as np
import torch
import random
from ultralytics import YOLO 

imgsz = 320
conf_thres = 0.35
infer_every_n = 3

frame_i = 0
last_boxes = []

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = YOLO("yolov8n.pt")

w, h = 900, 650
canvas = np.zeros((h,w,3), dtype=np.uint8)

particles = []
max_particles = 1500

def spawn(x,y,n,vmin,vmax,life_min,life_max):
    for _ in range(n):
        vx = random.uniform(vmin, vmax)
        vy = random.uniform(vmin, vmax)
        life = random.randint(life_min, life_max)
        particles.append([float(x), float(y), vx, vy, life])

gravity = 0.05
friction = 0.985
fade = 0.90

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("cam not opened")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (w,h))
    
    if frame_i % infer_every_n == 0:
        results = model.predict(source=frame, imgsz=imgsz, device=device, verbose=False)
        r0 = results[0]
        last_boxes = []

        if r0.boxes is not None and len(r0.boxes) > 0:
            cls = r0.boxes.cls.detach().cpu().numpy().astype(int)
            conf = r0.boxes.conf.detach().cpu().numpy()
            xyxy = r0.boxes.xyxy.detach().cpu().numpy()

            for c, cf, box in zip(cls, conf, xyxy):
                if c == 0 and cf >= conf_thres:
                    last_boxes.append((float(cf), box))

        for cf, box in last_boxes:
            x1, y1, x2, y2 = box.astype(int)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            spawn(cx, cy, n=40, vmin=-4.0, vmax=4.0, life_min=18, life_max=45)

    canvas = (canvas * fade).astype(np.uint8)
    
    if len(particles) > max_particles: 
        particles = particles[-max_particles]
    
    new_particles = []
    for x, y, vx, vy, life in particles:
        vy += gravity
        vx *= friction
        vy *= friction

        x += vx
        y += vy

        life -= 1
        if life > 0 and 0 <= x < w and 0 <= y < h:
            intensity = int(255 * (life / 45))
            cv2.circle(canvas, (int(x), int(y)), 2, (intensity, 0, 255), -1)
            new_particles.append([x, y, vx, vy, life])

    particles = new_particles

    for cf, box in last_boxes:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"person {cf:.2f}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out = cv2.addWeighted(frame, 1.0, canvas, 1.0, 0.0)

    cv2.imshow("yolo aura (q)", out)
    if cv2.waitKey(1) & 0xff in (ord("q"), ord("Q")):
        break

    frame_i += 1

cap.release()
cv2.destroyAllWindows()