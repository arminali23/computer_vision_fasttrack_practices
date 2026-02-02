import cv2 
from ultralytics import YOLO
import torch
import time

model_name = "yolov8n.pt"
imgsz = 320
infer_every_n = 3

alpha = 0.2
present_on = 0.45
present_off = 0.25

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = YOLO(model_name)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("cam could not opened")

frame_i = 0
conf_ema = 0
present = False
fps_ema = None
alpha_fps = 0.2
prev_t = False

last_infer_ms = 0.0

while True:
    t_loop0 = 0.0
    ret,frame = cap.read()
    if not ret:
        break
    
    if frame_i % infer_every_n == 0:
        t0 = time.time()
        results = model.predict(source=frame, imgsz=imgsz, device=device, verbose=False)
        last_infer_ms = (time.time()-t0) * 10000.0
        
        r0 = results[0]
        frame_conf = 0.0
        
        if r0.boxes is not None and len(r0.boxes) > 0:
            cls = r0.boxes.cls.detach().cpu().numpy().astype(int)
            conf = r0.boxes.cls.detach().cpu().numpy()
            person_confs = [float(cf) for c, cf, in zip(cls, conf) if c==0]
            frame_conf = max(person_confs) if person_confs else 0.0
            
        conf_ema = (1-alpha) * conf_ema + alpha * frame_conf
        
        if present:
            present = conf_ema >= present_off
        else :
            present = conf_ema >= present_on
    now = time.time()
    fps = 1.0/max(now-prev_t, 1e-6)
    prev_t = now
    fps_ema = fps if fps_ema is None else (1 - alpha_fps) * fps_ema + alpha_fps * fps
    
    loop_ms = (time.time() - t_loop0) * 1000.0
    

    cv2.putText(frame, f"present: {present}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"conf_ema: {conf_ema:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"fps_ema: {fps_ema:.1f}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"infer_ms: {last_infer_ms:.1f} loop_ms: {loop_ms:.1f} every_n: {infer_every_n}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("person presence", frame)

    if cv2.waitKey(1) & 0xff in (ord("q"), ord("Q")):
        break

    frame_i += 1

cap.release()
cv2.destroyAllWindows()