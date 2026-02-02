import torch
import cv2
from ultralytics import YOLO

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = YOLO("yolov8n.pt")
imgsize = 320
infer_every_n = 2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("camera could not be opened")

frame_i = 0
last_person_present = False
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_i % infer_every_n == 0:
        results = model.predict(frame, device=device, imgsz = imgsize, verbose = False)
        r0 = results[0]
        person_present = False
        if r0.boxes is not None and len(r0.boxes) > 0:
            cls = r0.boxes.cls.detach().cpu().numpy().astype(int)
            person_present = (cls == 0).any()
    
        last_person_present = person_present
    
    cv2.putText(frame, f"person_present (cached): {last_person_present}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"infer_every_n: {infer_every_n} | frame_i: {frame_i}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("yolo frameskip minimal", frame)

    if cv2.waitKey(1) & 0xff in (ord("q"), ord("Q")):
        break

    frame_i += 1

cap.release()
cv2.destroyAllWindows()    