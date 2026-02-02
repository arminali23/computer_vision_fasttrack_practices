import torch
import cv2
from ultralytics import YOLO

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("camera could not be opened")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model.predict(frame, device=device, imgsz = 320, verbose = False)
    r0 = results[0]
    person_present = False
    
    if r0.boxes is not None and len(r0.boxes)>0:
        cls = r0.boxes.cls.detach().cpu().numpy().astype(int)
        person_present = (cls == 0).any()

    cv2.putText(frame, f"person_present: {person_present}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("yolo person check", frame)
    if cv2.waitKey(1) & 0xff in (ord("q"), ord("Q")):
        break

cap.release()
cv2.destroyAllWindows()
