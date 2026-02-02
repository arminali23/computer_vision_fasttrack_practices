import time
import csv
import cv2
import numpy as np
import torch
from ultralytics import YOLO

model_name = "yolov8n.pt"
imgsz = 320
infer_every_n = 2
alpha_conf = 0.2
alpha_motion = 0.2

present_thres = 0.35
motion_thres = 1.5

duration_sec = 30
log_every_sec = 1.0
csv_path = "realtime/attention_log.csv"

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = YOLO(model_name)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("camera could not be opened")

ret, old_frame = cap.read()
if not ret:
    raise RuntimeError("could not read from camera")

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

feature_params = dict(maxCorners=250, qualityLevel=0.01, minDistance=7, blockSize=7)
lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

conf_ema = 0.0
motion_ema = 0.0

frame_i = 0
last_person_boxes = []
last_infer_ms = 0.0

rows = []
start_t = time.time()
last_log_t = start_t

while True:
    now_t = time.time()
    if now_t - start_t >= duration_sec:
        break

    t_loop0 = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xff
    if key in (ord("q"), ord("Q")):
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)

    if p1 is not None and st is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        if len(good_new) > 0:
            diff = good_new - good_old
            dists = np.linalg.norm(diff, axis=1)
            motion_score = float(np.mean(dists))
        else:
            motion_score = 0.0

        motion_ema = (1 - alpha_motion) * motion_ema + alpha_motion * motion_score

        old_gray = gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    else:
        old_gray = gray.copy()
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    if frame_i % infer_every_n == 0:
        t0 = time.time()
        results = model.predict(source=frame, imgsz=imgsz, device=device, verbose=False)
        last_infer_ms = (time.time() - t0) * 1000.0

        r0 = results[0]
        last_person_boxes = []
        frame_conf = 0.0

        if r0.boxes is not None and len(r0.boxes) > 0:
            xyxy = r0.boxes.xyxy.detach().cpu().numpy()
            cls = r0.boxes.cls.detach().cpu().numpy().astype(int)
            conf = r0.boxes.conf.detach().cpu().numpy()

            for (x1, y1, x2, y2), c, cf in zip(xyxy, cls, conf):
                if c == 0:
                    last_person_boxes.append((int(x1), int(y1), int(x2), int(y2), float(cf)))

            if last_person_boxes:
                frame_conf = max(b[4] for b in last_person_boxes)

        conf_ema = (1 - alpha_conf) * conf_ema + alpha_conf * frame_conf

    person_present = conf_ema >= present_thres
    moving = motion_ema >= motion_thres

    if not person_present:
        state = "no_person"
    elif person_present and not moving:
        state = "idle"
    else:
        state = "active"

    if now_t - last_log_t >= log_every_sec:
        rows.append({
            "t_sec": round(now_t - start_t, 2),
            "state": state,
            "conf_ema": round(conf_ema, 4),
            "motion_ema": round(motion_ema, 4),
        })
        last_log_t = now_t

    for x1, y1, x2, y2, cf in last_person_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"person {cf:.2f}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    loop_ms = (time.time() - t_loop0) * 1000.0

    cv2.putText(frame, f"state: {state}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"conf_ema: {conf_ema:.2f} present={person_present}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(frame, f"motion_ema: {motion_ema:.2f} moving={moving}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(frame, f"infer_ms: {last_infer_ms:.1f} loop_ms: {loop_ms:.1f}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("attention mini project", frame)

    frame_i += 1

cap.release()
cv2.destroyAllWindows()

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["t_sec", "state", "conf_ema", "motion_ema"])
    writer.writeheader()
    writer.writerows(rows)

total = len(rows)
active_n = sum(1 for r in rows if r["state"] == "active")
idle_n = sum(1 for r in rows if r["state"] == "idle")
no_n = sum(1 for r in rows if r["state"] == "no_person")

active_ratio = (active_n / total) if total else 0.0
idle_ratio = (idle_n / total) if total else 0.0
no_ratio = (no_n / total) if total else 0.0

print("\n=== summary ===")
print("rows:", total)
print("active_ratio:", round(active_ratio, 3))
print("idle_ratio:", round(idle_ratio, 3))
print("no_person_ratio:", round(no_ratio, 3))
print("saved:", csv_path)