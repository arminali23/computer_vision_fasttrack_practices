import cv2
import torch
from ultralytics import YOLO

imgsz = 320
conf_thres = 0.35
infer_every_n = 3

frame_i = 0
last_boxes = []  # list of (conf, box_xyxy)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = YOLO("yolov8n.pt")

# tracking state
next_id = 0
tracks = {}  # id -> (cx, cy)
max_distance = 80  # px, match threshold


def box_center_xyxy(box):
    x1, y1, x2, y2 = box.astype(int)
    return (x1 + x2) // 2, (y1 + y2) // 2


def dist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("cam not opened")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # run detection only every N frames
    if frame_i % infer_every_n == 0:
        results = model.predict(source=frame, imgsz=imgsz, device=device, verbose=False)
        r0 = results[0]

        last_boxes = []

        if r0.boxes is not None and len(r0.boxes) > 0:
            cls = r0.boxes.cls.detach().cpu().numpy().astype(int)
            conf = r0.boxes.conf.detach().cpu().numpy()
            xyxy = r0.boxes.xyxy.detach().cpu().numpy()

            for c, cf, box in zip(cls, conf, xyxy):
                if c == 0 and float(cf) >= conf_thres:
                    last_boxes.append((float(cf), box))

        # update tracking using the latest detections
        new_tracks = {}
        used_old_ids = set()

        for cf, box in last_boxes:
            cxy = box_center_xyxy(box)

            matched_id = None
            best_d = 1e9

            for tid, prev_cxy in tracks.items():
                if tid in used_old_ids:
                    continue
                d = dist(cxy, prev_cxy)
                if d < best_d and d < max_distance:
                    best_d = d
                    matched_id = tid

            if matched_id is None:
                matched_id = next_id
                next_id += 1

            new_tracks[matched_id] = cxy
            used_old_ids.add(matched_id)

        tracks = new_tracks

    # draw detections (cached) + ids
    track_items = list(tracks.items())  # [(id, (cx,cy)), ...]

    for cf, box in last_boxes:
        x1, y1, x2, y2 = box.astype(int)
        cx, cy = box_center_xyxy(box)

        # find closest track id for labeling
        best_id = None
        best_d = 1e9
        for tid, txy in track_items:
            d = dist((cx, cy), txy)
            if d < best_d:
                best_d = d
                best_id = tid

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID {best_id}  {cf:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        # optional: draw center point
        cv2.circle(frame, (cx, cy), 3, (255, 255, 0), -1)

    # show tracking centers too (even if no box drawn)
    for tid, (cx, cy) in tracks.items():
        cv2.putText(
            frame,
            f"{tid}",
            (cx + 6, cy + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

    cv2.imshow("yolo tracker (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
        break

    frame_i += 1

cap.release()
cv2.destroyAllWindows()