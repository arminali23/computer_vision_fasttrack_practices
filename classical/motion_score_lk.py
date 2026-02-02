import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("camera could not be opened")

ret, old_frame = cap.read()
if not ret:
    raise RuntimeError("could not read from camera")
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

feature_params = dict(
    maxCorners=250,
    qualityLevel=0.01,
    minDistance=7,
    blockSize=7
)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

ema = 0.0
alpha = 0.15

print("Controls:")
print("  R: re-detect points")
print("  Q: quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("r") or p0 is None:
        old_gray = frame_gray.copy()
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        ema = 0.0
        continue
    
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is None or st is None:
        old_gray = frame_gray.copy()
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        ema = 0.0
        continue
    
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    
    if len(good_new) > 0:
        diff = good_new - good_old
        dists = np.linalg.norm(diff, axis=1)  
        motion_score = float(np.mean(dists))
    else:
        motion_score = 0.0
        
    ema = (1 - alpha) * ema + alpha * motion_score

    vis = frame.copy()
    for x, y in good_new:
        cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)

    cv2.putText(vis, f"motion score (instant): {motion_score:.3f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(vis, f"motion score (smoothed): {ema:.3f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(vis, f"tracked points: {len(good_new)} (R reset)", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("LK Motion Score", vis)
    
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()