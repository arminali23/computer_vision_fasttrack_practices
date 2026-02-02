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
    maxCorners=200,        
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

mask = np.zeros_like(old_frame)

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
        mask = np.zeros_like(frame)
        continue

    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is None:
        old_gray = frame_gray.copy()
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        mask = np.zeros_like(frame)
        continue

    good_new = p1[st == 1]
    good_old = p0[st == 1]

    for (new, old) in zip(good_new, good_old):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 3, (0, 255, 0), -1)

    output = cv2.add(frame, mask)

    cv2.putText(output, f"tracked points: {len(good_new)}  (R: reset)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("LK Optical Flow", output)

    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()