import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("camera could not be opened")

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

template_gray = None
template_kp = None
template_des = None
template_h, template_w = None, None

MIN_GOOD = 25  

print("Controls:")
print("  SPACE: capture template frame")
print("  Q: quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):
        template_gray = gray.copy()
        template_h, template_w = template_gray.shape[:2]
        template_kp, template_des = orb.detectAndCompute(template_gray, None)
        n_kp = 0 if template_kp is None else len(template_kp)
        print(f"Template captured. keypoints={n_kp}")

    if template_gray is None or template_des is None or des is None:
        vis = cv2.drawKeypoints(
            frame, kp if kp is not None else [], None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        cv2.putText(vis, "Press SPACE to capture template", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("ORB Homography Live", vis)

        if key == ord("q"):
            break
        continue

    matches_knn = bf.knnMatch(template_des, des, k=2)

    good = []
    ratio = 0.75
    for m_n in matches_knn:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            good.append(m)

    out = frame.copy()
    cv2.putText(out, f"good matches: {len(good)} (min {MIN_GOOD})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if len(good) >= MIN_GOOD:
        src_pts = np.float32([template_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is not None:
            corners = np.float32([
                [0, 0],
                [template_w - 1, 0],
                [template_w - 1, template_h - 1],
                [0, template_h - 1]
            ]).reshape(-1, 1, 2)

            projected = cv2.perspectiveTransform(corners, H)

            cv2.polylines(out, [np.int32(projected)], True, (0, 255, 0), 3)

            inliers = int(mask.sum()) if mask is not None else 0
            cv2.putText(out, f"inliers: {inliers}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("ORB Homography Live", out)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()