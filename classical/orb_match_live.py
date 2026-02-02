import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError('cam not opened')

orb = cv2.ORB_create()

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

template_gray = None
template_kp = None
template_des = None

print("Controls:")
print("  SPACE: capture template frame")
print("  Q: quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    kp, des = orb.detectAndCompute(gray, None)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord(" "):
        template_gray = gray.copy()
        template_kp, template_des = orb.detectAndCompute(template_gray, None)

        n_kp = 0 if template_kp is None else len(template_kp)
        print(f"Template captured. keypoints={n_kp}, des={'OK' if template_des is not None else 'None'}")

    if template_gray is None or template_des is None or des is None:
        vis = cv2.drawKeypoints(
            frame, kp if kp is not None else [], None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        cv2.putText(
            vis,
            "Press SPACE to capture template",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        cv2.imshow("ORB Live Matching", vis)

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

    match_vis = cv2.drawMatches(
        template_gray, template_kp,
        gray, kp,
        good[:60],  
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.putText(
        match_vis,
        f"good matches: {len(good)}  (SPACE recapture)",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("ORB Live Matching", match_vis)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()