import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("camera could not be opened")

orb = cv2.ORB_create()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    keypoints, descriptors = orb.detectAndCompute(gray, None)

    vis = cv2.drawKeypoints(
        frame, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    cv2.putText(
        vis,
        f"keypoints: {0 if keypoints is None else len(keypoints)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.imshow("ORB Keypoints - press q", vis)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()