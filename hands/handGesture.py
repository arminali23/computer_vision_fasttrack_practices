import cv2
from cvzone.HandTrackingModule import HandDetector
import math

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("cam not opened")

detector = HandDetector(maxHands=1, detectionCon=0.7)

alpha = 0.8
smooth_x, smooth_y = None, None

pinch_on = 40
pinch_off = 50
pinch = False
prev_pinch = False

btn_x, btn_y = 300, 200
btn_h, btn_w = 200, 100

dragged = False
offset_x, offset_y = 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hands, frame = detector.findHands(frame)

    hover = False

    if hands:
        lmlist = hands[0]["lmList"]

        ix, iy = lmlist[8][:2]
        tx, ty = lmlist[4][:2]
        ix, iy, tx, ty = int(ix), int(iy), int(tx), int(ty)

        if smooth_x is None:
            smooth_x, smooth_y = ix, iy
        else:
            smooth_x = int(alpha * ix + (1 - alpha) * smooth_x)
            smooth_y = int(alpha * iy + (1 - alpha) * smooth_y)

        d = math.hypot(ix - tx, iy - ty)

        if pinch:
            pinch = d < pinch_off
        else:
            pinch = d < pinch_on

        hover = (btn_x <= smooth_x <= btn_x + btn_w and btn_y <= smooth_y <= btn_y + btn_h)

        if pinch and not prev_pinch and hover:
            dragged = True
            offset_x = smooth_x - btn_x
            offset_y = smooth_y - btn_y

        if dragged:
            btn_x = smooth_x - offset_x
            btn_y = smooth_y - offset_y

        if not pinch:
            dragged = False

        prev_pinch = pinch 

    if dragged:
        color = (0, 0, 255)
    elif hover:
        color = (0, 255, 255)
    else:
        color = (255, 255, 255)

    cv2.rectangle(frame, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), color, -1)
    cv2.putText(frame, "DRAG ME", (btn_x + 20, btn_y + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)

    cv2.imshow("pinch drag button (q)", frame)
    if cv2.waitKey(1) & 0xff in (ord("q"), ord("Q")):
        break

cap.release()
cv2.destroyAllWindows()