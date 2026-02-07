import cv2
import numpy as np

w, h = 600, 800
canvas = np.zeros((h,w,3),dtype=np.uint8)

trail = []
max_trail = 50

def mouse_move(event, x,y,flags,param):
    if event == cv2.EVENT_MOUSEMOVE:
        trail.append((x,y))
        if len(trail)> max_trail:
            trail.pop(0)
        
cv2.namedWindow("mouse trail")
cv2.setMouseCallback("mouse trail", mouse_move)

while True:
    canvas[:] = (0,0,0)   
    for i in range (1,len(trail)):
        alpha = i / len(trail)
        thickness = int(1 + alpha * 6)
        color = (0, int(255 * alpha), 255)
        cv2.line(canvas, trail[i - 1], trail[i], color, thickness)

    cv2.imshow("mouse trail", canvas)

    if cv2.waitKey(16) & 0xff in (ord("q"), ord("Q")):
        break
cv2.destroyAllWindows()