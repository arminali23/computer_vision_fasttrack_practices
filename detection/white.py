import cv2
import numpy as np
from utils import get_limits
from PIL import Image

white = [0,255,255]
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError('cam not opened')

while True:
    ret, frame = cap.read()
    if not ret: 
        break
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower, upper = get_limits(color= white)
    
    mask = cv2.inRange(hsv, lower, upper)
    mask_ = Image.fromarray(mask)
    bbox = mask_.getbbox()
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()