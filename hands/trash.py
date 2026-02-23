import cv2
from cvzone.HandTrackingModule import HandDetector
import math


detector = HandDetector(maxHands=1, detectionCon=0.7)
alpha = 0.8

smooth_x, smooth_y = None, None
pinch = False
pinch_on = 35
pinch_off = 50
prev_pinch = False
clicked = False

draggin = False
offset_x, offset_y =0, 0

boxes = [
    [120, 140, 180, 90],
    [380, 220, 220, 110],
    [200, 360, 200, 100],
]
active_i = None

trash_w, trash_h = 220, 120
trash_margin = 20

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("cam not opened")

while True:
    ret, frame = cap.read()
    if not ret: 
        break
    
    frame = cv2.flip(frame, 1)
    in_trash = False
    h, w = frame.shape[:2]
    trash_x = w - trash_w - trash_margin
    trash_y = h - trash_h - trash_margin
    hands, frame = detector.findHands(frame)
    
    hover_i = None
    
    if hands: 
        
        lmlist = hands[0]['lmList']
        ix, iy = lmlist[8][:2]
        tx, ty = lmlist[4][:2]
        ix, iy = int(ix), int(iy)
        tx, ty = int(tx), int(ty)
        
        d = math.hypot(ix- tx, iy - ty)
        
        if smooth_x is None and smooth_y is None: 
            smooth_x = ix
            smooth_y = iy
        else: 
            smooth_x = int(alpha * ix + (1-alpha) * smooth_x)
            smooth_y = int(alpha * iy + (1-alpha)* smooth_y)
        if pinch:
            pinch = d < pinch_off
        else:
            pinch = d < pinch_on 
        
        clicked = pinch and (not prev_pinch)
                
        for i in range (len(boxes)-1,-1, -1):
            bx, by, bw, bh = boxes[i]
            if bx <= smooth_x <= bx + bw and by <= smooth_y <= by + bh:
                hover_i = i
                break

        if clicked and hover_i is not None:
            active_i= hover_i
            draggin = True
            bx, by, bw, bh = boxes[active_i]
            offset_x = smooth_x - bx
            offset_y = smooth_y - by

        if draggin and active_i is not None:
            boxes[active_i][0] = smooth_x - offset_x
            boxes[active_i][1] = smooth_y - offset_y

            h, w = frame.shape[:2]
            boxes[active_i][0] = max(0, min(boxes[active_i][0], w - boxes[active_i][2]))
            boxes[active_i][1] = max(0, min(boxes[active_i][1], h - boxes[active_i][3]))
            
        if active_i is not None:
            bx, by, bw, bh = boxes[active_i]
            cx = bx + bw // 2
            cy = by + bh // 2
            in_trash = (trash_x <= cx <= trash_x + trash_w and trash_y <= cy <= trash_y + trash_h)

        if not pinch:
            if draggin and active_i is not None and in_trash:
                boxes.pop(active_i)
                draggin = False
                active_i = None
            
        trash_color = (0, 0, 255) if in_trash else (40, 40, 40)
        cv2.rectangle(frame, (trash_x, trash_y), (trash_x + trash_w, trash_y + trash_h), trash_color, -1)
        cv2.putText(frame, "TRASH", (trash_x + 45, trash_y + 75),cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)        

        prev_pinch = pinch
        
        cv2.circle(frame, (smooth_x, smooth_y), 12, (0, 255, 0), -1)

        if clicked:
            cv2.putText(frame, "CLICK", (50, 110),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
        if pinch:    
            cv2.putText(frame, "PINCH", (50,50),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),3)
    
    else : 
        draggin = False
        prev_pinch = False
        pinch = False
        hover_i = None
        active_i = None
        
    for i, (bx,by,bw,bh) in enumerate(boxes):
        if i == active_i:
            color = (0,0,255)
        elif i == hover_i:
            color = (0,255,255)
        else: 
            color = (255,255,255)
    
        cv2.rectangle(frame, (bx,by),(bx+bw, by+bh), color, -1)
        cv2.putText(frame, f"box{i+1}", (bx+15,by+60),cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,0,0),3)
                
    cv2.imshow("frame", frame)
    
    if cv2.waitKey(1) & 0xff in (ord('q'),ord('Q')):
        break
    
cap.release()
cv2.destroyAllWindows()