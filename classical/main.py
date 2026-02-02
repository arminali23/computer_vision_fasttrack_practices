import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError('camera could not be opened')


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    vis = frame.copy()
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 600: 
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y),(x+w,y+h), (0,255,0),2)
    
    
    cv2.imshow("press q to quit",frame)
    cv2.imshow('boxes',vis)
    cv2.imshow('edges',edges)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()