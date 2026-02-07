import cv2
import numpy as np
import random

w, h = 600, 800 
canvas = np.zeros((h,w, 3), dtype=np.uint8)

particles = []
max_particles = 1200

mouse_pos = (w // 2, h // 2)
def on_mouse(event, x, y, flags, params):
    global mouse_pos
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_pos = (x, y)
        for _ in range(12):
            vx = random.uniform(-3.0, 3.0)
            vy = random.uniform(-3.0, 3.0)
            life = random.randint(25, 55)
            particles.append([float(x), float(y), vx, vy, life])
            
    if event == cv2.EVENT_LBUTTONDOWN:
        for _ in range(140):
            vx = random.uniform(-9.0,9.0)
            vy = random.uniform(-9.0,9.0)
            life = random.randint(35,80)
            particles.append([float(x),float(y),vx,vy,life])
            
cv2.namedWindow("particles")
cv2.setMouseCallback("particles", on_mouse)

gravity = 0.08
friction = 0.98

while True:
    canvas = (canvas * 0.90).astype(np.uint8)
    
    if len(particles) > max_particles:
        particles = particles[-max_particles:]
        
    new_particles = []
    
    for x,y, vx, vy, life in particles:
        vy += gravity
        vx *= friction
        vy *= friction
        x += vx
        y += vy
        life -= 1
        
        if life > 0 and 0 <= x < w and 0 <= y < h:
            intensity = int(255 * (life / 55))
            radius = 2

            speed = (vx*vx + vy*vy) ** 0.5
            speed_norm = min(speed / 6.0, 1.0)

            r = int(255 * speed_norm)
            b = int(255 * (1 - speed_norm))

            cv2.circle(canvas, (int(x), int(y)), radius, (b, 0, r), -1)
            new_particles.append([x, y, vx, vy, life])

            particles = new_particles

    cv2.imshow("particles", canvas)

    if cv2.waitKey(16) & 0xff in (ord("q"), ord("Q")):
        break

cv2.destroyAllWindows()