import cv2
import numpy as np
import random

w, h = 900, 650
canvas = np.zeros((h, w, 3), dtype=np.uint8)

particles = []
max_particles = 1200

def spawn(x, y, n, vmin, vmax, life_min, life_max):
    for _ in range(n):
        vx = random.uniform(vmin, vmax)
        vy = random.uniform(vmin, vmax)
        life = random.randint(life_min, life_max)
        particles.append([float(x), float(y), vx, vy, life])

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        spawn(x, y, n=12, vmin=-3.0, vmax=3.0, life_min=25, life_max=55)

    if event == cv2.EVENT_LBUTTONDOWN:
        spawn(x, y, n=140, vmin=-9.0, vmax=9.0, life_min=35, life_max=80)

cv2.namedWindow("particles bounce")
cv2.setMouseCallback("particles bounce", on_mouse)

gravity = 0.08
friction = 0.98
fade = 0.90
bounce_loss = 0.8

while True:
    canvas = (canvas * fade).astype(np.uint8)

    if len(particles) > max_particles:
        particles = particles[-max_particles:]

    new_particles = []

    for x, y, vx, vy, life in particles:
        vy += gravity
        vx *= friction
        vy *= friction

        x += vx
        y += vy

        if x <= 0 or x >= w:
            vx = -vx * bounce_loss
        if y <= 0 or y >= h:
            vy = -vy * bounce_loss

        life -= 1

        if life > 0:
            if x < 0:
                x = 0.0
            if x > w - 1:
                x = float(w - 1)
            if y < 0:
                y = 0.0
            if y > h - 1:
                y = float(h - 1)

            speed = (vx * vx + vy * vy) ** 0.5
            speed_norm = min(speed / 6.0, 1.0)

            r = int(255 * speed_norm)
            b = int(255 * (1 - speed_norm))

            radius = 2
            cv2.circle(canvas, (int(x), int(y)), radius, (b, 0, r), -1)
            new_particles.append([x, y, vx, vy, life])

    particles = new_particles

    cv2.imshow("particles bounce", canvas)
    if cv2.waitKey(16) & 0xff in (ord("q"), ord("Q")):
        break

cv2.destroyAllWindows()