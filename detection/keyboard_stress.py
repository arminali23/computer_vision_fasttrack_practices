import time
import threading
import sys

alpha = 0.2
high_thres = 0.7
low_thres = 0.4

duration_sec = 30
window_sec = 1.0

max_press_per_sec = 8

press_count = 0
lock = threading.Lock()

stress_ema = 0.0
stressed = False
stop_flag = False

def listen_space():
    global press_count, stop_flag
    while not stop_flag:
        try:
            s = sys.stdin.readline()
        except Exception:
            break
        if not s:
            continue
        if s.strip().lower() == "q":
            stop_flag = True
            break
        if s.strip() == "":
            with lock:
                press_count += 1
        else:
            with lock:
                press_count += 1

listener = threading.Thread(target=listen_space, daemon=True)
listener.start()

print("space'e bas (terminalde). cikmak icin q + enter.")
print(f"{duration_sec} saniye calisacak.\n")

start_t = time.time()
last_t = start_t

while True:
    now = time.time()
    if stop_flag or now - start_t >= duration_sec:
        break

    if now - last_t >= window_sec:
        with lock:
            c = press_count
            press_count = 0

        signal = min(1.0, c / max_press_per_sec)

        stress_ema = (1 - alpha) * stress_ema + alpha * signal

        if stressed:
            stressed = stress_ema >= low_thres
        else:
            stressed = stress_ema >= high_thres

        print(
            f"presses/sec={c:2d}  "
            f"signal={signal:.2f}  "
            f"stress_ema={stress_ema:.2f}  "
            f"stressed={stressed}"
        )

        last_t = now

stop_flag = True
print("\nbittti.")  


