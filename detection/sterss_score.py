import random
import time

alpha = 0.2
high_tresh = 0.7
low_thresh = 0.4

stress_ema = 0.0
stressed = False 

def read_signal():
    base = random.uniform(0.2,0.8)
    noise = random.uniform(-0.2,0.2)
    return max(0.0,min(1.0,base+noise))

for t in range(1,61):
    signal = read_signal()

    stress_ema = (1 - alpha) * stress_ema + alpha * signal

    if stressed:
        stressed = stress_ema >= low_thresh
    else:
        stressed = stress_ema >= high_tresh

    print(
        f"t={t:02d}  "
        f"signal={signal:.2f}  "
        f"stress_ema={stress_ema:.2f}  "
        f"stressed={stressed}"
    )

    time.sleep(0.2)