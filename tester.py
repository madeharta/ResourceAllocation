import numpy as np

LOG_NORMAL_MEAN = 0
LOG_NORMAL_STD_MS = 8
for i in range(2):
    print(f" Random {i} is {np.random.normal(LOG_NORMAL_MEAN, LOG_NORMAL_STD_MS)}")

print(f"Cos value {np.cos(np.radians(90))}")
print(f"Sin value {np.sin(np.radians(90))}")
