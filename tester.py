import numpy as np
from BaseStation import BaseStation 

LOG_NORMAL_MEAN = 0
LOG_NORMAL_STD_MS = 8
for i in range(2):
    print(f" Random {i} is {np.random.normal(LOG_NORMAL_MEAN, LOG_NORMAL_STD_MS)}")

print(f"Cos value {np.cos(np.radians(90))}")
print(f"Sin value {np.sin(np.radians(90))}")

bs = BaseStation(1, 1, 10, [2,2], 50)
print(f"Radius Base Station : {bs.get_radius()}")
