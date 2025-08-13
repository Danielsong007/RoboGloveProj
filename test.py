import sys
import time
from xyz_demo.xyz_utils import xyz_utils
from sensors.dayang import DaYangSensor
import numpy as np
import matplotlib.pyplot as plt
from collections import deque



Samples_Rope_L = deque(maxlen=10)  # 自动丢弃旧数据
print(Samples_Rope_L)
Samples_Rope_L.append(0)
print(Samples_Rope_L)
Ave_Rope_L=np.mean(Samples_Rope_L)
print(Ave_Rope_L)
