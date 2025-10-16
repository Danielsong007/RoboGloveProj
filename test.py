import sys
import time
import threading
from xyz_demo.xyz_utils import xyz_utils
from sensors.dayang import DaYangSensor
from sensors.ligan import LiganSensor

# 全局变量存储传感器数据
rope_value = 0
touch_value = 0

def read_rope_sensor(sensor):
    global rope_value
    while True:
        rope_value = sensor.read_angles()
        time.sleep(0.005)

def read_touch_sensor(sensor):
    global touch_value
    while True:
        touch_value = sensor.read_data()[1]
        time.sleep(0.005)

def main():
    try:
        # 初始化传感器
        Srope = DaYangSensor('/dev/ttyUSB0', 0)
        Sligan = LiganSensor(port='/dev/ttyUSB3')
        threading.Thread(target=read_rope_sensor, args=(Srope,), daemon=True).start()
        threading.Thread(target=read_touch_sensor, args=(Sligan,), daemon=True).start()
        
        while True:
            time.sleep(0.01)  # 主循环频率约100Hz
            print(f'Rope_S: {rope_value}, Touch_S: {touch_value}')

    except KeyboardInterrupt:
        print("Ctrl-C is pressed!")
        
    finally:
        Srope.close()
        Sligan.close()
        sys.exit(0)

if __name__ == "__main__":
    main()