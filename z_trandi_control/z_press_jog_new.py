import sys
import time
import pygame
from xyz_demo.xyz_utils import xyz_utils
from sensors.ruifen import RuiFenSensor
from sensors.dayang import DaYangSensor

def main():
    try:
        dayang = DaYangSensor('/dev/ttyUSB0',0)
        myXYZ = xyz_utils()
        myXYZ.OpenEnableZero_ALL()
        myXYZ.AxisMode_Jog(1,3,0)
        myXYZ.AxisMode_Jog(2,3,0)

        while True:
            time.sleep(0.05)
            force=dayang.read_angles()
            print(f"force: {force}")
            if force > 300: # Up
                myXYZ.AxisMode_Jog(3,30,-2*force)
            elif force < 25: # Suspend
                myXYZ.AxisMode_Jog(3,30,0)
            else: # Down
                myXYZ.AxisMode_Jog(3,30,1000)

    except KeyboardInterrupt:
        print("Ctrl-C is pressed!")
    finally:
        myXYZ.SafeQuit()
        dayang.close()
        sys.exit(0)

if __name__ == "__main__":
    main()


            