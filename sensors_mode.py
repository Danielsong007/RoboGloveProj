import sys
import time
import pygame
from xyz_demo.xyz_utils import xyz_utils
from sensors.ruifen import RuiFenSensor
from sensors.dayang import DaYangSensor

def main():
    try:
        ruifen = RuiFenSensor()
        dayang = DaYangSensor()
        myXYZ = xyz_utils()
        myXYZ.OpenEnableZero_ALL()

        while True:
            time.sleep(0.05)
            x_angle, y_angle = ruifen.read_angles()
            x_angle=x_angle-(0)
            y_angle=y_angle-(0)
            # print(f'x: {x_angle}, y: {y_angle}')

            force=dayang.read_angles()
            print(f"force: {force}")
            if force > 200: # Up
                myXYZ.AxisMode_Jog(1,3,0)
                myXYZ.AxisMode_Jog(2,3,0)
                myXYZ.AxisMode_Jog(3,30,-3*force)
            elif force < 8: # Suspend
                myXYZ.AxisMode_Jog(1,3,0)
                myXYZ.AxisMode_Jog(2,3,0)
                myXYZ.AxisMode_Jog(3,30,0)
            else: # Down
                myXYZ.AxisMode_Jog(3,30,1200)
                # if abs(x_angle) < 20:
                #     myXYZ.AxisMode_Jog(2,2,40*x_angle)
                # if abs(y_angle) < 20:
                #     myXYZ.AxisMode_Jog(1,6,70*y_angle)

    except KeyboardInterrupt:
        print("Ctrl-C is pressed!")
    finally:
        myXYZ.SafeQuit()
        ruifen.close()
        dayang.close()
        sys.exit(0)

if __name__ == "__main__":
    main()


            