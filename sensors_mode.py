import sys
import time
from xyz_demo.xyz_utils import xyz_utils
from sensors.ruifen import RuiFenSensor
from sensors.dayang import DaYangSensor

def main():
    try:
        ruifen = RuiFenSensor()
        dayang = DaYangSensor()
        myXYZ = xyz_utils()
        myXYZ.OpenEnableZero_ALL()
        force_init=dayang.read_angles()
        movable=0
        while True:
            time.sleep(0.05)
            x_angle, y_angle = ruifen.read_angles()
            x_angle=x_angle-(-1)
            y_angle=y_angle-(4.3)
            print(f'x: {x_angle}, y: {y_angle}')

            force=dayang.read_angles()-force_init
            print(f"force: {force}")
            if force > 20: # Up
                myXYZ.AxisMode_Jog(3,30,-10*force)
                movable=0
            elif force < 3: # Down
                myXYZ.AxisMode_Jog(3,30,200)
                movable=0
            else: # Suspend
                myXYZ.AxisMode_Jog(3,30,0)
                movable=1
            if movable==1:
                if abs(x_angle) < 20:
                    myXYZ.AxisMode_Jog(2,20,30*x_angle)
                if abs(y_angle) < 20:
                    myXYZ.AxisMode_Jog(1,30,30*y_angle)

    except KeyboardInterrupt:
        print("Ctrl-C is pressed!")
    finally:
        myXYZ.SafeQuit()
        ruifen.close()
        dayang.close()
        sys.exit(0)

if __name__ == "__main__":
    main()
