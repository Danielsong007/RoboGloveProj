import sys
import time
from xyz_demo.xyz_utils import xyz_utils
from sensors.dayang import DaYangSensor

def main():
    try:
        Srope = DaYangSensor('/dev/ttyUSB0')
        Shand = DaYangSensor('/dev/ttyUSB1')
        myXYZ = xyz_utils()
        myXYZ.OpenEnableZero_ALL()

        while True:
            time.sleep(0.5)
            Frope=Srope.read_angles()
            Fhand=Shand.read_angles()
            print(f"Frope: {Frope}, Fhand: {Fhand}")
            Goal_Frope=Fhand
            myXYZ.AxisMode_Jog(3,30,5*(Frope-Goal_Frope))

    except KeyboardInterrupt:
        print("Ctrl-C is pressed!")
    finally:
        myXYZ.SafeQuit()
        Srope.close()
        Shand.close()
        sys.exit(0)

if __name__ == "__main__":
    main()

