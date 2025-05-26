import sys
import time
from xyz_demo.xyz_utils import xyz_utils
from sensors.dayang import DaYangSensor
import numpy as np

def main():
    try:
        Srope = DaYangSensor('/dev/ttyUSB0')
        Shand = DaYangSensor('/dev/ttyUSB1')
        myXYZ = xyz_utils()
        myXYZ.AxisMode_Torque(3)
        myXYZ.OpenEnableZero_ALL()
        
        while True:
            dt=0.1
            time.sleep(dt)
            # Frope=Srope.read_angles()
            Fhand=Shand.read_angles()
            Fgoal=(0.15*Fhand-75)
            if Fgoal>200:
                Fgoal=200
            print(Fgoal)
            myXYZ.Set_Torque(3,Fgoal)

    except KeyboardInterrupt:
        print("Ctrl-C is pressed!")
        
    finally:
        myXYZ.SafeQuit()
        Srope.close()
        Shand.close()
        sys.exit(0)

if __name__ == "__main__":
    main()

