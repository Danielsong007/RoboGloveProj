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
            Fgoal=Fhand-650
            if Fgoal>1000:
                Fgoal=1000
            CurPos=myXYZ.Get_Pos(3)/1000000
            if (CurPos>13 and Fgoal>0) or (CurPos<0.02 and Fgoal<0):
                Vgoal=0
            else:
                Vgoal=0.2*Fgoal
            print(Vgoal)
            myXYZ.Set_Torque(3,Vgoal)

    except KeyboardInterrupt:
        print("Ctrl-C is pressed!")
        
    finally:
        myXYZ.SafeQuit()
        Srope.close()
        Shand.close()
        sys.exit(0)

if __name__ == "__main__":
    main()

