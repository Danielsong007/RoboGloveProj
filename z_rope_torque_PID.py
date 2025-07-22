import sys
import time
from xyz_demo.xyz_utils import xyz_utils
from sensors.dayang import DaYangSensor
import numpy as np

def main():
    try:
        Srope = DaYangSensor('/dev/ttyUSB0',1)
        Shand = DaYangSensor('/dev/ttyUSB1',1)
        myXYZ = xyz_utils()
        myXYZ.AxisMode_Torque(3)
        myXYZ.OpenEnableZero_ALL()
        Tgoal_L=0
        
        while True:
            dt=0.005
            time.sleep(dt)
            Frope=Srope.read_angles()
            Fhand=Shand.read_angles()      
            CurPos=myXYZ.Get_Pos(3)/1000000
            print('Fhand:',Fhand,'\t','Frope:',Frope,'\t','Tgoal_L:',Tgoal_L)

            if Fhand>800:
                Fgoal=1800
                err=Fgoal-Frope
                diff=30
                if err>0:
                    Tgoal=Tgoal_L+diff
                else:
                    Tgoal=Tgoal_L-diff

                Tgoal_min=-100
                Tgoal_max=0.8*Fgoal
                if Tgoal<Tgoal_min:
                    Tgoal=Tgoal_min
                if Tgoal>Tgoal_max:
                    Tgoal=Tgoal_max
            else:
                Tgoal=0

            if CurPos>8:
                Tgoal=-20
            if CurPos<0 and Tgoal<0:
                Tgoal=-5
            myXYZ.Set_Torque(3,Tgoal)
            Tgoal_L=Tgoal

    except KeyboardInterrupt:
        print("Ctrl-C is pressed!")
        
    finally:
        myXYZ.SafeQuit()
        Srope.close()
        Shand.close()
        sys.exit(0)

if __name__ == "__main__":
    main()


