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
        Cgoal_last=0
        
        while True:
            dt=0.005
            time.sleep(dt)
            Frope=Srope.read_angles()
            Fhand=Shand.read_angles()      
            CurPos=myXYZ.Get_Pos(3)/1000000
            print('Fhand:',Fhand,'\t','Frope:',Frope,'\t','Cgoal_last:',Cgoal_last)

            if Fhand>800:
                Fgoal=1600
                err=Fgoal-Frope
                Cgoal_min=-20
                Cgoal_max=1*Fgoal
                if err>0:
                    err=500
                else:
                    err=-500
                Cgoal=Cgoal_last+0.01*err
                if Cgoal<Cgoal_min:
                    Cgoal=Cgoal_min
                if Cgoal>Cgoal_max:
                    Cgoal=Cgoal_max
                print('going going going')
            else:
                Cgoal=0
                print('Down Down Down')

            if CurPos>8:
                Cgoal=-20
            if CurPos<0 and Cgoal<0:
                Cgoal=-5
            myXYZ.Set_Torque(3,Cgoal)
            Cgoal_last=Cgoal

    except KeyboardInterrupt:
        print("Ctrl-C is pressed!")
        
    finally:
        myXYZ.SafeQuit()
        Srope.close()
        Shand.close()
        sys.exit(0)

if __name__ == "__main__":
    main()


