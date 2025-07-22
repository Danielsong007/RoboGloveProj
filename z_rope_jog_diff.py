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
        myXYZ.OpenEnableZero_ALL()
        Vgoal_L=0
        
        while True:
            dt=0.005
            time.sleep(dt)
            Frope=Srope.read_angles()
            Fhand=Shand.read_angles()      

            if Fhand>800:
                Fgoal=2500
            else:
                Fgoal=50
            
            err=Fgoal-Frope
            Vgoal_N=25*err
            Vgoal=Vgoal_L+(Vgoal_N-Vgoal_L)*0.03
            # Vgoal=25*err

            # diff=10
            # if err>0:
            #     Vgoal=Vgoal_L+diff
            # else:
            #     Vgoal=Vgoal_L-diff
            
            Vgoal_min=-1000
            Vgoal_max=1000
            if Vgoal<Vgoal_min:
                Vgoal=Vgoal_min
            if Vgoal>Vgoal_max:
                Vgoal=Vgoal_max

            myXYZ.AxisMode_Jog(3,30,Vgoal)
            Vgoal_L=Vgoal
            print('Fhand:',Fhand,'\t','Frope:',Frope,'\t','Vgoal:',Vgoal)


    except KeyboardInterrupt:
        print("Ctrl-C is pressed!")
        
    finally:
        myXYZ.SafeQuit()
        Srope.close()
        Shand.close()
        sys.exit(0)

if __name__ == "__main__":
    main()


