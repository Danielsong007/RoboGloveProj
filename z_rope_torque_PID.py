import sys
import time
from xyz_demo.xyz_utils import xyz_utils
from sensors.dayang import DaYangSensor
import numpy as np
import matplotlib.pyplot as plt


def main():
    try:
        Srope = DaYangSensor('/dev/ttyUSB0',1)
        Shand = DaYangSensor('/dev/ttyUSB1',1)
        myXYZ = xyz_utils()
        myXYZ.OpenEnableZero_ALL()
        myXYZ.AxisMode_Torque(3)
        V_last=0
        ThreS_Lift=550
        Mode=0
        
        while True:
            dt=0.001
            time.sleep(dt)
            Rope_S=Srope.read_angles()
            Hand_S=Shand.read_angles()

            if Hand_S>50:
                Mode=1
                Rope_B=2500 # Box
                # Rope_B=400 # Luggage
                err=Rope_B-Rope_S
                Tgoal=100*err

                Tgoal_min=10
                if Tgoal<Tgoal_min:
                    Tgoal=Tgoal_min
                Tgoal_max=350 # Box
                # Tgoal_max=220 # Luggage
                if Tgoal>Tgoal_max:
                    Tgoal=Tgoal_max
            else:
                if Rope_S<ThreS_Lift:
                    Mode=2
                    Rope_B=30 # Null
                    err=Rope_B-Rope_S
                    if err<0:
                        Tgoal=-120
                    else:
                        Tgoal=55
                else:
                    Mode=3
                    Tgoal=int(50-(Rope_S-ThreS_Lift)*0.01)
                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!',Tgoal)
            
            CurPos=myXYZ.Get_Pos(3)
            if CurPos>1983000000:
                Tgoal=-70
            if CurPos<1970000000:
                Tgoal=70
            myXYZ.Set_Torque(3,Tgoal)
            T_Sent,T_Actual,V_act=myXYZ.Read_Paras(3)
            print('Mode:',Mode,'\t','Hand_S:',Hand_S,'\t','Rope_S:',Rope_S,'\t','T_Goal:',Tgoal,'\t','T_Sent:',T_Sent,'\t','T_Actual:',T_Actual)

            # A_cur=(V_act-V_last)/1000
            V_last=V_act

    except KeyboardInterrupt:
        print("Ctrl-C is pressed!")
        
    finally:
        myXYZ.SafeQuit()
        Srope.close()
        Shand.close()
        sys.exit(0)

if __name__ == "__main__":
    main()


