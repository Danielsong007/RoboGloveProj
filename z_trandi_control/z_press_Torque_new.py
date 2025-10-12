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
        myXYZ.AxisMode_Torque(3)

        while True:
            dt=0.02
            time.sleep(dt)
            Rope_S=Srope.read_angles()
            Hand_S=Shand.read_angles()
            Tgoal=(Hand_S-50)*1
            T_max=250
            if Tgoal>T_max:
                Tgoal=T_max
            myXYZ.Set_Torque(3,Tgoal)
            myXYZ.Set_Torque(3,Tgoal)


            CurPos=myXYZ.Get_Pos(3)
            T_Sent,T_Actual,V_act=myXYZ.Read_Paras(3)
            print('Hand_S:',Hand_S,'\t','Rope_S:',Rope_S,'\t','T_Goal:',Tgoal,'\t','T_Sent:',T_Sent,'\t','T_Actual:',T_Actual)            

            if CurPos>1982000000:
                myXYZ.Set_Torque_Multi(3,-70)
                print('SSSSSSSSSSSSSSSSSSSSSSSSSSSS')
            if CurPos<1970000000:
                myXYZ.Set_Torque_Multi(3,70)
                print('SSSSSSSSSSSSSSSSSSSSSSSSSSSS')

    except KeyboardInterrupt:
        print("Ctrl-C is pressed!")
        
    finally:
        myXYZ.SafeQuit()
        Srope.close()
        Shand.close()
        sys.exit(0)

if __name__ == "__main__":
    main()

