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
        tc=0
        t_coll=[]
        T_sent_coll=[]
        T_act_coll=[]
        V_last=0
        
        while True:
            dt=0.001
            tc=tc+dt
            time.sleep(dt)
            Rope_S=Srope.read_angles()
            Hand_S=Shand.read_angles()

            if Hand_S>50:
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
                Rope_B=30 # Null
                err=Rope_B-Rope_S
                if err<0:
                    Tgoal=-120
                else:
                    Tgoal=80

            # CurPos=myXYZ.Get_Pos(3)/1000000
            # if CurPos>8:
            #     Tgoal=-20
            # if CurPos<0 and Tgoal<0:
            #     Tgoal=-5

            T_sent,T_act,V_act=myXYZ.Set_Torque(3,Tgoal)
            print('Hand_S:',Hand_S,'\t','Rope_S:',Rope_S,'\t','Goal_T:',Tgoal,'\t','Sent_T:',T_sent,'\t','Actual_T:',T_act)
            # A_cur=(V_act-V_last)/1000
            # print(A_cur)

            V_last=V_act
            # t_coll.append(tc)
            # T_sent_coll.append(T_sent)
            # T_act_coll.append(T_act)

    except KeyboardInterrupt:
        print("Ctrl-C is pressed!")
        
    finally:
        myXYZ.SafeQuit()
        Srope.close()
        Shand.close()

        # # 创建散点图
        # plt.figure(figsize=(8, 6))  # 设置图形大小
        # plt.scatter(t_coll, T_sent_coll, c='blue', marker='o', alpha=0.4, s=50)
        # plt.scatter(t_coll, T_act_coll, c='red', marker='.', alpha=0.8, s=50)
        # plt.show()

        sys.exit(0)

if __name__ == "__main__":
    main()


