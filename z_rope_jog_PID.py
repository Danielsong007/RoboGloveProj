import sys
import time
from xyz_demo.xyz_utils import xyz_utils
from sensors.dayang import DaYangSensor
import matplotlib.pyplot as plt
import numpy as np

def main():
    try:
        Srope = DaYangSensor('/dev/ttyUSB0')
        Shand = DaYangSensor('/dev/ttyUSB1')
        myXYZ = xyz_utils()
        myXYZ.OpenEnableZero_ALL()

        FlastG=0
        Fgoal=50
        err_prev=Fgoal
        err_coll=[]
        t_coll=[]
        tc=0
        Ierr=0

        while True:
            dt=0.05
            time.sleep(dt)
            tc=tc+dt

            Frope=Srope.read_angles()
            # Fhand=Shand.read_angles()
            # Fgoal=FlastG+0.05*(Fhand-FlastG)
            err=Frope-Fgoal
            Derr=(err-err_prev)/dt
            Ierr=Ierr+err
            Vrope=3*err/abs(err)*(abs(err)**1.5)
            # Vrope=20*err+0*Derr+0*Ierr
            myXYZ.AxisMode_Jog(3,30,Vrope)
            err_prev=err
            
            err_coll.append(err)
            t_coll.append(tc)
            # print(f"Frope: {Frope}, err: {err}, Derr: {Derr}")

    except KeyboardInterrupt:
        print("Ctrl-C is pressed!")
        
    finally:
        myXYZ.SafeQuit()
        Srope.close()
        Shand.close()

        # 创建散点图
        plt.figure(figsize=(8, 6))  # 设置图形大小
        plt.scatter(t_coll, err_coll, c='blue', marker='o', alpha=0.6, s=50)
        plt.show()

        sys.exit(0)

if __name__ == "__main__":
    main()

