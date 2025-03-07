import sys
import time
from xyz_demo.xyz_utils import xyz_utils
from RUIFEN.SerialMode import RuiFenAngleSensor


def main():
    try:
        ruifen = RuiFenAngleSensor()
        myXYZ = xyz_utils()
        myXYZ.OpenEnableZero_ALL()
        x_angle_init, y_angle_init = ruifen.read_angles(measurement_range=10)
        while True:
            time.sleep(0.05)
            x_angle, y_angle = ruifen.read_angles(measurement_range=10)
            velocity=150*(x_angle)
            if abs(x_angle) < 10:
                myXYZ.AxisMode_Jog(1,80,velocity)
            print(f"Velocit: {velocity}")
            print()
            # print(f"X轴角度: {x_angle}°, Y轴角度: {y_angle}°")

    except KeyboardInterrupt:
        print("Ctrl-C is pressed!")
    finally:
        myXYZ.SafeQuit()
        ruifen.close()
        sys.exit(0)

if __name__ == "__main__":
    main()

