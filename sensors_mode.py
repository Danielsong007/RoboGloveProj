import sys
import time
import pygame
from xyz_demo.xyz_utils import xyz_utils
from sensors.ruifen import RuiFenSensor
from sensors.dayang import DaYangSensor


pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0) # Attention
joystick.init()
print("Name of gamepad:", joystick.get_name())


def main():
    try:
        ruifen = RuiFenSensor()
        dayang = DaYangSensor()
        myXYZ = xyz_utils()
        myXYZ.OpenEnableZero_ALL()
        force_init=dayang.read_angles()

        while True:
            time.sleep(0.05)
            x_angle, y_angle = ruifen.read_angles()
            x_angle=x_angle-(0)
            y_angle=y_angle-(0)
            print(f'x: {x_angle}, y: {y_angle}')

            # force=dayang.read_angles()-force_init
            # print(f"force: {force}")
            # if force > 50: # Up
            #     myXYZ.AxisMode_Jog(1,3,0)
            #     myXYZ.AxisMode_Jog(2,3,0)
            #     myXYZ.AxisMode_Jog(3,30,-15*force)
            # elif force < 10: # Down
            #     myXYZ.AxisMode_Jog(1,3,0)
            #     myXYZ.AxisMode_Jog(2,3,0)
            #     myXYZ.AxisMode_Jog(3,30,700)
            # else: # Suspend
            #     myXYZ.AxisMode_Jog(3,30,0)
            #     if abs(x_angle) < 20:
            #         myXYZ.AxisMode_Jog(2,2,40*x_angle)
            #     if abs(y_angle) < 20:
            #         myXYZ.AxisMode_Jog(1,6,70*y_angle)
            
            for event in pygame.event.get():
                if event.type == pygame.JOYAXISMOTION:
                    print("JoyAxis {} Moved to {}".format(event.axis, event.value))
                    if event.axis == 1:
                        print('Z mode')
                        myXYZ.AxisMode_Jog(1,3,0)
                        myXYZ.AxisMode_Jog(2,3,0)
                        myXYZ.AxisMode_Jog(3,6,1000*event.value) # Axis, Acc, Vel
                    elif event.axis == 3:
                        print('XY MODE')
                        myXYZ.AxisMode_Jog(3,30,0)
                        if abs(x_angle) < 20:
                            myXYZ.AxisMode_Jog(2,2,30*x_angle)
                        if abs(y_angle) < 20:
                            myXYZ.AxisMode_Jog(1,6,40*y_angle)
                    elif event.axis == 4:
                        print('XY MODE')
                        myXYZ.AxisMode_Jog(3,30,0)
                        if abs(x_angle) < 20:
                            myXYZ.AxisMode_Jog(2,2,30*x_angle)
                        if abs(y_angle) < 20:
                            myXYZ.AxisMode_Jog(1,6,40*y_angle)
                elif event.type == pygame.JOYBUTTONDOWN:
                    print("Button {} Down".format(event.button))
                elif event.type == pygame.JOYBUTTONUP:
                    print("Button {} Up".format(event.button))
                else:
                    print('No gamepad command!!!')

    except KeyboardInterrupt:
        print("Ctrl-C is pressed!")
    finally:
        myXYZ.SafeQuit()
        ruifen.close()
        dayang.close()
        sys.exit(0)

if __name__ == "__main__":
    main()


            