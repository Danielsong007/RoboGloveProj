import pygame
import sys
import time
from xyz_demo.xyz_utils import xyz_utils

pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0) # Attention
joystick.init()
print("Name of gamepad:", joystick.get_name())


def main():
    try:
        myXYZ = xyz_utils()
        myXYZ.OpenEnableZero_ALL()
        # myXYZ.dll.GA_ECatGetSdoValue(3, 6077 , 0, *pSdoValue, short*pPdoFlag,short nLen,short nSignFlag)
        while True:
            time.sleep(0.1)
            for event in pygame.event.get():
                if event.type == pygame.JOYAXISMOTION:
                    print("JoyAxis {} Moved to {}".format(event.axis, event.value))
                    if event.axis == 1:
                        myXYZ.AxisMode_Jog(3,6,1000*event.value) # Axis, Acc, Vel
                    elif event.axis == 3:
                        myXYZ.AxisMode_Jog(1,6,-700*event.value)
                    elif event.axis == 4:
                        myXYZ.AxisMode_Jog(2,2,400*event.value)
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
        pygame.quit()
        sys.exit(0)

if __name__ == "__main__":
    main()
