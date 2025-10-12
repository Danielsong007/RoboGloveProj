import pygame
import sys

pygame.init()
pygame.joystick.init()
if pygame.joystick.get_count() == 0:
    print("No gamepad")
    pygame.quit()
    sys.exit()

joystick = pygame.joystick.Joystick(0) # Attention
joystick.init()
print("Name of gamepad:", joystick.get_name())

try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.JOYBUTTONDOWN:
                print("Button {} Down".format(event.button))
            elif event.type == pygame.JOYBUTTONUP:
                print("Button {} Up".format(event.button))
            if event.type == pygame.JOYAXISMOTION:
                print("JoyAxis {} Moved to {}".format(event.axis, event.value))

except KeyboardInterrupt:
    print("End!!!")
finally:
    pygame.quit()