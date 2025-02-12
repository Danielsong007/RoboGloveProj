import pygame
import sys

# 初始化pygame
pygame.init()

# 初始化手柄
pygame.joystick.init()

# 检查手柄数量
if pygame.joystick.get_count() == 0:
    print("没有检测到手柄")
    pygame.quit()
    sys.exit()

# 获取手柄
joystick = pygame.joystick.Joystick(0)
joystick.init()

print("游戏手柄名称:", joystick.get_name())

# 事件循环
try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # 检查手柄按键事件
            if event.type == pygame.JOYBUTTONDOWN:
                print(f"按钮 {event.button} 被按下")
            elif event.type == pygame.JOYBUTTONUP:
                print(f"按钮 {event.button} 被释放")

            # 检查摇杆移动
            if event.type == pygame.JOYAXISMOTION:
                print(f"摇杆 {event.axis} 移动到 {event.value}")

except KeyboardInterrupt:
    print("程序被终止")
finally:
    pygame.quit()