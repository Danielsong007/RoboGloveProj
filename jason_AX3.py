from ctypes import *
import ctypes
# import numpy as np
import struct 
import time
import math
import signal
import sys

dll = ctypes.CDLL('./libGAS-LINUX-DLL.so')
RetV=dll.GA_OpenByIP(b'192.168.0.200',b'192.168.0.1',0,0)
RetV=dll.GA_Reset() # 复位板卡
RetV=dll.GA_ECatInit() # 初始化总线
time.sleep(2)

def SafeQuit():
    RetV=dll.GA_AxisOff(1)
    RetV=dll.GA_AxisOff(2)
    RetV=dll.GA_AxisOff(3)
    RetV=dll.GA_AxisOff(4)
    RetV=dll.GA_Close()

def GetState(axis_id):
    dPrfPos = c_double(0.0)
    a = dll.GA_GetPrfPos(axis_id, byref(dPrfPos),1,0) # 获取轴1脉冲位置
    print('Position of axis', axis_id, 'is:', dPrfPos)

def AxisConfig_Trap(axis_id):
    RetV=dll.GA_AxisOn(axis_id) # Enable
    time.sleep(6)
    print('Enabled!!!!!!!!!!!!!!!')
    RetV=dll.GA_ZeroPos(axis_id,1)
    GetState(axis_id)
    RetV=dll.GA_PrfTrap(axis_id) # 点位模式
    RetV=dll.GA_SetTrapPrmSingle(axis_id,c_double(1.0),c_double(1.0),c_double(0.0),0)

def PosControl_Trap(axis_id,target_pos,target_vel):
    RetV=dll.GA_SetPos(axis_id,target_pos) # 设置轴1运动目标位置为20000脉冲的位置
    RetV=dll.GA_SetVel(axis_id,c_double(target_vel)) # 设置轴1运动速度为7.5脉冲/毫秒
    RetV=dll.GA_Update(0XFF) # 启动轴1运动
    time.sleep(5)
    GetState(axis_id)

def AxisConfig_Jog(axis_id):
    RetV=dll.GA_AxisOn(axis_id) # Enable
    time.sleep(6)
    print('Enabled!!!!!!!!!!!!!!!')
    RetV=dll.GA_ZeroPos(axis_id,1)
    GetState(axis_id)
    RetV=dll.GA_PrfJog(axis_id)
    RetV=dll.GA_SetJogPrmSingle(axis_id,c_double(1),c_double(1),c_double(0.0)) # ACC DEC SMOOTH 0.5 脉冲/毫秒^2
    RetV=dll.GA_SetVel(axis_id,c_double(100)) # 设置 JOG 运动速度为 100 脉冲/毫秒
    RetV=dll.GA_Update(0XFF)
    time.sleep(5)

try:
    axis_id=3
    # AxisConfig_Trap(axis_id)
    # PosControl_Trap(axis_id,200000,100)
    AxisConfig_Jog(axis_id)
    SafeQuit()
except KeyboardInterrupt:
    print("Ctrl-C pressed!")
    SafeQuit()
    sys.exit(0)
