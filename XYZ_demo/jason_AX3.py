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
RetV=dll.GA_Reset()
RetV=dll.GA_ECatInit()
RetV=dll.GA_Stop(0XFF,0)
time.sleep(1)

def EnableZeroAll():
    L_R1=dll.GA_AxisOn(1) # Enable
    L_R2=dll.GA_AxisOn(2) # Enable
    L_R3=dll.GA_AxisOn(4) # Enable
    L_R4=dll.GA_AxisOn(3) # Enable
    time.sleep(6)
    L_R5=dll.GA_ZeroPos(1,1)
    L_R6=dll.GA_ZeroPos(2,1)
    L_R7=dll.GA_ZeroPos(4,1)
    L_R8=dll.GA_ZeroPos(3,1)
    print('ALL Enabled & ALL Zero:', L_R1,L_R2,L_R3,L_R4,L_R5,L_R6,L_R7,L_R8)

def SetGear(id_slave,id_master):
    L_R1=dll.GA_PrfGear(id_slave,0)
    L_R2=dll.GA_SetGearMaster(id_slave,id_master,2)
    L_R3=dll.GA_SetGearRatio(id_slave,1,1,0,0) # 1:1
    L_R4=dll.GA_SetGearEvent(id_slave,1,0,0)
    L_R5=dll.GA_GearStart(0X0001<<(id_slave-1))
    print('SetGear:', L_R1,L_R2,L_R3,L_R4,L_R5)

def SafeQuit():
    L_R1=dll.GA_AxisOff(1)
    L_R2=dll.GA_AxisOff(2)
    L_R3=dll.GA_AxisOff(3)
    L_R4=dll.GA_AxisOff(4)
    L_R5=dll.GA_Close()

def Get_Pos(axis_id):
    dPrfPos=c_double(0.0)
    L_R1=dll.GA_GetPrfPos(axis_id,byref(dPrfPos),1,0) # Planned position
    dEncPos=c_double(0.0)
    L_R2=dll.GA_GetAxisEncPos(axis_id,byref(dEncPos),1,0) # Encoder position
    print('Axis',axis_id,':', 'Planned pos',dPrfPos, ',', 'Encoder pos',dEncPos)

def AxisMode_Trap(axis_id,target_pos,target_vel):
    L_R1=dll.GA_PrfTrap(axis_id)
    L_R2=dll.GA_SetTrapPrmSingle(axis_id,c_double(1.0),c_double(1.0),c_double(0.0),0)
    L_R3=dll.GA_SetPos(axis_id,c_long(target_pos)) # Pulse
    L_R4=dll.GA_SetVel(axis_id,c_double(target_vel)) # Pulse/ms
    L_R5=dll.GA_Update(0X0001<<(axis_id-1))

def AxisMode_Jog(axis_id,vel):
    L_R1=dll.GA_PrfJog(axis_id)
    L_R2=dll.GA_SetJogPrmSingle(axis_id,c_double(1),c_double(1),c_double(0.0)) # ACC DEC SMOOTH
    L_R1=dll.GA_SetVel(axis_id,c_double(vel))
    L_R2=dll.GA_Update(0X0001<<(axis_id-1)) # 0XFF    

try:
    axis_id=2
    EnableZeroAll()
    SetGear(4,2) # Axis 4 follows Axis 2
    AxisMode_Trap(axis_id,600000,300) # Axis, Pos, Vel
    time.sleep(3)
    # AxisMode_Jog(axis_id,50) # Axis, Vel
    # time.sleep(2)
    # AxisMode_Jog(axis_id,0) # Axis, Vel
    SafeQuit()
except KeyboardInterrupt:
    print("Ctrl-C pressed!")
    SafeQuit()
    sys.exit(0)
