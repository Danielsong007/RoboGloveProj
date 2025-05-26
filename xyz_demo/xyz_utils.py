from ctypes import *
import ctypes
import time
import sys

class xyz_utils():
    def __init__(self):
        self.dll = ctypes.CDLL('xyz_demo/libGAS-LINUX-DLL.so')
        ReV1=self.dll.GA_OpenByIP(b'192.168.0.200',b'192.168.0.1',0,0)
        ReV2=self.dll.GA_Reset()
        ReV3=self.dll.GA_ECatInit()
        # ReV4=self.dll.GA_Stop(0XFF,0)
        time.sleep(0.1)

    def OpenEnableZero_ALL(self):
        ReV5=self.dll.GA_AxisOn(1) # Enable
        ReV6=self.dll.GA_AxisOn(2)
        ReV7=self.dll.GA_AxisOn(4)
        ReV8=self.dll.GA_AxisOn(3)
        time.sleep(6)
        ReV9=self.dll.GA_ZeroPos(1,1)
        ReV10=self.dll.GA_ZeroPos(2,1)
        ReV11=self.dll.GA_ZeroPos(4,1)
        ReV12=self.dll.GA_ZeroPos(3,1)
        ReV13=self.SetGearFollow(4,2) # Axis 4 follows Axis 2
        ReV14 = self.dll.GA_LmtsOn(1,-1)
        ReV15 = self.dll.GA_LmtsOn(2,-1)
        ReV16=self.dll.GA_SetHardLimN(1,0,0,5) # Hard Limit
        ReV17=self.dll.GA_SetHardLimP(1,0,0,4)
        ReV18=self.dll.GA_SetHardLimN(2,0,0,3)
        ReV19=self.dll.GA_SetHardLimP(2,0,0,2)
        softlimit=1000*1000*3
        ReV20=self.dll.GA_SetSoftLimit(3,softlimit,-softlimit) # Soft Limit
        time.sleep(1)
        print('ALL: Enabled & Zero & Axis 4 follows Axis 2 & Hard Limit')

    def SetGearFollow(self,id_slave,id_master):
        ReV1=self.dll.GA_PrfGear(id_slave,0)
        ReV2=self.dll.GA_SetGearMaster(id_slave,id_master,2)
        ReV3=self.dll.GA_SetGearRatio(id_slave,1,1,0,0) # 1:1
        ReV4=self.dll.GA_SetGearEvent(id_slave,1,0,0)
        ReV5=self.dll.GA_GearStart(0X0001<<(id_slave-1))
        print('SetGearFollow:', ReV1,ReV2,ReV3,ReV4,ReV5)

    def SafeQuit(self):
        ReV1=self.dll.GA_AxisOff(1)
        ReV2=self.dll.GA_AxisOff(2)
        ReV3=self.dll.GA_AxisOff(3)
        ReV4=self.dll.GA_AxisOff(4)
        ReV5=self.AxisMode_Jog(1,1,0)
        ReV6=self.AxisMode_Jog(2,1,0)
        ReV7=self.AxisMode_Jog(3,1,0)
        ReV8=self.AxisMode_Jog(4,1,0)
        ReV9=self.dll.GA_Close()

    def Get_Pos(self,axis_id):
        dPrfPos=c_double(0.0)
        ReV1=self.dll.GA_GetPrfPos(axis_id,byref(dPrfPos),1,0) # Planned position
        dEncPos=c_double(0.0)
        ReV2=self.dll.GA_GetAxisEncPos(axis_id,byref(dEncPos),1,0) # Encoder position
        print('Axis',axis_id,':', 'Planned pos',dPrfPos, ',', 'Encoder pos',dEncPos)

    def AxisMode_Trap(self,axis_id,target_pos,target_vel): # AxisMode_Trap(axis_id,600000,300)
        ReV1=self.dll.GA_PrfTrap(axis_id)
        ReV2=self.dll.GA_SetTrapPrmSingle(axis_id,c_double(1.0),c_double(1.0),c_double(0.0),0)
        ReV3=self.dll.GA_SetPos(axis_id,c_long(target_pos)) # Pulse
        ReV4=self.dll.GA_SetVel(axis_id,c_double(target_vel)) # Pulse/ms
        ReV5=self.dll.GA_Update(0X0001<<(axis_id-1))

    def AxisMode_Jog(self,axis_id,acc,vel): # AxisMode_Jog(3,1,-100)
        ReV1=self.dll.GA_PrfJog(axis_id)
        ReV2=self.dll.GA_SetJogPrmSingle(axis_id,c_double(acc),c_double(acc),c_double(0.0)) # ACC DEC SMOOTH
        ReV1=self.dll.GA_SetVel(axis_id,c_double(vel))
        ReV2=self.dll.GA_Update(0X0001<<(axis_id-1)) # 0XFF
    
    def AxisMode_Torque(self,axis_id):
        self.dll.GA_ECatSetSdoValue(axis_id, 0x6060, 0, 4, 1)  # 6060h=4（PT模式）

    def Set_Torque(self,axis_id,torque): # 设置目标转矩(6071h), 0.1%单位, 100表示10%额定转矩
        IntTorque=int(torque)
        a = self.dll.GA_ECatSetSdoValue(axis_id, 0x6071, 0, IntTorque, 2)

# 使用示例
if __name__ == "__main__":
    myXYZ = xyz_utils()
    myXYZ.AxisMode_Torque(3)
    myXYZ.OpenEnableZero_ALL()
    try:
        while True:
            myXYZ.Set_Torque(3,-60)
            # myXYZ.AxisMode_Jog(3,30,1000)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Ctrl-C is pressed!")
    finally:
        myXYZ.SafeQuit()
        sys.exit(0)
