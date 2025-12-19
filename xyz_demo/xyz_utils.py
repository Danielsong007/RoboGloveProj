from ctypes import *
import ctypes
import time
import sys

class xyz_utils():
    def __init__(self):
        self.dll = ctypes.CDLL('xyz_demo/libGAS-LINUX-DLL.so')
        ReV1=self.dll.GA_OpenByIP(b'192.168.0.200',b'192.168.0.1',0,0) # IP: My computer, Board
        ReV2=self.dll.GA_Reset()
        ReV2=self.dll.GA_ECatLoadPDOConfig()
        ReV3=self.dll.GA_ECatInit()
        ReV4=self.dll.GA_Stop(0XFF,0)
        time.sleep(0.1)

    def OpenEnableZero_ALL(self):
        ReV5=self.dll.GA_AxisOn(1) # Enable
        ReV6=self.dll.GA_AxisOn(2)
        ReV7=self.dll.GA_AxisOn(4)
        ReV8=self.dll.GA_AxisOn(3)
        time.sleep(6)
        ReV13=self.SetGearFollow(4,2) # Axis 4 follows Axis 2
        ReV14 = self.dll.GA_LmtsOn(1,-1)
        ReV15 = self.dll.GA_LmtsOn(2,-1)
        ReV16=self.dll.GA_SetHardLimN(1,0,0,5) # Hard Limit
        ReV17=self.dll.GA_SetHardLimP(1,0,0,4)
        ReV18=self.dll.GA_SetHardLimN(2,0,0,3)
        ReV19=self.dll.GA_SetHardLimP(2,0,0,2)
        time.sleep(1)
        print('ALL: Enabled & Zero & Axis 4 follows Axis 2 & Hard Limit')

    def SetGearFollow(self,id_slave,id_master):
        ReV1=self.dll.GA_PrfGear(id_slave,0)
        ReV2=self.dll.GA_SetGearMaster(id_slave,id_master,2)
        ReV3=self.dll.GA_SetGearRatio(id_slave,1,1,0,0) # 1:1
        ReV4=self.dll.GA_SetGearEvent(id_slave,1,0,0)
        ReV5=self.dll.GA_GearStart(0X0001<<(id_slave-1))
        print('SetGearFollow:', ReV1,ReV2,ReV3,ReV4,ReV5)

    def AxisMode_Trap(self,axis_id,target_pos,target_vel): # AxisMode_Trap(axis_id,600000,300)
        ReV1=self.dll.GA_PrfTrap(axis_id)
        ReV2=self.dll.GA_SetTrapPrmSingle(axis_id,c_double(1.0),c_double(1.0),c_double(0.0),0)
        ReV3=self.dll.GA_SetPos(axis_id,c_long(target_pos)) # Pulse
        ReV4=self.dll.GA_SetVel(axis_id,c_double(target_vel)) # Pulse/ms
        ReV5=self.dll.GA_Update(0X0001<<(axis_id-1))

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

    def Safe_Jog(self):
        InitialPos=self.Get_Pos(3)
        ReV9=self.dll.GA_ZeroPos(1,1)
        ReV10=self.dll.GA_ZeroPos(2,1)
        ReV11=self.dll.GA_ZeroPos(4,1)
        ReV12=self.dll.GA_ZeroPos(3,1)
        SoftLimitUp=int(1964000000-InitialPos)
        SoftLimitDown=int(1949000000-InitialPos)
        ReV20=self.dll.GA_SetSoftLimit(3,SoftLimitUp,SoftLimitDown) # Soft Limit

        axis_id = 3
        FVAULE = c_int32(0)
        nFlag=c_int16(0)
        i=0
        while i<3:
            i=i+1
            a = self.dll.GA_ECatSetSdoValue(axis_id, 0x607F, 0, 2147483647, 4) # Max Velocity
            time.sleep(0.05)
            a = self.dll.GA_ECatSetSdoValue(axis_id, 0x6065, 0, 10048576, 4) # Allowed speed err
            time.sleep(0.05)
        a = self.dll.GA_ECatGetSdoValue(axis_id, 0x607F, 0, byref(FVAULE), byref(nFlag), 4, 0)
        print ('607F is reset to: ', FVAULE)
        self.dll.GA_ECatGetSdoValue(axis_id, 0x6065, 0, byref(FVAULE), byref(nFlag), 4, 0)
        print('6065 is reset to: ', FVAULE)

        return(int(InitialPos))

    def Get_Pos(self,axis_id):
        dPrfPos=c_double(0.0)
        ReV1=self.dll.GA_GetPrfPos(axis_id,byref(dPrfPos),1,0) # Planned position
        dEncPos=c_double(0.0)
        ReV2=self.dll.GA_GetAxisEncPos(axis_id,byref(dEncPos),1,0) # Encoder position
        return dEncPos.value
        # print('Axis',axis_id,':', 'Planned pos',dPrfPos, ',', 'Encoder pos',dEncPos)

    def AxisMode_Jog(self,axis_id,acc,vel): # AxisMode_Jog(3,1,-100)
        ReV1=self.dll.GA_PrfJog(axis_id)
        ReV2=self.dll.GA_SetJogPrmSingle(axis_id,c_double(acc),c_double(acc),c_double(0.0)) # ACC DEC SMOOTH
        ReV1=self.dll.GA_SetVel(axis_id,c_double(-vel))
        ReV2=self.dll.GA_Update(0X0001<<(axis_id-1)) # 0XFF
    
    def AxisMode_Torque(self,axis_id):
        i=0
        while i<3:
            i=i+1
            self.dll.GA_ECatSetSdoValue(axis_id, 0x6060, 0, 4, 1)  # PT模式
            time.sleep(0.05)
        FVAULE = c_int32(0)
        nFlag=c_int16(0)
        a = self.dll.GA_ECatGetSdoValue(axis_id, 0x6060, 0, byref(FVAULE), byref(nFlag), 1, 0)
        print (a,FVAULE)

        i=0
        while i<3:
            i=i+1
            Value=100000 # Max Acc
            a = self.dll.GA_ECatSetSdoValue(axis_id, 0x6087, 0, Value, 4)
            time.sleep(0.05)
        FVAULE = c_int32(0)
        nFlag=c_int16(0)
        a = self.dll.GA_ECatGetSdoValue(axis_id, 0x6087, 0, byref(FVAULE), byref(nFlag), 4, 0)
        print (a,FVAULE)

        i=0
        while i<3:
            i=i+1
            Value=c_int16(1000) # Max Torq
            a = self.dll.GA_ECatSetSdoValue(axis_id, 0x6072, 0, Value, 2)
            time.sleep(0.05)
        FVAULE = c_int16(0)
        nFlag=c_int16(0)
        a = self.dll.GA_ECatGetSdoValue(axis_id, 0x6072, 0, byref(FVAULE), byref(nFlag), 4, 0)
        print (a,FVAULE)

    def Set_Torque_Multi(self,axis_id,torque): # 设置目标转矩(6071h), 0.1%单位, 100表示10%额定转矩
        i=0
        while i<4:
            IntTorque=c_int16(torque)
            a0 = self.dll.GA_ECatSetSdoValue(axis_id, 0x6071, 0, IntTorque, 2)
            i=i+1
            time.sleep(0.07)

    def Set_Torque(self,axis_id,torque): # 设置目标转矩(6071h), 0.1%单位, 100表示10%额定转矩
        IntTorque=c_int16(torque)
        a0 = self.dll.GA_ECatSetSdoValue(axis_id, 0x6071, 0, IntTorque, 2)
        time.sleep(0.02)
        IntTorque=c_int16(torque)
        a0 = self.dll.GA_ECatSetSdoValue(axis_id, 0x6071, 0, IntTorque, 2)
    
    def Read_Paras(self,axis_id):
        nFlag=c_int16(0)
        T_Sent = c_int16(0)
        self.dll.GA_ECatGetSdoValue(axis_id, 0x6071, 0, byref(T_Sent), byref(nFlag), 2, 0)
        T_Actual = c_int16(0)
        self.dll.GA_ECatGetSdoValue(axis_id, 0x6077, 0, byref(T_Actual), byref(nFlag), 2, 0)
        V_Actual = c_int32(0)
        self.dll.GA_ECatGetSdoValue(axis_id, 0x606C, 0, byref(V_Actual), byref(nFlag), 4, 0)        
        return T_Sent.value,T_Actual.value,V_Actual.value
    
    def Read_IOs(self):
        Value = c_int(0)
        a=self.dll.GA_GetExtDiBit(0,0,byref(Value))
        # print(a,Value.value)
        return Value.value


# # Torque Demo
# if __name__ == "__main__":
#     myXYZ = xyz_utils()
#     myXYZ.OpenEnableZero_ALL()
#     myXYZ.AxisMode_Torque(3) # Only current mode
#     try:
#         while True:
#             myXYZ.Set_Torque(3,80)
#             myXYZ.Read_Paras(3)
#             print(myXYZ.Get_Pos(3))
#             time.sleep(0.01)
#     except KeyboardInterrupt:
#         print("Ctrl-C is pressed!")
#     finally:
#         myXYZ.Set_Torque(3,0)
#         myXYZ.SafeQuit()
#         sys.exit(0)

# Jog Demo
if __name__ == "__main__":
    myXYZ = xyz_utils()
    myXYZ.OpenEnableZero_ALL()
    myXYZ.Safe_Jog()
    try:
        while True:
            # myXYZ.AxisMode_Jog(1,6,-400)
            # myXYZ.AxisMode_Jog(2,2,-200) # - to me
            myXYZ.AxisMode_Jog(3,30,2000)
            print(myXYZ.Get_Pos(3))
            # IOvalue=myXYZ.Read_IOs()
            # print(IOvalue)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Ctrl-C is pressed!")
    finally:
        myXYZ.SafeQuit()
        sys.exit(0)

