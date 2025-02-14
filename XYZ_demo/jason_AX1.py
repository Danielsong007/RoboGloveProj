from ctypes import *
import ctypes
#import numpy as np
import struct 
import time
import math


dll = ctypes.CDLL('./libGAS-LINUX-DLL.so')
print(dll)

dll.GA_StartDebugLog(1)

print('测试开始')
a=dll.GA_OpenByIP(b'192.168.0.200',b'192.168.0.1',0,0)
#a=dll.GA_Open(1,"COM1")
print('打开板卡GA_Open返回值:',a)

a=dll.GA_Reset()
print('复位板卡GA_Reset返回值:',a)




a=dll.GA_ECatInit()
print('初始化总线')

time.sleep(5)
StationCount = c_short(0)
a=dll.GA_ECatGetSlaveCount(byref(StationCount))
print('从站数量:',StationCount)

time.sleep(5)
InitSlaveNum = c_short(0)
pMode = c_short(0)
pModeStep = c_short(0)
pPauseStatus = c_short(0)

a=dll.GA_ECatGetInitStep(byref(InitSlaveNum),byref(pMode),byref(pModeStep),byref(pPauseStatus))
print('正在初始化从站：',InitSlaveNum)




a=dll.GA_EncOff(1)
print('关闭轴1编码器:',a)

a=dll.GA_ZeroPos(1,1)
print('清零轴1零位，返回值:',a)

a=dll.GA_AxisOn(1)
print('使能轴1返回值:',a)

a=dll.GA_PrfTrap(1)
print('设置轴1进入点位模式，返回值:',a)

a=dll.GA_SetTrapPrmSingle(1,c_double(1.0),c_double(1.0),c_double(0.0),0)
print('设置轴1点位运动参数，返回值:',a)

while 1:
    print('打开输出口Y5')
    a=dll.GA_SetExtDoBit(0, 5, 1)
    a=dll.GA_SetPos(1,200000)
    print('设置轴1运动目标位置为20000脉冲的位置，返回值:',a)
    a=dll.GA_SetVel(1,c_double(100))
    print('设置轴1运动速度为7.5脉冲/毫秒，返回值:',a)
    a=dll.GA_Update(1)
    print('启动轴1运动')
    print('延时5秒钟')
    time.sleep(5)
    dPrfPos = c_double(0.0)
    a = dll.GA_GetPrfPos(1, byref(dPrfPos),1,0)
    dValue = dPrfPos
    print('获取轴1脉冲位置，返回值：',a,'获取值：',dValue)

    lSts = c_long(0)
    a = dll.GA_GetSts(1, byref(lSts),1,0)
    print('获取轴1状态，返回值：',a,'获取值：',lSts)


    
    print('关闭输出口Y5')
    a = dll.GA_SetExtDoBit(0, 5, 0)
    a=dll.GA_SetPos(1,0)
    print('设置轴1运动目标位置为0脉冲的位置，返回值:',a)
    a=dll.GA_Update(1)
    print('启动轴1运动')
    print('延时5秒钟')
    time.sleep(5)
    dPrfPos = c_double(0)
    a = dll.GA_GetPrfPos(1, byref(dPrfPos),1,0)
    dValue = dPrfPos
    print('获取轴1脉冲位置，返回值：',a,'获取值：',dValue)
    
a=dll.GA_Close()
print('测试结束')