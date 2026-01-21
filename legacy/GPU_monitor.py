from pynvml import *
import time

nvmlInit()     #初始化
print("Driver: ",nvmlSystemGetDriverVersion())  #显示驱动信息
#>>> Driver: 384.xxx

#查看设备
deviceCount = nvmlDeviceGetCount()
for i in range(deviceCount):
    handle = nvmlDeviceGetHandleByIndex(i)
    print("GPU", i, ":", nvmlDeviceGetName(handle))


#查看显存、温度、风扇、电源
for i in range(100):
    handle = nvmlDeviceGetHandleByIndex(0)
    print("Power usage",nvmlDeviceGetPowerUsage(handle)/1000)
    time.sleep(1)


#最后要关闭管理工具
nvmlShutdown()