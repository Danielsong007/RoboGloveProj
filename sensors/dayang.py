import serial
import time

class DaYangSensor:
    def __init__(self,port_name,zero):
        self.ser = serial.Serial(port=port_name, baudrate=19200, bytesize=8, parity=serial.PARITY_NONE, stopbits=1, timeout=1)
        if zero==1:
            self.ser.write(bytes([0x01, 0x10, 0x06, 0x2A, 0x00, 0x02, 0x04, 0x00, 0x00, 0x00, 0x01, 0x9B, 0xA8])) # Zero

    def read_angles(self): # 读取传感器角度数据
        self.ser.flushInput()  # 清空串口接收缓冲区
        self.ser.write(bytes([0x01, 0x03, 0x9C, 0x40, 0x00, 0x02, 0xEB, 0x8F]))
        response = self.ser.read(9)  # 读取传感器应答命令
        # print(f"Response in HEX: {response.hex().upper()}")
        force_data = int.from_bytes(response[3:7], byteorder='big')
        if force_data>20000:
            return 0
        else:
            return force_data

    def close(self): # 关闭串口连接
        self.ser.close()

# 使用示例
if __name__ == "__main__":
    sensor = DaYangSensor('/dev/ttyUSB1',1)
    try:
        last_t=time.time()
        while True:
            time.sleep(0.00001)  # 等待 100ms，确保命令已发送
            cur_t=time.time()
            dt=cur_t-last_t
            last_t=cur_t
            force=sensor.read_angles()
            print(force,1/dt)
    except KeyboardInterrupt:
        print("KeyboardExit")
    finally:
        sensor.close()

