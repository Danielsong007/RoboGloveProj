import serial
import time

class DaYangSensor:
    def __init__(self):
        self.ser = serial.Serial(port="/dev/ttyUSB0", baudrate=19200, bytesize=8, parity=serial.PARITY_NONE, stopbits=1, timeout=1)

    def read_angles(self): # 读取传感器角度数据
        self.ser.flushInput()  # 清空串口接收缓冲区
        self.ser.write(bytes([0x01, 0x03, 0x9C, 0x40, 0x00, 0x02, 0xEB, 0x8F]))
        response = self.ser.read(9)  # 读取传感器应答命令
        # print(f"Response in HEX: {response.hex().upper()}")
        Force_data = int.from_bytes(response[3:7], byteorder='big')
        if Force_data< 10000:
            print(f"x_data: {Force_data}")
        else:
            print(f"x_data: 0")

    def close(self): # 关闭串口连接
        self.ser.close()

# 使用示例
if __name__ == "__main__":
    sensor = DaYangSensor()
    try:
        while True:
            sensor.read_angles()
            time.sleep(0.1)  # 等待 100ms，确保命令已发送
    except KeyboardInterrupt:
        print("KeyboardExit")
    finally:
        sensor.close()

