import serial
import time

class RuiFenSensor:
    def __init__(self,port_name):
        self.ser = serial.Serial(port=port_name, baudrate=115200, bytesize=8, parity=serial.PARITY_EVEN, stopbits=1, timeout=1)

    def set_mode(self): # 设置传感器模式
        # cmd_disable_auto = bytes([0x01, 0x06, 0x00, 0x13, 0x00, 0x00, 0x78, 0x0F])  # 应答模式; CRC16
        # cmd_disable_auto = bytes([0x01, 0x06, 0x00, 0x13, 0x00, 0x01, 0xB9, 0xCF])  # 自动输出10HZ
        # cmd_disable_auto = bytes([0x01, 0x06, 0x00, 0x13, 0x00, 0x02, 0xF9, 0xCE])  # 自动输出25HZ
        cmd_disable_auto = bytes([0x01, 0x06, 0x00, 0x13, 0x00, 0x03, 0x38, 0x0E])  # 自动输出50HZ
        for _ in range(5):  # 尝试发送 5 次
            self.ser.write(cmd_disable_auto)
            time.sleep(0.01)  # 等待 100ms，确保命令已发送

    def parse_angle_data(self,response, measurement_range): # 解析角度数据
        x_data = int.from_bytes(response[3:7], byteorder='little')
        y_data = int.from_bytes(response[7:11], byteorder='little')
        TotalPoints = (measurement_range * 2) * 100
        x_angle = (x_data - TotalPoints) * 0.01 + measurement_range
        y_angle = (y_data - TotalPoints) * 0.01 + measurement_range
        return x_angle, y_angle

    def read_angles(self): # 读取传感器角度数据
        self.ser.flushInput()  # 清空串口接收缓冲区
        self.ser.write(bytes([0x01, 0x03, 0x00, 0x02, 0x00, 0x04, 0xE5, 0xC9]))
        response = self.ser.read(13)  # 读取传感器应答命令
        # print(f"Response in HEX: {response.hex().upper()}")
        x_angle, y_angle = self.parse_angle_data(response, 10)
        return x_angle, y_angle

    def close(self): # 关闭串口连接
        self.ser.close()

# 使用示例
if __name__ == "__main__":
    sensor = RuiFenSensor('/dev/ttyUSB2')
    try:
        # sensor.set_mode()
        while True:
            x_angle, y_angle = sensor.read_angles()
            print(f"X轴角度: {x_angle}°, Y轴角度: {y_angle}°")
            time.sleep(0.05)  # 等待 100ms，确保命令已发送
    except KeyboardInterrupt:
        print("KeyboardExit")
    finally:
        sensor.close()