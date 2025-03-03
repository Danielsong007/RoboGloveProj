import serial
import time

ser = serial.Serial(port="/dev/ttyUSB0", baudrate=115200, bytesize=8, parity=serial.PARITY_EVEN, stopbits=1, timeout=1)
def SetMode():
    # cmd_disable_auto = bytes([0x01, 0x06, 0x00, 0x13, 0x00, 0x00, 0x78, 0x0F])  # 应答模式; CRC16
    # cmd_disable_auto = bytes([0x01, 0x06, 0x00, 0x13, 0x00, 0x01, 0xB9, 0xCF])  # 自动输出10HZ
    # cmd_disable_auto = bytes([0x01, 0x06, 0x00, 0x13, 0x00, 0x02, 0xF9, 0xCE])  # 自动输出25HZ
    cmd_disable_auto = bytes([0x01, 0x06, 0x00, 0x13, 0x00, 0x03, 0x38, 0x0E])  # 自动输出50HZ
    for i in range(5):  # 尝试发送 3 次
        ser.write(cmd_disable_auto)
        time.sleep(0.01)  # 等待 100ms，确保命令已发送

def parse_angle_data(data_frame, measurement_range):
    x_data = int.from_bytes(data_frame[3:7], byteorder='little')
    y_data = int.from_bytes(data_frame[7:11], byteorder='little')
    TotalPoints = (measurement_range * 2) * 100
    x_angle = (x_data - TotalPoints) * 0.01 + measurement_range
    y_angle = (y_data - TotalPoints) * 0.01 + measurement_range
    return x_angle, y_angle

try:
    while True:
        ser.flushInput() # 清空串口接收缓冲区
        ser.write(bytes([0x01, 0x03, 0x00, 0x02, 0x00, 0x04, 0xE5, 0xC9]))
        response = ser.read(13)  # 读取传感器应答命令
        print(f"Response in HEX: {response.hex().upper()}")
        # response = bytes.fromhex("01 03 08 00 00 00 00 00 23 00 00 64 1D")  # 示例数据帧
        x_angle, y_angle = parse_angle_data(response, 10) # Range(-10,10)
        print(f"X轴角度: {x_angle}°, Y轴角度: {y_angle}°")
        time.sleep(0.02)  # 等待 100ms，确保命令已发送

except KeyboardInterrupt:
    print("KeyboardExit")

finally:
    ser.close()


