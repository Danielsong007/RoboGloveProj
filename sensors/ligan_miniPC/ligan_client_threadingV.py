from concurrent.futures import thread
import socket
import time
import serial
import heapq  # 用于获取最大的10个数
import threading


class LiganSensor:
    def __init__(self, port='COM3'): # Windows为COMx，Linux为/dev/ttyUSBx
        self.port = port
        self.ser = serial.Serial(port=self.port, baudrate=460800, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=1)
        self.Pdata=[0] * 36
    
    def close(self):
        self.ser.close()
    
    def print_data(self):
        for row in range(6):
            start_idx = row * 6
            end_idx = start_idx + 6
            row_data = self.Pdata[start_idx:end_idx]
            # print(f"传感器 {start_idx+1:2d}-{end_idx:2d}: {[f'{x:5d}' for x in row_data]}")
    
    def read_data(self):
        while self.ser.read() != b'\xFF':
            pass
        remaining_data = self.ser.read(77)
        frame_data = b'\xFF' + remaining_data
        pressure_values = []
        for i in range(36):
            idx = 4 + i * 2 # 每个数据点占2字节，从第4字节开始
            if idx + 1 < len(frame_data):
                value = (frame_data[idx] << 8) | frame_data[idx + 1]
                pressure_values.append(value)
        calculated_checksum = sum(frame_data[2:76]) & 0xFFFF
        frame_checksum = (frame_data[76] << 8) | frame_data[77]
        checksum_valid = (calculated_checksum == frame_checksum)
        if checksum_valid and len(remaining_data) == 77 and frame_data[1] == 0x66:
            self.Pdata=pressure_values
        else:
            print('Data is not right.')
        top_num=10
        Top_Ave=int(sum(heapq.nlargest(top_num, self.Pdata))/top_num)
        return [self.Pdata,Top_Ave]

def read_threading(sensor):
    global Touch_S
    try:
        while True:
            data=sensor.read_data()
            Touch_S=data[1]
            # print('Touch_S: ', Touch_S)
    except KeyboardInterrupt:
        print("\n User Interrupt!")
        sensor.close()

Touch_S=0
liganS = LiganSensor(port='/dev/ttyUSB0')
threading.Thread(target=read_threading, args=(liganS,), daemon=True).start()

# 使用示例
if __name__ == "__main__":
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(('192.168.10.105', 65432))
            print("Connected to server")
            Pnum = 0
            while True:
                message = "Data {}".format(Touch_S)
                s.sendall(message.encode())
                Pnum += 1
                if Pnum % 29 == 0:
                    print("Sent: {}".format(message))
                time.sleep(0.001)
    except KeyboardInterrupt:
        print("\n User Interrupt!")
    


