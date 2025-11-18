# 发送端
Touch_S=[0,0,0]
liganS = LiganSensor(port='/dev/ttyUSB0')
threading.Thread(target=read_threading, args=(liganS,), daemon=True).start()
if __name__ == "__main__":
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(('192.168.10.105', 65432))
            print("Connected to server")
            while True:
                message = "{}, {} \n".format(Touch_S[0], Touch_S[1])
                s.sendall(message.encode())
                print("Sent: {}".format(message.strip()))
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n User Interrupt!")


# 接收端 
def touch_sensor_server(host='0.0.0.0', port=65432):
    global Touch_S
    global buffer_dyn_Stouch
    global buffer_weight_Stouch
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"Server started, waiting for connection on {port}...")
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            buffer = b""
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                buffer += data
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    if b',' in line:
                        try:
                            time_str, avg_str = line.decode().split(',', 1)
                            send_time = float(time_str)
                            avg_value = int(avg_str)
                            print(f"send_time: {send_time}, avg_value: {avg_value}")
                        except (ValueError, UnicodeDecodeError):
                            pass



