import speech_recognition as sr

def list_microphones():
    mic_list = sr.Microphone.list_microphone_names()
    for i, mic_name in enumerate(mic_list):
        print(f"设备 {i}: {mic_name}")

def main():
    list_microphones()
    input_device_index = int(input("请输入要使用的麦克风设备索引: "))

    recognizer = sr.Recognizer()

    with sr.Microphone(device_index=input_device_index) as source:
        print("请说话...")
        while True:
            try:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=5)
                print("录音数据:", audio)  # 打印录音数据

                # text = recognizer.recognize_google(audio, language='zh-CN')
                text = recognizer.recognize_sphinx(audio)
                print(f"你说: {text}")

            except sr.UnknownValueError:
                print("抱歉，我无法理解你说的话。")
            except sr.RequestError as e:
                print(f"无法请求结果; {e}")
            except KeyboardInterrupt:
                print("停止识别。")
                break

if __name__ == "__main__":
    main()