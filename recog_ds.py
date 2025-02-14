import pyaudio
import wave
import speech_recognition as sr

# 配置参数
FORMAT = pyaudio.paInt16  # 16位深度
CHANNELS = 1  # 单声道
RATE = 48000  # 采样率
CHUNK = 1024  # 每个缓冲区的帧数
RECORD_SECONDS = 5  # 每次录制的时长（秒）
INPUT_DEVICE_INDEX = None  # 默认麦克风设备索引（None 表示使用默认设备）

# 初始化语音识别器
recognizer = sr.Recognizer()

# 列出所有音频设备
def list_audio_devices():
    p = pyaudio.PyAudio()
    print("可用音频设备：")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"设备索引 {i}: {info['name']} (输入通道: {info['maxInputChannels']})")
    p.terminate()

# 实时语音转文字
def real_time_speech_to_text(input_device_index=None):
    # 初始化 PyAudio
    p = pyaudio.PyAudio()

    # 打开音频流
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=input_device_index)

    print("开始录音...")

    try:
        while True:
            frames = []
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)

            # 将音频数据转换为 AudioData 对象
            audio_data = sr.AudioData(b''.join(frames), RATE, 2)

            # 使用 Google Web API 进行语音识别
            try:
                text = recognizer.recognize_google(audio_data, language="zh-CN", timeout=5)
                print("识别结果:", text)
            except sr.UnknownValueError:
                print("无法识别语音")
            except sr.RequestError as e:
                print(f"请求失败: {e}")

    except KeyboardInterrupt:
        print("停止录音")

    # 关闭流
    stream.stop_stream()
    stream.close()
    p.terminate()

# 主程序
if __name__ == "__main__":
    # 列出所有音频设备
    list_audio_devices()

    # 提示用户输入设备索引
    input_device_index = int(input("请输入要使用的麦克风设备索引: "))

    # 开始实时语音转文字
    real_time_speech_to_text(input_device_index)