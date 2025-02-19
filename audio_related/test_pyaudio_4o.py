import pyaudio
import wave


def device_info():
    audio = pyaudio.PyAudio()
    print("音频设备列表：")
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        print(f"{i}: {info['name']} - {info['maxInputChannels']} - {info['maxOutputChannels']} - {info['defaultSampleRate']}")
    audio.terminate()

# 配置参数
FORMAT = pyaudio.paInt16  # 音频格式
CHANNELS = 1  # 单声道
RATE = 48000  # 采样率
input_device_index=2
output_device_index=2
CHUNK = 1024  # 块大小
RECORD_SECONDS = 5  # 录音时长（秒）
OUTPUT_FILE = "test_audio.wav"  # 输出文件名

def record_audio():
    """录制音频"""
    audio = pyaudio.PyAudio()

    # 打开麦克风流
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=input_device_index,
                        frames_per_buffer=CHUNK)

    print("正在录音... 请对着麦克风说话")
    frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("录音结束")

    # 停止并关闭流
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # 保存音频文件
    with wave.open(OUTPUT_FILE, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"音频已保存到 {OUTPUT_FILE}")

def play_audio():
    """播放录制的音频"""
    audio = pyaudio.PyAudio()

    # 打开音频文件
    with wave.open(OUTPUT_FILE, 'rb') as wf:
        stream = audio.open(format=audio.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True,
                            output_device_index=output_device_index)

        print("正在播放录音...")
        data = wf.readframes(CHUNK)

        while data:
            stream.write(data)
            data = wf.readframes(CHUNK)

        print("播放结束")

    # 停止并关闭流
    stream.stop_stream()
    stream.close()
    audio.terminate()

if __name__ == "__main__":
    device_info()
    record_audio()
    play_audio()
    