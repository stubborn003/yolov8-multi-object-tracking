import pyttsx3
import threading

# 初始化语音引擎
engine = pyttsx3.init()
# 添加一个锁对象
engine_lock = threading.Lock()

def play_voice_alert():
    with engine_lock:
        engine.say("警告！非机动车闯入机动车道")
        engine.runAndWait()