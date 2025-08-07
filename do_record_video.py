import cv2
import pyaudio
import wave
import threading
import time
import os

class MediaRecorder:
    def __init__(self, audio_filename="recording.wav", video_filename="recording.avi"):
        self.audio_filename = audio_filename
        self.video_filename = video_filename
        self.audio_thread = None
        self.video_thread = None
        self.audio_recording = False
        self.video_recording = False
        
    def record_audio(self):
        """录音功能"""
        # 设置音频参数
        chunk = 1024
        format = pyaudio.paInt16
        channels = 2
        rate = 44100
        
        # 初始化PyAudio
        p = pyaudio.PyAudio()
        
        # 打开音频流
        stream = p.open(format=format,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=chunk)
        
        print("开始录音...")
        frames = []
        
        # 录音循环
        while self.audio_recording:
            data = stream.read(chunk)
            frames.append(data)
        
        print("录音结束.")
        
        # 停止并关闭流
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # 保存音频文件
        wf = wave.open(self.audio_filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        
    def record_video(self):
        """录像功能"""
        # 打开摄像头
        cap = cv2.VideoCapture(0)
        
        # 设置视频编码参数
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.video_filename, fourcc, 20.0, (640, 480))
        
        print("开始录像...")
        
        # 录像循环
        while self.video_recording:
            ret, frame = cap.read()
            if ret:
                # 写入帧
                out.write(frame)
                
                # 显示当前帧（可选）
                cv2.imshow('Recording...', frame)
                
                # 按'q'键停止录像
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        
        print("录像结束.")
        
        # 释放资源
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    def start_audio_recording(self):
        """开始录音"""
        if not self.audio_recording:
            self.audio_recording = True
            self.audio_thread = threading.Thread(target=self.record_audio)
            self.audio_thread.start()
            print("音频录制已启动")
        else:
            print("音频已在录制中")
    
    def stop_audio_recording(self):
        """停止录音"""
        if self.audio_recording:
            self.audio_recording = False
            if self.audio_thread:
                self.audio_thread.join()
            print("音频录制已停止")
        else:
            print("音频未在录制")
            
    def start_video_recording(self):
        """开始录像"""
        if not self.video_recording:
            self.video_recording = True
            self.video_thread = threading.Thread(target=self.record_video)
            self.video_thread.start()
            print("视频录制已启动")
        else:
            print("视频已在录制中")
    
    def stop_video_recording(self):
        """停止录像"""
        if self.video_recording:
            self.video_recording = False
            if self.video_thread:
                self.video_thread.join()
            print("视频录制已停止")
        else:
            print("视频未在录制")
    
    def start_recording(self):
        """同时开始录音和录像"""
        self.start_audio_recording()
        self.start_video_recording()
        
    def stop_recording(self):
        """同时停止录音和录像"""
        self.stop_audio_recording()
        self.stop_video_recording()

# 使用示例
