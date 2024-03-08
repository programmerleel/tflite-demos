# -*- coding: utf-8 -*-
# @Time    : 2024/03/08 11:32
# @Author  : LiShiHao
# @FileName: show.py
# @Software: PyCharm

import base64
import cv2
import json
from PyQt5.QtCore import QThread, pyqtSignal
import time

# 视频解码过程在子线程中进行 子线程将解码结果返回到主线程进行显示
class VideoThread(QThread):
    # 设置接受来自主线程解码视频的信号
    open_video_signal = pyqtSignal(str)
    def __init__(self,signal):
        super().__init__()
        # 主线程信号绑定对应函数
        self.open_video_signal.connect(self.open_video)
        # 子线程返回主线程信号
        self.complete_video_signal = signal

    def open_video(self,file_path):
        cap = cv2.VideoCapture(file_path)
        while True:
            # 读取到图像帧之后 传入推理接口 后续图像处理 推理 后处理等在infer中进行多线程处理
            ret,frame = cap.read()
            if ret is False:
                break
            # 转换图像
            image_json = self.image2json(frame)
            # 向主线程发送信号
            self.complete_video_signal.emit(image_json)
            cv2.waitKey(1)

    def image2json(self,image):
        retval, buf = cv2.imencode(".jpg",image)
        buf2bytes = base64.b64encode(buf)
        bytes2str = buf2bytes.decode("ascii")
        str2json = json.dumps({"image_json":bytes2str})
        return str2json

    def run(self):
        while True:
            time.sleep(1)

