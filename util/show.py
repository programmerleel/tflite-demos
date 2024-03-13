# -*- coding: utf-8 -*-
# @Time    : 2024/03/08 11:32
# @Author  : LiShiHao
# @FileName: show.py
# @Software: PyCharm

import base64
import json
from PyQt5.QtCore import QThread, pyqtSignal
import time
from util.infer import *

# 视频解码过程在子线程中进行 子线程将解码结果返回到主线程进行显示
class VideoThread(QThread):
    # 设置接受来自主线程解码视频的信号
    open_video_signal = pyqtSignal(list)
    def __init__(self,signal):
        super().__init__()
        # 主线程信号绑定对应函数
        self.open_video_signal.connect(self.open_video)
        # 子线程返回主线程信号
        self.complete_video_signal = signal

    def open_video(self,args):
        file_path, model_path, label_path, conf_thresh, score_thresh, nms_thresh, show_box,show_class,show_score = args
        cap = cv2.VideoCapture(file_path)
        image_queue = queue.Queue()
        pre_queue = queue.Queue()
        infer_queue = queue.Queue()
        post_queue = queue.Queue()
        send_processor = SendProcessor(post_queue,None,self.complete_video_signal)
        send_processor.run()
        post_processor = PostProcessor(infer_queue,post_queue,label_path,conf_thresh, score_thresh, nms_thresh,show_box, show_class, show_score)
        post_processor.run()
        infer_processor = InferProcessor(pre_queue, infer_queue,model_path)
        infer_processor.run()
        pre_processor = PreProcessor(image_queue, pre_queue)
        pre_processor.run()
        while True:
            # 读取到图像帧之后 传入推理接口 后续图像处理 推理 后处理等在infer中进行多线程处理
            ret, frame = cap.read()
            if ret is False:
                break
            image_queue.put(frame)

    def run(self):
        while True:
            time.sleep(0.1)

