# -*- coding: utf-8 -*-
# @Time    : 2024/03/08 09:31
# @Author  : LiShiHao
# @FileName: main.py
# @Software: PyCharm

# TODO 精度问题 与pytorch原生检测精度上有差距
# TODO 最终结果从队列取出展示 需要调整尺寸
# TODO 字体显示的颜色需要调整
# TODO 线程启动时 卡顿
# TODO 视频检测完成后 停止视频（kill thread）绑定按钮
# TODO 结果按钮

import base64
import cv2
import json
import numpy as np
import os
from program.yolov5_hamlet_detection import Ui_MainWindow
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog,QMessageBox,QFrame
import sys
from util.show import VideoThread

ALLOW_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm", "asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv"


class DetectionMainWindow(QMainWindow,Ui_MainWindow):
    # 子线程返回主线程信号
    complete_video_signal = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # 设置样式
        self.set_style()
        # 设置默认阈值
        self.set_threshold()
        # 设置默认显示选项
        self.set_show()
        # 绑定选择模型文件模型文件
        self.toolButton_1.clicked.connect(lambda: self.select_file(self.toolButton_1.objectName()))
        # 绑定选择标签文件
        self.toolButton_2.clicked.connect(lambda: self.select_file(self.toolButton_2.objectName()))
        # 绑定选择检测文件
        self.toolButton_3.clicked.connect(lambda: self.select_file(self.toolButton_3.objectName()))
        # 绑定选择结果文件夹
        self.toolButton_4.clicked.connect(lambda: self.select_file(self.toolButton_4.objectName()))
        # 主线程绑定函数
        self.complete_video_signal.connect(self.view_video)
        # 创建子线程
        self.video_thread = VideoThread(self.complete_video_signal)
        # 运行子线程
        self.video_thread.start()
        # 向子线程发送信号
        self.pushButton_1.clicked.connect(self.open_video)

    # 设置文件路径回显label样式
    def set_style(self):
        self.label_9.setFrameShape(QFrame.Box)
        self.label_9.setFrameShadow(QFrame.Sunken)
        self.label_9.setLineWidth(1)
        self.label_9.setStyleSheet("background-color: rgb(255,255,255)")
        self.label_10.setFrameShape(QFrame.Box)
        self.label_10.setFrameShadow(QFrame.Sunken)
        self.label_10.setLineWidth(1)
        self.label_10.setStyleSheet("background-color: rgb(255,255,255)")
        self.label_11.setFrameShape(QFrame.Box)
        self.label_11.setFrameShadow(QFrame.Sunken)
        self.label_11.setLineWidth(1)
        self.label_11.setStyleSheet("background-color: rgb(255,255,255)")
        self.label_12.setFrameShape(QFrame.Box)
        self.label_12.setFrameShadow(QFrame.Sunken)
        self.label_12.setLineWidth(1)
        self.label_12.setStyleSheet("background-color: rgb(255,255,255)")

    # 选择文件
    def select_file(self,object_name):
        if object_name == "toolButton_1":
            file_path, _ = QFileDialog.getOpenFileName(directory="", filter="All Files(*)")
            base_name = os.path.basename(file_path)
            if not base_name.endswith("xml",-3):
                QMessageBox.critical(self,"错误","模型文件选择错误，请重新选择！",QMessageBox.Yes|QMessageBox.No,QMessageBox.Yes)
            else:
                self.label_9.setText(file_path)
        elif object_name == "toolButton_2":
            file_path, _ = QFileDialog.getOpenFileName(directory="", filter="All Files(*)")
            base_name = os.path.basename(file_path)
            if not base_name.endswith("yaml",-4):
                QMessageBox.critical(self,"错误","标签文件选择错误，请重新选择！",QMessageBox.Yes|QMessageBox.No,QMessageBox.Yes)
            else:
                self.label_10.setText(file_path)
        elif object_name == "toolButton_3":
            file_path, _ = QFileDialog.getOpenFileName(directory="", filter="All Files(*)")
            base_name = os.path.basename(file_path)
            if base_name[-3:] not in ALLOW_FORMATS:
                QMessageBox.critical(self,"错误","检测文件选择错误，请重新选择！",QMessageBox.Yes|QMessageBox.No,QMessageBox.Yes)
            else:
                self.label_11.setText(file_path)
        elif object_name == "toolButton_4":
            dir_path = QFileDialog.getExistingDirectory(directory="")
            self.label_12.setText(dir_path)

    # 设置默认阈值
    def set_threshold(self):
        self.doubleSpinBox_1.setValue(0.25)
        self.doubleSpinBox_2.setValue(0.25)
        self.doubleSpinBox_3.setValue(0.45)

    # 获取输入文件
    def get_file(self):
        file_path = self.label_11.text()
        return file_path

    # 获取权重文件路径与标签文件路径
    def get_path(self):
        model_path = self.label_9.text()
        label_path = self.label_10.text()
        return model_path,label_path

    # 获取阈值
    def get_threshold(self):
        conf_thresh = self.doubleSpinBox_1.value()
        score_thresh = self.doubleSpinBox_2.value()
        nms_thresh = self.doubleSpinBox_3.value()
        return conf_thresh,score_thresh,nms_thresh

    # 设置默认显示设置
    def set_show(self):
        self.checkBox_1.setChecked(True)
        self.checkBox_1.setEnabled(False)
        self.checkBox_3.setChecked(True)
        self.checkBox_4.setChecked(True)

    # 获取显示设置
    def get_show(self):
        show_box = self.checkBox_1.isChecked()
        show_class = self.checkBox_3.isChecked()
        show_score = self.checkBox_4.isChecked()
        return show_box,show_class,show_score

    def open_video(self):
        # 发送读取视频的信号
        file_path = self.get_file()
        model_path,label_path = self.get_path()
        conf_thresh, score_thresh, nms_thresh = self.get_threshold()
        show_box, show_class, show_score = self.get_show()
        self.video_thread.open_video_signal.emit([file_path,model_path,label_path,conf_thresh, score_thresh, nms_thresh,show_box, show_class, show_score])

    # TODO 显示图像时 固定MainWindow和label的高度，按照label的高度来转换宽度，label和MainWindow按照转换的宽度自由伸缩
    def view_video(self,json_data):
        image = cv2.resize(self.json2image(json_data),(self.label_8.height(),self.label_8.width()))
        q_image = QImage(image, image.shape[1], image.shape[0], QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        self.label_8.setPixmap(pixmap)
        proportion = pixmap.height() / self.label_8.height()
        pixmap.setDevicePixelRatio(proportion)

    def json2image(self,json_data):
        json2str = json.loads(json_data)["image_json"]
        str2bytes = json2str.encode("ascii")
        bytes2buf = base64.b64decode(str2bytes)
        image = cv2.imdecode(np.frombuffer(bytes2buf,dtype=np.uint8),cv2.IMREAD_COLOR)
        return image



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DetectionMainWindow()
    window.show()
    sys.exit(app.exec_())
