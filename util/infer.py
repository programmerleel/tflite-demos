# -*- coding: utf-8 -*-
# @Time    : 2024/03/08 11:32
# @Author  : LiShiHao
# @FileName: infer.py
# @Software: PyCharm

"""
思路：生产者 消费者模型 + 流水线（队列串联）
    图像前处理：转换到640*640尺寸 （消费者：从读取的图像队列取图像 生产者：将前处理图像放入队列）
    openvino推理 （消费者：从前处理图像队列取图像 生产者：将推理结果放入队列）
    结果后处理 （消费者：推理结果队列取结果 生产者：将后处理结果放入队列）
    结果转换到匹配pyqt界面尺寸
"""

import multiprocessing as mp
import openvino as ov
import queue
from threading import Thread
import numpy as np
from math import tanh
from time import time, sleep, perf_counter as pc
from queue import Empty, Full

import cv2

class_names = ["hamlet","head"]
counts = []

class Product():
    def __init__(self):
        super().__init__()


class Processor():
    def __init__(self, previous_queue, next_queue):
        super().__init__()
        # 产品来源
        self.previous_queue = previous_queue
        # 产品去向
        self.next_queue = next_queue
        # 线程
        self.thread = Thread(target=self.process)

    # 处理产品
    def process(self):
        pass


class PreProduct(Product):
    def __init__(self, im, ratio, dw, dh, new_shape,count):
        super().__init__()
        self.im = im
        self.ratio = ratio
        self.dw = dw
        self.dh = dh
        self.new_shape = new_shape
        self.count = count


class PreProcessor(Processor):
    def __init__(self, previous_queue=None, next_queue=None):
        super().__init__(previous_queue, next_queue)

    def run(self):
        self.thread.start()

    def process(self):
        count = 0
        while True:
            try:
                image = self.previous_queue.get()
                im, ratio, dw, dh, new_shape = self.letterbox(image)
                self.next_queue.put(PreProduct(im, ratio, dw, dh, new_shape,count))
                count = count + 1
            except Exception as e:
                print(e)

    def letterbox(self, im, new_shape=640, color=(114, 114, 114)):
        shape = im.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, dw, dh, new_shape


class InferProduct(Product):
    def __init__(self, data, im, ratio, dw, dh, new_shape,count):
        super().__init__()
        self.data = data
        self.im = im
        self.ratio = ratio
        self.dw = dw
        self.dh = dh
        self.new_shape = new_shape
        self.count = count


class InferProcessor(Processor):
    def __init__(self, previous_queue=None, next_queue=None, model_path=None):
        super().__init__(previous_queue, next_queue)
        self.model_path = model_path
        self.core = ov.Core()
        self.read_model = self.core.read_model(self.model_path)
        self.pop = ov.preprocess.PrePostProcessor(self.read_model)
        self.input_info = self.pop.input()
        self.input_info.tensor().set_element_type(ov.runtime.Type.u8)
        self.model = self.pop.build()
        self.compile_model = self.core.compile_model(self.model)
        self.infer_request = self.compile_model.create_infer_request()

    def run(self):
        self.thread.start()

    def process(self):
        while True:
            try:
                pre_product = self.previous_queue.get()
                im = pre_product.im
                ratio = pre_product.ratio
                dw = pre_product.dw
                dh = pre_product.dh
                count = pre_product.count
                new_shape = pre_product.new_shape
                input_tensor = ov.Tensor(np.expand_dims(pre_product.im, 0).transpose((0,3,1,2)))
                self.infer_request.set_input_tensor(input_tensor)
                self.infer_request.infer()
                output_tensor = self.infer_request.get_output_tensor()
                result = output_tensor.data
                while True:
                    if count == len(counts):
                        counts.append(count)
                        print(count)

                        self.next_queue.put(InferProduct(im, ratio, dw, dh, new_shape, result,count))
                        break
            except Exception as e:
                print(e)

class PostProduct(Product):
    def __init__(self, im, ratio, dw, dh, new_shape,count):
        super().__init__()
        self.im = im
        self.ratio = ratio
        self.dw = dw
        self.dh = dh
        self.new_shape = new_shape
        self.count = count


class PostProcessor(Processor):
    def __init__(self, previous_queue=None, next_queue=None):
        super().__init__(previous_queue, next_queue)

    def run(self):
        self.thread.start()

    def process(self):
        while True:
            try:
                infer_product = self.previous_queue.get()
                im = infer_product.im
                ratio = infer_product.ratio
                dw = infer_product.dw
                dh = infer_product.dh
                count = infer_product.count
                result = infer_product.data
                new_shape = infer_product.new_shape
                boxes = []
                confidences = []
                scores = []
                ids = []
                for i in range(result.shape[1]):
                    confidence = result[i][4]
                    if (confidence < 0.25):
                        continue
                    score = result[i][5:7]
                    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(score)
                    if (maxVal>0.25):
                        cx = result[i][0]
                        cy = result[i][1]
                        w = result[i][2]
                        h = result[i][3]
                        left = int((cx - w / 2 - dw) / ratio)
                        top = int((cy - h / 2 - dh) / ratio)
                        width = int(w / ratio)
                        height = int(h / ratio)
                        scores.append(maxVal)
                        boxes.append([left,top,width,height])
                        confidences.append(confidence)
                        ids.append(maxLoc[0])
                    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)
                    for i in range(len(indices)):
                        index = indices[i]
                        id = ids[index]
                        cv2.rectangle(im, boxes[index], (0,0,0), 2, 8)
                        label = class_names[id]
                        cv2.putText(im, label, (boxes[index][0],boxes[index][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255))
                    cv2.imshow("im", im)
            except Exception as e:
                print(e)


# class Customer():
#     def __init__(self, processes):
#         super().__init__()
#         # 产品来源
#         self.previous_queue = None
#         # 产品去向
#         self.nets_queue = None
#         # 线程
#         self.threads = [Thread(target=self.process) for _ in range(processes)]
#
#     # 处理产品
#     def process(self):
#         pass

if __name__ == '__main__':
    path = r"D:\data\test_01.mp4"
    cap = cv2.VideoCapture(path)
    image_queue = queue.Queue()
    pre_queue = queue.Queue()
    infer_queue = queue.Queue()
    post_queue = queue.Queue()
    # TODO 开启多个线程 导致顺序改变 需要同步
    InferProcessor(pre_queue, infer_queue,
                   r"C:\yolov5-hamlet-detection\yolov5\runs\train\exp\weights\best_openvino_model\best.xml").run()
    InferProcessor(pre_queue, infer_queue,
                   r"C:\yolov5-hamlet-detection\yolov5\runs\train\exp\weights\best_openvino_model\best.xml").run()
    InferProcessor(pre_queue, infer_queue,
                   r"C:\yolov5-hamlet-detection\yolov5\runs\train\exp\weights\best_openvino_model\best.xml").run()
    InferProcessor(pre_queue, infer_queue,
                   r"C:\yolov5-hamlet-detection\yolov5\runs\train\exp\weights\best_openvino_model\best.xml").run()
    InferProcessor(pre_queue, infer_queue,
                   r"C:\yolov5-hamlet-detection\yolov5\runs\train\exp\weights\best_openvino_model\best.xml").run()
    InferProcessor(pre_queue, infer_queue,
                   r"C:\yolov5-hamlet-detection\yolov5\runs\train\exp\weights\best_openvino_model\best.xml").run()
    InferProcessor(pre_queue, infer_queue,
                   r"C:\yolov5-hamlet-detection\yolov5\runs\train\exp\weights\best_openvino_model\best.xml").run()
    InferProcessor(pre_queue, infer_queue,
                   r"C:\yolov5-hamlet-detection\yolov5\runs\train\exp\weights\best_openvino_model\best.xml").run()
    PostProcessor(infer_queue).run()
    PreProcessor(image_queue, pre_queue).run()

    # while True:
    #     try:
    #         item = pre_image_queue.get()
    #         print(item)
    #     except Exception:
    #         pass
    while True:
        # 读取到图像帧之后 传入推理接口 后续图像处理 推理 后处理等在infer中进行多线程处理
        ret, frame = cap.read()
        if ret is False:
            break
        image_queue.put(frame)
