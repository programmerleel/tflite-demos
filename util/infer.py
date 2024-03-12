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

# 产品
class Product():
    def __init__(self):
        super().__init__()

# 生产者 消费者
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

# 预处理图像
class PreProduct(Product):
    def __init__(self,im0, im, ratio, dw, dh, new_shape,count):
        super().__init__()
        # 原图
        self.im0 = im0
        # 预处理图像
        self.im = im
        # 缩放比例
        self.ratio = ratio
        # 宽 padding（half）
        self.dw = dw
        # 高 padding （half）
        self.dh = dh
        # 缩放后尺寸
        self.new_shape = new_shape
        # 队列中产品编号
        self.count = count

# 图像预处理 生产者 消费者
class PreProcessor(Processor):
    # 消费读取视频队列 生产与处理图像队列
    def __init__(self, previous_queue=None, next_queue=None):
        super().__init__(previous_queue, next_queue)

    # 开启线程
    def run(self):
        self.thread.start()

    def process(self):
        count = 0
        while True:
            try:
                # 获取视频图像
                image = self.previous_queue.get()
                # 缩放
                im, ratio, dw, dh, new_shape = self.letterbox(image)
                # 存入与处理队列
                self.next_queue.put(PreProduct(image,im, ratio, dw, dh, new_shape,count))
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
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, ratio, dw, dh, new_shape

# 推理结果
class InferProduct(PreProduct):
    def __init__(self, im0,im, ratio, dw, dh, new_shape,count,result):
        super().__init__(im0,im, ratio, dw, dh, new_shape,count)
        # 推理结果
        self.result = result

# 推理 生产者 消费者
class InferProcessor(Processor):
    # 消费预处理队列 生产结果队列
    def __init__(self, previous_queue=None, next_queue=None, model_path=None):
        super().__init__(previous_queue, next_queue)
        self.model_path = model_path
        self.core = ov.Core()
        self.read_model = self.core.read_model(self.model_path)
        self.pop = ov.preprocess.PrePostProcessor(self.read_model)
        self.input_info = self.pop.input()
        self.input_info.tensor().set_element_type(ov.runtime.Type.f32)
        self.model = self.pop.build()
        self.compile_model = self.core.compile_model(self.model)
        self.infer_request = self.compile_model.create_infer_request()

    def run(self):
        self.thread.start()

    def process(self):
        while True:
            try:
                pre_product = self.previous_queue.get()
                result,im0,im,ratio,dw,dh,new_shape,count = self.infer(pre_product)
                while True:
                    # 由于在推理时使用了多个线程，推理速度不同，会导致队列顺序的变化，通过编号（count）来对线程进行阻塞
                    # TODO 这个方法并不是最佳解决方案 长时间的视频（拉取摄像头资源 accounts会不断增加 导致内存爆炸） 需要寻找更合适的方法来解决放入队列顺序的问题
                    if count == len(counts):
                        counts.append(count)
                        self.next_queue.put(InferProduct(im0,im, ratio, dw, dh, new_shape, count,result))
                        break
            except Exception as e:
                print(e)

    def infer(self,pre_product):
        # 预处理图像转换模型输入
        input_tensor = ov.Tensor(np.expand_dims(pre_product.im, 0).transpose((0, 3, 1, 2)).astype(np.float32)/255.0)
        self.infer_request.set_input_tensor(input_tensor)
        # 推理
        self.infer_request.infer()
        # 推理结果
        output_tensor = self.infer_request.get_output_tensor()
        result = output_tensor.data
        return result,pre_product.im0,pre_product.im,pre_product.ratio,pre_product.dw,pre_product.dh,pre_product.new_shape,pre_product.count

class PostProduct(InferProduct):
    def __init__(self,im0, im, ratio, dw, dh, new_shape,count,result):
        super().__init__(im0,im, ratio, dw, dh, new_shape,count,result)

class PostProcessor(Processor):
    def __init__(self, previous_queue=None, next_queue=None):
        super().__init__(previous_queue, next_queue)

    def run(self):
        self.thread.start()

    def process(self):
        while True:
            # try:
            infer_product = self.previous_queue.get()
            self.post(infer_product)

            # except Exception as e:
            #     print(e)

    def post(self,infer_product):
        im0 = infer_product.im0
        im = infer_product.im
        ratio = infer_product.ratio
        dw = infer_product.dw
        dh = infer_product.dh
        count = infer_product.count
        result = infer_product.result
        new_shape = infer_product.new_shape

        # 置信度 类别编号 得分 框
        confidences,ids,scores,boxes = [],[],[],[]
        # 遍历一张图的所有结果
        for i in range(result.shape[1]):
            # 取置信度
            confidence = result[0][i][4]
            # 先从置信度排除部分结果 减轻后续压力
            if (confidence < 0.25):
                continue
            print(confidence)
            # 不同类别得分
            score = result[0][i][5:7]
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(score)
            # 判断类别得分
            if (maxVal > 0.25):
                cx = result[0][i][0]
                cy = result[0][i][1]
                w = result[0][i][2]
                h = result[0][i][3]
                left = int((float(cx) - float(w) / 2 - dw) / ratio[0])
                top = int((float(cy) - float(h) / 2 - dh) / ratio[0])
                width = int(float(w) / ratio[0])
                height = int(float(h) / ratio[0])
                scores.append(maxVal)
                boxes.append([left, top, width, height])
                confidences.append(confidence)
                print(maxLoc)
                ids.append(maxLoc[1])
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)
            print(indices)
            for i in range(len(indices)):
                index = indices[i]
                print(index)
                id = ids[index]
                print(id)
                cv2.rectangle(im0, (boxes[index][0],boxes[index][1]),(boxes[index][0]+boxes[index][2],boxes[index][1]+boxes[index][3]), (0, 0, 0), 2, 8)
                label = class_names[id]
                cv2.putText(im0, label, (boxes[index][0], boxes[index][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5,
                            (255, 255, 255))
            cv2.imshow("im0", im0)
            cv2.waitKey(0)


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
