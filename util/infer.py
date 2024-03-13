# -*- coding: utf-8 -*-
# @Time    : 2024/03/08 11:32
# @Author  : LiShiHao
# @FileName: infer.py
# @Software: PyCharm

import base64
import json
import cv2
import openvino as ov
import yaml
import queue
from threading import Thread

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

    # 开启线程
    def run(self):
        self.thread.start()

    # 关闭线程
    def close(self):
        self.thread.join()

    # 处理产品
    def process(self):
        pass


# 预处理图像
class PreProduct(Product):
    def __init__(self, im0, im, ratio, dw, dh, new_shape, t):
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
        # 记录处理时间
        self.t = t


# 图像预处理 生产者 消费者
class PreProcessor(Processor):
    # 消费读取视频队列 生产与处理图像队列
    def __init__(self, previous_queue=None, next_queue=None):
        super().__init__(previous_queue, next_queue)

    def process(self):
        while True:
            try:
                start = cv2.getTickCount()
                # 获取视频图像
                image = self.previous_queue.get()
                # 缩放
                im, ratio, dw, dh, new_shape = self.letterbox(image)
                # 存入与处理队列
                self.next_queue.put(PreProduct(image, im, ratio, dw, dh, new_shape,
                                               (cv2.getTickCount() - start) / float(cv2.getTickFrequency())))
                # print("pre:{}".format((cv2.getTickCount() - start) / float(cv2.getTickFrequency())))
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
        im = cv2.dnn.blobFromImage(im, scalefactor=1.0, size=new_shape, swapRB=True, ddepth=cv2.CV_32F) / 255.0
        return im, ratio, dw, dh, new_shape


# 推理结果
class InferProduct(PreProduct):
    def __init__(self, im0, im, ratio, dw, dh, new_shape, t, result):
        super().__init__(im0, im, ratio, dw, dh, new_shape, t)
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
        self.model = self.pop.build()
        self.compile_model = self.core.compile_model(self.model)
        self.infer_request = self.compile_model.create_infer_request()

    def process(self):
        while True:
            try:
                start = cv2.getTickCount()
                pre_product = self.previous_queue.get()
                result, im0, im, ratio, dw, dh, new_shape, t = self.infer(pre_product)
                self.next_queue.put(InferProduct(im0, im, ratio, dw, dh, new_shape,
                                                 t + (cv2.getTickCount() - start) / float(cv2.getTickFrequency()),
                                                 result))
            except Exception as e:
                print(e)

    # TODO 解决推理开启多个线程的顺序问题
    def infer(self, pre_product):
        result = self.compile_model(pre_product.im)["output0"]
        return result, pre_product.im0, pre_product.im, pre_product.ratio, pre_product.dw, pre_product.dh, pre_product.new_shape, pre_product.t


class PostProduct(InferProduct):
    def __init__(self, im0, im, ratio, dw, dh, new_shape, t, result):
        super().__init__(im0, im, ratio, dw, dh, new_shape, t, result)


class PostProcessor(Processor):
    def __init__(self, previous_queue=None, next_queue=None,label_path=None,conf_thresh=None, score_thresh=None, nms_thresh=None,show_box=None, show_class=None, show_score=None):
        super().__init__(previous_queue, next_queue)
        self.label_path = label_path
        self.conf_thresh = conf_thresh
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.show_box = show_box
        self.show_class = show_class
        self.show_score = show_score

    def load_label(self):
        label_file = open(self.label_path,"r")
        contents = yaml.load(label_file,Loader=yaml.FullLoader)
        # {'0': 'hamlet', '1': 'head'}
        labels = contents["names"]
        return labels

    def process(self):
        while True:
            try:
                start = cv2.getTickCount()
                infer_product = self.previous_queue.get()
                im0, im, ratio, dw, dh, new_shape, t, result = self.post(infer_product)
                self.next_queue.put(PostProduct(im0, im, ratio, dw, dh, new_shape,
                                                t + (cv2.getTickCount() - start) / float(cv2.getTickFrequency()),
                                                result))
            except Exception as e:
                print(e)

    def post(self, infer_product):
        im0 = infer_product.im0
        im = infer_product.im
        ratio = infer_product.ratio
        dw = infer_product.dw
        dh = infer_product.dh
        new_shape = infer_product.new_shape
        t = infer_product.t
        result = infer_product.result

        # 获取类别
        labels = self.load_label()

        # 置信度 类别编号 得分 框
        confidences, ids, scores, boxes = [], [], [], []

        # TODO 优化nms 减少循环
        for i in range(result.shape[1]):
            # 取置信度
            confidence = result[0][i][4]
            # 先从置信度排除部分结果 减轻后续压力
            if (confidence < self.conf_thresh):
                continue
            # 不同类别得分
            score = result[0][i][5:7]
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(score)
            # 判断类别得分
            if (maxVal > self.score_thresh):
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
                ids.append(maxLoc[1])
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.score_thresh, self.nms_thresh)
        results = {}
        for i in range(len(indices)):
            index = indices[i]
            id = ids[index]
            label = labels[id]
            if label not in results.keys():
                results[label] = 1
            else:
                results[label] = results[label] + 1
            # 显示box
            if self.show_box:
                cv2.rectangle(im0, (boxes[index][0], boxes[index][1]),
                          (boxes[index][0] + boxes[index][2], boxes[index][1] + boxes[index][3]), (255//(id+1), 255//(id+2), 255//(id+3)), 1, 8)
            if self.show_class:
                cv2.putText(im0, label, (boxes[index][0], boxes[index][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5,
                            (255 // (id + 2), 255 // (id + 3), 255 // (id + 4)))
            if self.show_score:
                score = scores[id]
                cv2.putText(im0, "{:.2f}".format(score), (boxes[index][0]+50, boxes[index][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5,
                            (255 // (id + 2), 255 // (id + 3), 255 // (id + 4)))
            detect_all,detect_head = 0,0
            for key,value in results.items():
                if key == "head":
                    detect_head = value
                detect_all = detect_all + value
            cv2.putText(im0, "检测到{}人，{}人未佩戴安全帽！".format(detect_all,detect_head), (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0,0,255))
            # TODO FPS并不准确，计算了各个阶段的时间，但是由于整个流程是以流水线形式进行，各个阶段会存在重叠，可以最后计算总的FPS
            return im0, im, ratio, dw, dh, new_shape, t, result

class SendProcessor(Processor):
    def __init__(self, previous_queue=None, next_queue=None,complete_video_signal=None):
        super().__init__(previous_queue, next_queue)
        self.complete_video_signal = complete_video_signal

    def run(self):
        self.thread.start()

    def process(self):
        while True:
            try:
                start = cv2.getTickCount()
                post_product = self.previous_queue.get()
                im0, im, ratio, dw, dh, new_shape, t, result = post_product.im0, post_product.im, post_product.ratio, post_product.dw, post_product.dh, post_product.new_shape, post_product.t, post_product.result
                # 转换图像
                image_json = self.image2json(im0)
                # 向主线程发送信号
                self.complete_video_signal.emit(image_json)
                cv2.waitKey(1)
                # self.next_queue.put(PostProduct(im0, im, ratio, dw, dh, new_shape,
                #                                  t + (cv2.getTickCount() - start) / float(cv2.getTickFrequency()),
                #                                  result))
                # print("post:{}".format((cv2.getTickCount() - start) / float(cv2.getTickFrequency())))
            except Exception as e:
                print(e)

    def image2json(self,image):
        retval, buf = cv2.imencode(".jpg",image)
        buf2bytes = base64.b64encode(buf)
        bytes2str = buf2bytes.decode("ascii")
        str2json = json.dumps({"image_json":bytes2str})
        return str2json


if __name__ == '__main__':
    path = r"D:\data\test_01.mp4"
    cap = cv2.VideoCapture(path)
    image_queue = queue.Queue()
    pre_queue = queue.Queue()
    infer_queue = queue.Queue()
    post_queue = queue.Queue()
    infer_processor = InferProcessor(pre_queue, infer_queue,
                                     r"C:\yolov5-hamlet-detection\yolov5\runs\train\exp\weights\best_openvino_model\best.xml")
    infer_processor.run()
    post_processor = PostProcessor(infer_queue, post_queue)
    post_processor.run()
    pre_processor = PreProcessor(image_queue, pre_queue)
    pre_processor.run()
    ret = True
    while True:
        # 读取到图像帧之后 传入推理接口 后续图像处理 推理 后处理等在infer中进行多线程处理
        ret, frame = cap.read()
        if ret is False:
            break
        image_queue.put(frame)

    while True:
        if image_queue.empty() and pre_queue.empty() and infer_queue.empty and not ret:
            infer_processor.close()
            post_processor.close()
            pre_processor.close()
