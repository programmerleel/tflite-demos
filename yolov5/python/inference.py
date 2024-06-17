# -*- coding: utf-8 -*-
# @Time    : 2024/6/17 22:26
# @Author  : Lee
# @Project ：yolov5 
# @File    : inference.py.py

import argparse
import cv2
import numpy as np
import os
import sys
import tflite_runtime.interpreter as tflite

def init(model_path):
    interpreter = tflite.Interpreter(model_path)
    # 获取输入输出张量
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter,input_details,output_details

def preprocess(image_path,input_details):
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    input_shape = input_details[0]['shape']
    len = max(w, h)
    ratio = len / input_shape[1]
    new_w, new_h = int(w / ratio), int(h / ratio)
    image = cv2.resize(image, (new_w, new_h))
    pad_w, pad_h = input_shape[2] - new_w, input_shape[1] - new_h
    image = cv2.copyMakeBorder(image, pad_h // 2, input_shape[1] - new_h - pad_h // 2, pad_w // 2, input_shape[2] - new_w - pad_w // 2,
                               borderType=cv2.BORDER_CONSTANT)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 添加一维度
    image_data = np.expand_dims(image_rgb, axis=0).astype(np.float32)
    image_data = image_data / 255.0
    return image,image_data

def infer(interpreter,input_details,output_details,image):
    interpreter.set_tensor(input_details[0]['index'], image)
    # 创建网络层
    interpreter.invoke()
    outputs = interpreter.get_tensor(output_details[0]['index'])[0]
    return outputs

def postprocess(outputs,conf_thresh,nms_thresh,image):
    scores = []
    boxes = []
    labels = []
    for i in range(outputs.shape[0]):
        # 判断conf
        if (outputs[i][4] > conf_thresh):
            scores.append(outputs[i][4])
            # 获取边框坐标
            # ymin = int(max(1, (boxes[i][0] * imH)))
            # xmin = int(outputs[i][0]*640)-int(outputs[i][2]*640)//2
            # ymin = int(outputs[i][1]*640)-int(outputs[i][3]*640)//2
            # xmax = int(outputs[i][0]*640)+int(outputs[i][2]*640)//2
            # ymax = int(outputs[i][1]*640)+int(outputs[i][3]*640)//2

            xmin = int(outputs[i][0]) - int(outputs[i][2]) // 2
            ymin = int(outputs[i][1]) - int(outputs[i][3]) // 2
            xmax = int(outputs[i][0]) + int(outputs[i][2]) // 2
            ymax = int(outputs[i][1]) + int(outputs[i][3]) // 2
            # 获取检测标签
            label = np.argmax(outputs[i][5:])
            labels.append(label)
            boxes.append([xmin,ymin,xmax,ymax])
    ids = cv2.dnn.NMSBoxes(boxes,scores,conf_thresh,nms_thresh)
    for id in ids:
        score = scores[id]
        box = boxes[id]
        xmin,ymin,xmax,ymax = box
        label = labels[id]
        # 画框
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=2)
        cv2.putText(image, str(score), (xmin, ymin-30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(image, str(label), (xmin, ymin-15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow('image', image)
    cv2.waitKey(0)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",type=str,help="模型地址",default=r"D:\tflite-demos\yolov5\saved_model\yolov5s_float32.tflite")
    parser.add_argument("--images_path",type=str,help="图片地址",default=r"D:\tflite-demos\yolov5\data\images")
    parser.add_argument("--conf_thresh",type=float,help="置信度阈值",default=0.25)
    parser.add_argument("--nms_thresh",type=float,help="nms阈值",default=0.45)

    return parser.parse_args(argv)

def main(args):
    interpreter,input_details,output_details = init(args.model_path)
    for item in os.scandir(args.images_path):
        image_path = item.path
        image,image_data = preprocess(image_path,input_details)
        outputs = infer(interpreter,input_details,output_details,image_data)
        postprocess(outputs,args.conf_thresh,args.nms_thresh,image)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))