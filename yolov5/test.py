#1.导入python库
#使用tflite_runtime来替换tensorflow，减少每次检测加载tf的时间
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
#import tensorflow as tf
#2.加载TFLite模型，具体模型文件根据模型的网络结构为主，解码不唯一，这个其中一种方法
interpreter = tflite.Interpreter(model_path=r"D:\tflite-demos\yolov5\yolov5m-fp16.tflite")
interpreter.allocate_tensors()
#3.获取模型输入、输出的数据的信息
input_details = interpreter.get_input_details()
print(input_details)
output_details = interpreter.get_output_details()
print(output_details)

#4.输入检测图像，处理图像矩阵（重点）
frame = cv2.imread(r"D:\tflite-demos\yolov5\data\images\bus.jpg")
#原图尺寸
h,w,_ = frame.shape
len = max(w,h)
ratio = len/640
new_w,new_h = int(w/ratio),int(h/ratio)
print(new_w,new_h)
frame = cv2.resize(frame,(new_w,new_h))
# cv2.imshow("frame",frame)
# cv2.waitKey()
pad_w,pad_h = 640-new_w,640-new_h
print(pad_w,pad_h)
frame = cv2.copyMakeBorder(frame,pad_h//2,640-new_h-pad_h//2,pad_w//2,640-new_w-pad_w//2,borderType=cv2.BORDER_CONSTANT)
print(frame.shape)
# cv2.imshow("frame",frame)
# cv2.waitKey()
#模型图尺寸
input_shape = input_details[0]['shape']
# RGB转BGR
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# 设置输入尺寸
frame_resized = cv2.resize(frame_rgb, (input_shape[1], input_shape[2]))
# 添加一维度
img = np.expand_dims(frame_resized, axis=0).astype(np.float32)
img = img/255.0
# float32浮点数，看模型量化类型unit8，不需要转换
#input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], img)

#5.创建检测网络层
interpreter.invoke()
#6.检测框/类别/置信度的检测
outputs = interpreter.get_tensor(output_details[0]['index'])[0]
print(type(outputs))
print(outputs.shape)
# classes = interpreter.get_tensor(output_details[1]['index'])[0]
# scores = interpreter.get_tensor(output_details[2]['index'])[0]
#7.绘画检测框
# 对于概率大于 50%的进行显示
for i in range(outputs.shape[0]):
    if (outputs[i][4] > 0.35):
        # 获取边框坐标
        # ymin = int(max(1, (boxes[i][0] * imH)))
        xmin = outputs[i][0]
        ymin = outputs[i][1]
        xmax = outputs[i][2]
        ymax = outputs[i][3]
        print(xmin,ymin,xmax,ymax)
        # 画框
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(10, 255, 0), thickness=2)
        # 获取检测标签
        # object_name = str(int(classes[i]))
        # label = '%s 0.%d' % (object_name, int(scores[i] * 100))
        # print("label", label)
        # 显示标记
        # frame = paint_chinese_opencv(frame, label, (xmin, ymin), (255, 0, 0))
cv2.imshow('object detect', frame)
cv2.waitKey(0)
