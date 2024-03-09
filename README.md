# yolov5-hamlet-detection

安全帽检测项目：
- 界面采用pyqt进行编写（在读取、展示视频时采用QThread开启两个线程分别执行）
- 采用ylov5n模型进行训练
- 利用openvino框架进行部署（在推理过程中使用生产者-消费者模型+异步流水线方式进行加速）

### pyqt 界面
