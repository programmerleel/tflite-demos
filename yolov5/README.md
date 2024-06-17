# yolov5



### 模型转换

- export.py转换tflite

  ```
  python export.py --weights yolov5s.pt --include tflite
  ```

  转换的tflite模型的精度为fp16：

  ![image-20240617214716297](assets\image-20240617214716297.png)

  查看代码yolov5/export.py 481行，将转换精度设置为fp16，将精度转换为fp32：

  ```
  converter.target_spec.supported_types = [tf.float16]
  # converter.target_spec.supported_types = [tf.float32]
  ```

  ![image-20240617214941669](assets\image-20240617214941669.png)

- onnx2tf转换tflite

  转换onnx模型

  ```
  python export.py --weights yolov5s.pt --include onnx
  ```

  转换tflite模型

  ```
  import onnx2tf
  
  onnx2tf.convert(
      input_onnx_file_path="yolov5s.onnx",
      batch_size=1,
  )
  ```

  **注意：onnx2tf转换的tflite模型并不需要对坐标进行转换（还未发现原因）**

  ![image-20240617220907394](assets\image-20240617220907394.png)

### 模型测试

- detect.py测试

  - yolov5s.pt

    ![image-20240617215202492](assets\image-20240617215202492.png)

  - yolov5s-fp32.tflite

    ![image-20240617215328139](assets\image-20240617215328139.png)

  - yolov5s-fp16.tflite

    ![image-20240617215342795](assets\image-20240617215342795.png)

- tflite_runtime编写inference.py

  https://github.com/NexelOfficial/tflite-runtime-win.git

  ![image-20240617230000385](assets\image-20240617230000385.png)
