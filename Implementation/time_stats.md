| Model Name | Model file | Device | Average Time per File (ms) | Peak Time per File (ms) | Fastest Time per File (ms) |
| --- | --- | --- | --- | --- | --- |
| MobilNet SSD (200) | pb | I7-10th |   52 |  147 |   26 |
| MobilNet SSD (17070) | pb | I7-10th |   47 |  100 |   28 |
| MobilNet SSD (17070) (1 threads)| f32 | I7-10th |   39 |  106 |   18 |
| MobilNet SSD (17070) (4 threads)| f32 | I7-10th |   21 |   71 |    0 |
| MobilNet SSD (17070) (4 threads)| f16 | I7-10th |   17 |   37 |    0 |
| yolov8n | pb | I7-10th |   26 |   47 |   13 |
| yolov8n (1 thread)| f32 | I7-10th |   34 |   63 |   18 |
| yolov8n (4 threads) | f32 | I7-10th |   15 |   37 |    0 |
| yolov8n (4 threads) | f16 | I7-10th |   13 |   31 |    0 |
| yolov8n (4 threads) | int_quant | I7-10th |   12 |   28 |    0 |
| yolov8n (4 threads) | full_int_quant | I7-10th |   12 |  38 |    0 |
| yolov5n | pb | I7-10th |   27 |   75 |   12 |
| yolov5n (1 thread)| f32 | I7-10th |   35 |   77 |   16 |
| yolov5n (4 thread)| f32 | I7-10th |   15 |   41 |    0 |
| yolov5n (4 thread)| f16 | I7-10th |   13 |   31 |    0 |
| yolov5n (4 thread)| full_int_quant | I7-10th |   12 |   28 |    0 |
| EfficientDet D0 (512x512) | pb | I7-10th |  249 |  453 |  183 |
| yolov8n | pb | Jetson |   53 |   82 |   48 |
| yolov5n | pb | Jetson |   47 |   70 |   43 |
| MobilNet SSD (200) | pb | Jetson |  103 |  713 |   98 |
| MobilNet SSD (17070) | pb | Jetson |  111 |  681 |  104 |


18 Bilder 0sec nur bei multithreaded
tflite act16 int8 > detection mehr als 0.75s!!
efficientdet_d0_512	not working on jetson? retrain with fixed_shape_resizeris instead of keep_aspect_ratio_resizer