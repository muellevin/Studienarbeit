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
| yolov8n (1 thread)| f32 | Jetson |  338 |  355 |  331 |
| yolov8n (4 thread)| f32 | Jetson |  179 |  226 |  166 |
| yolov8n (4 thread)| f32 | CM3    | 591  | 686 | 559 |
| yolov8n (4 thread)| full_int_quant | CM3    | 224  | 295 | 223 |
| yolov8n | full_int_quant | Jetson + ETPU |   17 |   20 |   16 |
| yolov8n | full_int_quant | CM3 + ETPU |   68 |   89 |   63 |
| yolov5n | pb | Jetson |   47 |   70 |   43 |
| yolov5n | full_int_quant | Jetson + ETPU |   23 |   26 |   21 |
| MobilNet SSD (200) | pb | Jetson |  103 |  713 |   98 |
| MobilNet SSD (17070) | pb | Jetson |  111 |  681 |  104 |


18 Bilder 0sec nur bei multithreaded
tflite act16 int8 > detection mehr als 0.75s!!
efficientdet_d0_512	not working on jetson? retrain with fixed_shape_resizeris instead of keep_aspect_ratio_resizer
tflite langsam -> https://github.com/tensorflow/tensorflow/issues/40706
trt nicht korrekt aber fp16 == int8 edgetpu time

wenn versuch ultralytics yolo direkt:
/usr/local/lib/python3.6/dist-packages/torchvision-0.11.3-py3.6-linux-aarch64.egg/torchvision/io/image.py:11: UserWarning: Failed to load image Python extension:
  warn(f"Failed to load image Python extension: {e}")

/home/levin/.local/lib/python3.6/site-packages/torch/cuda/__init__.py:121: UserWarning:
    Found GPU0 NVIDIA Tegra X1 which is of cuda capability 5.3.
    PyTorch no longer supports this GPU because it is too old.
    The minimum cuda capability supported by this library is 6.2.

  warnings.warn(old_gpu_warn % (d, name, major, minor, min_arch // 10, min_arch % 10))
/home/levin/.local/lib/python3.6/site-packages/torch/cuda/__init__.py:144: UserWarning:
NVIDIA Tegra X1 with CUDA capability sm_53 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_62 sm_72.
If you want to use the NVIDIA Tegra X1 GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/

  warnings.warn(incompatible_device_warn.format(device_name, capability, " ".join(arch_list), device_name))
Ultralytics YOLOv8.0.84 🚀 Python-3.6.9 torch-1.11.0a0+17540c5 CUDA:0 (NVIDIA Tegra X1, 3964MiB)
Traceback (most recent call last):
  File "testing.py", line 17, in <module>
    results = model.val(data=pp.YOLO_CONFIG_PATH,)  # evaluate model performance on the validation set
  File "/home/levin/.local/lib/python3.6/site-packages/torch/autograd/grad_mode.py", line 28, in decorate_context
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.6/dist-packages/ultralytics/yolo/engine/model.py", line 301, in val
    validator(model=self.model)
  File "/home/levin/.local/lib/python3.6/site-packages/torch/autograd/grad_mode.py", line 28, in decorate_context
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.6/dist-packages/ultralytics/yolo/engine/validator.py", line 112, in __call__
    model = AutoBackend(model, device=self.device, dnn=self.args.dnn, data=self.args.data, fp16=self.args.half)
  File "/usr/local/lib/python3.6/dist-packages/ultralytics/nn/autobackend.py", line 93, in __init__
    model = model.fuse(verbose=verbose) if fuse else model
  File "/usr/local/lib/python3.6/dist-packages/ultralytics/nn/tasks.py", line 101, in fuse
    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
  File "/usr/local/lib/python3.6/dist-packages/ultralytics/yolo/utils/torch_utils.py", line 121, in fuse_conv_and_bn
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
RuntimeError: CUDA error: no kernel image is available for execution on the device
CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.




[05/18/2023-21:52:15] [I] === Performance summary ===
[05/18/2023-21:52:15] [I] Throughput: 72.2544 qps
[05/18/2023-21:52:15] [I] Latency: min = 13.7075 ms, max = 14.8861 ms, mean = 13.8259 ms, median = 13.7999 ms, percentile(99%) = 14.8121 ms
[05/18/2023-21:52:15] [I] End-to-End Host Latency: min = 13.7175 ms, max = 14.9229 ms, mean = 13.8394 ms, median = 13.8109 ms, percentile(99%) = 14.831 ms
[05/18/2023-21:52:15] [I] Enqueue Time: min = 6.08826 ms, max = 9.51123 ms, mean = 7.35561 ms, median = 7.19287 ms, percentile(99%) = 9.49573 ms
[05/18/2023-21:52:15] [I] H2D Latency: min = 0.117554 ms, max = 0.13623 ms, mean = 0.120202 ms, median = 0.119202 ms, percentile(99%) = 0.128906 ms
[05/18/2023-21:52:15] [I] GPU Compute Time: min = 13.5784 ms, max = 14.7576 ms, mean = 13.6963 ms, median = 13.6698 ms, percentile(99%) = 14.6803 ms
[05/18/2023-21:52:15] [I] D2H Latency: min = 0.00830078 ms, max = 0.0100708 ms, mean = 0.00940053 ms, median = 0.00939941 ms, percentile(99%) = 0.0100098 ms
[05/18/2023-21:52:15] [I] Total Host Walltime: 3.03096 s
[05/18/2023-21:52:15] [I] Total GPU Compute Time: 2.99948 s
[05/18/2023-21:52:15] [I] Explanations of the performance metrics are printed in the verbose logs.
[05/18/2023-21:52:15] [I]
&&&& PASSED TensorRT.trtexec [TensorRT v8201] # /usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=here.engine --best