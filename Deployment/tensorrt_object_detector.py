import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np


import sys
import os
sys.path.append("../Detection_training/Tensorflow/")
sys.path.append('/usr/local/lib/python3.6/site-packages')
import time
from scripts.Paths import LABELS, paths, TEST_IMAGE

CLASSES = {}
for label in LABELS:
    CLASSES[label['id']] = label['name']

colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

MODEL_NAME = 'raccoon_yolov8n_320_B16_ep34'
# Load the TensorRT engine from the serialized file
def load_engine(engine_file_path):
    with open(engine_file_path, 'rb') as f:
        runtime = trt.Runtime(trt.Logger())
        return runtime.deserialize_cuda_engine(f.read())

# Preprocess the input image
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (320, 320))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img /= 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Postprocess the output tensor
def postprocess(output, threshold):
    boxes, classes, scores = [], [], []
    for detection in output:
        class_id = np.argmax(detection[1:4])
        confidence = detection[4]
        if confidence > threshold:
            print(detection)
            center_x = int(detection[:4][0] * 320)
            center_y = int(detection[:4][1] * 320)
            width = int(detection[:4][2] * 320)
            height = int(detection[:4][3] * 320)
            x = int(center_x - width / 2)
            y = int(center_y - height / 2)
            boxes.append([x, y, width, height])
            classes.append(class_id)
            scores.append(confidence)
    print((boxes, classes, scores))
    return boxes, classes, scores

# Run object detection on the input image
def detect_objects(engine, input_img, threshold=0.1):
    h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)

    # Preprocess the image
    preprocessed_img = preprocess_image(input_img)
    np.copyto(h_input, preprocessed_img.ravel())

    # Execute the inference
    with engine.create_execution_context() as context:
        # Transfer input data to the GPU
        cuda.memcpy_htod_async(d_input, h_input, stream)
        # Run inference
        context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        # Transfer predictions back from the GPU
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        # Synchronize the stream
        stream.synchronize()

    # Postprocess the output tensor
    print(h_output)
    output = np.reshape(h_output, (output_shapes[0][0], -1))
    boxes, classes, scores = postprocess(output, threshold)
    return boxes, classes, scores

# Allocate GPU buffers for input and output tensors
def allocate_buffers(engine):
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()
    return h_input, d_input, h_output, d_output, stream

# Define the paths to the model and engine files
onnx_model_path = 'path/to/yolov8.onnx'
trt_engine_path = 'path/to/yolov8.engine'
trt_engine_path = os.path.join(paths.MODEL_PATH, MODEL_NAME, 'export',
                                   'here.engine')

# Load the TensorRT engine
engine = load_engine(trt_engine_path)
output_shapes = [engine.get_binding_shape(i) for i in range(engine.num_bindings)]

# Provide the input image path
input_image_path = TEST_IMAGE

# Run object detection on the input image
boxes, classes, scores = detect_objects(engine, input_image_path)

# Display the results
img = cv2.imread(TEST_IMAGE)
img = cv2.resize(img, (320, 320))
for box, cls, score in zip(boxes, classes, scores):
    x, y, w, h = box
    label = f'Class: {cls}, Score: {score}'
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
cv2.imwrite('obj.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
