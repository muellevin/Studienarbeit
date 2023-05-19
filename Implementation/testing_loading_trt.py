import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os
import shutil

import sys
sys.path.append("../Detection_training/Tensorflow/")
from scripts.Paths import LABELS, paths, TEST_IMAGE

# Set parameters for preprocessing, postprocessing, and visualization
MODEL_WIDTH = 320
MODEL_HEIGHT = 320
SCORE_THRESHOLD = 0.5

MODEL_NAME = 'raccoon_yolov8n_320_B16_ep34'
LITE_NAME = 'yolo8n_int8.trt'

runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

trt_engine_path = os.path.join(paths.MODEL_PATH, MODEL_NAME, 'export',
                                'trt_engine', LITE_NAME)
with open(trt_engine_path, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

def preprocess_image(image: np.ndarray)-> np.ndarray:
    """
    Resize and expand dimensions of an input image for object detection.

    Args:
        image: An input image as a NumPy array.

    Returns:
        A preprocessed image as a NumPy array.
    """
    image = cv2.resize(image, (MODEL_WIDTH, MODEL_HEIGHT))
    image = np.expand_dims(image, axis=0)

    return image
input_dir = paths.DEVSET_PATH
output_dir = os.path.join(paths.IMAGE_PATH, 'testing_dir')
shutil.rmtree(output_dir, ignore_errors=True)
os.makedirs(output_dir, exist_ok=True)

input_image = preprocess_image(cv2.imread(TEST_IMAGE))

input_binding = engine[0].get_binding_index('input')  # Assuming the input binding is named 'input'
output_binding = engine[0].get_binding_index('output')  # Assuming the output binding is named 'output'
input_shape = engine[0].get_binding_shape(input_binding)
output_shape = engine[0].get_binding_shape(output_binding)
input_dtype = engine[0].get_binding_dtype(input_binding)
output_dtype = engine[0].get_binding_dtype(output_binding)

input_size = trt.volume(input_shape) * trt.int32.itemsize
output_size = trt.volume(output_shape) * trt.int32.itemsize

d_input = cuda.mem_alloc(input_size)
d_output = cuda.mem_alloc(output_size)

bindings = [int(d_input), int(d_output)]

cuda.memcpy_htod(d_input, input_image)

context.execute_v2(bindings)

h_output = np.empty(output_shape, dtype=output_dtype)
cuda.memcpy_dtoh(h_output, d_output)

print(h_output)
