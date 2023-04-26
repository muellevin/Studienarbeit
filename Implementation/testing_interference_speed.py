"""
This script performs object detection on a set of input images using a pre-trained model.
The input images are loaded from a directory and the output images with the detected objects are saved in another directory.

Functions:

    get_class_name_labels:  Get the class names of the object detection model.
    preprocess_image:       Resize and expand dimensions of an input image for object detection.
    postprocess_output:     Postprocess the object detection model output to filter out
                            low-confidence detections and format the output in a list of dictionaries
                            with the bounding box coordinates, score, and class ID for each detection.
    visualize_output:       Visualize the output detections by drawing the bounding box and label for
                            each detection on the input image.
"""
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Union

import cv2
import numpy as np
import tensorflow as tf

sys.path.append("../Detection_training/Tensorflow/")
from scripts.Paths import LABELS, paths, TEST_IMAGE

# Load the frozen graph
MODEL_NAME = 'raccoonModel_50k_B16_img17070_efficientdet_d0_512'
model_path = os.path.join(paths.MODEL_PATH, MODEL_NAME, 'export', 'saved_model')
model = tf.saved_model.load(model_path, tags=["serve"])
inputs = model.signatures['serving_default'].inputs
outputs = model.signatures['serving_default'].outputs

# Set parameters for preprocessing, postprocessing, and visualization
MODEL_WIDTH = 512
MODEL_HEIGHT = 512
SCORE_THRESHOLD = 0.5
YOLO = False


def get_class_name_labels() -> List[str]:
    """Gets class name labels from the LABELS list.
    
    Returns:
        A list of class name strings.
    """
    class_names = []
    for i in range(90):
        if len(LABELS) > i:
            class_names.append(LABELS[i]['name'])
        else:
            class_names.append('???')
    return class_names

class_names = get_class_name_labels()  # replace with your own class names


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
    
    if YOLO:
        # Convert the tensor to float32 dtype
        image = tf.cast(image, tf.float32)

    return image

def postprocess_output(output: Dict[str, Any]) -> List[Dict[str, Union[List[float], int]]]:
    """
    Postprocess the object detection model output to filter out low-confidence detections and format the output in a list of dictionaries with the bounding box coordinates, score, and class ID for each detection.

    csharp

    Args:
        output: The output of the object detection model as a dictionary.

    Returns:
        A list of dictionaries representing the detections with the bounding box coordinates, score, and class ID for each detection.
    """
    boxes = output['detection_boxes'][0].numpy()
    scores = output['detection_scores'][0].numpy()
    class_ids = output['detection_classes'][0].numpy().astype(np.int32)
    mask = scores > SCORE_THRESHOLD
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]
    boxes[:, 0] *= MODEL_HEIGHT
    boxes[:, 1] *= MODEL_WIDTH
    boxes[:, 2] *= MODEL_HEIGHT
    boxes[:, 3] *= MODEL_WIDTH
    detections = []
    for box, score, class_id in zip(boxes, scores, class_ids):
        detection = {
            'box': box.tolist(),
            'score': score.tolist(),
            'class_id': class_id.tolist()
        }
        detections.append(detection)
    return detections

def visualize_output(image: np.ndarray, detections: List[Dict[str, Union[List[float], int]]]) -> np.ndarray:
    """
    Visualize the output detections by drawing the bounding box and label for each detection on the input image.

    Args:
        image: The input image as a NumPy array.
        detections: The list of dictionaries representing the detections with the bounding box coordinates,
                    score, and class ID for each detection.

    Returns:
        The output image as a NumPy array with the detected objects visualized.
    """
    for detection in detections:
        box = detection['box']
        score = detection['score']
        class_id = detection['class_id']
        color = (0, 255, 0)  # green
        label = f"{class_names[class_id - 1]}: {score:.2f}"
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (y1, x1), (y2, x2), color, thickness=2)
        cv2.putText(image, label, (y1, x1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2)
    return image

# Define the input and output directories
input_dir = paths.DEVSET_PATH
output_dir = os.path.join(paths.IMAGE_PATH, 'testing_dir')
shutil.rmtree(output_dir, ignore_errors=True)
# os.makedirs(output_dir, exist_ok=True)

# Define lists to store the times
times = []
fastest_time = float('inf')
slowest_time = float('-inf')

# First interference is loading the model -> ignore that
# Run object detection on the input image
model(preprocess_image(cv2.imread(TEST_IMAGE)))

# Loop over the input images and perform object detection
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        # Load the input image
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)

        # Preprocess the input image
        input_image = preprocess_image(image)

        # Run object detection on the input image
        start_time = time.time()
        output = model(input_image)
        end_time = time.time()

        # Postprocess the output detections
        # detections = postprocess_output(output)

        # # Visualize the output detections
        # output_image = visualize_output(cv2.resize(image, (MODEL_WIDTH, MODEL_HEIGHT)), detections)

        # # Save the output image
        # output_path = os.path.join(output_dir, filename)
        # cv2.imwrite(output_path, output_image)

        # Calculate the time taken for this file
        file_time = end_time - start_time
        print(f"time needed for file {filename} is {file_time:.4f} seconds")
        times.append(file_time)

        if file_time < fastest_time:
            fastest_time = file_time

        if file_time > slowest_time:
            slowest_time = file_time

# Calculate the average time
average_time = sum(times) / len(times)

# Print the results
print(f"Interference time stats")
print(f"    Average time: {average_time:.4f} seconds")
print(f"    Fastest time: {fastest_time:.4f} seconds")
print(f"    Slowest time: {slowest_time:.4f} seconds")

print("| Model Name | Model file | Device | Average Time per File (s) | Peak Time per File (s) | Fastest Time per File (s) |")
print(f"| {MODEL_NAME} | pb | I7-10th | {average_time:.4f} | {slowest_time:.4f} | {fastest_time:.4f} |")
