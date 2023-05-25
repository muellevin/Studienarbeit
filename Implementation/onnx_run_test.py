from dataclasses import dataclass
import time
import sys
import os
import cv2
import onnxruntime as ort
import numpy as np

sys.path.append("../Detection_training/Tensorflow/")
from scripts.Paths import LABELS, paths, TEST_IMAGE

MODEL_NAME = 'raccoon_yolov8n_320_B16_ep34'

onnx_path = os.path.join(paths.MODEL_PATH, MODEL_NAME, 'export',
                                   'best.onnx')


EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
# np.set_printoptions(threshold=np.inf)
def load_model(model_path):
    # Load the ONNX model
    session = ort.InferenceSession(model_path, providers=EP_list)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session, input_name, output_name


def preprocess_image(image_path, input_shape):
    # Load and preprocess the input image
    original_image: np.ndarray = cv2.imread(image_path)
    [height, width, _] = original_image.shape
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image
    scale = length / 320
    
    image = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(320, 320), swapRB=True)
    return image, original_image, scale

def p_to(outputs, scale=0, confidence_threshold=0.25) -> list[dict]:
    outputs = np.squeeze(outputs).T
    print(outputs.shape)
    boxes = outputs[:,:4]
    classes_scores = outputs[:, 4:]
    valid_indices = np.where(classes_scores > confidence_threshold)[0]
    print(valid_indices)
    valid_boxes = boxes[valid_indices]
    valid_scores = classes_scores[valid_indices]
    valid_boxes[:,0] -=  0.5 * valid_boxes[:,2]
    valid_boxes[:,1] -= 0.5 * valid_boxes[:,3]
    valid_boxes[:,2] += valid_boxes[:,0]
    valid_boxes[:,3] += valid_boxes[:,1]
    valid_boxes *= scale

    class_ids = np.argmax(valid_scores, axis=1)
    valid_scores = np.max(valid_scores, axis=1)
    result_boxes = cv2.dnn.NMSBoxes(valid_boxes, valid_scores, confidence_threshold, 0.45, 0.5)
    detections = []
    for i in result_boxes:
        detections.append(
            {
                'box': np.round(valid_boxes[i]).astype(np.int32),
                'score': valid_scores[i],
                'class_id': class_ids[i]
            })
    return detections

def visualize_output(image: np.ndarray, detections) -> np.ndarray:
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
        label = f"{CLASSES[class_id - 1]}: {score:.2f}"
        x1, y1, x2, y2 = box
        print(box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2)
    return image

def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes

def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou

# Apply non-maxima suppression to suppress weak, overlapping bounding boxes
CLASSES = {}
for label in LABELS:
    CLASSES[label['id']] = label['name']

colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))
def run_object_detection(image_path, model_path, confidence_threshold=0.2):
    input_shape = (320, 320)  # Adjust the input shape according to your model's requirements

    # Load the model
    session, input_name, output_name = load_model(model_path)

    # Preprocess the input image
    image, original_image, scale = preprocess_image(image_path, input_shape)
    
    for i in range(10):
        start_time = time.time()

        output = session.run([output_name], {input_name: image})[0]
        end_time = time.time()
        file_time = end_time - start_time
        print(f"{file_time*1000:4.0f} seconds")
    file_time = end_time - start_time
    print(f"time needed for file {TEST_IMAGE} is {file_time*1000:4.0f} seconds")
    
    detections = p_to(output, scale)
    cv2.imshow('img', visualize_output(original_image, detections))
    cv2.waitKey()


# Example usage
image_path = TEST_IMAGE
model_path = onnx_path

run_object_detection(image_path, model_path)
