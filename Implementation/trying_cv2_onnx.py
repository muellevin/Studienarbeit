import argparse


import numpy as np
import sys
import os
sys.path.append("../Detection_training/Tensorflow/")
sys.path.append('/usr/local/lib/python3.6/site-packages')
import cv2.dnn
import time
from scripts.Paths import LABELS, paths, TEST_IMAGE

CLASSES = {}
for label in LABELS:
    CLASSES[label['id']] = label['name']

colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

MODEL_NAME = 'raccoon_yolov8n_320_B16_ep34'

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f'{CLASSES[class_id -1]} ({confidence:.2f})'
    print(label)
    color = colors[class_id -1]
    print(((x, y), (x_plus_w, y_plus_h)))
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main(onnx_model, input_image):
    model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(onnx_model)
    # To test:
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    original_image: np.ndarray = cv2.imread(input_image)
    [height, width, _] = original_image.shape
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image
    scale = length / 320
    
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(320, 320), swapRB=True)
    model.setInput(blob)
    for i in range(10):
        start_time = time.time()

        outputs = model.forward()
        end_time = time.time()
        file_time = end_time - start_time
        print(f"{file_time*1000:4.0f} seconds")
    file_time = end_time - start_time
    print(f"time needed for file {TEST_IMAGE} is {file_time*1000:4.0f} seconds")
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]
    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        # print(classes_scores)
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2], outputs[0][i][3]]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)
    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.99)

    detections = []
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = np.array(boxes[index])
        detection = {
            'class_id': class_ids[index],
            'class_name': CLASSES[class_ids[index]],
            'confidence': scores[index],
            'box': box,
            'scale': scale}
        detections.append(detection)
        x, y, x_plus_w, y_plus_h, = (round(box[0] * scale), round(box[1] * scale),
                          round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale))
        print((x, y, x_plus_w, y_plus_h))
        class_id = class_ids[index]
        label = f'{CLASSES[class_id -1]} ({scores[index]:.2f})'
        print(label)
        color = colors[class_id -1]
        cv2.rectangle(original_image, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(original_image, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imshow('image_teest.jpg', original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return detections


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trt_engine_path = os.path.join(paths.MODEL_PATH, MODEL_NAME, 'export',
                                   'best.onnx')
    parser.add_argument('--model', default=trt_engine_path, help='Input your onnx model.')
    parser.add_argument('--img', default=str(TEST_IMAGE), help='Path to input image.')
    args = parser.parse_args()
    main(args.model, args.img)
