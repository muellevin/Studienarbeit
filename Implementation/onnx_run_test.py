
import time
import sys
sys.path.append('/home/levin/.local/lib/python3.6/site-packages')
import os
import cv2
import threading
import onnxruntime as ort
import numpy as np
from camera_setup import vStream

sys.path.append(os.path.join(os.path.dirname(__file__), "../Detection_training/Tensorflow/"))
from scripts.Paths import LABELS, paths, TEST_IMAGE

MODEL_NAME = 'raccoon_yolov8n_320_B16_ep34'

onnx_path = os.path.join(paths.MODEL_PATH, MODEL_NAME, 'export',
                                   'best.onnx')


EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
# np.set_printoptions(threshold=np.inf)

MODEL_SHAPE = (320, 320)

class ThreadedDetection(threading.Thread):
    
    def __init__(self, frame_capture: vStream, model=onnx_path, threshold=0.25):
        threading.Thread.__init__(self)
        self.model = model
        self.threshold = threshold
        self.frame_capture = frame_capture
        # Load the model
        self.session, self.input_name, self.output_name = load_model(self.model)
        self.detections = [{}, None]
        self.daemon = True
        self.start()

    def run(self):
        while True:
            start_time = cv2.getTickCount()
            frame = self.frame_capture.getFrame()
            if frame is not None:
                blob_image = preprocess_frame(frame)
            
                # run interference
                output = self.session.run([self.output_name], {self.input_name: blob_image})[0]
            
                self.detections[0] = postprocess_onnx(output, confidence_threshold=self.threshold)
                self.detections[1] = frame
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - start_time)
                print(f'current fps: {fps}')

    def get_detections(self) -> dict:
        return self.detections


def load_model(model_path):
    # Load the ONNX model
    session = ort.InferenceSession(model_path, providers=EP_list)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session, input_name, output_name


def preprocess_image(image_path, input_shape):
    # Load and preprocess the input image
    original_image: np.ndarray = cv2.imread(image_path)
    original_image = cv2.resize(original_image, input_shape)
    [height, width, _] = original_image.shape
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image
    scale = length / 320

    image = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=MODEL_SHAPE, swapRB=True)
    return image, original_image, scale

# To test
def preprocess_frame(frame: np.ndarray):
    image = cv2.dnn.blobFromImage(frame, scalefactor=1 / 255, size=MODEL_SHAPE, swapRB=True)
    return image

def postprocess_onnx(outputs, scale=1, confidence_threshold=0.05):
    outputs = np.squeeze(outputs).T
    boxes = outputs[:,:4]
    classes_scores = outputs[:, 4:]
    valid_indices = np.where(classes_scores > confidence_threshold)[0]
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
                'class_name': CLASSES[class_ids[i] - 1]
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
        class_id = detection['class_name']
        color = (0, 255, 0)  # green
        label = f"{class_id}: {score:.2f}"
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2)
    return image

CLASSES = {}
for label in LABELS:
    CLASSES[label['id']] = label['name']

def run_object_detection(image_path, model_path, confidence_threshold=0.25):
    input_shape = (320, 320)  # Adjust the input shape according to your model's requirements

    # Load the model

    session, input_name, output_name = load_model(model_path)
    # Preprocess the input image

    for i in range(20):
        start_time = cv2.getTickCount()
        image, original_image, scale = preprocess_image(image_path, input_shape)
        output = session.run([output_name], {input_name: image})[0]
    
        detections = postprocess_onnx(output, scale, confidence_threshold=confidence_threshold)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - start_time)
        print(f'current fps: {fps}')
    cv2.imwrite('img.jpg', visualize_output(original_image, detections))
    cv2.waitKey()


def run_object_detection_on_cam(model_path, confidence_threshold=0.25):
    from camera_setup import vStream, gstreamer_pipeline
    CAM_LEFT = vStream(gstreamer_pipeline(flip_method=3))
    # CAM_RIGHT = vStream(gstreamer_pipeline(sensor_id=1, flip_method=1))
    # Load the model
    time.sleep(1)
    testi = ThreadedDetection(CAM_LEFT)
    # testi_2 = ThreadedDetection(CAM_RIGHT)
    start_time_t = cv2.getTickCount()
    # test for 10 seconds
    while int((cv2.getTickCount() - start_time_t) / cv2.getTickFrequency()) < 10:
        detections, frame = testi.get_detections()
        # detections_2, frame_2 = testi_2.get_detections()
        time.sleep(0.05)
    # detections_2, frame_2 = testi_2.get_detections()
    CAM_LEFT.capture.release()
    # CAM_RIGHT.capture.release()
    vis = visualize_output(frame, detections)
    # vis_2 = visualize_output(frame_2, detections_2)
    # myFrame3=np.hstack((vis, vis_2))
    cv2.imwrite('img.jpg', vis)
    # cv2.waitKey()

# Example usage
image_path = TEST_IMAGE
model_path = onnx_path
if __name__ == '__main__':
    # run_object_detection(image_path, model_path)
    run_object_detection_on_cam(model_path)
