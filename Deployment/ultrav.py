
from Deployment import MIN_SEPARATION_TIME_S
from depths_estimation_and_angle import Detection
import numpy as np
import cv2
from ultralytics.engine.results import Results
from ultralytics import YOLO
from typing import List, Tuple
from time import sleep
import threading
from camera_interaction import vStream, gstreamer_pipeline
# from scripts.Paths import LABELS, paths, TEST_IMAGE  # type: ignore
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../Detection_training/Tensorflow"))

COLOR = (0, 255, 0)  # green

class ThreadedDetection(threading.Thread):

    def __init__(self, frame_capture: vStream, model: os.PathLike, threshold=0.5):
        threading.Thread.__init__(self)
        self.model_path = model
        self.threshold = threshold
        self.frame_capture = frame_capture
        # Load the model
        self.model = YOLO(self.model_path, task='detect', verbose=False)  # type: ignore
        self.detections: Tuple[List[Detection], np.ndarray] = None # type: ignore
        self.daemon = True
        self.read_lock = threading.Lock()
        # make sure frames are available
        while self.frame_capture.get_frame().shape == ():
            sleep(0.1)
        self.start()
        
        # Make sure detection is available
        while not self.get_detections():
            sleep(1)

    @staticmethod
    def results_to_detection(yolo_result: Results) -> Detection:
        box = yolo_result.boxes.data.cpu().flatten().numpy()
        if len(box) > 0:
            return Detection.from_detect(box[0:4].astype(np.int16), box[4], yolo_result.names[int(box[5])])
        return None

    def run(self):
        while True:
            start_time = cv2.getTickCount()
            frame = self.frame_capture.get_frame()
            # pred = self.model.predict(frame, stream=True, verbose=True, conf=self.threshold, imgsz=320)
            pred = self.model(frame, stream=True, verbose=False, conf=self.threshold, imgsz=320, max_det=5)
            # dets = list(map(ThreadedDetection.results_to_detection, pred))
            dets = list(map(ThreadedDetection.results_to_detection, pred))
            with self.read_lock:
                self.detections = (dets, frame)

            sec = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()

            sleep(max(0, MIN_SEPARATION_TIME_S - sec))
            # print(f"Time needed for capture: {sec:.3f}")

    def get_detections(self) -> Tuple[List[Detection], np.ndarray]:
        with self.read_lock:
            det = self.detections
            if det is None:
                return None # type: ignore
            det = (det[0].copy(), det[1].copy())
        return det

    @staticmethod
    def visualize_output(image: np.ndarray, detections: List[Detection]) -> np.ndarray:
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
            label = f"{detection.classification}: {detection.confidence:.2f}"
            cv2.rectangle(image, (detection.xmin, detection.ymin), (detection.xmax, detection.ymax), COLOR, thickness=1)
            cv2.putText(image, label, (detection.xmin, detection.ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, thickness=1)
        return image


MODEL_NAME = 'raccoon_pre_yolov8n_320_B16_ep34_aug_alb4'

# onnx_path = os.path.join(paths.MODEL_PATH, MODEL_NAME, 'weights', 'best.onnx')

# model = YOLO(onnx_path, task='detect')

# im = cv2.imread("C:/dev/random_stuff/Studienarbeit/rac/left2023-06-03 11_22_21.433310.jpg")
# res: List[Results] = model.predict(im, save=False, imgsz=320, verbose=False)
# # xmin, ymin, xmax, ymax = res[0].boxes.xyxy[0].numpy().astype(np.int16)
# yolo_result = map(ThreadedDetection.results_to_detection, res)

if __name__ == '__main__':
    det = ThreadedDetection(vStream(gstreamer_pipeline(flip_method=3)), 'yolov8n.engine')  # type: ignore

    while not det.get_detections():
        sleep(1)

    for i in range(0, 100):
        try:
            dets = det.get_detections()
            fr = det.visualize_output(dets[1], dets[0])
            cv2.imwrite('image.jpg', fr)
            print('Image saved')
            sleep(0.1)
        except:
            continue
