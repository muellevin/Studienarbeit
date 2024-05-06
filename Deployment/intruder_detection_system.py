
from time import sleep

import cv2
import numpy as np
from Deployment import MIN_SEPARATION_TIME_S
from camera_interaction import vStream, gstreamer_pipeline
# from onnx_object_detector import ThreadedDetection
from Deployment.ultrav import ThreadedDetection
from hardware_interaction import SERIAL_COM
from depths_estimation_and_angle import Detection, targeter_angle
from contour_tracking import write_detections_and_image
from template_matching import template_match
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../Detection_training/Tensorflow"))

from scripts.Paths import paths  # type: ignore

MODEL_NAME = 'raccoon_pre_yolov8n_320_B16_ep34_aug_alb4'

MODEL_PATH = os.path.join(paths.MODEL_PATH, MODEL_NAME, 'weights', 'best.engine')

# initialize cameras:
CAM_RIGHT = vStream(gstreamer_pipeline(flip_method=3))
CAM_LEFT = vStream(gstreamer_pipeline(sensor_id=1, flip_method=1))

while not (CAM_LEFT.capture.grab() and CAM_RIGHT.capture.grab()):
    sleep(0.1)

WATER_SPEED = 10  # m/s
"""Reduces image size in in the vertical direction to reduce latency and resource consumption (CPU/RAM usage)."""
MAX_VERTICAL_TEMPLATE_SIZE: int = 20
# enables parallel detection
DETECTION_LEFT = ThreadedDetection(CAM_LEFT, model=MODEL_PATH, threshold=0.4)  # type: ignore
# DETECTION_RIGHT = ThreadedContourTracker(CAM_RIGHT)


def main():
    while True:
        start_time = cv2.getTickCount()
        left_det, frame_left = DETECTION_LEFT.get_detections()
        # right_det, frame_right = DETECTION_RIGHT.get_detections()

        num_left_detection = len(left_det)
        # num_right_detection = len(right_det)

        if num_left_detection > 0 and left_det[0] is not None:
            # estimate distance, degree and size of object
            frame_right = CAM_RIGHT.get_frame()
            best_detection = left_det[0]

            # reduce image size to small template to improve performance
            # template = frame_left[best_detection.ymin:best_detection.ymin +
            #                       HORIZONTAL_VIEW_PX, best_detection.xmin:best_detection.xmax]
            template = frame_left[best_detection.ymin:best_detection.ymin +
                                  MAX_VERTICAL_TEMPLATE_SIZE, best_detection.xmin:best_detection.xmax]
            right_top = template_match(frame_right[best_detection.ymin:best_detection.ymin +
                                       MAX_VERTICAL_TEMPLATE_SIZE, :], template, (best_detection.xmin, best_detection.ymin))

            right_det = Detection(right_top[0], right_top[1], right_top[0] + best_detection.width,
                                  right_top[1] + best_detection.height, 0, best_detection.classification)

            # calculate angles for water shot:
            target_angle = targeter_angle(WATER_SPEED, best_detection, right_det)
            # send them to Arduino to activate the targeter and repellent system
            SERIAL_COM.horizontal_pos = target_angle[0]
            SERIAL_COM.vertical_pos = target_angle[1]
            SERIAL_COM.start_toggle()

            # Save detection (and template match) to disk. Might be used for further improvements
            write_detections_and_image(left_det, frame_left, prefix='left')
            write_detections_and_image([right_det], frame_right, prefix='right')
            # fr = DETECTION_LEFT.visualize_output(frame_left, left_det)
            # fr_right = DETECTION_LEFT.visualize_output(frame_right, [right_det])
            # cv2.imwrite('image.jpg', np.hstack((fr, fr_right)))
            # print('Image saved')
        else:
            SERIAL_COM.stop()
        sec = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        sleep(max(0, MIN_SEPARATION_TIME_S - sec))


if __name__ == '__main__':
    main()
