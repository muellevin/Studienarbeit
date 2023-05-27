
import atexit
import numpy as np
from time import sleep
from camera_setup import vStream, gstreamer_pipeline
from onnx_run_test import CLASSES, ThreadedDetection
from arduino import SERIAL_COM
from depths import calculate_distance_and_angles
from contour_tracking import write_detections_and_image

# initialize cameras:
CAM_LEFT = vStream(gstreamer_pipeline(flip_method=3))
CAM_RIGHT = vStream(gstreamer_pipeline(sensor_id=1, flip_method=1))

# ensures frame is not None
sleep(1)
GRAVITY = 9.81 # m/s²
WATER_SPEED = 1
# enables parallel detection
DETECTION_LEFT = ThreadedDetection(CAM_LEFT, threshold=0.25)
DETECTION_RIGHT = ThreadedDetection(CAM_RIGHT, threshold=0.25)
TARGETER_OFFSET = 400

def main():
    while True:
        left_det, frame_left = DETECTION_LEFT.get_detections()
        right_det, frame_right = DETECTION_RIGHT.get_detections()
        
        num_left_detection = len(left_det)
        num_right_detection = len(right_det)
        
        if num_left_detection > 0 and num_right_detection > 0:
            # normally should be same (e.g. walking from one side does not detect on other size)
            # estimate distance, degree and size of object
            # normally enumeration should be same detection, first is highest confidence level
            distance, horizontal_angle, vertical_angle_object = calculate_distance_and_angles(left_det[0], right_det[0])
            
                # calculate angle for targeter:
            # unknown why *10 is needed
            h_dist = distance/10 * np.tan(np.radians(horizontal_angle))
            horizontal_angle_targeter = np.degrees(np.arctan2(distance/10, TARGETER_OFFSET -  h_dist))
            SERIAL_COM.horizontal_pos = horizontal_angle_targeter
            SERIAL_COM.start_toggle()
            
            # calculate distance for water shot:
            target_distance = np.sqrt((TARGETER_OFFSET -  h_dist**2) + distance**2)
            print(target_distance)
            
            # calculate vertical angle based on distance and relative object vertical orientation
            v_dist = distance/10 * np.tan(np.radians(vertical_angle_object))
            SERIAL_COM.vertical_pos = calculate_vertical_start_angle(target_distance, v_dist)
            
        write_detections_and_image(left_det, frame_left, prefix='left')
        write_detections_and_image(right_det, frame_right, prefix='right')



# ensures clean exit status
def cleanup():
    CAM_LEFT.capture.release()
    CAM_RIGHT.capture.release()


atexit.register(cleanup)

def calculate_vertical_start_angle(target_x, target_y) -> float:
    
    term = WATER_SPEED**4 - GRAVITY * (GRAVITY * target_x**2 + 2 * target_y * WATER_SPEED**2)
    if term < 0:
        # No real solution
        return 0

    angle1 = np.arctan2((WATER_SPEED**2 + np.sqrt(term)), (GRAVITY * target_x))
    # angle2 = np.atan((WATER_SPEED**2 - np.sqrt(term)) / (GRAVITY * x))

    return np.degrees(angle1)#, np.degrees(angle2)
