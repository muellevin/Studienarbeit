# MIT License
# Copyright (c) 2019-2022 JetsonHacks

# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV
# Drivers for the camera and OpenCV are included in the base image

import cv2
from threading import Thread
import time
import numpy as np

""" 
gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
Flip the image by setting the flip_method (most common values: 0 and 2)
display_width and display_height determine the size of each camera pane in the window on the screen
Default 1920x1080 displayd in a 1/4 size window
"""

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=3264,
    capture_height=2464,
    display_width=320,
    display_height=320,
    framerate=21,
    flip_method=0,
):
    return (
        "nvarguscamerasrc saturation=0.5 awblock=true wbmode=5 tnr-mode=2 tnr-strength=1  ee-mode=2 ee-strength=1 sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "videobalance hue=-0.12 contrast=1.1 ! appsink drop"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
    # return (
    #     "nvarguscamerasrc saturation=1 awblock=false wbmode=1 tnr-mode=1 tnr-strength=-1  ee-mode=1 ee-strength=-1 sensor-id=%d !"
    #     "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
    #     "nvvidconv flip-method=%d ! "
    #     "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
    #     "videoconvert ! "
    #     "video/x-raw, format=(string)BGR ! "
    #     "appsink"
    #     % (
    #         sensor_id,
    #         capture_width,
    #         capture_height,
    #         framerate,
    #         flip_method,
    #         display_width,
    #         display_height,
    #     )
    # )


class vStream:
    def __init__(self,src):

        self.capture=cv2.VideoCapture(src, cv2.CAP_GSTREAMER)
        # self.update()
        self.thread=Thread(target=self.update,args=())
        self.thread.daemon=True
        self.thread.start()
    def update(self):
        while True:
            _,self.frame=self.capture.read()

    def getFrame(self):
        return self.frame

def show_camera():
    window_title = "CSI Camera"

    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=3), cv2.CAP_GSTREAMER)
    video_capture1 = cv2.VideoCapture(gstreamer_pipeline(flip_method=1, sensor_id=1), cv2.CAP_GSTREAMER)

    if video_capture.isOpened():
        try:
            ret_val, frame = video_capture.read()
            ret_val, frame1 = video_capture1.read()

            myFrame3=np.hstack((frame,frame1))
            cv2.imwrite('ComboCam.jpg',myFrame3)
            cv2.imwrite("image.jpg", frame)

        finally:
            video_capture.release()
            video_capture1.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")


def take_mul_cam():
    cam1=vStream(gstreamer_pipeline(flip_method=3, sensor_id=0))
    cam2=vStream(gstreamer_pipeline(flip_method=1, sensor_id=1))
    font=cv2.FONT_HERSHEY_SIMPLEX
    startTime=time.time()
    dtav=0
    i = 0
    if True:
        try:
            myFrame1=cam1.getFrame()
            cv2.imwrite('ComboCam.jpg',myFrame1)
            myFrame2=cam2.getFrame()
            cv2.imwrite('ComboCam.jpg',myFrame2)
            myFrame3=np.hstack((myFrame1,myFrame2))
            cv2.imwrite('ComboCam.jpg',myFrame3)
    
    
    
        except:
            print('frame not available')
            
    cam1.capture.release()
    cam2.capture.release()
    cv2.destroyAllWindows()
    exit(1)

if __name__ == "__main__":
    take_mul_cam()
    # show_camera()
