# MIT License
# Copyright (c) 2019-2022 JetsonHacks

# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV
# Drivers for the camera and OpenCV are included in the base image

import cv2

""" 
gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
Flip the image by setting the flip_method (most common values: 0 and 2)
display_width and display_height determine the size of each camera pane in the window on the screen
Default 1920x1080 displayd in a 1/4 size window
"""

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=320,
    display_height=320,
    framerate=120,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
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


def show_camera():
    window_title = "CSI Camera"

    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=3), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        try:
            ret_val, frame = video_capture.read()
            # Convert the frame to HSV format
            # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # # Adjust the hue channel by subtracting a value from it
            # hue_adjust = 0  # Example: decrease hue by 50
            # hsv[:, :, 0] -= hue_adjust

            # # Convert the frame back to BGR format
            # frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imwrite("image.jpg", frame)
            # window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            # while True:
            #     ret_val, frame = video_capture.read()
            #     # Check to see if the user closed the window
            #     # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
            #     # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
            #     if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
            #         cv2.imshow(window_title, frame)
            #     else:
            #         break 
            #     keyCode = cv2.waitKey(10) & 0xFF
            #     # Stop the program on the ESC key or 'q'
            #     if keyCode == 27 or keyCode == ord('q'):
            #         break
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")


if __name__ == "__main__":
    show_camera()
