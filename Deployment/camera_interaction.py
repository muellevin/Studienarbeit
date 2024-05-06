import threading
from time import sleep
import cv2
from threading import Thread
import atexit
import numpy as np

# sudo service nvargus-daemon restart # Because too much errors occured

""" 
gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
Flip the image by setting the flip_method (most common values: 0 and 2)
display_width and display_height determine the size of each camera pane in the window on the screen
Default 1920x1080 displayd in a 1/4 size window
"""


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=320,
    display_height=320,
    framerate=30,
    flip_method=0,
):
    # return (f"nvarguscamerasrc sensor-id={sensor_id} ! "
    #         "video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080,format=(string)NV12, framerate=(fraction)30/1 ! "
    #         f"nvvidconv flip-method={flip_method} ! "
    #         f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
    #         "videoconvert !"
    #         "appsink drop")

    return (
        f"nvarguscamerasrc saturation=0.5 awblock=true wbmode=5 tnr-mode=2 tnr-strength=1  ee-mode=2 ee-strength=1 sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"videobalance hue=-0.12 contrast=1.1 ! appsink max-time=0.5 max-buffers=1 drop=true"
    )


class vStream:
    def __init__(self, src, max_invalid=100):

        self.capture = cv2.VideoCapture(src, cv2.CAP_GSTREAMER)
        self.frame = np.ndarray([])
        self.max_invalid = max_invalid
        # self.times = []
        atexit.register(self.capture.release)
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.read_lock = threading.Lock()
        self.thread.start()

    def update(self):

        invalid_counter = 0

        while True:
            grabbed, frame = self.capture.read()
            if grabbed and frame is not None:
                invalid_counter = 0
                with self.read_lock:
                    self.frame = frame
            else:
                invalid_counter += 1

            if invalid_counter >= self.max_invalid:
                self.capture.release()
                raise BufferError(f'Failed to read camera data {invalid_counter} times')

    def get_frame(self):
        with self.read_lock:
            frame = self.frame.copy()
        return frame


def show_camera():
    window_title = "CSI Camera"

    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=3), cv2.CAP_GSTREAMER)
    video_capture1 = cv2.VideoCapture(gstreamer_pipeline(flip_method=1, sensor_id=1), cv2.CAP_GSTREAMER)

    if video_capture.isOpened():
        try:
            ret_val, frame = video_capture.read()
            ret_val, frame1 = video_capture1.read()

            myFrame3 = np.hstack((frame, frame1))
            cv2.imwrite('ComboCam.jpg', myFrame3)
            cv2.imwrite("image.jpg", frame)

        finally:
            video_capture.release()
            video_capture1.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")


def take_mul_cam():
    cam1 = vStream(gstreamer_pipeline(flip_method=3, sensor_id=0))
    cam2 = vStream(gstreamer_pipeline(flip_method=1, sensor_id=1))
    while not (cam1.capture.grab() and cam2.capture.grab()):
        sleep(0.1)

    for _ in range(0, 100):
        try:
            myFrame1 = cam1.get_frame()
            # cv2.imwrite('own_images/left' +str(datetime.datetime.now()).replace(':', '_') +'.jpg', myFrame1)
            myFrame2 = cam2.get_frame()
            # cv2.imwrite('own_images/right' +str(datetime.datetime.now()).replace(':', '_') +'.jpg', myFrame2)
            myFrame3 = np.hstack((myFrame1, myFrame2))
            cv2.imwrite('image.jpg', myFrame3)

        except:
            print('frame not available')
        sleep(0.4)
    exit(1)


def check_fps():
    # read is ok
    cam = vStream(gstreamer_pipeline(flip_method=3))
    # cam = CSI_Camera()
    # cam.open(0)
    # cam.start()

    # while not cam.capture.grab():
    #     sleep(0.1)

    for i in range(0, 1000):
        #     # cv2.imshow('Images', cam.getFrame())
        #     # time.sleep(0.01)
        cv2.imwrite('image.jpg', cam.get_frame())
        sleep(0.01)
        # cam.get_frame()
    # cam.stop()
    # cam.release()
    # Calculate the average time
    # tt = cam.times.copy()
    # # Calculate the times
    # fastest_time = min(tt)
    # slowest_time = max(tt)
    # average_time = sum(tt) / len(tt)

    # # Print the results
    # print(f"Interference time stats")
    # print(f"    Average time: {average_time:.4f} seconds. Took: {len(tt)} pictures")
    # print(f"    Fastest time: {fastest_time:.4f} seconds")
    # print(f"    Slowest time: {slowest_time:.4f} seconds")
    # exit(1)


if __name__ == "__main__":
    take_mul_cam()
    # show_camera()
    # check_fps()
