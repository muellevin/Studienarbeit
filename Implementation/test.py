import cv2
import numpy as np
import sys
import os

#Constants
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
dispW = 640
dispH = 480
flip = 2
OBJECT_SIZE_THRESHOLD = dispH*dispW*0.01    # object tracked at 5% display size

#Paths
sys.path.append("../Detection_training/Tensorflow/")
from scripts.Paths import paths

paths.setup_paths()
#Functions
def create_tracker(tracker_type):
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()
    return tracker

def get_camera_settings():
    # Un-comment these next two lines for Pi Camera
    camSet = 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method=' + str(flip) + ' ! video/x-raw, width=' + str(dispW) + ', height=' + str(dispH) + ', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
    # cam = cv2.VideoCapture(camSet)

    # Or, if you have a web camera, uncomment the next line (If it does not work, try setting to '1' instead of '0')
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, dispH)

    return cam

def process_frames():
    #Initialize objects and variables
    cam = get_camera_settings()
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[2]
    # tracker = create_tracker(tracker_type)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    tracked_objects = {}
    next_id = len(os.listdir(paths.SAVED_MOVING))
    
    while True:
        track_objects(fgbg, None, tracked_objects, next_id, cam)
        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


def track_objects(fgbg, tracker, tracked_objects, next_id, cam):
        timer = cv2.getTickCount()
        ret, frame = cam.read()

        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10)))

        cv2.imshow('FGmaskComp',fgmask)
        cv2.moveWindow('FGmaskComp',0,530)

        contours,_=cv2.findContours(fgmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours=sorted(contours,key=lambda x:cv2.contourArea(x),reverse=True)
        for cnt in contours:
            area=cv2.contourArea(cnt)
            if area>=OBJECT_SIZE_THRESHOLD:
                (x,y,w,h)=cv2.boundingRect(cnt)

                # cv2.drawContours(frame,[cnt],0,(255,0,0),3)

                if not tracker:
                    # Initialize tracker
                    tracker = cv2.TrackerKCF_create()
                    ok = tracker.init(frame, (x, y, w, h))
                    obj_id = next_id
                    tracked_objects[obj_id] = tracker
                    next_id += 1
                else:
                    # Update tracker
                    ok, bbox = tracker.update(frame)
                    if ok:
                        # Tracking success
                        x, y, w, h = [int(v) for v in bbox]
                        obj_id = next_id - 1
                        tracked_objects[obj_id] = tracker
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
                    else:
                        # Tracking failure
                        tracker = None

        # Display FPS on frame
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

        # Display the resulting tracking frame
        cv2.imshow('nanoCam',frame)
        cv2.moveWindow('nanoCam',0,0)

        
process_frames()