import cv2
import numpy as np
import sys
import os
from camera_setup import gstreamer_pipeline, vStream
import datetime
from time import sleep
import atexit

sys.path.append(os.path.join(os.path.dirname(__file__), "../Detection_training/Tensorflow/"))
from scripts.Paths import paths

paths.setup_paths()
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

dispW=640
dispH=480
flip=3
OBJECT_SIZE_THRESHOLD = dispH*dispW*0.005    # object tracked at 5% display size

def get_jetson_temp():
    temp_file = "/sys/devices/virtual/thermal/thermal_zone0/temp"
    with open(temp_file, "r") as f:
        temp_string = f.readline().strip()
    return float(temp_string) / 1000.0

def cleanup():
    # Function to be called at exit
    print("Cleaning up...")
    camStreamed.capture.release()
    # cam.release()
    # out.release()
    cv2.destroyAllWindows()


def write_detections_and_image(detections, frame, prefix='img'):
    file = prefix + str(datetime.datetime.now()) + '.xml'
    img_path = os.path.splitext(file)[0] + '.jpg'
    cv2.imwrite(img_path, frame)
    
    objects = ""
    for box in detections:
        bbox = box['box']
        objects += f"""<object>
                        <name>{box['class_name']}</name>
                        <pose>Unspecified</pose>
                        <truncated>1</truncated>
                        <difficult>0</difficult>
                        <bndbox>
                            <xmin>{bbox[0]}</xmin>
                            <ymin>{bbox[1]}</ymin>
                            <xmax>{bbox[2]}</xmax>
                            <ymax>{bbox[3]}</ymax>
                        </bndbox>
                    </object>\n"""
    file_str = f"""<annotation verified="no">
                <folder>{os.path.dirname(file)}</folder>
                <filename>{os.path.basename(img_path)}</filename>
                <path>{img_path}</path>
                <source>
                    <database>Unknown</database>
                </source>
                <size>
                    <width>{320}</width>
                    <height>{320}</height>
                    <depth>3</depth>
                </size>
                <segmented>0</segmented>
                {objects}
            </annotation>"""
    with open(file, "w") as xml_file:
        xml_file.write(file_str)

def write_xml_boxes_and_image(bounding_boxes, file, frame):
    img_path = os.path.splitext(file)[0] + '.jpg'
    cv2.imwrite(img_path, frame)
    
    objects = ""
    for object_number, box in enumerate(bounding_boxes):
        objects += f"""<object>
                        <name>{object_number}</name>
                        <pose>Unspecified</pose>
                        <truncated>1</truncated>
                        <difficult>0</difficult>
                        <bndbox>
                            <xmin>{box[0]}</xmin>
                            <ymin>{box[1]}</ymin>
                            <xmax>{box[0] + box[2]}</xmax>
                            <ymax>{box[1] + box[3]}</ymax>
                        </bndbox>
                    </object>\n"""
    file_str = f"""<annotation verified="no">
                <folder>{os.path.dirname(file)}</folder>
                <filename>{os.path.basename(img_path)}</filename>
                <path>{img_path}</path>
                <source>
                    <database>Unknown</database>
                </source>
                <size>
                    <width>{dispW}</width>
                    <height>{dispH}</height>
                    <depth>3</depth>
                </size>
                <segmented>0</segmented>
                {objects}
            </annotation>"""
    with open(file, "w") as xml_file:
        xml_file.write(file_str)
    

#Uncomment These next Two Line for Pi Camera
# camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
# cam= cv2.VideoCapture(camSet)
# cam = cv2.VideoCapture(gstreamer_pipeline(flip_method=flip, display_width=dispW, display_height=dispH), cv2.CAP_GSTREAMER)
camStreamed = vStream(gstreamer_pipeline(flip_method=flip, display_width=dispW, display_height=dispH, sensor_id=0))
#Or, if you have a WEB cam, uncomment the next line
#(If it does not work, try setting to '1' instead of '0')
# cam=cv2.VideoCapture(0)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, dispH)

# Register the cleanup function with atexit
atexit.register(cleanup)

fgbg = cv2.createBackgroundSubtractorMOG2()
tracked_objects = {}
next_id = int(len(os.listdir(paths.SAVED_MOVING))/2)


# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
# Define the fps to be equal to 10. Also frame size is passed.
# frame_width = int(cam.get(3))
# frame_height = int(cam.get(4))
# out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
def merge_bounding_boxes(boxes, threshold=0):
    """
    Merge bounding boxes that are near or overlapping
    bounding_boxes: list of bounding boxes in the format of (x,y,w,h)
    threshold: overlap threshold to consider two boxes as near or overlapping
    return: a list of merged bounding boxes
    """
    # Convert bounding boxes to numpy array
    boxes = np.array(boxes)
    # Compute the coordinates of the corners of the boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    # Compute the area of the boxes
    area = boxes[:, 2] * boxes[:, 3]
    # Initialize the list of merged boxes
    merged_boxes = []

    # Loop over all boxes
    while boxes.shape[0] > 0:
        #First check if the objects are big enough
        too_small = np.where(area < OBJECT_SIZE_THRESHOLD)[0]
        deleted = 0
        for small in too_small:
            boxes = np.delete(boxes, small - deleted, axis=0)
            area = np.delete(area, small - deleted, axis=0)
            x1 = np.delete(x1, small - deleted, axis=0)
            y1 = np.delete(y1, small - deleted, axis=0)
            x2 = np.delete(x2, small - deleted, axis=0)
            y2 = np.delete(y2, small - deleted, axis=0)
            deleted += 1
        if len(boxes) == 0:
            continue
        # Take the first box and remove it from the list
        box = boxes[0, :]

        # Compute the coordinates of the corners of the box
        b_x1 = box[0]
        b_y1 = box[1]
        b_x2 = box[0] + box[2]
        b_y2 = box[1] + box[3]

        # Compute the intersection over union (IoU) with all remaining boxes
        x1_diff = np.maximum(x1, b_x1)
        y1_diff = np.maximum(y1, b_y1)
        x2_diff = np.minimum(x2, b_x2)
        y2_diff = np.minimum(y2, b_y2)
        w_diff = np.maximum(0, x2_diff - x1_diff)
        h_diff = np.maximum(0, y2_diff - y1_diff)
        inter_area = w_diff * h_diff
        iou = inter_area / (area + (box[2] * box[3]) - inter_area)

        # Find the boxes that have IoU greater than the threshold
        mask = iou > threshold
        indices = np.where(mask)[0]


        # Merge the box with the boxes that have IoU greater than the threshold
        deleted = 0
        if len(boxes) > 0:
            for index in indices:
                box = np.minimum(box, boxes[index - deleted, :])
                box[2] = boxes[index - deleted, 0] + boxes[index - deleted, 2] - box[0]
                box[3] = boxes[index - deleted, 1] + boxes[index - deleted, 3] - box[1]
                boxes = np.delete(boxes, index - deleted, axis=0)
                area = np.delete(area, index - deleted, axis=0)
                x1 = np.delete(x1, index - deleted, axis=0)
                y1 = np.delete(y1, index - deleted, axis=0)
                x2 = np.delete(x2, index - deleted, axis=0)
                y2 = np.delete(y2, index - deleted, axis=0)
                deleted += 1

        # Add the merged box to the list of merged boxes
        merged_boxes.append(box.tolist())
        # boxes = boxes[1:, :]

    return merged_boxes


sleep(1)
start_time = cv2.getTickCount()
fps_sum = 0
fps_count = 0
# while int((cv2.getTickCount() - start_time) / cv2.getTickFrequency()) < 10:
while True:
    jetson_temp = get_jetson_temp()
    if jetson_temp > 60:
        now = datetime.datetime.now()
        date_string = now.strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(os.path.dirname(__file__),"temperature.log"), "a") as f:
            f.write("{}: {:.2f} C\n".format(date_string, jetson_temp))
        sleep(0.5)
    timer = cv2.getTickCount()
    frame = camStreamed.getFrame()
    # _, frame = cam.read()
    # out.write(frame)

    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4)))

    # cv2.imshow('FGmaskComp',fgmask)
    # cv2.moveWindow('FGmaskComp',0,530)
    

    contours,_=cv2.findContours(fgmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours=sorted(contours,key=lambda x:cv2.contourArea(x),reverse=True)
    
    something_big_enough_detected = []
    for cnt in contours:
        # area=cv2.contourArea(cnt)
        # if area>=OBJECT_SIZE_THRESHOLD:
            # something_big_enough_detected.append(np.append(
            #     np.array(cv2.boundingRect(cnt)), [area], axis=0))
        something_big_enough_detected.append(np.array(cv2.boundingRect(cnt)))
            # (x,y,w,h)=cv2.boundingRect(cnt)
            #cv2.drawContours(frame,[cnt],0,(255,0,0),3)
            # cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)

    # Combine intersecting bounding boxes
    if len(something_big_enough_detected) > 0:
        combined_bounding_boxes = merge_bounding_boxes(something_big_enough_detected)
    else:
        combined_bounding_boxes = something_big_enough_detected
    for box in combined_bounding_boxes:
        x, y, w, h = box
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if len(combined_bounding_boxes) > 0:
        write_xml_boxes_and_image(combined_bounding_boxes, os.path.join(paths.SAVED_MOVING, f'Saving_Frame_{str(next_id)}.xml'), frame)
        next_id += 1
        # cv2.imwrite("image.jpg", frame)
    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    fps_sum += fps
    fps_count += 1
    print(f'current fps: {fps}')
    # Display FPS on frame
    # cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
    # Write the frame into the output video
    # out.write(frame)
    # cv2.imshow('nanoCam',frame)
    cv2.moveWindow('nanoCam',0,0)
    if cv2.waitKey(1)==ord('q'):
        break

print(f"average FPS: {fps_sum/fps_count}")
