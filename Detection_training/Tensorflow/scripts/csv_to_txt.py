import csv
import os
from os import PathLike as Path
import shutil
from .Paths import LABELS

IMAGE_EXTENSIONS = ('.jpg', '.JPG', '.PNG', '.png')

def convert_folder_to_yolov8(input_dir: Path, output_dir: Path):
    classes = {} # dictionary to store class names
    for label in LABELS:
        classes[label['name']] = label['id']

    shutil.rmtree(output_dir, ignore_errors=False, onerror=None)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    with open(os.path.join(input_dir, "detections.csv"), "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        data = [row for row in reader]
        for row in data[1:]:
            image_path = os.path.splitext(row[0])[0] # image name
            image_width = int(row[1])
            image_height = int(row[2])
            label = row[3]
            xmin = int(row[4]) # xmin of the bounding box
            ymin = int(row[5]) # ymin of the bounding box
            xmax = int(row[6]) # xmax of the bounding box
            ymax = int(row[7]) # ymax of the bounding box
            
            class_id = classes[label]
            
            # calculate the center and width and height of the bounding box
            width = xmax - xmin
            height = ymax - ymin
            x_center = xmin + (width / 2)
            y_center = ymin + (height / 2)
            
            # normalize the values between 0 and 1
            x_center /= image_width
            y_center /= image_height
            width /= image_width
            height /= image_height
            with open(os.path.join(output_dir, 'labels', image_path + '.txt'), "a") as output_file:
                # write the converted data to the output file
                output_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
    
    # write the class names to a separate file
    with open(os.path.join(output_dir, "classes.txt"), "w") as classes_file:
        for class_name, class_id in classes.items():
            classes_file.write(f"{class_name}\n")
    
    # Loop through each file in the source folder
    for filename in os.listdir(input_dir):
        # Check if the file has the desired extension
        if filename.endswith(IMAGE_EXTENSIONS):
            # Copy the file to the destination folder
            shutil.copyfile(os.path.join(input_dir, filename), os.path.join(output_dir, 'images', filename))
    
