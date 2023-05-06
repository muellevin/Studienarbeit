import csv
import os
from os import PathLike as Path
import shutil
from collections import namedtuple
from typing import List
from .Paths import LABELS

IMAGE_EXTENSIONS = ('.jpg', '.JPG', '.PNG', '.png')
CSV_DETECTION = namedtuple("detection", "image_path image_width image_height label xmin ymin xmax ymax")
classes = {} # dictionary to store class names
for label in LABELS:
    classes[label['name']] = label['id']


def remove_and_create_folders(output_dir: Path):
    shutil.rmtree(output_dir, ignore_errors=False, onerror=None)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)

def convert_folder_to_yolo(input_dir: Path, output_dir: Path):
    
    remove_and_create_folders(output_dir)
    detections = parse_csv_file(input_dir)
    for det in detections:
            class_id = classes[det.label]

            # calculate the center and width and height of the bounding box
            width = det.xmax - det.xmin
            height = det.ymax - det.ymin
            x_center = det.xmin + (width / 2)
            y_center = det.ymin + (height / 2)

            # normalize the values between 0 and 1
            x_center /= det.image_width
            y_center /= det.image_height
            width /= det.image_width
            height /= det.image_height
            
            out_file = os.path.join(output_dir, 'labels', det.image_path + '.txt')
            out_val = f"{class_id} {x_center} {y_center} {width} {height}\n"
            write_output_file(out_file, out_val)

    # write the class names to a separate file
    with open(os.path.join(output_dir, "classes.txt"), "w") as classes_file:
        for class_name, class_id in classes.items():
            classes_file.write(f"{class_name}\n")
    copy_images(input_dir, output_dir)

def parse_csv_file(input_dir: Path) -> List[CSV_DETECTION]:
    
    parsed_csv: List[CSV_DETECTION] = []
    with open(os.path.join(input_dir, "detections.csv"), "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        data = [row for row in reader]
        for row in data[1:]:
            parsed_csv.append(
                CSV_DETECTION(
                    image_path = os.path.splitext(row[0])[0], # image name
                    image_width = int(row[1]),
                    image_height = int(row[2]),
                    label = row[3],
                    xmin = int(row[4]), # xmin of the bounding box
                    ymin = int(row[5]), # ymin of the bounding box
                    xmax = int(row[6]), # xmax of the bounding box
                    ymax = int(row[7]) # ymax of the bounding box
                    )
                )
    return parsed_csv
            

def write_output_file(file_path: Path, file_value: str, mode: str='a'):
    with open(file_path, mode) as output_file:
        # write the converted data to the output file
        output_file.write(file_value)
        output_file.close()

def copy_images(input_dir: Path, output_dir: Path):
    # Loop through each file in the source folder
    for filename in os.listdir(input_dir):
        # Check if the file has the desired extension
        if filename.endswith(IMAGE_EXTENSIONS):
            # Copy the file to the destination folder
            shutil.copyfile(os.path.join(input_dir, filename),
                            os.path.join(output_dir, 'images',
                                         filename))
