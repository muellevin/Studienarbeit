import argparse
import csv
import os
import shutil

class Detection:
    def __init__(self, filename, width, height, class_, xmin, ymin, xmax, ymax):
        self.filename = filename
        self.width = int(width)
        self.height = int(height)
        self.class_ = class_
        self.xmin = int(xmin)
        self.ymin = int(ymin)
        self.xmax = int(xmax)
        self.ymax = int(ymax)

    def resize_height(self, new_height):
        if new_height != self.height:
            img_ratio = self.width / self.height
            new_width = new_height * img_ratio
            width_ratio = new_width / self.width
            height_ratio = new_height / self.height

            self.ymin = int(self.ymin * width_ratio)
            self.ymax = int(self.ymax * width_ratio)
            self.xmin = int(self.xmin * height_ratio)
            self.xmax = int(self.xmax * height_ratio)

            self.width = int(new_width)
            self.height = int(new_height)

parser = argparse.ArgumentParser(description='Resize detection in csv file')
parser.add_argument('input_file', type=str, help='Path to the input file')
parser.add_argument('output_file', type=str, help='Path to the output file')
parser.add_argument('resize_height', type=int, help='New height to resize detections to')


def resize_csv(input_file: str, output_file: str, resize_height: int):

    # Clean output directory from all files
    output_dir = os.path.dirname(output_file)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Write the list of available detections to the 'detections.csv' file in the output directory. If it exists it will join them.
    with open(input_file, newline='') as csvfile, open(output_file, 'w', newline='') as outfile:
        csvwriter = csv.writer(outfile)
        csvwriter.writerow(['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
        csvreader = csv.reader(csvfile)
        next(csvreader)  # skip header row
        for row in csvreader:
            record = Detection(*row)
            record.resize_height(resize_height)
            csvwriter.writerow([record.filename, record.width, record.height, record.class_, record.xmin, record.ymin, record.xmax, record.ymax])
            

if __name__ == "__main__":
    args = parser.parse_args()
    resize_csv(args.input_file, args.output_file, args.resize_height)
