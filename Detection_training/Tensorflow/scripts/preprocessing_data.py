""" Sample transformator for the given data

usage: preprocessing_data.py [-c MAX_IMAGE_HEIGHT] [-d PATH] [-o PATH]
optional arguments:
  -r MAX_IMAGE_HEIGHT, --resize_height MAX_IMAGE_HEIGHT
                        Resize height of images and manipulate the .xml-annotation files
  -d PATH, --directory PATH,
                        directory to images and annotation files
  -o PATH, --output PATH,
                        directory to save resized images and annotation file.
"""

import os
import argparse
import imutils
import cv2

parser = argparse.ArgumentParser(
    description="Sample transformator for the given data\nExample usage: python scripts/preprocessing_data.py -r 640 -d Tensorflow/workspace/images/testset/ -o Tensorflow/workspace/images/testset/")
parser.add_argument("-r",
                    "--resize_height",
                    help="Resize height of images and manipulate the .csv-annotation files",
                    type=int, default=0)
parser.add_argument("-d",
                    "--directory",
                    help="directory to images and csv file",
                    type=str)
parser.add_argument("-o",
                    "--output",
                    help="directory to save resized images and annotation file",
                    type=str, default=None)
parser.add_argument("-g",
                    "--grayscale",
                    help="set if image is set to be grayscaled",
                    type=bool, default=False)


IMAGE_EXTENSIONS = ('.jpg', '.JPG', '.PNG', '.png')


def resize_images(input_dir: os.path, output_dir: os.path, resize_height: int, grayscale: bool = False):
    os.makedirs(output_dir, exist_ok=True)
    img = None
    counter = 0
    for file in os.listdir(input_dir):
        if file.endswith(IMAGE_EXTENSIONS):
            counter += 1
            if grayscale:
                img = cv2.imread(os.path.join(input_dir, file), 0)
            else:
                img = cv2.imread(os.path.join(input_dir, file))

            if resize_height > 0:
                img = imutils.resize(img, height=resize_height)
            cv2.imwrite(os.path.join(output_dir, file), img)
    print("Resized {} images with option grayscale={}".format(counter, grayscale))


def main():
    args = parser.parse_args()
    if args.directory is None:
        print("No directory was specified")
        return
    output_dir = args.directory + '_resized'
    if args.output is not None:
        output_dir = args.output
    resize_images(args.directory, output_dir, args.resize_height, args.grayscale)


if __name__ == '__main__':
    main()
