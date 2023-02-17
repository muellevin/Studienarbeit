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

from lxml import etree
import os
import argparse
import imutils
import cv2

parser = argparse.ArgumentParser(
    description="Sample transformator for the given data\nExample usage: python scripts/preprocessing_data.py -r 640 -d Tensorflow/workspace/images/testset/ -o Tensorflow/workspace/images/testset/")
parser.add_argument("-r",
                    "--resize_height",
                    help="Resize height of images and manipulate the .xml-annotation files",
                    type=int, default=0)
parser.add_argument("-d",
                    "--directory",
                    help="directory to images and annotation files",
                    type=str)
parser.add_argument("-o",
                    "--output",
                    help="directory to save resized images and annotation files",
                    type=str, default=None)
parser.add_argument("-g",
                    "--grayscale",
                    help="set if image is set to be grayscaled",
                    type=bool, default=False)


IMAGE_EXTENSIONS = ('.jpg', '.JPG', '.PNG', '.png')

def preprocess_annotations(input_dir: os.path, output_dir: os.path, resize_height: int = 0):

    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith('.xml'):
            print('modifying    ' + file)
            xmlTree = etree.parse(os.path.join(input_dir, file), etree.XMLParser(remove_blank_text=True))
            root = xmlTree.getroot()

            if resize_height > 0:
                resize_annotations(root, resize_height)
            xmlTree.write(os.path.join(output_dir, file), pretty_print=True)

def resize_annotations(root: etree.Element, resize_height: int):
    picture_size = root.find('size')
    height_el = picture_size.find('height')
    width_el = picture_size.find('width')
    if height_el.text != resize_height:
        old_height = float(height_el.text)
        old_width = float(width_el.text)
        ratio = (float(width_el.text)/ old_height)
        height_el.text = str(resize_height)
        new_width = resize_height * ratio
        width_el.text = str(int(new_width))

        for bnd_box in root.findall('object/bndbox'):
            resize_el = bnd_box.find('ymin')
            resize_el.text = str(int(float(resize_el.text) * new_width/old_width))
            resize_el = bnd_box.find('ymax')
            resize_el.text = str(int(float(resize_el.text) * new_width/old_width))
            resize_el =bnd_box.find('xmin')
            resize_el.text = str(int(float(resize_el.text) * (resize_height/old_height)))
            resize_el = bnd_box.find('xmax')
            resize_el.text = str(int(float(resize_el.text) * (resize_height/old_height)))


def resize_images(input_dir: os.path, output_dir: os.path, resize_height: int, grayscale: bool = False):
    os.makedirs(output_dir, exist_ok=True)
    img = None
    for file in os.listdir(input_dir):
        if file.endswith(IMAGE_EXTENSIONS):
            print('resizing     ' + file)
            if grayscale:
                img = cv2.imread(os.path.join(input_dir, file), 0)
            else:
                img = cv2.imread(os.path.join(input_dir, file))

            if resize_height > 0:
                img = imutils.resize(img, height=resize_height)
            cv2.imwrite(os.path.join(output_dir, file), img)


def main():
    args = parser.parse_args()
    if args.directory is None:
        print("No directory was specified")
        return
    output_dir = args.directory + '_resized'
    if args.output is not None:
        output_dir = args.output
    resize_images(args.directory, output_dir, args.resize_height, args.grayscale)
    preprocess_annotations(args.directory, output_dir, args.resize_height)


if __name__ == '__main__':
    main()
