import csv
import os
from os import PathLike as Path
from .Paths import LABELS

IMAGE_EXTENSIONS = ('.jpg', '.JPG', '.PNG', '.png')

def convert_csv_to_voc(input_file: Path, output_dir: Path):
    classes = {} # dictionary to store class names
    for label in LABELS:
        classes[label['name']] = label['id']

    files = {}
    with open(input_file, "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        data = [row for row in reader]
        for row in data[1:]:
            image_name = os.path.splitext(row[0])[0] # image name
            voc_path = os.path.join(output_dir, image_name + '.xml')
            image_name += '.jpg'
            image_width = int(row[1])
            image_height = int(row[2])
            label = row[3]
            xmin = int(row[4]) # xmin of the bounding box
            ymin = int(row[5]) # ymin of the bounding box
            xmax = int(row[6]) # xmax of the bounding box
            ymax = int(row[7]) # ymax of the bounding box
            if voc_path not in files:
                files[voc_path] = f'<annotation verified="yes">\n'\
                        f'	<folder>{os.path.basename(output_dir)}</folder>\n'\
                        f'	<filename>{image_name}</filename>\n'\
                        f'	<path>{voc_path}</path>\n'\
                        f'	<source>\n'\
                                f'<database>Unknown</database>\n'\
                        f'	</source>\n'\
                        f'	<size>\n'\
                                f'<width>{image_width}</width>\n'\
                                f'<height>{image_height}</height>\n'\
                                f'<depth>3</depth>\n'\
                        f'	</size>\n'\
                        f'	<segmented>0</segmented>\n'\
                        f'	<object>\n'\
                                f'<name>{label}</name>\n'\
                                f'<pose>Unspecified</pose>\n'\
                                f'<truncated>1</truncated>\n'\
                                f'<difficult>0</difficult>\n'\
                                f'<bndbox>\n'\
                                    f'<xmin>{xmin}</xmin>\n'\
                                    f'<ymin>{ymin}</ymin>\n'\
                                    f'<xmax>{xmax}</xmax>\n'\
                                    f'<ymax>{ymax}</ymax>\n'\
                                f'</bndbox>\n'\
                        f'	</object>\n'
            else:
                files[voc_path] += f'	<object>\n'\
                                f'<name>{label}</name>\n'\
                                f'<pose>Unspecified</pose>\n'\
                                f'<truncated>1</truncated>\n'\
                                f'<difficult>0</difficult>\n'\
                                f'<bndbox>\n'\
                                    f'<xmin>{xmin}</xmin>\n'\
                                    f'<ymin>{ymin}</ymin>\n'\
                                    f'<xmax>{xmax}</xmax>\n'\
                                    f'<ymax>{ymax}</ymax>\n'\
                                f'</bndbox>\n'\
                        f'	</object>\n'\

    for keys, value in files.items():
        value += '</annotation>'
        with open(keys, 'w') as voc:
            voc.write(value)
            voc.close()
     