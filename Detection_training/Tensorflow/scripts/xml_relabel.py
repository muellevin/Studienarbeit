import os
import cv2

import xml.etree.ElementTree as ET
from Paths import paths, LABELS


labels = {}
for label in LABELS:
    labels.update({label['id'] : label['name']})


def get_bbox(object):
    for element in object:
        if element.tag == 'bndbox':
            x = 0
            y = 0
            width = 0
            height = 0

            for coord in element:
                if coord.tag == "xmin":
                    xmin = int(coord.text)
                elif coord.tag == "ymin":
                    ymin = int(coord.text)
                elif coord.tag == "xmax":
                    xmax = int(coord.text)
                elif coord.tag == "ymax":
                    ymax = int(coord.text)

            x = xmin
            y = ymin
            width = xmax - xmin
            height = ymax - ymin
            return (x, y, height, width)


def main():

    print('If Unkown (remove this BoundingBox), enter [j]')
    print(f'Avaiable Numbers and Labels: {str(labels)}')
    files = os.listdir(paths.SAVED_MOVING)
    print(f'Found {str(len(files)/2)} Images in {paths.SAVED_MOVING}')
    for file in files:
        if file.endswith('.jpg'):
            img_file = os.path.join(paths.SAVED_MOVING, file)

            xml_file = os.path.splitext(img_file)[0] + '.xml'
            if not os.path.exists(xml_file):
                print(f"removing {img_file} because of no xml file found")
                os.remove(img_file)
                continue
            
            xmlTree = ET.parse(xml_file)
            xml_root = xmlTree.getroot()

            frame = cv2.imread(img_file)
            bboxes = xml_root.findall("object")
            for object in bboxes:

                # Display bounding box
                x, y, h, w = get_bbox(object)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)

                cv2.putText(frame, "Name : " + str(os.path.basename(img_file)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
                cv2.imshow('Images', frame)
                
                key = cv2.waitKey(2000)
                if key == -1:
                    continue
                if key >= 48 and key <= 57:  # Check if key is a digit (ASCII codes 48-57)
                    inputChar = int(chr(key))

                else:
                    inputChar = chr(key & 0xFF)

                for element in object:
                    if element.tag  == "name":
                        if inputChar in labels.keys():
                            print(inputChar)
                            element.text = labels[inputChar]
                        elif(inputChar == 'j'):
                            xml_root.remove(object)
                        elif( inputChar == 'q'):
                            exit()
                        else:
                            print('Unknwon input: {}'.format(inputChar))

                # bbox.remove()
            if len(xml_root.findall("object")) <=0:
                    os.remove(xml_file)
                    os.remove(img_file)
            else:
                xmlTree.write(xml_file)

    cv2.destroyAllWindows()
main()
