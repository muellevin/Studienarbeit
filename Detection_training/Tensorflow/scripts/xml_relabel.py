import os
import cv2

from PIL import Image
import xml.etree.ElementTree as ET
from Paths import paths, LABELS


labels = {}
for label in LABELS:
    labels.update({label['id'] : label['name']})

def alreadyAGoalLabel(object) -> bool:
    for element in object:
        if element.tag == 'name':
            if element.text == labels[0]:
                return True
            elif element.text == labels[1]:
                return True
    return False

def getBoundingBox(object):
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

    for file in os.listdir(paths.SAVED_MOVING):
        if file.endswith('.jpg'):
            imgFilename = paths.SAVED_MOVING + '/' + file

            xmlFilename = os.path.splitext(imgFilename)[0] + '.xml'

            xmlTree = ET.parse(xmlFilename)
            rootElement = xmlTree.getroot()

            frame = cv2.imread(imgFilename)

            for object in rootElement.findall("object"):

                # Display bounding box
                x, y, h, w = getBoundingBox(object)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)

                print('If Unkown (remove this BoundingBox), enter [j]')
                cv2.putText(frame, "Name : " + str(os.path.basename(imgFilename)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
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
                            rootElement.remove(object)
                        elif( inputChar == 'q'):
                            exit()
                        else:
                            print('Unknwon input: {}'.format(inputChar))

                # bbox.remove()
                xmlTree.write(xmlFilename)

    cv2.destroyAllWindows()
main()
