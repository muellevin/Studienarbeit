from glob import glob
import cv2
from typing import Tuple
import time
import os
import xml.etree.ElementTree as ET

import numpy as np

from Detection_training.Tensorflow.scripts.xml_relabel import get_bbox

def template_match(image: np.ndarray, template: np.ndarray, coordinates: Tuple[int, int]) -> Tuple[int, int]:

    # reduce match size as we should have a match on horizontal line
    # match_img = image[coordinates[1]:coordinates[1] + HORIZONTAL_VIEW_PX, :]
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF)

    _, _, _, top_left = cv2.minMaxLoc(result)
    return (top_left[0], coordinates[1])

# pc 1 ms


def main():

    left_imgs = glob(os.path.join(".", "rac", "left*.jpg"))
    right_imgs = glob(os.path.join(".", "rac", "right*.jpg"))
    methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,  # no CCORR IS BROKEN
               cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

    for method in methods[:1]:
        times = []
        for a in range(0, 10):
            for i, img_file in enumerate(left_imgs):

                img_file_match = right_imgs[i]

                xml_file = os.path.splitext(img_file)[0] + '.xml'

                xmlTree = ET.parse(xml_file)
                xml_root = xmlTree.getroot()

                template_img = cv2.imread(img_file)

                detections_bb = xml_root.findall("object")
                match_img = cv2.imread(img_file_match)
                for object in detections_bb:

                    # Display bounding box
                    x, y, x_m, y_m = get_bbox(object)
                    start_time = time.time()
                    h_st = 20
                    match_img_cr = match_img[y:y+h_st, :]

                    detection = template_img[y:y+h_st, x:x_m]
                    # print(f"detection: {x, y, x+w, y+h}")

                    result = cv2.matchTemplate(match_img_cr, detection, method)
                    # Specify a threshold
                    # threshold = 0.45
                    _, _, min_loc, max_loc = cv2.minMaxLoc(result)
                    # Store the coordinates of matched area in a numpy array
                    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
                    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                        top_left = min_loc
                    else:
                        top_left = max_loc
                    w = x_m - x
                    top = (top_left[0], y - 7)
                    bottom_right = (top[0] + w, y_m)
                    # loc = np.where(result >= threshold)  # type: ignore

                    # Draw a rectangle around the matched region.
                    # for pt in zip(*loc[::-1]):
                    #     print(pt)
                    #     # print(f"detection: {pt[0], pt[1], pt[0]+w, pt[1]+h}")
                    #     cv2.rectangle(match_img, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 1)
                    # break
                    end_time = time.time()
                    # Calculate the time taken for this file
                    file_time = end_time - start_time
                    times.append(file_time)

                    # cv2.rectangle(template_img, (x, y), (x_m, y_m), (255, 0, 0), 1)
                    # cv2.rectangle(match_img, top, bottom_right, 255, 1)
                    # both_img = np.hstack((template_img, match_img))
                    # cv2.imwrite('image.jpg', both_img)
                    # cv2.waitKey(10)
                    # time.sleep(0.01)
                    break

        tt = times
        fastest_time = min(tt)
        slowest_time = max(tt)
        average_time = sum(tt) / len(tt)

        # Print the results
        print(f"Interference time stats method {method}")
        print(f"    Average time: {average_time:.5f} seconds. Took: {len(tt)} pictures")
        print(f"    Fastest time: {fastest_time:.5f} seconds")
        print(f"    Slowest time: {slowest_time:.5f} seconds")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
