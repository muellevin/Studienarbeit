from dataclasses import dataclass
import numpy as np
from typing import Tuple

TARGETER_OFFSET_X_CM = 40
GRAVITY_M_SS = 9.81  # m/s²
# Note because of Flip 3264 is height
TAKEN_RESOLUTION = (1920, 1080)

IMG_RESOLUTION = (320, 320)

TAKEN_RESOLUTION = IMG_RESOLUTION

RATIO = (TAKEN_RESOLUTION[0]*TAKEN_RESOLUTION[1]) / (IMG_RESOLUTION[0]*IMG_RESOLUTION[1])

FOV = 79.3

DIAG_ANGLE_PER_PIXEL = FOV/np.sqrt(TAKEN_RESOLUTION[0]**2 + TAKEN_RESOLUTION[1]**2)

HOR_PIXEL_ANGLE = DIAG_ANGLE_PER_PIXEL * TAKEN_RESOLUTION[1]
VER_PIXEL_ANGLE = DIAG_ANGLE_PER_PIXEL * TAKEN_RESOLUTION[0]

f_pixel = (320)/np.tan(np.radians(VER_PIXEL_ANGLE))


@dataclass
class Point:
    z: float
    y: float
    x: float


class Detection:

    def __init__(self, xmin: int, ymin: int, xmax: int, ymax: int, confidence: float, classification: str):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.width = xmax - xmin
        self.height = ymax - ymin
        self.confidence = confidence
        self.classification = classification

    @classmethod
    def from_detect(cls, box: Tuple[int, int, int, int], confidence: float, classification: str):
        return Detection(box[0], box[1], box[2], box[3], confidence, classification)


def targeter_angle(projectile_speed: float, left_box: Detection, right_box: Detection) -> Tuple[float, float]:
    """calculate targeter angles for a given projectile speed and target_disparity's

    Args:
        projectile_speed (float): average speed in m/s
        left_box (_type_): left box of target
        right_box (_type_): right box of target

    Returns:
        Tuple[float, float]: Tuple containing horizontal and vertical angles for targeting
    """

    z_distance_cm, x_angle, y_angle = calculate_z_and_xy_angles(left_box, right_box)
    # calculate vertical distance and relative object orientation in 3D space
    x_distance_cm = z_distance_cm * np.tan(np.radians(x_angle))
    y_distance_cm = z_distance_cm * np.tan(np.radians(y_angle))
    # print(f'z {z_distance_cm:.4f} cm, x {x_distance_cm:.4f} cm, y {y_distance_cm:.4f} cm')

    # position_of_camera = Point(distance, v_dist, h_dist)
    # remodify to use m and targeter position
    position_of_targeter_m = Point(z_distance_cm/100, y_distance_cm/100, (TARGETER_OFFSET_X_CM - x_distance_cm)/100)

    # Calculate absolute distance between targeter and target position to make a 2 dimensional problem (schräger Wurf)
    distance_in_x = np.sqrt(position_of_targeter_m.x**2 + position_of_targeter_m.z**2)
    horizontal_angle_targeter = np.arctan2(position_of_targeter_m.z, position_of_targeter_m.x)

    distance_in_y = position_of_targeter_m.y  # Can be relative as target position is already in the 2nd dimension
    vertical_angle = np.arctan((distance_in_y + 0.5 * (GRAVITY_M_SS/projectile_speed**2)
                               * distance_in_x**2) / distance_in_x)

    return np.degrees(horizontal_angle_targeter) - 90, np.degrees(vertical_angle)


def calculate_z_and_xy_angles(left_box: Detection, right_box: Detection,
                              focal_length=2.96, baseline=108.0, pixel_size=1.12*0.001) -> Tuple[float, float, float]:
    """
    Calculates the distance, horizontal angle, and vertical angle of an object based on stereo image bounding box coordinates,
    focal length, baseline, and image height.

    Args:
        (x1_left, y1_left, x2_left, y2_left): Bounding box coordinates (top-left and bottom-right) in the left image.
        (x1_right, y1_right, x2_right, y2_right): Bounding box coordinates (top-left and bottom-right) in the right image.
        focal_length: The focal length of the camera (mm).
        baseline: The baseline distance between the stereo camera lenses.
    Returns:
        A tuple containing the distance (in cm), horizontal angle (in degrees), and vertical angle (in degrees) of the object relative to the center between the cameras.
    """
    # Calculate object center points in each image
    object_center_left_x = float((left_box.xmin + left_box.xmax)) / 2.0
    object_center_left_y = float((left_box.ymin + left_box.ymax)) / 2.0
    object_center_right_x = float((right_box.xmin + right_box.xmax)) / 2.0
    # object_center_right_y = float((right_box[1] + right_box[3])) / 2.0 should be same as object_center_right_y

    # Calculate disparity (horizontal shift) between the object centers
    disparity = abs(object_center_left_x - object_center_right_x)

    # Calculate object distance using triangulation
    disparity = max(1, disparity)   # max distance
    # distance = (baseline * focal_length) / (disparity * pixel_size)
    d_distance = (baseline * 452.55) / disparity    # focal length in pixel? Wi finde ich den das?
    # This seems to be the most correct one
    # dd_distance = (baseline * f_pixel) / disparity    # focal length in pixel? Wi finde ich den das?

    # print(distance, d_distance, dd_distance)

    # Calculate horizontal angle from the camera's optical axis
    center_right_x_angle = (object_center_right_x/IMG_RESOLUTION[1] * HOR_PIXEL_ANGLE) - HOR_PIXEL_ANGLE/2
    center_left_x_angle = (object_center_left_x/IMG_RESOLUTION[1] * HOR_PIXEL_ANGLE) - HOR_PIXEL_ANGLE/2

    # Centered horizontal angle
    horizontal_angle = (center_left_x_angle + center_right_x_angle) / 2
    center_left_y_angle = (object_center_left_y / IMG_RESOLUTION[0] * VER_PIXEL_ANGLE * (-1)) + VER_PIXEL_ANGLE/2

    return d_distance*0.1, horizontal_angle, center_left_y_angle


if __name__ == '__main__':
    # Example usage
    pixel_position_left = Detection(40, 38, 318, 234, 0, 'none')  # (x, y, x1, y1) bounding box of the left object
    pixel_position_right = Detection(0, 38, 280, 234, 0, 'none')  # (x, y, x1, y1) bounding box of the right object

    # calculate angles for water shot:
    target_angles = targeter_angle(10, pixel_position_left, pixel_position_right)

    # vert_angle = calculate_vertical_start_angle(target_distance, v_dist)
    print(
        f"target angle: {target_angles}")
