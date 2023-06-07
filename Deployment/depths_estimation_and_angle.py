import numpy as np


# Note because of Flip 3264 is height
TAKEN_RESOLUTION = (3264, 2464)

IMG_RESOLUTION = (320, 320)

FOV = 79.3

DIAG_ANGLE_PER_PIXEL = FOV/np.sqrt(TAKEN_RESOLUTION[0]**2 + TAKEN_RESOLUTION[1]**2)

HOR_PIXEL_ANGLE = DIAG_ANGLE_PER_PIXEL * TAKEN_RESOLUTION[1]
VER_PIXEL_ANGLE = DIAG_ANGLE_PER_PIXEL * TAKEN_RESOLUTION[0]

# print((HOR_PIXEL_ANGLE, VER_PIXEL_ANGLE))

def calculate_distance_and_angles(left_box, right_box,
                                  focal_length=2.96, baseline=150.0, pixel_size=1.12/1000.0*10):
    """
    Calculates the distance, horizontal angle, and vertical angle of an object based on stereo image bounding box coordinates,
    focal length, baseline, and image height.

    Args:
        (x1_left, y1_left, x2_left, y2_left): Bounding box coordinates (top-left and bottom-right) in the left image.
        (x1_right, y1_right, x2_right, y2_right): Bounding box coordinates (top-left and bottom-right) in the right image.
        focal_length: The focal length of the camera.
        baseline: The baseline distance between the stereo camera lenses.
    Returns:
        A tuple containing the distance (in meters), horizontal angle (in degrees), and vertical angle (in degrees) of the object.
    """
    # Calculate object center points in each image
    object_center_left_x = float((left_box[0] + left_box[2])) / 2.0
    object_center_left_y = float((left_box[1] + left_box[3])) / 2.0
    object_center_right_x = float((right_box[0] + right_box[2])) / 2.0
    # object_center_right_y = float((right_box[1] + right_box[3])) / 2.0 should be same as object_center_right_y

    # Calculate disparity (horizontal shift) between the object centers
    disparity = object_center_left_x - object_center_right_x

    # Calculate object distance using triangulation
    distance = (baseline * focal_length) / (disparity * pixel_size)

    # Calculate horizontal angle from the camera's optical axis
    center_right_x_angle = (object_center_right_x/IMG_RESOLUTION[1] * HOR_PIXEL_ANGLE) - HOR_PIXEL_ANGLE/2
    center_left_x_angle = (object_center_left_x/IMG_RESOLUTION[1] * HOR_PIXEL_ANGLE) - HOR_PIXEL_ANGLE/2
    
    # Centered horizontal angle
    horizontal_angle = (center_left_x_angle + center_right_x_angle) / 2
    center_left_y_angle = (object_center_left_y / IMG_RESOLUTION[0] * VER_PIXEL_ANGLE * (-1)) + VER_PIXEL_ANGLE/2

    return distance, horizontal_angle, center_left_y_angle


if __name__ == '__main__':
    # Example usage
    pixel_position_left = (261, 148, 261, 148)  # (x, y, x1, y1) bounding box of the left object
    pixel_position_right = (404-320, 141, 404-320, 141)  # (x, y, x1, y1) bounding box of the right object

    # distance, degree = calculate_distance_degree_size(pixel_position_left, pixel_position_right)
    distance, h, v = calculate_distance_and_angles(pixel_position_left, pixel_position_right)
    print("Distance:", distance/10)
    print("Degree:", v)
    print("hor:", h)
