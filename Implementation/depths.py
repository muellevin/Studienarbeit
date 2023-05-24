import math

def calculate_distance_degree_size(pixel_position_left, pixel_position_right, baseline_distance=100, focal_length=2.96, fov_deg=77):
    # Calculate pixel disparity
    disparity = abs(pixel_position_left[0] - pixel_position_right[0])

    # Calculate distance using triangulation
    distance = (baseline_distance * focal_length) / disparity

    # Calculate degree
    pixel_center = (pixel_position_left[0] + pixel_position_right[0]) / 2
    degree = math.degrees(math.atan((pixel_center - focal_length) / focal_length))

    # Approximate object size ratio based on bounding box dimensions
    size_ratio = pixel_position_right[2] / pixel_position_left[2]

    return distance, degree, size_ratio


if __name__ == '__main__':
    # Example usage
    pixel_position_left = (100, 200, 50, 80)  # (x, y, w, h) bounding box of the left object
    pixel_position_right = (150, 200, 80, 80)  # (x, y, w, h) bounding box of the right object

    distance, degree, size_ratio = calculate_distance_degree_size(pixel_position_left, pixel_position_right)
    print("Distance:", distance)
    print("Degree:", degree)
    print("Size Ratio:", size_ratio)
