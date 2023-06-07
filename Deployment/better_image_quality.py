import cv2
import numpy as np

# image = cv2.imread('image.jpg')

# Load the infrared image
img = cv2.imread('image n.jpg')
com = cv2.imread('test.jpg')


# Convert the image to the HSV color space
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# # Adjust the color balance by modifying the hue and saturation channels
# hsv[:, :, 0] += 13  # Increase hue by 10 degrees
# # hsv[:, :, 1] += 13  # Decrease saturation by 30 units
# # hsv[:, :, 2] -= 10  # Decrease saturation by 30 units

# # Reduce the intensity of the red channel
# v_channel = hsv[:, :, 2]
# v_channel_norm = cv2.normalize(v_channel, None, 0, 255, cv2.NORM_MINMAX)
# hsv[:, :, 2] = v_channel_norm

# # Convert the image back to the BGR color space
# modified = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Convert the image to LAB color space
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
  
# Split the LAB image into separate channels
l, a, b = cv2.split(lab)
  
# Apply CLAHE to the L channel
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16,16))
l = clahe.apply(l)
  
# Merge the LAB channels back together
lab = cv2.merge((l,a,b))
  
# Convert the LAB image back to RGB color space
labed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
# Convert the image to the HSV color space
hsv = cv2.cvtColor(labed, cv2.COLOR_BGR2HSV)

# Adjust the color balance by modifying the hue and saturation channels
# hsv[:, :, 0] += 11  # Increase hue by 10 degrees
# hsv[:, :, 1] += 20  # Decrease saturation by 30 units


# Reduce the intensity of the red channel
v_channel = hsv[:, :, 2]
v_channel_norm = cv2.normalize(v_channel, None, 0, 255, cv2.NORM_MINMAX)
hsv[:, :, 2] = v_channel_norm

# Convert the image back to the BGR color space
modified = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Display the original and modified images side by side
output = cv2.hconcat([img, labed, modified, com])
cv2.imwrite("com.jpg", output)
cv2.imshow('Original vs Modified', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
