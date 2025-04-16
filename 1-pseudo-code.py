import cv2
import numpy as np

# Load the image
image = cv2.imread('RGB-cubes.jpg')

# Convert the image to HSV (Hue, Saturation, Value) color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Constants
min_s = 110
max_s = 255
min_v = 0
max_v = 255

# Define the range of colors to detect for red, green, and blue
lower_limit = np.array([10, min_s, min_v])
upper_limit = np.array([50, max_s, max_v])

# Create masks for each color
mask_color = cv2.inRange(hsv, lower_limit, upper_limit)
cv2.imshow('Color Mask', mask_color)
#cv2.bitwise_or(mask_1, mask_2)

# Create the corresponding color-separated image
color_image = cv2.bitwise_and(image, image, mask=mask_color)

cv2.imshow('Color Image', color_image)

# Press any key to close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
