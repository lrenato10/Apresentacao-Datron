import cv2
import numpy as np

# Load the image
image = cv2.imread('RGB-cubes.jpg')

# Convert the image to HSV (Hue, Saturation, Value) color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

min_s = 110
max_s = 255
min_v = 0
max_v = 255

# Define the range of colors to detect for red, green, and blue
lower_red1 = np.array([0, min_s, min_v])
upper_red1 = np.array([20, max_s, max_v])
lower_red2 = np.array([160, min_s, min_v])
upper_red2 = np.array([180, max_s, max_v])

lower_green = np.array([40, min_s, min_v])
upper_green = np.array([90, max_s, max_v])

lower_blue = np.array([100, min_s, min_v])
upper_blue = np.array([140, max_s, max_v])

# Create masks for each color
mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask_red = cv2.bitwise_or(mask_red1, mask_red2)  # Combine both red ranges

mask_green = cv2.inRange(hsv, lower_green, upper_green)
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

# Create the corresponding color-separated images
red_image = cv2.bitwise_and(image, image, mask=mask_red)
green_image = cv2.bitwise_and(image, image, mask=mask_green)
blue_image = cv2.bitwise_and(image, image, mask=mask_blue)

# Concatenate images to create a 2x2 grid
# First, combine the original and the red image horizontally
top_row = cv2.hconcat([image, red_image])

# Then, combine the green and blue images horizontally
bottom_row = cv2.hconcat([green_image, blue_image])

# Now combine the top and bottom rows vertically
combined_image = cv2.vconcat([top_row, bottom_row])

# Show the combined image with all four images in a 2x2 grid
cv2.imshow('2x2 Grid of Images', combined_image)
cv2.imwrite('segmented-cubes.jpg', combined_image)


# Press any key to close the window
cv2.waitKey(0)
cv2.destroyAllWindows()