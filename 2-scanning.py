import cv2
import numpy as np

def order_points(pts):
    # Initialize a list of coordinates that will be ordered
    rect = np.zeros((4, 2), dtype="float32")

    # Sum and diff to find corners
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]       # Top-left
    rect[2] = pts[np.argmax(s)]       # Bottom-right
    rect[1] = pts[np.argmin(diff)]    # Top-right
    rect[3] = pts[np.argmax(diff)]    # Bottom-left

    return rect

# Load the image
image = cv2.imread('Prova SCHP1.jpeg')

# Resize the image to make it easier to work with
height, width = image.shape[:2]
aspect_ratio = width / height
new_width = 800
new_height = int(new_width / aspect_ratio)
resized_image = cv2.resize(image, (new_width, new_height))

# Convert to grayscale
gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise and improve edge detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Use Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort the contours by area, in descending order
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Loop through the contours to find the document contour (the largest one)
document_contour = None
for contour in contours:
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    # the contours is a quadrilateral
    if len(approx) == 4:
        document_contour = approx
        break

# If the document contour is found
if document_contour is not None:
    # Draw the contour on the original image
    cv2.drawContours(resized_image, [document_contour], -1, (0, 255, 0), 2)

    # Apply perspective transform to get the scanned version of the document
    rect = order_points(document_contour.reshape(4, 2))
    width = int(np.linalg.norm(rect[0] - rect[1]) + np.linalg.norm(rect[2] - rect[3]))
    height = int(np.linalg.norm(rect[1] - rect[2]) + np.linalg.norm(rect[3] - rect[0]))

    # Order the points for perspective transformation
    ordered_rect = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype='float32')

    # Perform the perspective transformation
    matrix = cv2.getPerspectiveTransform(rect, ordered_rect)
    scanned_document = cv2.warpPerspective(resized_image, matrix, (width, height))

    # Show the original and scanned images
    cv2.imshow("Scanned Document", scanned_document)
    cv2.imwrite("scanned-document.jpg", scanned_document)
    cv2.imshow("Dedected Paper", resized_image)
    cv2.imshow("Edges Image", edges)
    cv2.imshow("Blurred Image", blurred)
    cv2.imshow("Gray Image", gray)
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("No document contour found!")