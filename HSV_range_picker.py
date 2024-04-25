import cv2
import numpy as np

def nothing(x):
    pass

# Create a window
cv2.namedWindow('picker')

# Create trackbars for color range
cv2.createTrackbar('H1','picker',0,179,nothing)
cv2.createTrackbar('S1','picker',0,255,nothing)

cv2.createTrackbar('H2','picker',0,179,nothing)
cv2.createTrackbar('S2','picker',0,255,nothing)

# Create picker image in HSV color space
picker_image = np.zeros((256, 180, 3), dtype=np.uint8)
picker_image[..., 1] = np.linspace(0, 255, 256).reshape(-1, 1)  # Saturation
picker_image[..., 2] = 255  # Value
picker_image[..., 0] = np.arange(180)  # Hue

while True:
    # Convert the HSV image to BGR color space for display
    bgr_image = cv2.cvtColor(picker_image, cv2.COLOR_HSV2BGR)

    # Get the current positions of the trackbars
    h1 = cv2.getTrackbarPos('H1','picker')
    s1 = cv2.getTrackbarPos('S1','picker')
    h2 = cv2.getTrackbarPos('H2','picker')
    s2 = cv2.getTrackbarPos('S2','picker')

    # Create a rectangle outline on top of the image
    cv2.rectangle(bgr_image, (h1, s1), (h2, s2), (0, 0, 0), 1)

    # Display the image
    # Scale the image by 4
    scaled_image = cv2.resize(bgr_image, (0, 0), fx=4, fy=2)
    # Display the image
    cv2.imshow('picker', scaled_image)

    if cv2.waitKey(1) == 27:  # Check if escape key is pressed
        break

cv2.destroyAllWindows()