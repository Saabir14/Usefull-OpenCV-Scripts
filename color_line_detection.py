import cv2
import numpy as np

# import the OpenCV library 

def color_line_detection_camera():
    """
    Perform color line detection using the camera.

    This function creates a window with trackbars to adjust the color range for line detection.
    It captures frames from the camera, blurs them, converts them to HSV color space, and applies a color filter based on the trackbar values.
    The resulting binary mask is displayed in the window until the 'esc' button is pressed.
    """
    # Create a window
    cv2.namedWindow('color_line_detection')

    # Create trackbar for blur
    cv2.createTrackbar('Blur', 'color_line_detection', 15, 100, lambda _: None)
    
    # Create trackbar for color change
    # Hue is from 0-180 for OpenCV
    cv2.createTrackbar('HLower', 'color_line_detection', 0, 180, lambda _: None)
    cv2.createTrackbar('SLower','color_line_detection',0,255, lambda _: None)
    cv2.createTrackbar('VLower','color_line_detection',0,255, lambda _: None)
    cv2.createTrackbar('HUpper','color_line_detection',180,180,lambda _: None)
    cv2.createTrackbar('SUpper','color_line_detection',35,255,lambda _: None)
    cv2.createTrackbar('VUpper','color_line_detection',100,255,lambda _: None)

    # Create trackbar for dilate
    cv2.createTrackbar('Dilate', 'color_line_detection', 0, 100, lambda _: None)

    # Open the camera
    cap = cv2.VideoCapture(0)
    
    while(True):
        # Capture frame-by-frame
        ret, feed = cap.read()

        if not ret:
            break

        if blur := cv2.getTrackbarPos('Blur', 'color_line_detection'):
            # Blur frame to remove unnecessary details
            frame = cv2.blur(feed, (blur, blur))

        # Convert the image frame to HSV
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Get the new values of the trackbar in real time as the user changes them
        hLower = cv2.getTrackbarPos('HLower','color_line_detection')
        sLower = cv2.getTrackbarPos('SLower','color_line_detection')
        vLower = cv2.getTrackbarPos('VLower','color_line_detection')
        hUpper = cv2.getTrackbarPos('HUpper','color_line_detection')
        sUpper = cv2.getTrackbarPos('SUpper','color_line_detection')
        vUpper = cv2.getTrackbarPos('VUpper','color_line_detection')

        # Set the lower and upper HSV range according to the value selected by the trackbar
        lower_range = (hLower, sLower, vLower)
        upper_range = (hUpper, sUpper, vUpper)

        # Filter the image and get the binary mask, where white represents your target color
        frame = cv2.inRange(frame, lower_range, upper_range)

        # Dilate the mask
        frame = cv2.dilate(frame, None, iterations=cv2.getTrackbarPos('Dilate', 'color_line_detection'))

        # Display the resulting frame
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        cv2.imshow('color_line_detection', np.hstack((feed, frame)))
        
        # the 'esc' button is set as the quitting button
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    # Release the camera and destroy all the windows 
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    color_line_detection_camera()