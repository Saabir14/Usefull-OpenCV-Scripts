# import the OpenCV library 
import cv2
import os
import numpy as np
from itertools import count
import time

def color_line_detection_single_frame_weighted_average():
    """
    Perform color line detection on a single frame.

    This function creates a window with trackbars to adjust the color range for line detection.
    It loads an image, blurs it, converts it to HSV color space, and applies a color filter based on the trackbar values.
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
    cv2.createTrackbar('Dilate', 'color_line_detection', 35, 100, lambda _: None)

    # Trackbar for skipping columns
    cv2.createTrackbar('Skip Columns', 'color_line_detection', 1, 100, lambda _: None)
    # Trackbar for skipping rows
    cv2.createTrackbar('Skip Rows', 'color_line_detection', 1, 100, lambda _: None)

    # Load the image
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, 'black_line.jpeg')
    image = cv2.imread(image_path)
    
    for frame_number in count(1):
        frame = image.copy()

        start_time = time.time()

        if blur := cv2.getTrackbarPos('Blur', 'color_line_detection'):
            # Blur frame to remove unnecessary details
            frame = cv2.blur(frame, (blur, blur))

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

        # Calculate the weighted average
        sum = 0
        total = 0

        if skipRows := cv2.getTrackbarPos('Skip Rows', 'color_line_detection'): pass
        else: skipRows = 1; cv2.setTrackbarPos('Skip Rows', 'color_line_detection', 1)
        if skipCols := cv2.getTrackbarPos('Skip Columns', 'color_line_detection'): pass
        else: skipCols = 1; cv2.setTrackbarPos('Skip Columns', 'color_line_detection', 1)

        # Assuming frame is a numpy array
        rows = np.arange(frame.shape[0])
        cols = np.arange(frame.shape[1])

        # Select rows and columns based on skipRows and skipCols
        selected_rows = rows[rows % skipRows == 0]
        selected_cols = cols[cols % skipCols == 0]

        # Create a grid of selected rows and columns
        selected_rows, selected_cols = np.meshgrid(selected_rows, selected_cols)

        # Calculate total
        total = np.sum(selected_cols / 2)

        # Calculate sum where frame value is not 0
        mask = frame[selected_rows, selected_cols] != 0
        sum = np.sum(selected_cols[mask] - frame.shape[1] // 2)

        w = sum/total if total != 0 else 0

        end_time = time.time()
        frame_time = end_time - start_time
        frame_rate = 1.0 / frame_time if frame_time > 0 else 0
        
        # Convert the resulting frame
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # Draw the value of w on the image frame
        cv2.putText(frame, f'w: {w}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (150, 150, 0), 1)
        cv2.putText(frame, f'Frame rate: {frame_rate:.2f} fps', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (150, 150, 0), 1)
        resolution = f'{frame.shape[1]}x{frame.shape[0]}'
        cv2.putText(frame, f'Resolution: {resolution}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (150, 150, 0), 1)
        # Display the resulting frame
        cv2.imshow('color_line_detection', frame) # np.hstack((image, frame))

        print(f'Frame n: {frame_number}\t w: {w}\t Frame rate: {frame_rate:.2f} fps\t')

        # the 'esc' button is set as the quitting button
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    # Destroy all the windows 
    cv2.destroyAllWindows()

if __name__ == "__main__":
    color_line_detection_single_frame_weighted_average()