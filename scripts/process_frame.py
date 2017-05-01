#
# Process a single frame
#

import cv2


def process_frame(frame):
    intermediary = []
    # intermediary.append(('original', frame));

    # Convert to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # intermediary.append(('grayscale', frame))

    # Gaussian blur to counter camera noise
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    # intermediary.append(('gaussian blur', frame))

    # Adaptive tresholding
    # In some cases performes better (Time.03.jpg, LouiseWritesHepto.03.jpg)
    gauss = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 201, 2)
    intermediary.append(('Adaptive Gauss Tresh', gauss))

    # Otsu tresholding
    treshold, otsu = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    intermediary.append(('Otsu', otsu))

    return frame, intermediary
