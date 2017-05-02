#
# Process a single frame
#

import cv2
import numpy as np
import random


def process_frame(frame):

	intermediary = []
	intermediary.append(("original", frame))

	# The resolution on which to work
	# curr_size = (1920, 1080) # Full HD (1080p)
	curr_size = (1280, 720) # HD Ready (720p)

	# Resize to computable resolution
	frame_resized = cv2.resize(frame, curr_size, interpolation = cv2.INTER_AREA)

	# Convert to grayscale
	grayscale = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2GRAY)

	# Gaussian blur to reduce camera noise
	blurred = cv2.GaussianBlur(grayscale, (3, 3), 0)

	# Adaptive tresholding: always creates a precise separation between
	# the glyph and everything else
	gauss = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY_INV,201,2)
	#intermediary.append(('Adaptive Gauss Tresh', gauss))

	# Detecting contours
	im2, contours, hierarchy = cv2.findContours(gauss, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	# Filtering contours too big or too small
	contours = [c for c in contours if 1000 <= cv2.contourArea(c) <= 60000]
	#intermediary.append(("contours", __draw_all_contours(gauss, contours)))

	# Filling the survived contours
	filled_poly = np.zeros(gauss.shape, dtype='uint8')
	filled_poly = cv2.fillPoly(filled_poly, pts = contours, color=(255, 255, 255))
	#intermediary.append(("filledContours", filled_poly))

	# Detecting circles in the image with the filled contours. The thresholds for detecting
	# a circle are very low: many circles are detected, but most of them are very close to the glyph
	circles = cv2.HoughCircles(filled_poly, cv2.HOUGH_GRADIENT, 1, 5, 
								param1=170, param2=22, 
								minRadius=70, maxRadius=150)

	# If no circles are detected, then there is no glyph in the frame.
	# An empty object is returned.
	if circles is None: 
		return (None, intermediary)

	# If circles are detected, the list of circles is extracted
	circles = circles[0]
	#intermediary.append(('circles', __draw_all_circles(frame, circles)))

	# Computers the mean of all the centers of all the circles found.
	# The resulting point will very probably be very close to the center of the glyph.
	x_mean, y_mean = 0, 0
	for x, y, radius in circles:
		x_mean += x
		y_mean += y

	x_mean = int(x_mean / len(circles))
	y_mean = int(y_mean / len(circles))

	# For each contour, the centroid of that contour is computed.
	# Then, the square of the distance between that centroid and the mean point
	# found before is computed and stored.
	res = []
	for cnt in contours:
		M = cv2.moments(cnt)
		cx = int(M['m10']/M['m00'])
		cy = int(M['m01']/M['m00'])

		res.append((cnt, (cx-x_mean)**2 + (cy-y_mean)**2))

	# The result with the minimum distance is found and is saved
	res_min = min(res, key = lambda t: t[1])
	choosen_contour = res_min[0]

	# The choosen contour is drawn
	choosen = np.zeros(gauss.shape, dtype='uint8')
	choosen = cv2.fillPoly(choosen, pts = [choosen_contour], color=(255, 255, 255))
	#intermediary.append(('choosen', choosen))

	# The bounding rect around the contour is detected and the
	# frame is clipped to that rect
	x,y,w,h = cv2.boundingRect(choosen_contour)
	final = choosen[y:y+h, x:x+w]

	# rect = cv2.minAreaRect(minC)
	# box = cv2.boxPoints(rect)
	# box = np.int0(box)

	# rotation_center_x = (box[0][0] + box[2][0]) / 2
	# rotation_center_y = (box[0][1] + box[2][1]) / 2
	# rot = cv2.getRotationMatrix2D((rotation_center_x,rotation_center_y),rect[2],1)
	# final = cv2.warpAffine(choosen,rot,choosen.shape)

	intermediary.append(('final', final))

	return final, intermediary



# Draws all the circles and their centers on a given frame.
# The drawn frame is returned.
def __draw_all_circles(frame, circles):
	res = np.copy(frame)

	for x, y, radius in circles:
		# outer circle
	    cv2.circle(res, (x,y), radius, (0, 255, 0), 2)
	    # center
	    cv2.circle(res, (x,y), 2, (0, 0, 255), 3)

	return res

# Draws all the contours on a black frame. Each contour is drawn 
# with a random color. The drawn frame is returned.
def __draw_all_contours(frame, contours):
	black = np.zeros(frame.shape)

	for cnt in contours:
		cv2.drawContours(black, [cnt], 0, __random_color(), 1)

	return black

# Returns a random BGR color as a tuple (b, g, r).
def __random_color():
	b = random.randint(0, 255)
	g = random.randint(0, 255)
	r = random.randint(0, 255)

	return (b, g, r)