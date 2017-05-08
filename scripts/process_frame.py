#
# Process a single frame
#
import traceback

import cv2
import numpy as np
import random
import math


def process_frame(frame, show=False, debug=False, read=False):
	data = []
	intermediary = []
	#intermediary.append(("original", frame))

	# The resolution on which to work
	# curr_size = (1920, 1080) # Full HD (1080p)
	#curr_size = (1280, 720) # HD Ready (720p)
	curr_size = (int(round(frame.shape[1]/frame.shape[0]*720)),720)

	# Resize to computable resolution
	frame_resized = cv2.resize(frame, curr_size, interpolation = cv2.INTER_AREA)
	intermediary.append(("resized", frame_resized))

	# Convert to grayscale
	grayscale = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2GRAY)

	# Gaussian blur to reduce camera noise
	blurred = cv2.GaussianBlur(grayscale, (3, 3), 0)

	# Adaptive tresholding: always creates a precise separation between
	# the glyph and everything else
	gauss = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY_INV,201,2)
	# intermediary.append(('Adaptive Gauss Tresh', gauss))

	# Detecting contours
	im2, contours, hierarchy = cv2.findContours(gauss, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	# Filtering contours too big or too small
	contours = [c for c in contours if 1000 <= cv2.contourArea(c) <= 60000]
	#intermediary.append(("contours", __draw_all_contours(gauss, contours)))

	# get a list of all contours and sort the contours according to it
	contAreas = [cv2.contourArea(x) for x in contours]
	sorted_contours = sorted(zip(contAreas, contours), key=lambda x: x[0], reverse=True)

	# square around best (chosen) and display its "arrivality", as well as returnig the chosen contour
	chosen, arrivality, chosen_cnt = get_best_contour(sorted_contours, gauss.shape, debug=debug)
	if show:
		if debug:
			if chosen_cnt is not None:
				frame_resized = cv2.fillPoly(frame_resized, pts=chosen_cnt, color=(255, 150, 150))

		cv2.imshow("frame", frame_resized)
	intermediary.append(('chosen contour', cv2.cvtColor(chosen, cv2.COLOR_GRAY2BGR)))
	# print("Arrivality: ", arrivality)

	if read:
		chosen_debug = None
		if chosen is not None:
			chosen_debug, data = read_circle_segment(chosen)
		if chosen_debug is not None:
			intermediary.append(('debug_image', chosen_debug))

	return chosen, intermediary, data


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


# computes angle of vector from center in reference to zero PI
def angleZeroRad(center, vector):
	angle = np.rad2deg(np.arctan2(vector[1] - center[1], vector[0] - center[0]))
	return angle

# gets distance between two points
def get_distance(p1, p2):
	return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# chooses best contour out of a list of (area, contour) - tuples
# needs shape of source-image
def get_best_contour(sorted_contours, shape, debug=False):
	arrivality = None
	best_mask = None
	best_cnt = None
	best_cnt_area = 0
	best_arr = 0
	for cnt in sorted_contours:
		# todo tune thresholds to values that makes sense. Big speedup possible.
		if 100000 > cnt[0] > 6000:
			# The choosen contour is drawn
			mask = np.zeros(shape, dtype='uint8')
			mask = cv2.fillPoly(mask, pts=[cnt[1]], color=(255, 255, 255))
			x, y, w, h = cv2.boundingRect(cnt[1])
			cnt_mask = mask[y:y + h, x:x + w]

			diagonal_half = int(round(math.sqrt(cnt_mask.shape[0]**2 + cnt_mask.shape[1]**2)/2))
			try:
				center, radius, arrivality = get_circle_from_mask(cnt_mask,diagonal_half)
			except:
				pass
			if best_arr is None or arrivality is not None and arrivality > best_arr:
				best_cnt_area = cnt[0]
				best_cnt = [cnt[1]]
				best_mask = cnt_mask
				best_arr = arrivality
		else:
			break

		# todo tune coefficient
		if best_arr > 60:
			break
	if debug:
		print("Arrivality: ", best_arr, "\tArea: ", best_cnt_area)

	return best_mask, best_arr, best_cnt

# gives back arrivality coefficient for whole contour. the higher the better
def get_total_arrivality(mask, center, radius):
	# todo tune coefficients on glyphs
	inner_radius = 0.8 * radius
	outer_radius = 1.3 * radius
	radius_a = get_arrivality(mask, center, radius)
	outer_a = get_arrivality(mask, center, outer_radius)
	inner_a = get_arrivality(mask, center, inner_radius)

	return radius_a - outer_a - inner_a

# gives back arrivality coefficient for single radius in contour.
def get_arrivality(mask, center, radius):
	arrivality = 0
	for ang in range(100):
		pX = int(round(radius * math.cos(ang) + center[0]))
		pY = int(round(radius * math.sin(ang) + center[1]))

		if pX >= 0 and pX < mask.shape[1] and pY >= 0 and pY < mask.shape[0]:
			if mask[pY][pX] > 0:
				arrivality +=1
	return arrivality


# computes the center of a circle defined by 3 points
def get_center_from_points(a,b,c):
	x, y, z = complex(a[0], a[1]), complex(b[0], b[1]), complex(c[0], c[1])
	w = z - x
	w /= y - x
	c = (x - y) * (w - abs(w) ** 2) / 2j / w.imag - x
	return (int(round(-c.real)), int(round(-c.imag))), int(round(abs(c + x)))

# computes the best fitting circle for a single contour-mask
# retruns center, radisu and arrivality
def get_circle_from_mask(mask, diagonal):
	center = (int(round(mask.shape[1]/2)), int(round(mask.shape[0]/2)))
	radius = None
	circle_points = []

	# Get distance Map
	dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
	dist = cv2.normalize(dist, dist, 0, 1, cv2.NORM_MINMAX)

	# cv2.imshow("distancemap",dist)
	# cv2.waitKey(0)

	for ang in [0, math.pi/2, math.pi, 3*math.pi/2]:
		lightest = 0
		for pixel in range(0, int(round(diagonal))):
			pX = int(round(pixel * math.cos(ang) + center[0]))
			pY = int(round(pixel * math.sin(ang) + center[1]))

			if pX >= 0 and pX < mask.shape[1] and pY >= 0 and pY < mask.shape[0]:
				if dist[pY][pX] > lightest:
					lightest = dist[pY][pX]
					lightest_point = (pX, pY)
			else:
				break
		if lightest > 0:
			circle_points.append(lightest_point)
		if len(circle_points) >= 3:
			break
	if len(circle_points) < 3:
		arrivalidity = 0
	else:
		center, radius = get_center_from_points(circle_points[0],circle_points[1],circle_points[2])
		arrivalidity = get_total_arrivality(dist, center, radius)

	return center, radius, arrivalidity


# reads a glyph radially from its center
# returns the mask(?) and a list of read data
# todo make independent from radius by choosing a fixed number for read splits
def read_data_from_center(mask, center, radius, angle, minDist, maxDistCheck):
	readSplits = 3600
	# reads data from frame in which a valid circle has been detected
	radAngle = (angle)/180 * math.pi
	#reads thickness of sign radially
	result = []
	for ang in np.linspace(0, 2 * math.pi, readSplits):
		pixelSum = 0
		for pixel in range(int(minDist), int(round(radius*maxDistCheck))):
			pX = int(round(pixel * math.cos(ang + radAngle) + center[0]))
			pY = int(round(pixel * math.sin(ang + radAngle) + center[1]))

			if pX >= 0 and pX < mask.shape[1] and pY >= 0 and pY < mask.shape[0]:
				if mask[pY][pX][0] > 0:
					pixelSum += 1
					# mask[pY,pX] = (255,0,0)
			else:
				break
		result.append(pixelSum)

	result = np.array(result)
	result = np.trim_zeros(result, trim="f")
	result = np.pad(result, (0, (readSplits - len(result)) % readSplits), 'constant')
	result = result/radius*1000
	# result[result == 0] = -1

	return mask, result


# finds the beginning and end of a given glyph
# returns the mask, (start, end) of gap, percentage of inliers and the minimum radius
def find_ends(mask, center, radius, minDist, maxDistCheck):
	start, end = None, None
	inliers = 0
	min_radius = None

	# compute necessary reads and double for good measure:
	read_splits = 2 * math.pi * radius * 2

	last = None
	this_coords = ()
	last_coords = ()
	gap = []

	for ang in np.linspace(0, 2 * math.pi, read_splits):
		found = False

		for pixel in range(int(round(radius*minDist)), int(round(radius*maxDistCheck))):
			pX = int(round(pixel * math.cos(ang) + center[0]))
			pY = int(round(pixel * math.sin(ang) + center[1]))
			this_coords = (pX,pY)

			if pX >= 0 and pX < mask.shape[1] and pY >= 0 and pY < mask.shape[0]:
				if(mask[pY][pX]) != 0:
					found = True
					inliers += 1
					distance_to_center = get_distance(center, this_coords)
					if min_radius is None or distance_to_center < min_radius:
						min_radius = distance_to_center
					break
			else:
				break

		if last != None:
			if found and not last:
				start = this_coords

			if not found and last:
				end = last_coords

			if start is not None and end is not None:
				gap = [start, end]
				# print("start: ", start)
				# print("end: ", end)


		last = found
		last_coords = this_coords


	mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
	cv2.circle(mask,center,1,(255,255,255),1)

	return mask, gap, inliers, min_radius


def read_circle_segment(sign):
	frame = sign.copy()
	minDist = 0  # start reading at minDist * detected radius
	maxDistCheck = int(round(math.sqrt(sign.shape[0]**2+sign.shape[1]**2)/2))  # longest readable distance from center(half of diagonal of image)
	read = []

	diagonal_half = int(round(math.sqrt(sign.shape[0]**2+sign.shape[1]**2)/2 ))     # diagonal/2 - short_side



	center, radius, arrivalidity = get_circle_from_mask(frame,diagonal_half)

	# Todo tune arrivality limit
	if arrivalidity > 150:
		readSplits = 4 * math.pi * radius

		black_bgr, gap, inlier, minDist = find_ends(frame, center, radius, minDist, maxDistCheck)

		if inlier >= (readSplits * 0) and gap != []:
			try:
				# draw stuff in the output image,

				circle_start = gap[0]
				circle_end = gap[1]
				if circle_start[0] and circle_start[1] and circle_end[0] and circle_end[1]:

					cv2.circle(black_bgr, center, int(round(radius*0.8)), (0,255,0), 2)
					cv2.circle(black_bgr, center, int(round(radius*1.3)), (0,255,0), 2)
					cv2.circle(black_bgr, center, radius, (255,0,0), 2)
					cv2.rectangle(black_bgr, (circle_start[0] - 3, circle_start[1] - 3), (circle_start[0] + 3, circle_start[1] + 3), (0, 255, 0), -1)
					cv2.rectangle(black_bgr, (circle_end[0] - 3, circle_end[1] - 3), (circle_end[0] + 3, circle_end[1] + 3), (0, 0, 255), -1)
					cv2.line(black_bgr, center, circle_start, (0, 255, 0))
					cv2.line(black_bgr, center, circle_end, (0, 0, 255))

				angle = 180 + angleZeroRad(gap[0], center)
			except Exception as x:
				traceback.print_exc()
				pass
			mask, read = read_data_from_center(black_bgr, center, radius, angle, minDist, maxDistCheck)
			read = read.tolist()
	else:
		black_bgr = None
		read = []

	return black_bgr, read