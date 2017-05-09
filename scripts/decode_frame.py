import math
import numpy as np

# reads a glyph radially from its center
# returns the mask(?) and a list of read data
def read_data_from_center(mask, center, radius, angle, min_dist, max_dist_check):
	# todo tune readsplits
	readSplits = 3600
	# reads data from frame in which a valid circle has been detected
	radAngle = (angle)/180 * math.pi
	#reads thickness of sign radially
	result = []
	for ang in np.linspace(0, 2 * math.pi, readSplits):
		pixelSum = 0
		for pixel in range(int(min_dist), int(round(radius*max_dist_check))):
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
