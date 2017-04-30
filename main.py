#
# Arrival Glyph Encoder and Decoder
#	by Matteo Nardini and Raphael Pisoni
#

import numpy as np
import cv2

from scripts.image_samples_reader import image_samples
from scripts.process_frame import process_frame
from scripts.result_display import displaySequential, cleanOutputFolder

def main():

	# ALl glyphs, all numbers
	# glyphs = []
	# numbers = ["01", "02", "03"]

	# Test
	glyphs = ["Human", "Time", "LouiseWritesHepto"]
	numbers = ["03"]
	
	cleanOutputFolder()

	for folder, file, img in image_samples(glyphs, numbers):
		print "Processing glpyh {0}".format(file)

		processed, intermediary = process_frame(img)
		displaySequential(file, intermediary, save=True)


if __name__ == "__main__":
    main()
