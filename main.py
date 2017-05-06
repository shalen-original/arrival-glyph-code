#
# Arrival Glyph Encoder and Decoder
#	by Matteo Nardini and Raphael Pisoni
#

import numpy as np
import cv2

from scripts.image_samples_reader import image_samples
from scripts.process_frame import process_frame
from scripts.result_display import display, display_sequential, clean_output_folder
import matplotlib.pyplot as plt

def main():

	# ALl glyphs, all numbers
	glyphs = []
	numbers = ["01", "02", "03"]
	show_results = True

	all_data = []

	# Test
	#glyphs = ["Human", "Time", "LouiseWritesHepto"]
	#numbers = ["02", "01"]
	
	clean_output_folder()

	for folder, file, img in image_samples(glyphs, numbers):
		print("Processing glpyh ", (file))

		processed, intermediary, data = process_frame(img)
		all_data.append(data)
		#display(file, intermediary, save=False)
		display_sequential(file, intermediary, save=True)

	# if show_results:
	# 	for d in all_data:
	# 		plt.plot(d)
	# 	plt.ylabel("read and normaized value")
	# 	plt.show()




if __name__ == "__main__":
    main()
