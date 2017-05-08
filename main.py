#
# Arrival Glyph Encoder and Decoder
#	by Matteo Nardini and Raphael Pisoni
#

import numpy as np
import cv2

from scripts.samples_reader import image_samples
from scripts.process_frame import process_frame
from scripts.result_display import display, display_sequential, clean_output_folder
import matplotlib.pyplot as plt
from scripts.process_file import *

def main():

	# ALl glyphs, all numbers
	glyphs = ["AbbotIsDead"]
	numbers = ["00"]
	show_results = False

	clean_output_folder()

	# all_data = process_all_pictures()
	# all_data =  process_all_videos()
	# all_data = process_pictures(glyphs, numbers)
	all_data =  process_videos(glyphs, numbers, save=True, show=True, debug=False, read=False)


	if show_results and all_data != []:
		for d in all_data:
			plt.plot(d)
		plt.ylabel("read and normaized value")
		plt.show()




if __name__ == "__main__":
    main()
