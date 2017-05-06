
import cv2
from scripts.samples_reader import image_samples
from scripts.process_frame import process_frame
from scripts.result_display import display, display_sequential, clean_output_folder

def process_pictures(glyphs, numbers):
	all_data = []
	for folder, file, img in image_samples(glyphs, numbers):
		print("Processing glpyh ", (file))

		processed, intermediary, data = process_frame(img)
		all_data.append(data)
		# display(file, intermediary, save=False)
		# display_sequential(file, intermediary, save=True)
		# cv2.imshow(intermediary[-1][0], intermediary[-1][1])
	return all_data


def process_all_pictures():
	glyphs = []
	numbers = []
	all_data = process_pictures(glyphs, numbers)
	return all_data



def process_all_videos():

	all_data = []

	#TOdo stuff

	return all_data