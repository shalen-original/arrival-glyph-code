import cv2
import numpy as np
from scripts.samples_reader import *
from scripts.process_frame import process_frame
from scripts.result_display import *


def process_pictures(glyphs, numbers, show=True, save=False, debug=False, read=False):
	all_read_data = []
	for folder, file, img in image_samples(glyphs, numbers):
		print("Processing glpyh ", file)

		processed, intermediary, read_data = process_frame(img, show=False, debug=debug, read=read)
		all_read_data.append(read_data)

		if show:
			if save:
				display(file, intermediary, save=True)
				# display_sequential(file, intermediary, save=True)
			else:
				display(file, intermediary)
				# display_sequential(file, intermediary)

	return all_read_data


# takes list of glyph-names and numbers.
# save: saves the output in output folder if true
# show: shows the video directly on the screen
# output_size: specifies the dimensions of the output video
# debug: shows debug information in the output video
# read: sets whether found glyphs should be decoded

def process_videos(glyphs, numbers, save=False, show=True, output_size=(1280, 720), debug=False, read=False):
	all_data = []

	if show:
		print("Press Q to interrupt and process move to next video.")

	for folder, file, video in video_samples(glyphs, numbers):
		print("Processing glpyh ", file)
		if save:
			fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
			filename =  file[0:file.index(".mp4")]
			print(filename)
			out = cv2.VideoWriter("out/" + filename + '.avi', fourcc, 20.0, output_size)

		cap = cv2.VideoCapture(video)
		while True:
			ret, frame = cap.read()
			if ret:
				processed, intermediary, data = process_frame(frame, show=show, debug=debug, read=read)
				all_data.append(data)

				if save:
					#todo ovethink what to save
					frame = img_resize(intermediary[1][1], output_size)
					out.write(frame)
			else:
				break

			# STOP if Q key is pressed
			if show and cv2.waitKey(1) & 0xFF == ord('q'):
				break
		if save:
			out.release()

		cap.release()
	cv2.destroyAllWindows()

	return all_data


# KEEPS ASPECT RATIO takes image and returns it resized to the shape give to it
# retruns 3-channel  image resized to the shape given to it
def img_resize(img, final_shape):
	if img is not None:
		if (img.shape[0]/img.shape[1] < final_shape[0]/final_shape[1]):
			targer_size = (int(round(img.shape[1] / img.shape[0] * final_shape[1])), final_shape[1])
			resized = cv2.resize(img, targer_size, interpolation = cv2.INTER_AREA)
			top_border = 0
			bottom_border = 0
			left_border = int(round((final_shape[0] - resized.shape[0]) / 2))
			right_border = final_shape[0] - targer_size[0] - left_border
		else:
			targer_size = (final_shape[0], int(round(img.shape[0] / img.shape[1] * final_shape[1])))
			resized = cv2.resize(img, targer_size, interpolation=cv2.INTER_AREA)
			top_border = int(round((final_shape[1] - resized.shape[1]) / 2))
			bottom_border = final_shape[1] - targer_size[1] - top_border
			left_border = 0
			right_border = 0
	else:
		resized = np.zeros((final_shape[1], final_shape[0]), dtype='uint8')
		top_border, bottom_border, left_border, right_border = 0,0,0,0

	if len(resized.shape) == 2:
		resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
	final = cv2.copyMakeBorder(resized, top=top_border, bottom=bottom_border, left=left_border, right=right_border, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
	return final