import cv2
from scripts.samples_reader import *
from scripts.process_frame import process_frame
from scripts.result_display import *

my_absolute_path = "/home/raf/PycharmProjects/arrival-glyph-code/"

def process_pictures(glyphs, numbers):
	all_data = []
	for folder, file, img in image_samples(glyphs, numbers):
		print("Processing glpyh ", file)

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


def process_videos(glyphs, numbers, save=False, show=True, debug=False, read=False):
	all_data = []


	for folder, file, video in video_samples(glyphs, numbers):
		if save:
			fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
			out = cv2.VideoWriter("out/" + file + '.avi', fourcc, 20.0, (1280, 720))
		print("Processing glpyh ", file)
		# path = my_absolute_path + video
		# print(path)
		cap = cv2.VideoCapture(video)
		while True:
			ret, frame = cap.read()
			if ret:
				processed, intermediary, data = process_frame(frame, show=show, debug=debug, read=read)
				all_data.append(data)


				if save:
					#todo ovethink what to save
					frame = cv2.resize(intermediary[1][1], (1280, 720), interpolation = cv2.INTER_AREA)
					out.write(frame)
			else:
				break

			# STOP if Q key is pressed
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		if save:
			out.release()
		cap.release()
		# display(file, intermediary, save=False)
		# display_sequential(file, intermediary, save=True)
		# cv2.imshow(intermediary[-1][0], intermediary[-1][1])


	cv2.destroyAllWindows()

	return all_data


def process_all_videos():
	glyphs = []
	numbers = []
	all_data = process_videos(glyphs, numbers)

	return all_data