#
# Defines method to plot graph in "the proper way".
#

import matplotlib.pyplot as plt
from os.path import join
import cv2

# The folder in which to store outputs
OUTPUT_FOLDER = "out"

# Takes a list of tuples (title, image) and displays all of them in a single
# matplotlib figure, with one image per subplot. If the "save" flag is given, the figure
# is saved to disk instead of being displayed. The function does not return anything.
# The function has the following parameters:
#	global_title: The title of the figure
#	images: a list of tuples (title, image) to display. The images are expected to be either
#			grayscale or BGR.
#   save: If True, the figure generated by the method is saved in the OUTPUT_FOLDER. If False,
#			the generated figure is shown.
def display(global_title, images, save = False):
	global OUTPUT_FOLDER

	# Settings of the figure
	fig = plt.figure(figsize=(15,7), dpi=80)
	fig.suptitle(global_title)
	fig.subplots_adjust(left = 0.01, bottom = 0.01, right = 0.99, top = 0.95, wspace = 0.02, hspace = 0)

	curr = 1 					# The current subplot being generated
	n = len(images)				# The total number of images
	rows = n // 3 + 1			# Number of rows in the figure
	cols = n if n < 3 else 3	# Number of columns in the figure

	for title, image in images:
		subp = fig.add_subplot(rows, cols, curr)
		curr += 1

		subp.set_title(title)
		subp.axis('off')

		# If the shape of an image has length two, the image is grayscale. Otherwise,
		# the image is expected to be in the BGR format.
		if 2 == len(image.shape):
			subp.imshow(image, cmap='gray')
		else:
			subp.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

	if save:
		fig.savefig(join(OUTPUT_FOLDER, "{0}.png".format(global_title)));
	else:
		plt.show()
		

# Takes a list of tuples (title, image) and displays all of them in sequence,
# waiting for a keyboard input before switching to the next. If the "save" flag is given, the
# images are saved to disk instead of being displayed. The function does not return anything.
# The function has the following parameters:
#	global_title: The title of the sequence
#	images: a list of tuples (title, image) to display. The images are expected to be either
#			grayscale or BGR.
#   save: If True, the images are saved in the OUTPUT_FOLDER. If False, they are shown.
def displaySequential(global_title, images, save = False):
	global OUTPUT_FOLDER

	curr = 1 # The current image being shown

	if not save:
		cv2.namedWindow(global_title,cv2.WINDOW_NORMAL)
		cv2.resizeWindow(global_title, 1200, 650)

	for title, image in images:
		if save:
			cv2.imwrite(join(OUTPUT_FOLDER, "{0}-{1}-{2}.png".format(global_title, curr, title)), image)
		else:
			cv2.imshow(global_title, image)
			cv2.waitKey(0)

		curr += 1

	if not save:
		cv2.destroyWindow(global_title)