#
# Utility to import test images
#

import cv2

from os import listdir
from os.path import isfile, isdir, join

# Generator function that loads one at the time the various samples from
# the folder SAMPLES_PATH. 
# The return value is a tuple (folder, sample) in which
# folder is the name of the folder containing the current glyph being examined and sample
# is either the file name or the image loaded from that file. Image are loaded in color mode.
# The function offers the following parameters:
#	path_only: If False, the element "sample" of the return tuple will contain the current already-loaded
#				image. The image is loaded in color mode. If True, the element "sample" of the return
#				tuple will contain the file name of the file currently reached.
#	only_glyphs: A list of folder names to analyze. If the list is empty, all folders in the SAMPLES_PATH
#					will be inspected. If the list is not empty, only the folders which name are in the
#					list will be analyzed.
#   only_numbers: A list of file numbers to analyze (each glyph is represented in multiple images, each
#					image has its number). If the list is empty, all numbers will be inspected. If the 
#					list is not empty, only the numbers which are in the list will be analyzed.
def image_samples(path_only = False, only_glyphs = [], only_numbers = []):

	SAMPLES_PATH = "samples" # Path to samples relative to main.py

	# Obtains the different foldes in SAMPLES_PATH. Each folder contains multiple
	# images of the same glyph. If only_glyphs is empty, all the folders are loaded, 
	# otherwise only the folder which name is in only_glyphs are loaded.
	if only_glyphs:
		glyphs_folders = [d for d in listdir(SAMPLES_PATH) if isdir(join(SAMPLES_PATH, d)) and d in only_glyphs]
	else:
		glyphs_folders = [d for d in listdir(SAMPLES_PATH) if isdir(join(SAMPLES_PATH, d))]

	# Similar to above, only for the list only_numbers. Apologies for the criptic list comprehension.
	if only_numbers:
		glyphs = {folder : [f for f in listdir(join(SAMPLES_PATH, folder)) if f[-6:-4] in only_numbers] 
					for folder in glyphs_folders}
	else:
		glyphs = {folder : listdir(join(SAMPLES_PATH, folder)) for folder in glyphs_folders}
		
	# For each file in every folder, if path_only the file name is returned, otherwise
	# the image file is loaded (color) and the image is returned
	for folder in glyphs:
		for file in glyphs[folder]: 
			if path_only:
				yield (folder, file)
			else:
				img = cv2.imread(join(SAMPLES_PATH, folder, file), cv2.IMREAD_COLOR)
				yield (folder, img) 