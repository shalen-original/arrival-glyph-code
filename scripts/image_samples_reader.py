#
# Utility to import test images
#

import cv2

from os import listdir
from os.path import isfile, isdir, join

# Generator function that loads one at the time the various samples from
# the folder SAMPLES_PATH. 
# The return value is a tuple (folder, file, img) in which
# folder is the name of the folder containing the current glyph being examined, file
# is the file name of the image being inspected and img is the image loaded from that file. 
# Image are loaded in color mode.
# The function offers the following parameters:
#	only_glyphs: A list of folder names to analyze. If the list is empty, all folders in the SAMPLES_PATH
#					will be inspected. If the list is not empty, only the folders which name are in the
#					list will be analyzed.
#   only_numbers: A list of file numbers to analyze (each glyph is represented in multiple images, each
#					image has its number). If the list is empty, all numbers will be inspected. If the 
#					list is not empty, only the numbers which are in the list will be analyzed.
def image_samples(only_glyphs = [], only_numbers = []):

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
		
	# For each file in every folder, the tuple containing the folder, the file name and
	# the loaded image is returned
	for folder in glyphs:
		for file in glyphs[folder]: 
				img = cv2.imread(join(SAMPLES_PATH, folder, file), cv2.IMREAD_COLOR)
				yield (folder, file, img) 