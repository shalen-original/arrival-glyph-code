#
# Arrival Glyph Encoder and Decoder
#	by Matteo Nardini and Raphael Pisoni
#

from scripts.process_file import *

def main():

	# ALl glyphs, all numbers
	# glyps = []
	# numbers = []

	glyphs = ["AbbotIsDead"]
	numbers = []

	show_results = False

	clean_output_folder()

	# all_data = process_pictures(glyphs, numbers, show=True, save=False, debug=False, read=False)
	all_data = process_videos(glyphs, numbers, save=True, show=True, debug=False, read=False)

	if show_results:
		print_graphs(all_data)



if __name__ == "__main__":
	main()
