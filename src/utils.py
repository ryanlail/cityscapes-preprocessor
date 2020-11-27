import numpy as np
import webcolors

# Convert RGB to Colour Name
###################################################################################################
# https://stackoverflow.com/a/9694246

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.HTML4_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        colour_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        colour_name = closest_colour(requested_colour)
    return colour_name

###################################################################################################


###################################################################################################
# https://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/
def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    num_labels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=num_labels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    # return the histogram
    return hist
###################################################################################################

