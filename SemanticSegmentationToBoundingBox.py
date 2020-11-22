import json
import cv2
import numpy as np

import sys
import os

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse

# Convert RGB to Colour Name
###############################################################################

import webcolors

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

#############################################################################

# https://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/
#############################################################################
def centroid_histogram(clt):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)
	# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()
	# return the histogram
	return hist

#############################################################################

def bounding_box_dominant_colour(img, x1, y1, w, h):
    # given a bounding box definitions and an image, return the dominant colour for that box via k-means clustering
    
    roi = img[int(y1):int(y1+h), int(x1):int(x1+w)]
    
    # reshape the image to be a list of pixels
    roi = roi.reshape((roi.shape[0] * roi.shape[1], 3))

    # cluster the pixel intensities
    clt = KMeans(n_clusters = 5)
    clt.fit(roi)

    # build a histogram of clusters and then create a figure
    # representing the number of pixels labeled to each color
    hist = centroid_histogram(clt)

    max_cluster = np.argmax(hist)
    rgb = clt.cluster_centers_[max_cluster].astype(int)

    return get_colour_name(rgb)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Extract bounding box and colour data from the cityscapes dataset")
    parser.add_argument(
        'gtFine_directory',
        type=str,
        nargs='?',
        help='location of the gtFine directory containing test train val annotations')
    parser.add_argument(
        'leftimg8bit_directory',
        type=str,
        nargs='?',
        help='location of the leftimg8bit directory containing test train val images')
    parser.add_argument(
        'output_directory',
        type=str,
        nargs='?',
        help='location of where generated annotation files should be produced')


    """
    make bounding box optional. Could extend to colour extraction too.

    parser.add_argument(
        "-b",
        "--generate_bounding_boxes",
        action='store_true',
        help="generate boudning boxes in resultant dataset")
    """

    args = parser.parse_args()





    # go into gtFine_directory
    # go into each test, train, val
    for sub_set in os.listdir(args.gtFine_directory):
        sub_set_path = os.path.join(args.gtFine_directory, sub_set)

        # go into each city
        for city in os.listdir(sub_set_path):
            city_path = os.path.join(sub_set_path, city)

            for annotation in os.listdir(city_path):
                annotation_path = os.path.join(city_path, annotation)
                if annotation.endswith(".json"):
 
                    # for every json load corresponsing colour image from leftimg8bit_directory
                    image_path = os.path.join(args.leftimg8bit_directory, sub_set, city, annotation.replace("gtFine_polygons.json", "leftImg8bit.png"))
                    image = cv2.imread(image_path)
                    
                    # generate new json
                    with open(annotation_path) as fh:
                        data = json.load(fh)
                        segments = data["objects"]
                        
                        bounding_boxes = []

                        for segment in segments:
                            # take labels x and y min and max to generate a bbox
                            
                            label = segment["label"]
                            vertices = segment["polygon"]
                            
                            x_vals = [coord[0] for coord in vertices]
                            y_vals = [coord[1] for coord in vertices]
                            
                            # max to ensure co-ordinate is at least 0
                            x1 = max(min(x_vals), 0)
                            y1 = max(min(y_vals), 0)
                            w = max(x_vals) - x1
                            h = max(y_vals) - y1
                            
                            # draw bounding box
                            # cv2.rectangle(image, (x1, y1), (x1+w, y1+h), 1, 5)

                            object_attributes = {"label": segment["label"], "x1": x1, "y1": y1, "w": w, "h": h}
                            
                            if segment["label"] == "car":
                                # only capture bottom half of car for best colour representation
                                cv2.rectangle(image, (x1, y1), (x1+w, y1+h), 1, 5)
                                colour = bounding_box_dominant_colour(image, x1, y1+(h/2), w, h/2)
                                object_attributes["colour"] = colour
                            

                            # add bounding box to list
                            bounding_boxes.append(object_attributes)

                    # save to output directory maintaining structure
                    generated_annotations = {"imgHeight": data["imgHeight"], "imgWidth": data["imgWidth"], "objects": bounding_boxes}
                    output_path = os.path.join(args.output_directory, sub_set, city, annotation.replace("gtFine_polygons", "bounding_boxes"))
                    
                    if not os.path.exists(os.path.dirname(output_path)):
                        os.makedirs(os.path.dirname(output_path))

                    with open(output_path, 'w') as fh:
                        json.dump(generated_annotations, fh)



    """


    ##################################################################################
    filename = "../CityScapes/gtFine_trainvaltest/gtFine/train/zurich/zurich_000000_000019_gtFine_polygons.json"

    image = cv2.imread("../CityScapes/leftimg8bit_trainvaltest/leftImg8bit/train/zurich/zurich_000000_000019_leftImg8bit.png")
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with open(filename) as fh:
        data = json.load(fh)
        segments = data["objects"]

        for segment in segments:
            # take labels x and y min and max to generate a bbox
            
            label = segment["label"]
            vertices = segment["polygon"]
            
            x_vals = [coord[0] for coord in vertices]
            y_vals = [coord[1] for coord in vertices]
            
            x1 = min(x_vals)
            y1 = min(y_vals)
            w = max(x_vals) - x1
            h = max(y_vals) - y1
            
            #print(x1, y1, w, h)
            #cv2.rectangle(image, (x1, y1), (x1+w, y1+h), 1, 5)

            if segment["label"] == "car":
                # only capture bottom half of car for best colour representation
                cv2.rectangle(image, (x1, y1), (x1+w, y1+h), 1, 5)
                bounding_box_dominant_colour(image, x1, y1+(h/2), w, h/2)

        cv2.imshow("test", image)
        k = cv2.waitKey(0)

    """
