import os
import argparse
import json
import cv2
import numpy as np
from sklearn.cluster import KMeans
from utils import get_colour_name
from utils import centroid_histogram

def bounding_box_dominant_colour(img, x1, y1, w, h):
    # given a bounding box definitions and an image, return the dominant colour for that box via
    # k-means clustering

    roi = img[int(y1):int(y1+h), int(x1):int(x1+w)]

    # reshape the image to be a list of pixels
    roi = roi.reshape((roi.shape[0] * roi.shape[1], 3))

    # cluster the pixel intensities
    clt = KMeans(n_clusters=5)
    clt.fit(roi)

    # build a histogram of clusters and then create a figure
    # representing the number of pixels labeled to each color
    hist = centroid_histogram(clt)

    max_cluster = np.argmax(hist)
    rgb = clt.cluster_centers_[max_cluster].astype(int)

    return get_colour_name(rgb)

def generate_bounding_box(segment, box_attributes):
    # take labels x and y min and max to generate a bbox

    vertices = segment["polygon"]

    x_vals = [coord[0] for coord in vertices]
    y_vals = [coord[1] for coord in vertices]

    # max to ensure co-ordinate is at least 0
    x1 = max(min(x_vals), 0)
    y1 = max(min(y_vals), 0)
    w = max(x_vals) - x1
    h = max(y_vals) - y1

    box_attributes["x1"] = x1
    box_attributes["y1"] = y1
    box_attributes["w"] = w
    box_attributes["h"] = h

    return box_attributes

def generate_colour(segment, box_attributes, image):

    x1 = box_attributes["x1"]
    y1 = box_attributes["y1"]
    w = box_attributes["w"]
    h = box_attributes["h"]

    if segment["label"] == "car":
        # only capture bottom half of car for best colour representation
        colour = bounding_box_dominant_colour(image, x1, y1+(h/2), w, h/2)
        box_attributes["colour"] = colour

    return box_attributes

def generate_annotations(annotation_path, image_path):

    image = cv2.imread(image_path)

    # generate new json
    with open(annotation_path) as fh:
        data = json.load(fh)
        segments = data["objects"]

        bounding_boxes = []

        for segment in segments:

            box_attributes = dict()
            box_attributes["label"] = segment["label"]

            box_attributes = generate_bounding_box(segment, box_attributes)

            box_attributes = generate_colour(segment, box_attributes, image)

            # add bounding box to list
            bounding_boxes.append(box_attributes)

    return {"imgHeight": data["imgHeight"], "imgWidth": data["imgWidth"], "objects": bounding_boxes}

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

    ARGS = parser.parse_args()

    # go into gtFine_directory
    # go into each test, train, val
    for sub_set in os.listdir(ARGS.gtFine_directory):
        sub_set_path = os.path.join(ARGS.gtFine_directory, sub_set)

        # go into each city
        for city in os.listdir(sub_set_path):
            city_path = os.path.join(sub_set_path, city)

            for annotation in os.listdir(city_path):
                annotation_path = os.path.join(city_path, annotation)
                if annotation.endswith(".json"):

                    # for every json load corresponsing colour image from leftimg8bit_directory
                    image_path = os.path.join(ARGS.leftimg8bit_directory, sub_set, city,
                                              annotation.replace("gtFine_polygons.json",
                                                                 "leftImg8bit.png"))

                    generated_annotations = generate_annotations(annotation_path, image_path)

                    # save to output directory maintaining structure
                    output_path = os.path.join(ARGS.output_directory, sub_set, city,
                                               annotation.replace("gtFine_polygons",
                                                                  "bounding_boxes"))

                    if not os.path.exists(os.path.dirname(output_path)):
                        os.makedirs(os.path.dirname(output_path))

                    with open(output_path, 'w') as fh:
                        json.dump(generated_annotations, fh)
