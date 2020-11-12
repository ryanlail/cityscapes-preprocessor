import json
import cv2

filename = "../CityScapes/gtFine_trainvaltest/gtFine/train/zurich/zurich_000000_000019_gtFine_polygons.json"
image = cv2.imread("../CityScapes/gtFine_trainvaltest/gtFine/train/zurich/zurich_000000_000019_gtFine_color.png")

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
        
        print(x1, y1, w, h)
        cv2.rectangle(image, (x1, y1), (x1+w, y1+h), 1, 5)

    cv2.imshow("test", image)
    k = cv2.waitKey(0)

