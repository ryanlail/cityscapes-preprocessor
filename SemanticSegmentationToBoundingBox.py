import json

with fh.open(filename) as fh:
    data = json.load(fh)
    labels = data["objects"]

    for label in labels:
        # take labels x and y min and max to generate a bbox
        pass
