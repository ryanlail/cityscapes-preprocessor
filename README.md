# cityscapes-preprocessor

A package containing various preprocessing tools for the [Cityscapes dataset](https://www.cityscapes-dataset.com/). This tool extracts further annotations from the original dataset. Tools include:

- [Bounding box generation](#bounding-box-generation)
- [Car colour extraction](#car-colour-extraction)

# Usage

1. Download and extract the `gtFine_trainvaltest.zip (241MB)` and `leftImg8bit_trainvaltest.zip (11GB)` datasets from https://www.cityscapes-dataset.com/ to a directory of your choosing.

2. 
```
$ git clone https://github.com/tobybreckon/DoG-saliency.git
$ cd DoG-saliency
$ python3.x -m pip install -r requirements.txt
```

# Documentation

JSON files are generated with the same file structure as `gtFine_trainvaltest` and `leftImg8bit_trainvaltest`, i.e. 

```
generated
│
└───test
│   │
│   └───berlin
│       │   berlin_000000_000019_bounding_boxes.json
│       │   ...
|   
│   └───...
└───train
    │   ...
...
```

Each object's annotation in each image is given in the corresponding json, with key-value pairs.

## Bounding box generation

Each boudning box is denoted by (x1, x2, w, h) where:
- x1 - x value for leftmost pixel in bounding box
- y1 - y value for leftmost pixel in bounding box
- w - width of boudning box from x1
- h - height of bounding box from y1

## Car colour extraction

The colour of cars are given in the object annotation under key "colour". The colours are given in standard HTML4 format, see [here](https://www.w3.org/TR/2002/WD-css3-color-20020418/#html4)
