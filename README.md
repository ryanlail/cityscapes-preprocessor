# cityscapes-preprocessor

A package containing various preprocessing tools for the [Cityscapes dataset](https://www.cityscapes-dataset.com/). This tool extracts further annotations from the original dataset. Tools include:

- [Bounding box generation](#bounding-box-generation)
- [Car colour extraction](#car-colour-extraction)

# Installation

1. Download and extract the `gtFine_trainvaltest.zip (241MB)` and `leftImg8bit_trainvaltest.zip (11GB)` datasets from https://www.cityscapes-dataset.com/ to a directory of your choosing.

2. 

        $ git clone https://github.com/ryanlail/cityscapes-preprocessor.git
        $ cd cityscapes-preprocessor
        $ python3.x -m pip install -r requirements.txt
        
# Instructions to use

Run the following command to generate the new annotations from the dataset:

```
$ python3.x cityscapes-preprocessor.py [-h] [gtFine_directory] [leftimg8bit_directory] [output_directory]
```
positional arguments:
-   `gtFine_directory`&nbsp;&nbsp;specify the location of the directory called `gtFine` (containing sub-directories `train`, `test`, and `val`)
-   `leftimg8bit_directory`&nbsp;&nbsp;specify the location of the directory called `leftimg8bit` (containing sub-directories `train`, `test`, and `val`)
-   `output_directory`&nbsp;&nbsp;specify the location where you would like the generated annoatations to be stored


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

Example annotation file:

```
{
    "imgHeight":1024,
    "imgWidth":2048,
    "objects":[
        {
            "label":"ego vehicle",
            "x1":271,
            "y1":844,
            "w":1757,
            "h":179
        },
        {
            "label":"out of roi",
            "x1":0,
            "y1":0,
            "w":2048,
            "h":1024
        }
    ]
}
```

## Bounding box generation

Each boudning box is denoted by (x1, x2, w, h) where:
- x1 - x value for leftmost pixel in bounding box
- y1 - y value for leftmost pixel in bounding box
- w - width of boudning box from x1
- h - height of bounding box from y1

## Car colour extraction

The colour of cars are given in the object annotation under key "colour". The colours are given in standard HTML4 format, see [here](https://www.w3.org/TR/2002/WD-css3-color-20020418/#html4)
