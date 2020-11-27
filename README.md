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

## Bounding box generation

## Car colour extraction

blah blah blah
