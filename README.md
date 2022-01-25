# data_splitter
Scripts to split the dataset and to collect metrics on the rg-data

- `make_split_files.py` creates the splits in txt format, so to be easily accessed and shared
- `split_data.py` creates the actual folders from the generated text files
- `count_objects.py` counts the number of objects and does some statistics on the splits
- `parse_data.py` parses splitted data, one split at a time, and puts the output into parsed/{data_dir}. To run the script on all splits, run `bash parse_splits.sh`

## parse_data
Collection of data parsers in different formats for training neural network models on radioastronomical datasets
Parses a single split of the dataset, so the split has to be preventively done when running this script

### COCO Parser
Converts FITS mask data in [COCO format](https://cocodataset.org/#format-data)

### YOLO Parser
Converts FITS mask data in [YOLO format](https://github.com/cvjena/darknet/blob/master/README.md)

### Args
- `-p` Type of parser (default: coco)
- `-m`, Path of file that lists all mask file paths (trainset.dat)
- `-d` Destination directory for converted data (default: coco)
- `-c` Contrast value for conversion from FITS to PNG (default: 0.15)

### Directory Structure
```
parent_folder
└───data_processor
│   │───main.py
│   │───...
│   │───README.md (**YOU ARE HERE**)    
│   │
└───data_dir (e.g. MLDataset_cleaned)
    │
    └───train
    │   │───imgs
    │   │───annotations
    │   │───masks
    │   │───imgs_png
    │   │───...
    └───val
    │   │───imgs
    │   │───annotations
    │   │───masks
    │   │───imgs_png
    │   │───...
    └───test
        │───imgs
        │───annotations
        │───masks
        │───imgs_png
        │───...

```
