# data_splitter
Scripts to split the dataset and to collect metrics on the rg-data

- `make_split_files.py` creates the splits in txt format, so to be easily accessed and shared
- `split_data.py` creates the actual folders from the generated text files
- `count_objects.py` counts the number of objects and does some statistics on the splits
- `parse_data.py` parses splitted data, one split at a time, and puts the output into parsed/{data_dir}. To run the script on all splits, run `bash parse_splits.sh`