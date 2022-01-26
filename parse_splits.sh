for split in train val test; do
    echo $split
    python parse_data.py --data_dir no_meerkat --masks mask_list.dat --split $split --parser yolo
done