#!/bin/sh

# Kinetics
# convert_mp4_to_jpeg('../data/' + args.data_dir)
python convert_mp4_to_jpeg.py -d kinetics-dataset/k400/train
python convert_mp4_to_jpeg.py -d kinetics-dataset/k400/test
python convert_mp4_to_jpeg.py -d kinetics-dataset/k400/val

# python convert_mp4_to_jpeg.py -d kinetics-600/train
# python convert_mp4_to_jpeg.py -d kinetics-600/test
# python convert_mp4_to_jpeg.py -d kinetics-600/val