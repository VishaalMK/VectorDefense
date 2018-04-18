#!/bin/bash

# Script to vectorize and rasterize mnist images
counter=0
# Vectorize using potrace
for img in adv_images/0000{0..9}.png;
do
    new=$(printf "%05d.svg" "$counter")
    convert -channel GRAY -compress None ${img} bmp:- | potrace --turnpolicy black --turdsize 5 -s - -o vectorize/$new
    counter=$(expr ${counter} + 1)
done


# Rasterize
counter=0
for img in vectorize/0000{0..9}.svg;
do
    new=$(printf "%05d.png" "$counter")
    inkscape -z -e rasterize/$new -w 28 -h 28 ${img}
    counter=$(expr ${counter} + 1)
done


