#!/bin/bash
cd ./kitti_data

for FILE in *.zip;
do
    unzip $FILE
    rm $FILE
    find ./ -name '*.png' | parallel 'convert {.}.png {.}.jpg && rm {}'
done
