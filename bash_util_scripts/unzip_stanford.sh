#!/usr/bin/env bash
root="../../../../mnt/data2/bchao/lf"
root="$root/stanford/"

cd $root

files=($(ls -d *))
for zipf in "${files[@]}"
do 
    echo "Unzipping $zipf ..."
    filename=${zipf%.zip}
    rm -rf ${filename##*/}
    unzip $zipf -d ${filename##*/}
done