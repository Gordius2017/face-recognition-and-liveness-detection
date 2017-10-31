#!/bin/bash

d=s$1

eval "$(mkdir $d)"
for i in `seq 1 40`;
do
    name=$d"/"$i".jpg"
    echo $name
    eval "$(fswebcam --resolution 640x480 --no-banner --no-underlay --delay 1 --save $name)"
done
namedir=$d"/"name.txt
eval "$(echo $2 > $namedir)"
