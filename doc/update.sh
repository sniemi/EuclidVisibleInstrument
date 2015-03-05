#! /bin/bash

make html
make latexpdf
cp -r _build/html/* /Volumes/disk_www/smn2/

#scp -r _build/html/* euclid_vis@www:/home/euclid_vis/euclid_vis/

cp _build/latex/VIS.pdf /Volumes/disk_www/smn2/Manual.pdf


exit