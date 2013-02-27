#! /bin/bash

make html
make latexpdf
cp -r _build/html/* /Volumes/disk_www/smn2/
cp _build/latex/VIS.pdf /Volumes/disk_www/smn2/Manual.pdf

exit