#!/usr/bin/env bash

REL_SCRIPT_DIR=$(dirname "$0")
INITIAL_DIR=$(pwd)
cd $REL_SCRIPT_DIR
ABS_SCRIPT_DIR=$(pwd)


# Get keypoints data
wget http://filecremers3.informatik.tu-muenchen.de/~dl4cv/training.zip
unzip training.zip
rm training.zip


cd $INITIAL_DIR
