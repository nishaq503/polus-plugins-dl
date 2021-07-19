#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-cell-nuclei-segmentation:${version}
