#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-smp-training-plugin:${version}