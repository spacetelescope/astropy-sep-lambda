#!/usr/bin/env bash

# Download the 2018 docker repo for amazon linux
docker pull amazonlinux:2018.03

# Build the container
docker run -v $(pwd):/outputs -it amazonlinux:2018.03 /bin/bash /outputs/py3build.sh