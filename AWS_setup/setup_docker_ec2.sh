#!/usr/bin/env bash

# install docker
sudo yum install docker

# start the docker daemon
sudo service docker start

# give the ec2-user permission to run run
sudo usermod -a -G docker ec2-user