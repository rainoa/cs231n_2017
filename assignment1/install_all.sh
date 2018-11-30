#!/usr/bin/env bash

# This runs everyting for assignment 1
# Author: nirbenz
export LC_ALL="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"
sudo dpkg-reconfigure -f noninteractive locales
sudo apt-get install -y unzip zip
wget http://cs231n.stanford.edu/assignments/2017/spring1617_assignment1.zip && unzip spring1617_assignment1.zip
cd assignment1 && \
cp setup_googlecloud.sh setup_googlecloud_orig.sh
sed -i 's/sudo apt-get install /sudo apt-get install -y /g' setup_googlecloud.sh 
sed -i 's/sudo apt-get build-dep /sudo apt-get build-dep -y /g' setup_googlecloud.sh 
./setup_googlecloud.sh
