# image-denoising

# Abstract
This toolchain attempts to denoise images by using a set of small filters and a reinforcement learning agent to select the most appropriate one.

# Set up

To run this toolchain you will need to install and compile OpenCV library. (Run these following commands in a command prompt)
### To compile OpenCV you will need a C++ compiler and Cmake
    sudo apt install -y g++ cmake make wget unzip

### Install minimal prerequisites (Ubuntu 18.04 as reference)
    sudo apt update && sudo apt install -y cmake g++ wget unzip
### Download and unpack sources
    wget -O opencv.zip https://github.com/opencv/opencv/archive/master.zip
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/master.zip
    unzip opencv.zip
    unzip opencv_contrib.zip
### Create build directory and switch into it
    mkdir -p build && cd build
### Configure
    cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-master/modules -DBUILD_LIST=quality ../opencv-master
### Build
(you can speed up the compilation by increasing the number after -j, but it's limited by the number of processors)

    make -j4

### Install
    sudo make install
    sudo ldconfig
