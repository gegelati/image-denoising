#This script is used to noise the dataset
#Compile noise_cifar-10.cpp
g++ noise_cifar-10.cpp -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_quality -o noise_dataset

#Execute
./noise_dataset