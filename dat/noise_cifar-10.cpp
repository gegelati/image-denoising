#ifndef NOISE_CIFAR_10_CPP
#define NOISE_CIFAR_10_CPP

#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include "../src/cifar/cifar10_reader.hpp"
#include <sys/stat.h> 
#include <sys/types.h>
#include <string>

using namespace std;
using namespace cv;

#define NB_BYTE_CHAN 1024
#define NB_IMG_BATCH 10000
#define CIFAR_10_DATA "cifar-10/cifar-10-batches-bin"
#define NOISY_CIFAR_10_DATA "cifar-10/noisy-cifar-10-batches-bin"

namespace noise{

    void bin_to_Mat(vector<uchar> in_bin_img, Mat out_img){
        for (int chan = 0; chan <3 ; chan++){//three chanels
            int x = 0,y =0;
            for (int byte = 0; byte < NB_BYTE_CHAN; byte++){
                out_img.at<Vec3b>(x,y)[chan] = (uchar)(in_bin_img[byte + NB_BYTE_CHAN*chan]);   //Read values from the vector
                y++;
                if (y == 32){
                    y = 0;
                    x++;
                }
            }
        }
    }

    void Mat_to_bin(Mat in_img, vector<uchar> & out_bin_img){
        for (int chan = 0; chan < 3; chan++){
            for(int row =0; row < in_img.rows; row++){
                for (int col = 0; col < in_img.cols; col++){
                    out_bin_img[col + in_img.cols * row + chan * NB_BYTE_CHAN] = (uchar)(in_img.at<Vec3b>(row,col)[chan]);
                }
            }
        }
    }


    Mat add_gaussian_noise(const Mat & img, int mean, int sigma){
        Mat noisy_img = img.clone();
        Mat my_noise(img.rows,img.cols,CV_MAKE_TYPE(CV_64F,img.channels()));

        vector<int> m(img.channels(),mean);
        vector<int> s(img.channels(),sigma);

        randn(my_noise,m,s); //add noise

        return noisy_img += my_noise;
    }

    
    int cifar_add_noise(void){
        
        //Load dataset
        cout << "Loading dataset" << endl;
        auto dataset = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(CIFAR_10_DATA);
        Mat img(32,32,CV_8UC3,Scalar(0));

        String path;
        String folder = "cifar-10/cifar-10-batches/train";
        char path_folder[100] = "cifar-10/cifar-10-batches/";
        String str(path_folder);

        if (mkdir(path_folder,0755) != 0){
            cout << "Cannot create folder at " + folder << endl;
        }

        for(int img_num =0; img_num < 20 ;img_num++){
            bin_to_Mat(dataset.training_images[img_num],img);

            path = folder + to_string(img_num) + ".jpg";
            bool check = imwrite(path, img);

            if (check == false) {
            cout << "Mission - Saving the image, FAILED" << endl;
            }                     
        }
        cout << "Successfully saved the image : 1 to 20 at " + str << endl;

        //Noising training images
        cout << "Noising training images" << endl;
        for(auto &img_bin : dataset.training_images){
            bin_to_Mat(img_bin,img);
            Mat img_noisy = add_gaussian_noise(img,0,50);
            Mat_to_bin(img_noisy,img_bin);
        }

        //Noising testing images
        cout << "Noising testing images" << endl;
        for(auto &img_bin_t : dataset.test_images){
            bin_to_Mat(img_bin_t,img);
            Mat img_noisy = add_gaussian_noise(img,0,50);
            Mat_to_bin(img_noisy,img_bin_t);
        }

        //Saving noisy images in new binaries files
        String filename_data_batch = "/data_batch_";
        String filename_test_batch = "/test_batch";
        String noisy_cifar_path = NOISY_CIFAR_10_DATA;
        String filename;
        ofstream wf;

        if (mkdir(NOISY_CIFAR_10_DATA,0755) != 0){
            cout << "Cannot create folder at " + noisy_cifar_path << endl;
        }

        //Save training batches
        cout << "Saving training images" << endl;
        for(int batch_num =1; batch_num <= 5 ;batch_num++){
            filename = noisy_cifar_path + filename_data_batch + to_string(batch_num) + ".bin";

            wf.open(filename, ios::out | ios::binary); //Binaries files stream creation

             if(!wf) {
                cout << "Cannot open file !" << endl;
                return EXIT_FAILURE;
            }
            
            for(int num_img = 0; num_img < NB_IMG_BATCH; num_img++){
                wf.write((char *) &dataset.training_labels[num_img + NB_IMG_BATCH * (batch_num-1)], sizeof(dataset.training_labels[0]));
                for (int i = 0; i < NB_BYTE_CHAN * 3; i++){
                    wf.write((char *) &dataset.training_images[num_img + NB_IMG_BATCH * (batch_num-1)][i], sizeof(dataset.training_images[0][0]));
                }   

            }
            wf.close();
            if(!wf.good()) {
                cout << "Error occurred at writing time !" << endl;
                return EXIT_FAILURE;
            }
        }

        //Save test batch
        cout << "Saving testing images" << endl;
        filename =  noisy_cifar_path + filename_test_batch + ".bin";

        wf.open(filename, ios::out | ios::binary); //Binaries files stream creation

        if(!wf) {
            cout << "Cannot open file !" << endl;
            return EXIT_FAILURE;
        }
        
        for(int num_img = 0; num_img < NB_IMG_BATCH; num_img++){
            wf.write((char *) &dataset.test_labels[num_img], sizeof(dataset.test_labels[0]));
            for (int i = 0; i < NB_BYTE_CHAN * 3; i++){
                wf.write((char *) &dataset.test_images[num_img][i], sizeof(dataset.test_images[0][0]));
            }
        }
        wf.close();
        if(!wf.good()) {
            cout << "Error occurred at writing time !" << endl;
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
    }


    void test_read_cifar_noisy(){
        cout << endl << "Test loading noisy dataset" << endl;
        //Load dataset
        
        auto dataset = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(NOISY_CIFAR_10_DATA);
        
        Mat img(32,32,CV_8UC3,Scalar(0));
        
        String path;
        
        String folder = "cifar-10/noisy-cifar-10-batches/train";
        char path_folder[100] = "cifar-10/noisy-cifar-10-batches/";
        String str(path_folder);

        if (mkdir(path_folder,0755) != 0){
            cout << "Cannot create folder at " + folder << endl;
        }

        for(int img_num =0; img_num < 20 ;img_num++){
            bin_to_Mat(dataset.training_images[img_num],img);

            path = folder + to_string(img_num) + ".jpg";
            bool check = imwrite(path, img);

            if (check == false) {
            cout << "Mission - Saving the image, FAILED" << endl;
            }
                            
        }
        cout << "Successfully saved the image : 1 to 20 at " + str << endl;
    }

}

#endif //NOISE_CIFAR_10_CPP


int main(int argc,char ** argv){
    noise::cifar_add_noise();
    noise::test_read_cifar_noisy();
    return 0;
}