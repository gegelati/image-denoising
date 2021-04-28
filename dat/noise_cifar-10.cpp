#ifndef NOISE_CIFAR_10_CPP
#define NOISE_CIFAR_10_CPP

#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include "../src/cifar/cifar10_reader.hpp"
#include <opencv4/opencv2/quality/qualitymse.hpp>
#include <sys/stat.h> 
#include <sys/types.h>
#include <string>

//Noise parameters
#define SIGMA_GAUSSIAN 50     /// Value of sigma parameter for Gaussian law
#define GAUSSIAN_BLUR_RATIO 0.25 ///Ratio that determines the size of the kernel (based on the size of an image) for blurring an image (0 to 1)
#define QUALITY_JPEG 10       /// Quality (0 to 100) setting for JPEG compression
#define SUPERRESOLUTION_DOWN 0.25 ///Ratio for downsampling and upsampling an image to add superresoltion noise (0 to 1)
#define BERNOUILLI_AMOUNT 0.10 ///Amount of black and white pixel (0 to 1) for Bernouilli noise (salt and pepper)   


#define NB_BYTE_CHAN 1024   /// Number of bytes in a single channel 
#define NB_IMG_BATCH 10000   ///Number of images in a single batch
#define CIFAR_10_DATA "cifar-10/cifar-10-batches-bin" /// Path to the binary CIFAR-10 dataset
#define NOISY_CIFAR_10_DATA "cifar-10/noisy-cifar-10-batches-bin"   /// Path to the binary noisy CIFAR-10 dataset

using namespace std;

namespace noise{
    /**
     * \brief Regroups all type of noise
     */ 
    enum Type_noise{
        GAUSSIAN_NOISE,GAUSSIAN_BLUR,JPEG_BLOCKING,SUPER_RESOLUTION,BERNOUILLI,TYPE_NOISE_MAX
    };


    /**
     * \brief Convert a vector of pixel (1024 * 3) into an image 32 * 32 with 3 channel RGB
     * \param[in] in_bin_img Vector of pixel (1024 * 3)
     * \param[in] out_img Image (Mat under Opencv) of 32 * 32 with 3 channel RGB
     */
    void bin_to_Mat(const vector<uchar>& in_bin_img,cv::Mat out_img){
        for (int chan = 0; chan <3 ; chan++){//three chanels
            int x = 0,y =0;
            for (int byte = 0; byte < NB_BYTE_CHAN; byte++){
                out_img.at<cv::Vec3b>(x,y)[chan] = (uchar)(in_bin_img[byte + NB_BYTE_CHAN*chan]);   //Read values from the vector
                y++;
                if (y == 32){
                    y = 0;
                    x++;
                }
            }
        }
    }

    /**
     * \brief Convert an image 32 * 32 with 3 channel RGB into a vector of pixel (1024 * 3)
     * \param[in] out_img Image (Mat under Opencv) of 32 * 32 with 3 channel RGB
     * \param[in] in_bin_img Vector of pixel (1024 * 3)
     */
    void Mat_to_bin(const cv::Mat in_img, vector<uchar> & out_bin_img){
        for (int chan = 0; chan < 3; chan++){
            for(int row =0; row < in_img.rows; row++){
                for (int col = 0; col < in_img.cols; col++){
                    out_bin_img[col + in_img.cols * row + chan * NB_BYTE_CHAN] = (uchar)(in_img.at<cv::Vec3b>(row,col)[chan]);
                }
            }
        }
    }

    /**
     * \brief Compute the MSE between two images
     * \param[in] img Input image
     * \param[in] noisy_img Input noisy image 
     * \return MSE value
     */ 
    double MSE_compute(const cv::Mat & img, const cv::Mat & noisy_img){
        cv::Scalar mse;
        //Calculation of MSE
        mse = cv::quality::QualityMSE::compute(img,noisy_img,cv::noArray());
        //cout << "MSE opencv : " << mse[0] << endl;
        return mse[0];
    }

    /**
     * \brief Add Gaussian noise to an image
     * \param[in] img Input image 
     * \param[in] mean Mean parameter for the Gaussian law
     * \param[in] sigma Sigma (standard deviation) parameter for the Gaussian law
     * \return Noisy image
     */ 
    cv::Mat add_gaussian_noise(const cv::Mat & img, int mean, int sigma){
        cv::Mat noisy_img = img.clone();
        cv::Mat my_noise(img.rows,img.cols,CV_MAKE_TYPE(CV_64F,img.channels())); //Creation of our noise

        vector<int> m(img.channels(),mean);
        vector<int> s(img.channels(),sigma);

        cv::randn(my_noise,m,s); //add noise

        return noisy_img += my_noise;
    }


    /**
     * \brief Add Gaussian Blur noise to an image
     * \param[in] img Input image
     * \param[in] ratio Ratio that determines the size of the kernel for blurring an image (0 to 1)
     * \return Noisy image
     */ 
    cv::Mat add_gaussian_blur(const cv::Mat & img, float ratio){
        cv::Mat img_blur = img.clone();

        //Compute the size of the kernel thanks to the image size
        int width = (int)(ratio*img.rows); //Computed numbers must be odd numbers
        if(width %2 ==0){
            width--;
        }
        int height = (int)(ratio*img.cols); //Computed numbers must be odd numbers
        if(height %2 ==0){
            height--;
        }

        //Adding gaussian blur
        cv::GaussianBlur(img_blur,img_blur,cv::Size(width,height),0);
        
        return img_blur;
    }


    /**
     * \brief Add JPEG-blocking to an image
     * \param[in] img Input image
     * \param[in] quality Quality parameter for compression
     * \return Noisy image
     */ 
    cv::Mat add_JPEG_blocking(const cv::Mat & img, int quality){
        vector<int> compression_params; //Define imwrite vector parameter for compression
        compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
        compression_params.push_back(quality);

        //Saving the image
        cv::imwrite("cifar-10/JPEG_block.jpg",img,compression_params);
                
        return cv::imread("cifar-10/JPEG_block.jpg");
    }

    /**
     * \brief Add supperresolution noise to an image 
     * \param[in] img Input image
     * \param[in] ratio Ratio for downsampling and upsampling an image 
     * \return Noisy image
     */ 
    cv::Mat add_superresolution_noise(const cv::Mat & img, float ratio){
        cv::Mat super_image = img.clone();
        //Downsampling of the image
        cv::resize(super_image,super_image, cv::Size(),ratio,ratio,cv::INTER_AREA);

        //Upsampling (recover original size)
        cv::resize(super_image,super_image,cv::Size(img.rows,img.cols), 0, 0,cv::INTER_LINEAR);

        return super_image;
    }


     /**
     * \brief Add Bernouilli noise to an image (salt and pepper noise)
     * \param[in] img Input image
     * \param[in] amount Amount of black and white pixel (0 to 1)
     * \return Noisy image
     */ 
    cv::Mat add_Bernouilli_noise(const cv::Mat & img, float amount){
        cv::Mat saltpepper_noise = cv::Mat::zeros(img.rows, img.cols,CV_8U);

        cv::randu(saltpepper_noise,0,255); //Uniform law 

        cv::Mat black = saltpepper_noise < (BERNOUILLI_AMOUNT/2.0) * 255; //Keep pixel under a value threshold and set them to 1
        cv::Mat white = saltpepper_noise > 255 - ((BERNOUILLI_AMOUNT/2.0) * 255); //Keep pixel above a value threshold and set them to 1

        cv::Mat saltpepper_img = img.clone();
        saltpepper_img.setTo(255,white); //Add white pixels
        saltpepper_img.setTo(0,black);  //Add black pixels

        return saltpepper_img;
    }


    /**
     * \brief It applies a randomly chosen noise to an image
     * \param[in] img Input image
     * \param[in] wf_type_noise Ofstream to print the chosen noise and applied to the image
     * \return Noisy image
     */ 
    cv::Mat apply_rand_noise(const cv::Mat & img, ofstream & wf_type_noise){
        //Chooses a random value between 0 and the maximum number of type of noise available
        int choice = rand() % TYPE_NOISE_MAX; 
        switch (choice)
        {
        case GAUSSIAN_NOISE:
            wf_type_noise <<" : Gaussian noise";
            return add_gaussian_noise(img,0,SIGMA_GAUSSIAN);

        case GAUSSIAN_BLUR:
            wf_type_noise <<  " : Gaussian blur";
            return add_gaussian_blur(img,GAUSSIAN_BLUR_RATIO);

        case JPEG_BLOCKING:
            wf_type_noise << " : JPEG blocking";
            return add_JPEG_blocking(img,QUALITY_JPEG);
        
        case SUPER_RESOLUTION:
            wf_type_noise << " : Super resolution";
            return add_superresolution_noise(img,SUPERRESOLUTION_DOWN);

        case BERNOUILLI:
            wf_type_noise << " : Bernouilli";
            return add_Bernouilli_noise(img,BERNOUILLI_AMOUNT);

        default:
            return img;
        }

    }

    /**
     * \brief Add noise the CIFAR-10 dataset
     * Various type of noise are added randomly on all of the dataset : Gaussian noise, JPEG-blocking, Bernouilli, Superresolution, Gaussian Blur 
     */ 
    int cifar_add_noise(void){
        
        //Load dataset
        cout << "Loading dataset" << endl;
        auto dataset = cifar::read_dataset<vector, vector, uint8_t, uint8_t>(CIFAR_10_DATA);
        cv::Mat img(32,32,CV_8UC3);

        string path;
        string folder = "cifar-10/cifar-10-batches/train";
        char path_folder[100] = "cifar-10/cifar-10-batches/";
        string str(path_folder);

        //Creation of a folder to store 20 noise-free dataset images
        if (mkdir(path_folder,0755) != 0){
            cout << "Cannot create folder at " + folder << endl;
        }

        //Saving 20 images of the dataset
        for(int img_num =0; img_num < 20 ;img_num++){
            bin_to_Mat(dataset.training_images[img_num],img);

            path = folder + to_string(img_num) + ".jpg";
            bool check = imwrite(path, img);

            if (check == false) {
            cout << "Mission - Saving the image, FAILED" << endl;
            }                     
        }
        cout << "Successfully saved the image : 1 to 20 at " + str << endl;

        //Creating a file to print the recap of noising
        ofstream wf_type_noise;
        int img_index =0;
        wf_type_noise.open("Noise_dataset_recap.txt", ios::out);
        wf_type_noise << "Recap of noise for training batches(0 to 49999)" << endl;
        //Noising training images
        cout << "Noising training images" << endl;
        for(auto &img_bin : dataset.training_images){
            bin_to_Mat(img_bin,img);
            wf_type_noise <<  img_index;
            //Noise application
            cv::Mat img_noisy = apply_rand_noise(img,wf_type_noise);
            wf_type_noise <<" | MSE = " <<MSE_compute(img,img_noisy) << endl;
            img_index++;
            Mat_to_bin(img_noisy,img_bin);
        }

        //Noising testing images
        wf_type_noise << "Recap of noise for testing batch(50000 to 59999)" << endl;
        cout << "Noising testing images" << endl;
        for(auto &img_bin_t : dataset.test_images){
            bin_to_Mat(img_bin_t,img);
            wf_type_noise <<  img_index;
            //Noise application
            cv::Mat img_noisy = apply_rand_noise(img,wf_type_noise);
            wf_type_noise <<" | MSE = " <<MSE_compute(img,img_noisy) << endl;
            img_index++;
            Mat_to_bin(img_noisy,img_bin_t);
        }

        wf_type_noise.close();
        //Tests that file is well written
        if(!wf_type_noise.good()) {
            cout << "Error occurred at writing time !" << endl;
            return EXIT_FAILURE;
        }

        //Saving noisy images in new binaries files
        string filename_data_batch = "/data_batch_";
        string filename_test_batch = "/test_batch";
        string noisy_cifar_path = NOISY_CIFAR_10_DATA;
        string filename;
        ofstream wf;

        if (mkdir(NOISY_CIFAR_10_DATA,0755) != 0){
            cout << "Cannot create folder at " + noisy_cifar_path << endl;
        }

        //Save training batches
        cout << "Saving training images" << endl;
        for(int batch_num =1; batch_num <= 5 ;batch_num++){
            filename = noisy_cifar_path + filename_data_batch + to_string(batch_num) + ".bin";

            wf.open(filename, ios::out | ios::binary); //Binary file stream creation

             if(!wf) {
                cout << "Cannot open file !" << endl;
                return EXIT_FAILURE;
            }
            
            //Saving of noisy images in a binary file
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

        wf.open(filename, ios::out | ios::binary); //Binary file stream creation

        if(!wf) {
            cout << "Cannot open file !" << endl;
            return EXIT_FAILURE;
        }
        
        //Saving of noisy images in a binary file
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

    /**
     * \brief Test of the reading of the noisy dataset and saving sample of images 
     */ 
    void test_read_cifar_noisy(){
        cout << endl << "Test loading noisy dataset" << endl;
        //Load dataset
        
        auto dataset = cifar::read_dataset<vector, vector, uint8_t, uint8_t>(NOISY_CIFAR_10_DATA);
        
        cv::Mat img(32,32,CV_8UC3);
        
        string path;
        
        string folder = "cifar-10/noisy-cifar-10-batches/train";
        char path_folder[100] = "cifar-10/noisy-cifar-10-batches/";
        string str(path_folder);

        //Creation of a folder to store 20 noisy images of the noisy dataset
        if (mkdir(path_folder,0755) != 0){
            cout << "Cannot create folder at " + folder << endl;
        }

        //Saving noisy images
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