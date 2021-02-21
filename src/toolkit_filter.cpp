#include "toolkit_filter.h"


void toolkit::mean_filter_3x3(vector<double>& in_img,vector<double>& out_img){
    Mat img(32,32,CV_8UC3);
    toolkit::bin_to_Mat(in_img,img);
    boxFilter(img,img,-1,Size(3,3));
    toolkit::Mat_to_bin(img,out_img);
}

void toolkit::mean_filter_5x5(vector<double>& in_img,vector<double>& out_img){
    Mat img(32,32,CV_8UC3);
    toolkit::bin_to_Mat(in_img,img);
    boxFilter(img,img,-1,Size(5,5));
    toolkit::Mat_to_bin(img,out_img);
}

void toolkit::median_filter_3x3(vector<double>& in_img,vector<double>& out_img){
    Mat img(32,32,CV_8UC3);
    toolkit::bin_to_Mat(in_img,img);
    medianBlur(img,img,3);
    toolkit::Mat_to_bin(img,out_img);
}

void toolkit::median_filter_5x5(vector<double>& in_img,vector<double>& out_img){
    Mat img(32,32,CV_8UC3);
    toolkit::bin_to_Mat(in_img,img);
    medianBlur(img,img,5);
    toolkit::Mat_to_bin(img,out_img);
}

void toolkit::gaussian_filter_3x3(vector<double>& in_img,vector<double>& out_img){
    Mat img(32,32,CV_8UC3);
    toolkit::bin_to_Mat(in_img,img);
    GaussianBlur(img,img,Size(3,3),0);
    toolkit::Mat_to_bin(img,out_img);
}

void toolkit::gaussian_filter_5x5(vector<double>& in_img,vector<double>& out_img){
    Mat img(32,32,CV_8UC3);
    toolkit::bin_to_Mat(in_img,img);
    GaussianBlur(img,img,Size(5,5),0);
    toolkit::Mat_to_bin(img,out_img);
}


double toolkit::mse_compute_v1(const Mat& img, const Mat& noisy){
    double sum =0.0;
    int diff;
    //cout << "chan : " << img.channels() << endl;

    for (int i = 0; i < img.channels(); i++){
        for (int row = 0; row < img.rows; row++){
            for (int col = 0; col < img.cols; col++){
                //cout << " (" << (int)img.at<uchar>(row,col) << "," << (int)noisy.at<uchar>(row,col)  << ")" ;
                diff = ((double)img.at<uchar>(row,col)) - ((double)noisy.at<uchar>(row,col));
                sum +=   diff *diff;
            }
        } 
    }    

    //mean compute

    double mse = (double)sum /(double) (img.channels() * img.total());
    cout << "sum : " << sum << "| MSE : " << mse << endl;

    return mse;
}

double toolkit::MSE_compute(const vector<double>& img,const vector<double>& noisy_img){
    Mat img_mat(32,32,CV_8UC3);
    Mat noisy_img_mat(32,32,CV_8UC3);
    toolkit::bin_to_Mat(img,img_mat);
    toolkit::bin_to_Mat(noisy_img,noisy_img_mat);
    Scalar mse;
    mse = quality::QualityMSE::compute(img_mat,noisy_img_mat,noArray());
    //cout << "MSE opencv : " << mse[0] << endl;
    return mse[0];
}

double toolkit::PSNR_compute(const vector<double>& img,const vector<double>& noisy_img){
    Mat img_mat(32,32,CV_8UC3);
    Mat noisy_img_mat(32,32,CV_8UC3);
    toolkit::bin_to_Mat(img,img_mat);
    toolkit::bin_to_Mat(noisy_img,noisy_img_mat);
    Scalar psnr;
    psnr = quality::QualityPSNR::compute(img_mat,noisy_img_mat,noArray());
    //cout << "PSNR opencv : " << psnr[0] << endl;
    return psnr[0];
}

void toolkit::bin_to_Mat(const vector<double>& in_bin_img, Mat& out_img){
    for (int chan = 0; chan <3 ; chan++){//three chanels
        int x = 0,y =0;
        for (int byte = 0; byte < NB_BYTE_CHAN; byte++){
            out_img.at<Vec3b>(x,y)[chan] = (uchar)(in_bin_img[byte + NB_BYTE_CHAN*chan]);
            y++;
            if (y == 32){
                y = 0;
                x++;
            }
        }
    }
}

void toolkit::Mat_to_bin(const Mat& img_in,vector<double> & out_bin_img){
    for (int chan = 0; chan < 3; chan++){
        for(int row =0; row < img_in.rows; row++){
            for (int col = 0; col < img_in.cols; col++){
                out_bin_img[col + img_in.cols * row + chan * NB_BYTE_CHAN] = (double)(img_in.at<Vec3b>(row,col)[chan]);
            }
        }
    }
}

bool toolkit::save_Image(const vector<double> &img_in, u_int64_t index) {
    Mat img_mat(32,32,CV_8UC3);
    bin_to_Mat(img_in,img_mat);
    String path = RESULT_DENOISING_CIFAR_10_LOCATION;
    path = path + "/img" + to_string(index) + ".jpg";
    char * char_path;
    char_path = &path[0];

    return imwrite(char_path,img_mat);
}
