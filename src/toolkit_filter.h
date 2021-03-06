#ifndef TOOLKIT_FILTER_H
#define TOOLKIT_FILTER_H

#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/quality/qualitymse.hpp>
#include <opencv4/opencv2/quality/qualitypsnr.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/xphoto.hpp>
#include "cifar/cifar10_reader.hpp"

using namespace std;

#define NB_BYTE_CHAN 1024

namespace toolkit {

    void mean_filter_3x3(vector<double>& in_img,vector<double>& out_img);

    void mean_filter_5x5(vector<double>& in_img,vector<double>& out_img);

    void median_filter_3x3(vector<double>& in_img,vector<double>& out_img);

    void median_filter_5x5(vector<double>& in_img,vector<double>& out_img);

    void gaussian_filter_3x3(vector<double>& in_img,vector<double>& out_img);

    void gaussian_filter_5x5(vector<double>& in_img,vector<double>& out_img);

    void bm3d_filter(vector<double>& in_img,vector<double>& out_img, float filter_strength);

    double mse_compute_v1(const cv::Mat &img, const cv::Mat &noisy);

    double MSE_compute(const vector<double>& img,const vector<double>& noisy_img);

    double PSNR_compute(const vector<double>& img,const vector<double>& noisy_img);

    void bin_to_Mat(const vector<double>& in_bin_img, cv::Mat& out_img);

    void Mat_to_bin(const cv::Mat& img_in,vector<double> & out_bin_img);

    bool save_Image(const vector<double> & img_in, u_int64_t index);

}

#endif //TOOLKIT_FILTER_H