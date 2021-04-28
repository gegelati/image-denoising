#ifndef TOOLKIT_FILTER_H
#define TOOLKIT_FILTER_H

#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/quality/qualitymse.hpp>
#include <opencv4/opencv2/quality/qualitypsnr.hpp>
#include <opencv4/opencv2/quality/qualityssim.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/xphoto.hpp>
#include "cifar/cifar10_reader.hpp"

using namespace std;

#define NB_BYTE_CHAN 1024

/**
 * \brief Contains the filtering actions used by the toolchain and other functions for image management.
 * The filters present in the toolbox are very basic (mean, median, gaussian) with 1 a little more complex the bm3d
 * It contains also function to convert image in vector, to save images and to compute MSE and PSNR for noise calcultion
 */
namespace toolbox {

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

    double SSIM_compute(const vector<double>& img,const vector<double>& noisy_img);

    void bin_to_Mat(const vector<double>& in_bin_img, cv::Mat& out_img);

    void Mat_to_bin(const cv::Mat& img_in,vector<double> & out_bin_img);

    bool save_Image(const vector<double> & img_in, u_int64_t index);

}

#endif //TOOLKIT_FILTER_H