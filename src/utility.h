#pragma once
#include <opencv2/opencv.hpp>
#include <string>
int createDirectory(std::string path);
std::string GetTimeString();
template<typename T>
int getCvType()
{
    int cvType = 0;
    if(typeid(T) == typeid(cv::float16_t))
        cvType = CV_16FC1;
    if(typeid(T) == typeid(float))
        cvType = CV_32FC1;
    if(typeid(T) == typeid(double))
        cvType = CV_64FC1;
    return cvType;
}

cv::Mat matread(const std::string& filename);
void matwrite(const std::string& filename, const cv::Mat& mat);
