//
// Created by bytelai on 2022/1/16.
//

#ifndef DFDWILD_MYTYPES_H
#define DFDWILD_MYTYPES_H
#include <opencv2/opencv.hpp>
template<typename T>
struct weight_st {
    int indx;
    T weight;
};

typedef struct aux_st{
    cv::Mat Nq;
    cv::Mat dBar;
    cv::Mat dBarVar;
} AUX_TYPE;
#endif //DFDWILD_MYTYPES_H
