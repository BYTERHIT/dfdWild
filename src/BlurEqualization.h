//
// Created by bytelai on 2021/12/5.
//

#ifndef DFDWILD_BLUREQUALIZATION_H
#define DFDWILD_BLUREQUALIZATION_H
#include <opencv2/opencv.hpp>
#include "mat_vector.h"
mat_vector BlurEqualization(cv::Mat g1, cv::Mat g2, double beta, double thresh, int kerSize);


#endif //DFDWILD_BLUREQUALIZATION_H
