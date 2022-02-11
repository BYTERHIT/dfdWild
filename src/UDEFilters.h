//
// Created by bytelai on 2021/12/27.
//

#ifndef DFDWILD_UDEFILTERS_H
#define DFDWILD_UDEFILTERS_H
#include <opencv2/opencv.hpp>
#include <vector>
#include "parameters.h"
#include "BCCB.h"
typedef struct
{
    BCCB G1;
    BCCB G2;
    double dep;
    double simga1;
    double simga2;
    cv::Mat pr;
} UDEF;//unbiased defocus equlization filters

typedef struct qp_filter_st{
    double dpStar = 0.;
    double qp =0.;
    double simgap_2=0.;//simgap^-2
    double Qp(double d,cv::Point2d v=cv::Point2d(0,0)){
       return simgap_2 * pow(d-dpStar,2) + qp;
    }
}QP_FITER;
typedef struct qp_fitter_mat{
    cv::Mat dpStar;
    cv::Mat qp;
    cv::Mat sigmapm2;//sigmap^-2
    cv::Mat Qp(cv::Mat d){
        cv::Mat tmp;
        cv::pow(d-dpStar,2,tmp);
        return sigmapm2.mul(tmp)  + qp;
    }
} QP_FITER_MAT;
QP_FITER_MAT GetQpFiter(cv::Mat i1, cv::Mat i2,int kerSize, double sigmai, double depMax, double depMin);


#endif //DFDWILD_UDEFILTERS_H
