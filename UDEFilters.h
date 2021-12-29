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
    double pr;
} UDEF;//unbiased defocus equlization filters

typedef struct {
    double dpStar = 0.;
    double qp =0.;
//    cv::Point2d vpStar = cv::Point2d(0,0);//TODO 先不考虑光流
    double simgap_2=0.;//simgap^-2
    double Qp(double d,cv::Point2d v=cv::Point2d(0,0)){
       return simgap_2 * pow(d-dpStar,2) + qp;
    }
} QP_FITER;
class UDEFilters {
private:
    double _sigmai;//图像白噪声，需要通过图像标定计算
    std::vector<UDEF> _udefList;
    cv::Size _Sz;
public:
    void init(double depthMin, double depthMax, int kerSize,double sigmai, int patchWidth, int patchHeight);
    QP_FITER GetQpFiter(cv::Mat i1, cv::Mat i2);
};


#endif //DFDWILD_UDEFILTERS_H
