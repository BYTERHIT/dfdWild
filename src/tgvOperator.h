//
// Created by laiwenjie on 2021/10/26.
//

#ifndef DEPTHINPAINTER_TGVOPERATOR_H
#define DEPTHINPAINTER_TGVOPERATOR_H
#include <opencv2/opencv.hpp>
#include "mat_vector.h"
#define d_u_w
#define USING_L2
//#define USING_BACKWARD

typedef struct {
    int idx;//addresss offset
    //grad normerlized grad [dy*dy,-dx*dy,-dy*dx,dx*dx]
    double tGradProjMtx[2][2];//切线方向的投影矩阵
} EDGE_GRAD;

typedef struct {
    cv::Mat norm;
    double min;
    double max;
} MAX_MIN_NORM;

mat_vector GetWSteps(int rows, int cols);
mat_vector GetUstepsUsingMat(std::vector<EDGE_GRAD> edgeGrad, int rows, int cols);
mat_vector GetSteps(mat_vector edgeGrad, int rows, int cols, double alpha_u, double alpha_w, double alpha);
double GetFidelityCost(cv::Mat g, cv::Mat u, double lambda);
double GetTgvCost(cv::Mat u, mat_vector w, mat_vector edgeGrad, double alpha_u, double alpha_w);
double GetTgvCost(cv::Mat u, mat_vector w, std::vector<EDGE_GRAD> edgeGrad, double alpha_u, double alpha_w);
double GetEnerge(cv::Mat u,cv::Mat g, mat_vector w, std::vector<EDGE_GRAD> edgeGrad, double lambda = 1., double alpha_u = 1.0, double alpha_w=2.0);
double GetEnerge(cv::Mat u,cv::Mat g, mat_vector w, mat_vector edgeGrad, double lambda, double alpha_u, double alpha_w);
cv::Mat G_OPERATOR(cv::Mat g, cv::Mat uBar,double to, double lambda);
cv::Mat G_OPERATOR(cv::Mat g, cv::Mat uBar, cv::Mat to, double lambda, double thresh);
cv::Mat G_OPERATOR(cv::Mat g, cv::Mat uBar, cv::Mat to, cv::Mat lambda, double thresh);
mat_vector F_STAR_OPERATOR(mat_vector pBar, double alpha);
mat_vector D_OPERATOR(std::vector<EDGE_GRAD> edgeGrad, mat_vector du);
mat_vector D_OPERATOR(mat_vector edgeGrad, mat_vector du);
mat_vector secondOrderDivergenceForward(mat_vector second_order_derivative, bool ciculant = false);
mat_vector secondOrderDivergenceBackward(mat_vector grad);
mat_vector symmetrizedSecondDerivativeForward(mat_vector grad,bool circulant = false);
mat_vector symmetrizedSecondDerivativeBackward(mat_vector grad);
cv::Mat divergenceForward(mat_vector grad, bool circulant = false);
cv::Mat divergenceBackward(mat_vector grad );
mat_vector  derivativeForward(cv::Mat input,bool circulant = false);
mat_vector  derivativeBackward(cv::Mat input);
mat_vector  GetDGradMtx(cv::Mat grayImg, double gama = 0.75 , double beta = 10.);
mat_vector GetTensor(cv::Mat spMap, cv::Mat grayImg, cv::Mat depth = cv::Mat());
MAX_MIN_NORM MaxMinNormalizeNoZero(cv::Mat input);


#endif //DEPTHINPAINTER_TGVOPERATOR_H
