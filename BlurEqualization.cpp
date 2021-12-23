//
// Created by bytelai on 2021/12/5.
//
/*
 * Xian T, Subbarao M. Depth-from-defocus: Blur equalization technique[C]//
 * Two-and Three-Dimensional Methods for Inspection and Metrology IV.
 * International Society for Optics and Photonics, 2006, 6382: 63820E.
 *
*/
#include "BlurEqualization.h"

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
#define EPSILON 1e-5f

Mat SolvePloy2nd(Mat coef, Mat&root)
{
    double a = coef.at<double>(0, 0);
    double b = coef.at<double>(0, 1);
    double b2 = b*b;
    double c = coef.at<double>(0, 2);
    double delta = b2 - 4 * a * c;
    double tmp;
    double root1,root2;
    if(a==0)
    {
        Mat ret = Mat::zeros(2,1,CV_64FC1);
        root1 = -c / b;
        ret.at<double>(0,0) = root1;
        return ret;
    }
    else if(delta >= 0)
    {
        Mat ret = Mat::zeros(2,1,CV_64FC1);
        tmp = sqrt(delta);
        root1 = (-b + tmp) / 2 / a;
        root2 = (-b - tmp) / 2 / a;
        ret.at<double>(0,0) = root1;
        ret.at<double>(1,0) = root2;
        root = ret.clone();
        return ret;
    }
    else
        return Mat();
}
Mat SolvePolyEq(Mat a, double b, Mat c)
{
    Mat sigma = Mat::zeros(a.rows,a.cols,CV_64FC1);
    int rows = a.rows;
    int cols = a.cols;
    for(int i =0;i<rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            double coefA = a.at<double>(i, j);
            double coefC = c.at<double>(i, j);
            Mat coef = (Mat_<double>(1,3) << coefA, b, coefC);
            Mat roots;
            SolvePloy2nd(coef, roots);
            double iniX = -c.at<double>(i,j) / b;//参考公式16,如果存在多重根的情况，应该和理想情况的sigma的解相近。
            if (roots.total() != 0) {
                double minResidu = abs(roots.ptr<double>(0)[0] - iniX);
                unsigned mink = 0;
                for (unsigned k = 0; k < roots.total(); k++) {
                    const double &root = roots.ptr<double>(0)[k];
                    if ((-EPSILON > root) || (root > EPSILON)) {
                        double residu = abs(roots.ptr<double>(0)[k] - iniX);
                        if (residu < minResidu) {
                            minResidu = residu;
                            mink = k;
                        }
                    }
                }
                float root = roots.ptr<double>(0)[mink];
//				if(root<-0.52 || root> 1)
//				    continue;
                sigma.at<double>(i,j) = root;
//			}
            } else {
                sigma.at<double>(i,j) = 0;
                continue;
            }
        }
    }
    return sigma;
}

Mat AverageInRoi(Mat sigma, Mat msk, int kerSize, int borderType)
{
    Mat sigmaAvg;
    Mat mskDen;
    Mat valid = sigma.mul(msk);
    blur(valid,sigmaAvg,{kerSize,kerSize},Point(-1,-1),borderType);
    blur(msk,mskDen,{kerSize,kerSize},Point(-1,-1),borderType);
    divide(sigmaAvg,mskDen,sigmaAvg);
    return sigmaAvg;
}

Mat GenMsk(Mat a, Mat lapG, Mat c, double b, double thresh)
{
    Mat delta;
    delta = b*b - 4*a.mul(c);
    Mat msk0,msk1,mskf;
    threshold(abs(lapG),msk0, thresh,1,THRESH_BINARY);
    threshold(delta,msk1,0,1,THRESH_BINARY);
    mskf = msk0.mul(msk1);
    return mskf;
}

mat_vector BlurEqualization(Mat g1, Mat g2, double beta, double thresh, int kerSize)
{
    Mat lapG1, lapG2;
    Mat G1,G2;
    g1.convertTo(G1,CV_64FC1);
    g2.convertTo(G2,CV_64FC1);
    Laplacian(G1,lapG1,CV_64FC1);
    Laplacian(G2,lapG2,CV_64FC1);
    Mat L1,L2;
    int borderType = BORDER_REFLECT101;
    blur(abs(lapG1),L1,{kerSize,kerSize},Point(-1,-1),borderType);
    blur(abs(lapG2),L2,{kerSize,kerSize},Point(-1,-1),borderType);

    Mat a1, a2, c1,c2;

    divide(lapG2,lapG1+DBL_EPSILON,a1);
    a1 = a1 - 1.;

    double b1 = 2*beta;
    double b2 = b1;
    divide(4*(G1-G2),lapG1 + DBL_EPSILON,c1);
    c1 = -c1 - beta*beta;

    divide(lapG1,lapG2 + DBL_EPSILON,a2);
    a2 = 1. - a2;

    divide(4*(G1-G2),lapG2 + DBL_EPSILON,c2);
    c2 = -c2 + beta*beta;

    Mat mskf1 = GenMsk(a1,lapG2,c1,b1,thresh);
    Mat mskf2 = GenMsk(a2,lapG1,c2,b2,thresh);

    Mat sigma1 = SolvePolyEq(a1,b1,c1);
    Mat sigma2 = SolvePolyEq(a2,b2,c2);
    Mat sigmaAvg1 = AverageInRoi(sigma1,mskf1,kerSize,borderType);
    Mat sigmaAvg2 = AverageInRoi(sigma2,mskf2,kerSize,borderType);
    Mat sigmaF1 = Mat::zeros(g1.rows,g1.cols,CV_64FC1);
    Mat sigmaF2 = Mat::zeros(g1.rows,g1.cols,CV_64FC1);

    double *sigmaF1Ptr = (double*) sigmaF1.data;
    double *sigmaF2Ptr = (double*) sigmaF2.data;
    double *sigmaAvg1Ptr = (double*) sigmaAvg1.data;
    double *sigmaAvg2Ptr = (double*) sigmaAvg2.data;
    double *L1Ptr = (double*) L1.data;
    double *L2Ptr = (double*) L2.data;

    for(int i = 0 ; i< g1.rows*g1.cols; i++){
       if(*L1Ptr >= *L2Ptr)
       {
           *sigmaF1Ptr = *sigmaAvg1Ptr;
           *sigmaF2Ptr = *sigmaAvg1Ptr - beta;
       }
       else
       {
           *sigmaF1Ptr = *sigmaAvg2Ptr + beta;
           *sigmaF2Ptr = *sigmaAvg2Ptr;
       }
       sigmaAvg1Ptr++;
       sigmaAvg2Ptr++;
       sigmaF1Ptr++;
       sigmaF2Ptr++;
       L1Ptr++;
       L2Ptr++;
    }
    mat_vector ret;
    ret.addItem(sigmaF1);
    ret.addItem(sigmaF2);
    return ret;
}