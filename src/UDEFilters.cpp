//
// Created by bytelai on 2021/12/27.
//

#include "UDEFilters.h"
#include <opencv2/opencv.hpp>
#include "BCCB.h"
#include "cmath"
using namespace cv;
using namespace std;
double GetBlurRadii(double lensPower, double distance, double v0, double pixSize,double aperSize)
{
    return abs((lensPower - 1./distance - 1./v0 )*aperSize*v0/pixSize);
}

UDEF GetUnbiasedDefocusEqulizationFilter(double dep,int kerSize, int M, int N)
{
    double sigma1 = GetBlurRadii(P2,dep,V0,PIX_SIZE,APERTURE_SIZE);
    double sigma2 = GetBlurRadii(PG,dep,V0,PIX_SIZE,APERTURE_SIZE);
    Mat h1 =  getGaussianKernel(kerSize, sigma1);//
    h1 = h1 * h1.t();
    Mat h2 = getGaussianKernel(kerSize, sigma2); //
    h2 = h2 * h2.t();
    BCCB H1(h1,N, M, true);
    BCCB H2(h2,N, M, true);
    Mat Fh1 = H1.GetEigen();
    Mat Fh2 = H2.GetEigen();
    Mat FpH1, FpH2;
    mulSpectrums(Fh1,Fh1,FpH1,0,true);
    mulSpectrums(Fh2,Fh2,FpH2,0,true);
    
    Mat FpH2Re,FpH1Re;
    if(FpH1.channels() >1)
    {
        Mat planes[2];
        split(FpH1,planes);
        FpH1Re = planes[0].clone();
    }
    else
        FpH1Re = FpH1.clone();

    if(FpH2.channels() >1)
    {
        Mat planes[2];
        split(FpH2,planes);
        FpH2Re = planes[0].clone();
    }
    else
        FpH2Re = FpH2.clone();
    Mat Den;
    sqrt(FpH1Re+FpH2Re,Den);
    Den = Den+ EPSILON;
    Mat Fg1,Fg2,g1,g2;
    if(Fh1.channels() >1)
    {
        Mat planes[2];
        Mat re,im;
        split(Fh1,planes);
        divide(planes[0],Den,re);
        divide(planes[1],Den,im);
        planes[0] = re;
        planes[1] = im;
        merge(planes,2,Fg2);
        dft(Fg2,g2,DFT_INVERSE + DFT_SCALE, 0);
        split(g2, planes);
        g2 = planes[0].clone();
    }
    if(Fh2.channels() >1)
    {
        Mat planes[2];
        Mat re,im;
        split(Fh2,planes);
        divide(planes[0],Den,re);
        divide(planes[1],Den,im);
        planes[0] = re;
        planes[1] = im;
        merge(planes,2,Fg1);
        dft(Fg1,g1,DFT_INVERSE + DFT_SCALE, 0);
        split(g1, planes);
        g1 = planes[0].clone();
    }
    BCCB G1(g1,N,M,false);
    BCCB G2(g2,N,M,false);
    UDEF ret;
    ret.G1 = G1;
    ret.G2 = G2;
    ret.simga1 = sigma1;
    ret.simga2 = sigma2;
    ret.dep = dep;
    return ret;

}

/*
 * @brief -logPr
 */
double logPr(BCCB G1, BCCB G2, Mat i1, Mat i2, double sigmai)
{
    Mat ig1 = G1*i1;
    Mat ig2 = G2*i2;
    double alpha = sum(ig1.mul(ig2))[0] / sum(ig2.mul(ig2))[0];
    Mat err = (ig1 - alpha * ig2);
    double energe = sum(err.mul(err))[0];
    return -energe / 2 / pow(sigmai, 2);
}
/*
 * i1,i2 should be gray and normalize
 */
Mat logPrMat(BCCB G1, BCCB G2, Mat i1, Mat i2, double sigmai)
{
    Size patchSz;
    patchSz.height = G1.imgRows() - i1.rows + 1;
    patchSz.width = G1.imgCols() - i1.cols + 1;
    Rect roi = Rect((patchSz.width - 1) / 2, (patchSz.height - 1) / 2, i1.cols, i1.rows);
    Mat ig1 = G1*i1;
    Mat ig2 = G2*i2;
    Mat sumg1g2,sumg2g2, sumg1g1;
    ig1 = ig1(roi);
    ig2 = ig2(roi);
    boxFilter(ig1.mul(ig2),sumg1g2,-1,patchSz,Point(-1,-1), false,BORDER_REFLECT101);
    boxFilter(ig2.mul(ig2),sumg2g2,-1,patchSz,Point(-1,-1), false,BORDER_REFLECT101);
    boxFilter(ig1.mul(ig1),sumg1g1,-1,patchSz,Point(-1,-1), false,BORDER_REFLECT101);
//    boxFilter(ig2,sumg2,-1,patchSz,Point(-1,-1), false,BORDER_REFLECT101);
    Mat alpha;

    divide(sumg1g2,sumg2g2 + DBL_EPSILON,alpha);
    //Mat tmp2 = ig2.mul(alpha);
    //Mat tmp = ig1 - tmp2;
    //Mat tmp3 = ig2 - tmp2;
    //Mat tmp4 = ig1 - ig2;
    Mat err = sumg1g1 -2* alpha.mul( sumg1g2) + alpha.mul(alpha).mul(sumg2g2);
    Mat tmp = err / 2 / pow(sigmai, 2);
    Mat energe = err;
    return -energe / 2 / pow(sigmai, 2);
}

/*
 * i1 image1, i2 image2, full size
 */
QP_FITER_MAT GetQpFiter(cv::Mat i1, cv::Mat i2, int kerSize, double sigmai, double depMax, double depMin)
{
    Mat grayI1, grayI2;
    if(i1.channels() == 3)
        cvtColor(i1,grayI1,COLOR_BGR2GRAY);
    else
        grayI1 = i1;
    if(i2.channels() == 3)
        cvtColor(i2,grayI2,COLOR_BGR2GRAY);
    else
        grayI2 = i2;
    grayI1.convertTo(grayI1, CV_64FC1);
    grayI1 = grayI1 / 255.;
    grayI2.convertTo(grayI2, CV_64FC1);
    grayI2 = grayI2 / 255.;
    int M = i1.rows + kerSize -1;
    int N = i1.cols + kerSize -1;

    Mat minCost = DBL_MAX * Mat::ones(i2.size(), CV_64FC1);
    Mat depStar = Mat::zeros(i2.size(), CV_64FC1);
    UDEF bestFilters;
    Mat qp = Mat::zeros(i1.size(),CV_64FC1);
    Mat sigmaPm2 = Mat::zeros(i1.size(),CV_64FC1);
    
    Mat f=Mat::zeros(i1.size(), CV_64FC1);
    Mat d=Mat::zeros(i1.size(), CV_64FC1);
    double deltaDep = (depMax - depMin) / double(DEPTH_PRE_CACULATE);
    Mat prv[DEPTH_PRE_CACULATE];
    for(int i = 0 ;i<DEPTH_PRE_CACULATE;i++)
    {
        Mat pr = Mat::zeros(i1.size(), CV_64FC1);
        double dep = depMin + i * deltaDep;
        UDEF udef= GetUnbiasedDefocusEqulizationFilter(dep, kerSize, M, N);
        Mat cost = -logPrMat(udef.G1,udef.G2, grayI1, grayI2,sigmai);
        exp(-cost,pr);
        prv[i] = pr;
        double* depStarPtr = (double*)depStar.data;
        double* costPtr = (double*)cost.data;
        double* minCostPtr = (double*)minCost.data;
        for(int i = 0; i< i1.size().area();i++)
        {
            if(*minCostPtr > *costPtr)
            {
                *minCostPtr = *costPtr;
                *depStarPtr = dep;
            }
            minCostPtr++;
            costPtr++;
            depStarPtr++;
        }
    }
    for(int i = 0 ;i<DEPTH_PRE_CACULATE;i++)
    {
        double dep = depMin + i * deltaDep;
        Mat tmp;
        pow(dep-depStar,2,tmp);
        f += prv[i].mul(tmp);
        d += prv[i];
    }
    qp = minCost/pow(sigmai,2)/2.;
    divide(d,f + EPSILON,sigmaPm2);
    //sigmaPm2 = d/f;//simgap^-2
    QP_FITER_MAT qpFiterMat;
    qpFiterMat.dpStar = depStar;
    qpFiterMat.qp = qp;
    qpFiterMat.sigmapm2 = sigmaPm2;
    return qpFiterMat;
}
