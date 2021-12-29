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
    return (lensPower - 1./distance - 1./v0 )*aperSize*v0/pixSize;
}

UDEF GetUnbiasedDefocusEqulizationFilter(double dep,int kerSize, int M, int N)
{
    double sigma1 = GetBlurRadii(P1,dep,V0,PIX_SIZE,APERTURE_SIZE);
    double sigma2 = GetBlurRadii(P2,dep,V0,PIX_SIZE,APERTURE_SIZE);
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
    Mat planes[2];
    Mat FpH2Re,FpH1Re;
    if(FpH1.channels() >1)
    {
        split(FpH1,planes);
        FpH1Re = planes[0];
    }
    else
        FpH1Re = FpH1;

    if(FpH2.channels() >1)
    {
        split(FpH2,planes);
        FpH2Re = planes[0];
    }
    else
        FpH2Re = FpH2;
    Mat Den;
    sqrt(FpH1Re+FpH2Re,Den);
    Den = Den + FLT_EPSILON;
    Mat Fg1,Fg2,g1,g2;
    if(Fh1.channels() >1)
    {
        Mat re,im;
        split(Fh1,planes);
        divide(planes[0],Den,re);
        divide(planes[1],Den,im);
        planes[0] = re;
        planes[1] = im;
        merge(planes,2,Fg2);
        dft(Fg2,g2,DFT_INVERSE + DFT_SCALE, 0);
    }
    if(Fh2.channels() >1)
    {
        Mat re,im;
        split(Fh2,planes);
        divide(planes[0],Den,re);
        divide(planes[1],Den,im);
        planes[0] = re;
        planes[1] = im;
        merge(planes,2,Fg1);
        dft(Fg1,g1,DFT_INVERSE + DFT_SCALE, 0);
    }
    BCCB G1(g1,N,M,true);
    BCCB G2(g2,N,M,true);
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

void UDEFilters::init(double depthMin, double depthMax, int kerSize,double sigmai, int patchWidth, int patchHeight) {
    double deltaDep = (depthMax - depthMin) / double(DEPTH_PRE_CACULATE);
    int M = patchHeight + kerSize -1;
    int N = patchWidth + kerSize -1;
    _Sz.width = N;
    _Sz.height = M;
    _sigmai = sigmai;
    for(int i = 0 ;i<DEPTH_PRE_CACULATE;i++)
    {
        double dep = depthMin + i * deltaDep;
        UDEF udef= GetUnbiasedDefocusEqulizationFilter(dep, kerSize, M, N);
        _udefList.push_back(udef);
    }
}
QP_FITER UDEFilters::GetQpFiter(cv::Mat i1, cv::Mat i2)
{

    double minCost = DBL_MAX;
    double depStar = 0;
    UDEF bestFilters;
    double qp, sigmaP2 = 0.;
    for(auto iter = _udefList.begin(); iter<_udefList.end(); iter++)
    {
        double cost = -logPr(iter->G1,iter->G2,i1,i2,_sigmai);
        double pr = exp(-cost);
        iter->pr = pr;
        if(minCost > cost)
        {
            minCost = cost;
            bestFilters = *iter;
            depStar = iter->dep;
        }
    }
    double f=0,d=0;
    for(auto iter = _udefList.begin();iter<_udefList.end();iter++)
    {
        f += pow(iter->dep-depStar,2)*iter->pr;
        d += iter->pr;
    }
    qp = minCost/pow(_sigmai,2)/2.;
    sigmaP2 = f/d;//simgap^2
}
