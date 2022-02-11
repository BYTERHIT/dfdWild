//
// Created by bytelai on 2022/1/5.
//
#include "AlgOperator.h"
#include "imageFeature.h"
#include <opencv2/opencv.hpp>
#include "utility.h"
#include <fstream>
using namespace cv;
using namespace std;

/*
 * not sqrt
 */
double ControlPointDistanceP2(CONTROL_POINT_INFO Cm, CONTROL_POINT_INFO Cn)
{
    return pow(Cm.f.pixLocation[0] - Cn.f.pixLocation[0],2) - pow(Cm.f.pixLocation[1] - Cm.f.pixLocation[1],2);
}


/***************************************
* xgv -- 【输入】指定X输入范围
* ygv -- 【输入】指定Y输入范围
* X   -- 【输出】Mat
* Y   -- 【输出】Mat
****************************************/
void meshgrid(const cv::Range &xgv, const cv::Range &ygv, cv::Mat &X, cv::Mat &Y) {
    std::vector<int> t_x, t_y;
    for (int i = xgv.start; i <= xgv.end; i++) t_x.push_back(i);
    for (int j = ygv.start; j <= ygv.end; j++) t_y.push_back(j);

    cv::repeat(cv::Mat(t_x).t(), t_y.size(), 1, X);
    cv::repeat(cv::Mat(t_y), 1, t_x.size(), Y);
}
#define USING_MAT


///*
// * @param simgadm2 simgad^-2
// */
//Mat PsiMn(PLANE_FITER Dm, PLANE_FITER Dn, Size imgSz, Mat simgadm2)
//{
//    Mat Dmp = GetDq<double>(Dm,imgSz);
//    Mat Dnp = GetDq<double>(Dn,imgSz);
//    Mat L2Norm;
//    pow(Dmp-Dnp, 2,L2Norm);
//    Mat psimn = 2.0*simgadm2.mul(L2Norm);
//    return psimn;
//}

/*
 * @param
 */
double Psi(PLANE_FITER Dm, PLANE_FITER Dn, Point p, double sigmad2)
{
    double Dmp = Dm.Dq(p);
    double Dnp = Dn.Dq(p);
    double L2Norm = pow(Dmp-Dnp,2);
    double sigma = abs(sigmad2);
    if (sigma == 0)
        sigma = EPSILON;

    double psi = 0.5 * L2Norm / sigma;
    return psi;
}




/*
 * @brief simgD^2 = D(X) = E(X^2)-(E(X))^2;
 */
Mat GetVariance(cv::Mat depth, int winWidth, int winHeight) {
    Mat dep2 = depth.mul(depth);
    Mat dep2Mean;
    Size ksize = Size(winWidth,winHeight);
    blur(dep2,dep2Mean,ksize);
    Mat depMean;
    blur(depth,depMean,ksize);
    Mat depMean2= depMean.mul(depMean);
    Mat d = abs(dep2Mean - depMean2);
    return d;
}

/*
 * @brief simgD^2 = D(X) = E(X^2)-(E(X))^2;
 */
double GetVariance(cv::Mat depth) {
    Mat mean,stdDev;
    meanStdDev(depth,mean,stdDev);
    return pow(stdDev.at<double>(0,0),2);
}


//int oIdx(int i,int j)
//{
//    assert(j>i);
//    return (SEG_NUMBER-1 + SEG_NUMBER -1-i-1)*(i)/2 + j - i-1;
//}

/*
 * Get sum(Bij)-Bij
 */
Mat LapMatrix(Mat Bij)
{
    Mat rowSum;
    reduce(Bij,rowSum,1,REDUCE_SUM);
    Mat dia = Mat::diag(rowSum);
    Mat ret = dia - Bij;
    return ret;
}

Mat getF(mat_vector B)
{
    int N = B[0].cols;
    Mat F = Mat::zeros(3*N,3*N,B[0].type());
    Mat F11 = LapMatrix(B[0]);
    Mat F12 = LapMatrix(B[1]);
    Mat F13 = LapMatrix(B[2]);
    Mat F22 = LapMatrix(B[3]);
    Mat F23 = LapMatrix(B[4]);
    Mat F33 = LapMatrix(B[5]);
    F11.copyTo(F(Rect(0, 0,N,N)));
    F12.copyTo(F(Rect(N, 0,N,N)));
    F12.copyTo(F(Rect(0, N,N,N)));
    F13.copyTo(F(Rect(2*N, 0,N,N)));
    F13.copyTo(F(Rect(0, 2*N,N,N)));
    F22.copyTo(F(Rect(N, N,N,N)));
    F23.copyTo(F(Rect(2*N, N,N,N)));
    F23.copyTo(F(Rect(N, 2*N,N,N)));
    F33.copyTo(F(Rect(2*N, 2*N,N,N)));
    return F;
}

cv::Mat fastGuidedFilter(cv::Mat I_org, cv::Mat p_org, int r, double eps, int s)
{
    /*
    % GUIDEDFILTER   O(N) time implementation of guided filter.
    %
    %   - guidance image: I (should be a gray-scale/single channel image)
    %   - filtering input image: p (should be a gray-scale/single channel image)
    %   - local window radius: r
    %   - regularization parameter: eps
    */

    cv::Mat I,_I;
    I_org.convertTo(_I, CV_64FC1, 1.0 / 255);

    resize(_I,I,Size(),1.0/s,1.0/s,1);



    cv::Mat p,_p;
    p_org.convertTo(_p, CV_64FC1, 1.0 / 255);
    //p = _p;
    resize(_p, p, Size(),1.0/s,1.0/s,1);

    //[hei, wid] = size(I);
    int hei = I.rows;
    int wid = I.cols;

    r = (2 * r + 1)/s+1;//因为opencv自带的boxFilter（）中的Size,比如9x9,我们说半径为4

    //mean_I = boxfilter(I, r) ./ N;
    cv::Mat mean_I;
    cv::boxFilter(I, mean_I, CV_64FC1, cv::Size(r, r));

    //mean_p = boxfilter(p, r) ./ N;
    cv::Mat mean_p;
    cv::boxFilter(p, mean_p, CV_64FC1, cv::Size(r, r));

    //mean_Ip = boxfilter(I.*p, r) ./ N;
    cv::Mat mean_Ip;
    cv::boxFilter(I.mul(p), mean_Ip, CV_64FC1, cv::Size(r, r));

    //cov_Ip = mean_Ip - mean_I .* mean_p; % this is the covariance of (I, p) in each local patch.
    cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);

    //mean_II = boxfilter(I.*I, r) ./ N;
    cv::Mat mean_II;
    cv::boxFilter(I.mul(I), mean_II, CV_64FC1, cv::Size(r, r));

    //var_I = mean_II - mean_I .* mean_I;
    cv::Mat var_I = mean_II - mean_I.mul(mean_I);

    //a = cov_Ip ./ (var_I + eps); % Eqn. (5) in the paper;
    cv::Mat a = cov_Ip / (var_I + eps);

    //b = mean_p - a .* mean_I; % Eqn. (6) in the paper;
    cv::Mat b = mean_p - a.mul(mean_I);

    //mean_a = boxfilter(a, r) ./ N;
    cv::Mat mean_a;
    cv::boxFilter(a, mean_a, CV_64FC1, cv::Size(r, r));
    Mat rmean_a;
    resize(mean_a, rmean_a, Size(I_org.cols, I_org.rows),1);

    //mean_b = boxfilter(b, r) ./ N;
    cv::Mat mean_b;
    cv::boxFilter(b, mean_b, CV_64FC1, cv::Size(r, r));
    Mat rmean_b;
    resize(mean_b, rmean_b, Size(I_org.cols, I_org.rows),1);

    //q = mean_a .* I + mean_b; % Eqn. (8) in the paper;
    cv::Mat q = rmean_a.mul(_I) + rmean_b;

    return q;
}

Mat standardize(Mat input, double stdev)
{
    
    Mat ret;
    Mat planes[3];
    if(input.channels() >1)
    {
        split(input,planes);
        for (int i = 0; i < 3; i++)
        {
            Mat u, dev, tmp;
            planes[i].convertTo(tmp, CV_64FC1);
            meanStdDev(tmp, u, dev);
            double sigma = dev.at<double>(0);
            if(sigma == 0)
                sigma = EPSILON;
            planes[i] = (tmp - u.at<double>(0)) / sigma * stdev;
        }
        merge(planes, 3, ret);

    }
    else
    {
        Mat u, dev,tmp;
        input.convertTo(tmp, CV_64FC1);
        meanStdDev(tmp, u, dev);
        double sigma = dev.at<double>(0);
        if(sigma == 0)
            sigma = EPSILON;
        ret = (tmp - u.at<double>(0)) / sigma * stdev;
    }
    
    
    return ret;
}

/*
 * equation A36
 * @param d [Nq,dBar,dBarVar]
 */
Mat CaculateEq_0(vector<Mat> aux, Mat sigmaD2, double tau_0, int patchWidth, int patchHeight) {
    double patchSize = patchHeight * patchWidth;
    Mat Nq;
    Mat dBar = aux[1];
    Mat dBarVar = aux[2];
    aux[0].convertTo(Nq, dBar.type());

    Mat tmp1 = Nq.mul(dBarVar,0.5);
    double sigma;
    if(sigmaD2.type() == CV_64FC1)
        sigma = sigmaD2.at<double>(0,0);
    else
        sigma = sigmaD2.at<float>(0,0);
    if(sigma == 0)
        sigma = EPSILON;
    if(sigmaD2.size() == Nq.size())
        divide(tmp1,abs(sigmaD2) + EPSILON,tmp1);
    else
        tmp1 = tmp1 * (1/sigma);
    Mat tmp = (patchSize - Nq) * tau_0;
    Mat eq_0 = tmp1 + tmp;
    return eq_0;
}

/*
 * equation A36
 * @param d [Nq,dBar,dBarVar]
 */
mat_vector CaculateEq_0(vector<AUX_TYPE> aux, Mat sigmaD2, double tau_0, int patchWidth, int patchHeight, int segNum) {
    double patchSize = patchHeight * patchWidth;
    mat_vector eq0V;
    for(int seg = 0; seg<segNum; seg++)
    {
        Mat Nq;
        Mat dBar = aux[seg].dBar;
        Mat dBarVar = abs(aux[seg].dBarVar);
        aux[seg].Nq.convertTo(Nq, dBar.type());
        Mat tmp1 = Nq.mul(dBarVar) * 0.5;
        double sigma;
        if(sigmaD2.type() == CV_64FC1)
            sigma = sigmaD2.at<double>(0,0);
        else
            sigma = sigmaD2.at<float>(0,0);
        if(sigma == 0)
            sigma = EPSILON;

        if(sigmaD2.size() == aux[0].Nq.size())
            divide(tmp1,abs(sigmaD2) + EPSILON,tmp1);
        else
            tmp1 = tmp1*(1/sigma);
        Mat tmp = (Nq - patchSize) * (-tau_0);
        Mat eq_0 = tmp1 +tmp;
        eq0V.addItem(eq_0);
    }

    return eq0V;
}





double Vpq(uchar sp, uchar sq, PARAM par)
{
    if(sp == sq)
        return 0;
    else
        return par.lambda_b;
}

std::ostream& operator<<(std::ostream& os,const PLANE_FITER & D ) {
    return os << D.cx << " " << D.cy << " " << D.b;
}

std::istream& operator>>(std::istream& is, PLANE_FITER & D ) {
    return is >> D.cx >> D.cy >> D.b;
}
std::ostream& operator<<(std::ostream& os,const IMAGE_FEATURE & f )
{
    //空格分开,方便stream的读入
    std::string output;
    output = std::to_string(f.pixLocation[0]) + " ";
    output += std::to_string(f.pixLocation[1]) + " ";
    output += std::to_string(f.lab[0]) + " " + std::to_string(f.lab[1]) + " " + std::to_string(f.lab[2]) + " ";
    for(int i = 0; i<EIG_NUMBER; i++)
    {
        output += std::to_string(f.spec_cluster_eigenv[i]) + " ";
    }
    output += std::to_string(f.dep) + " ";
    return os<<output;

}

std::istream& operator>>(std::istream& is, IMAGE_FEATURE & f )
{
    is>>f.pixLocation[0]>>f.pixLocation[1];
    is>>f.lab[0]>>f.lab[1]>>f.lab[2];
    for(int i = 0; i<EIG_NUMBER; i++)
    {
        is>>f.spec_cluster_eigenv[i];
    }
    is >> f.dep;
    return is;
}

std::ostream& operator<<(std::ostream& os,const CONTROL_POINT_INFO & controlP )
{
    //空格分开,方便stream的读入
    return os<<controlP.f<<" "<<controlP.t<<" "<<controlP.D;
}

std::istream& operator>>(std::istream& is, CONTROL_POINT_INFO & controlP )
{
    //空格分开,方便stream的读入
    return is>>controlP.f>>controlP.t>>controlP.D;
}
