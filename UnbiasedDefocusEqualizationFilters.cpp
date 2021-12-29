//
// Created by bytelai on 2021/12/1.
//

#include "UnbiasedDefocusEqualizationFilters.h"
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
/*
 * @brief 获取模糊半径
 * @param p 液晶透镜加电压下光学系统光焦度
 * @param pg 液晶不加电压的光学系统光焦度
 * @param u0 初始对焦距离
 * @param a 光圈半径
 * @param d 物体深度
 * @return d处的点光源在p下的模糊核半径
 */
//double GetBlurRadii(double p,double pg, double u0, double a, double d)
//{
//    double v0 = 1./(pg-1./u0);
//    double v = p - 1./d;
//    double r = a*v0*(1./v0 - 1./v);
//    return abs(r);
//}
/*
 * @brief 获取圆盘模型的模糊psf的空域表示
 * @param radius 模糊核半径
 * @return 圆盘模型的psf的空域表示
 */
Mat GetPillowBoxFilter(double radius)
{
    int r = radius;
    int half = r * 2;
    int width = half * 2 + 1;
    int height = width;
    int rr = r*r;
    Mat tmp = Mat::zeros(height,width,CV_64FC1);
    double sum = 0;
    for(int i = -r; i<=r; i++)
    {
        int ii = i*i;
        for(int j = -r; j<=r; j++)
        {
            int jj = j*j;
            if(ii+jj <= rr)
            {
                tmp.at<double>(i+half,j+half) = 1.;
                sum++;
            }
        }
    }
    return tmp/sum;
}
