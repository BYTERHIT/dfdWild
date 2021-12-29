//
// Created by bytelai on 2021/12/7.
//

#ifndef DFDWILD_CIRCULANTMATRIX_H
#define DFDWILD_CIRCULANTMATRIX_H
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include "mat_vector.h"
/*利用 downshift permutation matrix 的方式表示卷积矩阵，g = H*u; H是循环矩阵
 * 参考 file:///D:/lwj/doc/lect_Toeplitz.pdf
 *
 *
 */
class circulantMatrix{
private:
    std::map<int, double> _pts;
    int _ImgWidth = 0;//图像的宽度
    int _ImgHeight = 0; //图像的高度
    int _MtxCols = 0; //循环矩阵的大小
    int _MaxNoZero = 0;
    cv::Point _kerTl = {10000,10000};
    cv::Point _kerBr = {0,0};
    cv::Mat _kernel;//卷积核
    cv::Mat _eigen;
    cv::Mat _eigenInverse;

    void set(int i, double v);
    void init(int width, int height);
public:
    int imgRows();
    int imgCols();
//    circulantMatrix(int n);
    circulantMatrix(int width, int height);

    circulantMatrix(cv::Mat kernel, int winWidth,int winHeight);
    cv::Mat GetKernel();//卷积核
    void UpdateKernel();
    void CaculateEigenValue();
    cv::Mat GetInverseEigen();
    circulantMatrix transpose();
    double at(int i);
    double operator[](int i);
    circulantMatrix operator + (circulantMatrix b);

    circulantMatrix& operator = (const circulantMatrix &b);

    circulantMatrix operator - (circulantMatrix b);

    circulantMatrix operator * (circulantMatrix b);

    circulantMatrix leftDivide(circulantMatrix b);

    cv::Mat operator * (cv::Mat b);
    Eigen::MatrixXd operator * (Eigen::MatrixXd b);

    template<typename T>
    circulantMatrix operator*(T b)
    {
        circulantMatrix ret(_ImgWidth, _ImgHeight);
        //std::map<int, double> pts = this->_pts;
        for(auto iter = this->_pts.begin(); iter != this->_pts.end();iter++)
        {
            int key = iter->first;
            double val = iter->second;
            ret.set(key,val*b);
        }
        ret.UpdateKernel();
        return ret;
    }
};

template<typename T>
circulantMatrix operator*(T lem, circulantMatrix cmat){
    return cmat*lem;}
#endif //DFDWILD_CIRCULANTMATRIX_H
