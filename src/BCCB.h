//
// Created by bytelai on 2021/12/22.
//

#ifndef DFDWILD_BCCB_H
#define DFDWILD_BCCB_H
#include <opencv2/opencv.hpp>

class BCCB {
private:
    int _ImgWidth = 0;//图像的宽度
    int _ImgHeight = 0; //图像的高度
    cv::Mat _kernel;//
    cv::Mat _kernelOrig;
    cv::Mat _eigen;
    cv::Mat _eigenInverse;

    void init(int width, int height);

public:
    BCCB(cv::Mat kernel, int winWidth, int winHeight,bool reshuffle = true);
    BCCB(int width, int height);
    BCCB();
    BCCB operator + (BCCB b);
    BCCB& operator = (const BCCB &b);
    BCCB operator - (BCCB b);
    BCCB operator * (BCCB b);
    BCCB leftDivide (BCCB b);
    void UpdateKernel();
    cv::Mat GetKernel();
    BCCB transpose();

    cv::Mat operator * (cv::Mat b);

    int imgCols();
    int imgRows();

    void CaculateEigenValue();
    cv::Mat GetInverseEigen();
    cv::Mat GetEigen();
    template<typename T>
    BCCB operator*(T b)
    {
        cv::Mat kernel;
        kernel = this->_kernel*b;
        BCCB ret(kernel,_ImgWidth, _ImgHeight,false);
        return ret;
    }
};

template<typename T>
BCCB operator*(T lem, BCCB cmat){
    return cmat*lem;}


#endif //DFDWILD_BCCB_H
