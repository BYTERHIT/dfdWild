//
// Created by bytelai on 2021/12/22.
//

#include "BCCB.h"
/*
C=[C0,  Cn-1,Cn-2,...,C1;
   C1,  C0,  Cn-1,...,C2;
   ....
   Cn-2,Cn-3,Cn-3,...,Cn-1;
   Cn-1,Cn-2,Cn-2,...,C0]
Ci = [Ci0,  Cin-1,...,Ci1;
      Ci1,  Ci0  ,...,Ci2;
      .
      .
      .
      Cin-2,Cin-3,...,Cin-1
      Cin-1,Cin-2,...,Ci0  ];
m = [C00,  C10,  C20,...,Cn-10;
     C01,  C11,  C21,...,Cn-12;
       .    .     .        . ;
       .    .     .        . ;
       .    .     .        . ;
     C0n-1,C1n-1,C2n-1,...Cn-1n-1]';
C'=[C0', C1',C2' ,,...,Cn-1';
    Cn-1',
    Cn-2'
     ....
    C2'
    C1']'
*/
using namespace cv;


int BCCB::imgCols()
{
    return _ImgWidth;
}

int BCCB::imgRows()
{
    return _ImgHeight;
}
void BCCB::init(int width, int height) {
    _ImgWidth = width;
    _ImgHeight = height;
    _kernel = Mat::zeros(_ImgHeight, _ImgWidth, CV_64FC1);
    _kernelOrig = Mat::zeros(_ImgHeight, _ImgWidth, CV_64FC1);
}
BCCB::BCCB(int width, int height) {
    init(width,height);
}

BCCB::BCCB(Mat kernel, int winWidth, int winHeight, bool reshuffle) {
    init(winWidth, winHeight);
    int margin[2] = { _ImgHeight - kernel.rows ,_ImgWidth - kernel.cols };//bottom,right;
    if(reshuffle)
    {
        for(int i = 0; i< kernel.rows; i++)
        {
            int ii = ((kernel.rows-1) / 2  - i + winHeight ) % winHeight;//20
            for(int j = 0; j < kernel.cols; j++)
            {
                int jj = ((kernel.cols -1) / 2  - j + winWidth ) % winWidth;//20
                _kernel.at<double>(jj,ii) =kernel.at<double>(i,j);
            }
        }
        copyMakeBorder(kernel,_kernelOrig,(margin[0]-1)/2,(margin[0]+1)/2,(margin[1]-1)/2,(margin[1]+1)/2,BORDER_CONSTANT,Scalar(0.));
    }
    else
    {
        _kernel = kernel.clone();
        UpdateKernel();
    }
}

bool isSameSize(BCCB a,BCCB b)
{
    return (a.imgCols() == b.imgCols()) && (a.imgRows() == b.imgRows());
}
BCCB BCCB::operator + (BCCB b)
{
    assert(isSameSize(*this,b));
    Mat kernel = this->_kernel + b._kernel;
    BCCB ret(kernel,_ImgWidth,_ImgHeight,false);
    return ret;
}

BCCB BCCB::operator - (BCCB b)
{
    assert(isSameSize(*this,b));
    Mat kernel = this->_kernel - b._kernel;
    BCCB ret(kernel,_ImgWidth,_ImgHeight,false);
    return ret;
}

BCCB BCCB::operator * (BCCB b)
{
    assert(isSameSize(*this,b));

    if(_eigen.empty())
        CaculateEigenValue();
    Mat tmp;
    Mat bF;
    cv::dft(b._kernel,bF,DFT_COMPLEX_OUTPUT ,0);
    mulSpectrums(bF,_eigen,tmp,0,false);
    dft(tmp, tmp, DFT_INVERSE + DFT_SCALE, 0);

    Mat planes[2];
    if(tmp.channels() >1)
    {
        split(tmp,planes);
    }
    Mat Re = planes[0];
    BCCB ret(Re.clone(),_ImgWidth,_ImgHeight,false);

    return ret;
}
void BCCB::CaculateEigenValue() {
    cv::dft(_kernel,_eigen,DFT_COMPLEX_OUTPUT ,0);
}
Mat BCCB::GetInverseEigen() {
    if(!_eigenInverse.empty())
        return _eigenInverse.clone();
    if(_eigen.empty())
        CaculateEigenValue();
    Mat planes[2];
    split(_eigen,planes);
    Mat mag;
    Mat Re = planes[0]; Mat Im = planes[1];
    Mat tmp1,tmp2;
    pow(Re, 2, tmp1);
    pow(Im, 2,tmp2);
    mag = tmp1 + tmp2;
    divide(Re,mag + DBL_EPSILON,Re);
    divide(-Im,mag + DBL_EPSILON,Im);
    planes[0] = Re;
    planes[1] = Im;
    merge(planes,2,_eigenInverse);
    return _eigenInverse.clone();
}
/*
 * @brief 使用same模式输出，进行卷积计算，使用filter2d来进行计算，opencv内部会根据计算复杂程度，优化的选择fft或者卷积进行实现。
 */
Mat BCCB::operator * (Mat b)//p = conj(y)*x caculate corr not conv
{
    if(_eigen.empty())
        CaculateEigenValue();
    Mat bExpand;
    int margin[2] = { _ImgHeight - b.rows ,_ImgWidth - b.cols };//bottom,right;
    copyMakeBorder(b, bExpand, margin[0]/2, margin[0] / 2, margin[1] / 2, margin[1] / 2, BORDER_REFLECT101);//, Scalar(0.));
//    bExpand = bExpand.t();
    //v(roi) = bRow.clone();
    Mat tmp;
    Mat vF;
    bExpand = bExpand.t();
    cv::dft(bExpand,vF,DFT_COMPLEX_OUTPUT ,0);
    mulSpectrums(vF,_eigen,tmp,0,false);
    dft(tmp, tmp, DFT_INVERSE + DFT_SCALE, 0);

    Mat planes[2];
    if(tmp.channels() >1)
    {
        split(tmp,planes);
    }
    Mat Re = planes[0];
    Mat ret;
    ret = Re.clone();
    ret = ret.t();
    Mat g = ret(Rect(margin[0]/2, margin[1]/2, b.cols, b.rows)).clone();
    Mat kernel = GetKernel();
    Mat imf;
    filter2D(b,imf,CV_64FC1,kernel,Point((kernel.cols -1)/2,(kernel.rows-1)/2));
    //imf = imf * sum(kernel)[0];
    double err = sum(abs(imf - g))[0];
    return ret;
}

BCCB& BCCB::operator = (const BCCB& b) {
    if (this != &b)
    {
        _eigen = b._eigen.clone();
        _eigenInverse = b._eigenInverse.clone();
        _kernel = b._kernel.clone();
        _kernelOrig = b._kernelOrig.clone();
        _ImgWidth = b._ImgWidth;
        _ImgHeight = b._ImgHeight;
    }
    return *this;
}

void BCCB::UpdateKernel() {
    for(int i = 0; i< _kernelOrig.rows; i++)
    {
        int ii = ((_kernelOrig.rows-1) / 2  - i + _kernel.rows ) % _kernel.rows;//20
        for(int j = 0; j < _kernelOrig.cols; j++)
        {
            int jj = ((_kernelOrig.cols-1) / 2  - j + _kernel.cols ) % _kernel.cols;//20
            _kernelOrig.at<double>(i,j) = _kernel.at<double>(jj,ii);
        }
    }
}

Mat BCCB::GetKernel() {
    UpdateKernel();
    return _kernelOrig.clone();
}

/*
 * @brief b \ a namely b.inverse * a
 */
BCCB BCCB::leftDivide(BCCB b){
    if (_eigen.empty())
    {
        CaculateEigenValue();
    }
    Mat invEig;
    invEig = b.GetInverseEigen();
    Mat eig;
    mulSpectrums(_eigen, invEig, eig, 0, false);
    Mat z;
    dft(eig,z,DFT_INVERSE + cv::DFT_SCALE,0);
    Mat planes[2];
    if(z.channels() >1)
    {
        split(z,planes);
    }
    Mat Re = planes[0];
    BCCB ret(Re,_ImgWidth,_ImgHeight,false);
    return ret;
}
BCCB BCCB::transpose(){
    BCCB ret(_ImgWidth,_ImgHeight);
    Mat kernel = Mat::zeros(_ImgHeight,_ImgWidth,CV_64FC1);
    for(int i = 0; i<_ImgHeight; i++)
    {
        int ii = 0;
        int jj = 0;
        if(i > 0)
            ii= _ImgHeight  - i;
        for(int j = 0; j < _ImgWidth; j++)
        {
            if(j > 0)
                jj = _ImgWidth - j;
            ret._kernel.at<double>(ii, jj) =this->_kernel.at<double>(i,j);
//            kernel.at<double>(ii,jj) = this->_kernel.at<double>(i,j);
        }
    }
    ret.UpdateKernel();
    return ret;
}
