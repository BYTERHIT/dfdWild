//
// Created by bytelai on 2021/12/7.
//
#include "circulantMatrix.h"
#include "assert.h"
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
int circulantMatrix::imgCols()
{
    return _ImgWidth;
}

int circulantMatrix::imgRows()
{
    return _ImgHeight;
}
circulantMatrix::circulantMatrix(Mat kernel, int winWidth, int winHeight) {
    init(winWidth, winHeight);
    int anchor = (kernel.rows-1) / 2 * winWidth + (kernel.cols-1) / 2;
    int offset = _MtxCols - ((kernel.rows -1) * winWidth + kernel.cols -1);
    for(int i = 0; i< kernel.rows; i++)
    {
        int linOffset = i * winWidth;
        for(int j = 0; j < kernel.cols; j++)
        {
            int key = linOffset + j;
            key = key - anchor + _MtxCols;
            key = key % _MtxCols;
            double val = kernel.at<double>(i,j);
            set(key,val);
        }
    }
    UpdateKernel();
}
void circulantMatrix::init(int width, int height) {
    _ImgWidth = width;
    _ImgHeight = height;
    _MtxCols = width * height;
    _kernel = Mat::zeros(_ImgHeight, _ImgWidth, CV_64FC1);
}
circulantMatrix::circulantMatrix(int width, int height)
{
    init(width, height);
}
 circulantMatrix circulantMatrix::transpose(){
    circulantMatrix ret(_ImgWidth,_ImgHeight);
    int n = this->_pts.size();
    for(auto iter = this->_pts.begin(); iter!= this->_pts.end();iter++)
    {
        int key = iter->first;
        double val = iter->second;
        int newKey = (_MtxCols-key) % _MtxCols;
        ret.set(newKey,val);
    }
    ret.UpdateKernel();
    return ret;
}

double circulantMatrix::at(int i)
{
    auto iter = this->_pts.find(i);
    if(iter!=this->_pts.end())
    {
        return iter->second;
    }
    else
        return 0;
}

double circulantMatrix::operator[](int i)
{
    auto iter = this->_pts.find(i);
    if(iter!=this->_pts.end())
    {
        return iter->second;
    }
    else
        return 0;
}

void circulantMatrix::set(int indx, double val)
{
    int i = indx / _ImgWidth;
    int j = indx % _ImgWidth;
    if(_MaxNoZero < indx)
        _MaxNoZero = indx;

    if( _kerTl.x > j)
        _kerTl.x = j;
    if(_kerTl.y > i)
        _kerTl.y = i;
    if(_kerBr.x < j)
        _kerBr.x = j;
    if(_kerBr.y < i)
        _kerBr.y = i;
    //std::map<int, double> pts = this->_pts;
    if(this->operator[](indx) !=0)
        this->_pts.erase(indx);
    this->_pts.insert(pair<int,double>(indx,val));
}

bool isSameSize(circulantMatrix a,circulantMatrix b)
{
    return (a.imgCols() == b.imgCols()) && (a.imgRows() == b.imgRows());
}
circulantMatrix circulantMatrix::operator + (circulantMatrix b)
{
    assert(isSameSize(*this,b));
    circulantMatrix ret(_ImgWidth,_ImgHeight);
    for(auto iter = this->_pts.begin(); iter!=this->_pts.end();iter++)
    {
        int key = iter->first;
        double val = iter->second;
        ret.set(key,val + b[key]);
    }

    for(auto iter = b._pts.begin(); iter!=b._pts.end();iter++)
    {
        int key = iter->first;
        double val = iter->second;
        if(this->operator[](key) == 0)
            ret.set(key, val);
    }
    ret.UpdateKernel();
    return ret;
}

circulantMatrix circulantMatrix::operator - (circulantMatrix b)
{
    assert(isSameSize(*this,b));
    circulantMatrix ret(_ImgWidth,_ImgHeight);
    for(auto iter = this->_pts.begin(); iter!=this->_pts.end();iter++)
    {
        int key = iter->first;
        double val = iter->second;
        ret.set(key, val - b[key]);
    }

    for(auto iter = b._pts.begin(); iter!=b._pts.end();iter++)
    {
        int key = iter->first;
        double val = iter->second;
        if(this->operator[](key) == 0)
            ret.set(key,-val);
    }
    ret.UpdateKernel();
    return ret;
}

circulantMatrix circulantMatrix::operator * (circulantMatrix b)
{
    assert(isSameSize(*this,b));
    circulantMatrix ret(_ImgWidth,_ImgHeight);
    int j = 0;
    for(auto iterA = this->_pts.begin(); iterA!=this->_pts.end();iterA++)
    {
        j++;
        int aKey = iterA->first;
        double aVal = iterA->second;
        if (aVal == 0)
            continue;
        int i = 0;
        for(auto iterB = b._pts.begin(); iterB!=b._pts.end();iterB++)
        {
            i++;
            int bKey = iterB->first;
            double bVal = iterB->second;
            if (bVal == 0)
                continue;
            int key = (aKey + bKey) % _MtxCols;
            double val = aVal * bVal;
            ret.set(key,ret[key] + val);
        }
    }
    ret.UpdateKernel();
    return ret;
}
void circulantMatrix::UpdateKernel() {
    double *ptrKernel = (double*)_kernel.data;
    for(auto iter = this->_pts.begin();iter!=this->_pts.end();iter++)
    {
        int k = iter->first;
        double v = iter->second;
        if (k >= _MtxCols)
            continue;
        *(ptrKernel + k) = v;
    }
}

Mat circulantMatrix::GetKernel() {
    return _kernel.clone();
}
circulantMatrix& circulantMatrix::operator = (const circulantMatrix& b) {
    if (this != &b)
    {
        _eigen = b._eigen.clone();
        _eigenInverse = b._eigenInverse.clone();
        _kerTl = b._kerTl;
        _kerBr = b._kerBr;
        _kernel = b._kernel.clone();
        _ImgWidth = b._ImgWidth;
        _ImgHeight = b._ImgHeight;
        _MtxCols = b._MtxCols;
        _MaxNoZero = b._MaxNoZero;
        for (auto iter = b._pts.begin(); iter!=b._pts.end(); iter++)
        {
            int key = iter->first;
            double val = iter->second;
            this->set(key,val);
        }
    }
    return *this;
}
/*
 * @brief 使用same模式输出，进行卷积计算，使用filter2d来进行计算，opencv内部会根据计算复杂程度，优化的选择fft或者卷积进行实现。
 */
Mat circulantMatrix::operator * (Mat b)//p = conj(y)*x caculate corr not conv
{
    if(_eigen.empty())
        CaculateEigenValue();
    int optSize = _MtxCols;// getOptimalDFTSize(_MtxCols);
    Mat bExpand;
    int margin[2] = { _ImgHeight - b.rows ,_ImgWidth - b.cols };//bottom,right;
    copyMakeBorder(b, bExpand, margin[0] / 2, margin[0] / 2, margin[1] / 2, margin[1] / 2, BORDER_REFLECT101);// BORDER_CONSTANT, Scalar(0.));
    Mat v = Mat::zeros(1, optSize,CV_64FC1);
    Mat bRow = bExpand.reshape(0,1);
    Rect roi = Rect(0,0,bRow.cols,bRow.rows);
    bRow.copyTo(v(roi));
    //v(roi) = bRow.clone();
    Mat tmp;
    Mat vF;
    cv::dft(v,vF,DFT_COMPLEX_OUTPUT ,0);
    mulSpectrums(vF,_eigen,tmp,0,true);
    dft(tmp, tmp, DFT_INVERSE + DFT_SCALE, 0);

    Mat planes[2];
    if(tmp.channels() >1)
    {
        split(tmp,planes);
    }
    Mat Re = planes[0];
    Mat ret;
    ret = Re(roi).clone();
    ret = ret.reshape(0, bExpand.rows);
    Mat g = ret(Rect(margin[0]/2, margin[1]/2, b.cols, b.rows)).clone();
    return ret;
//    Mat dst;
//    Rect kerVaildRegion = Rect(_kerTl,_kerBr);
//    Mat kernel = _kernel(kerVaildRegion);
//    Point anchor = Point(kerVaildRegion.width / 2 -1,kerVaildRegion.height / 2 - 1);
//    Mat bExpand;
//    copyMakeBorder(b,bExpand,0,kerVaildRegion.height / 2, 0, kerVaildRegion.width / 2, BORDER_WRAP);
//    filter2D(bExpand,dst,CV_64FC1,_kernel,anchor,0,BORDER_WRAP);
//    return dst(Rect(0,0,b.rows,b.cols)).clone();
}
void circulantMatrix::CaculateEigenValue() {
    int optSize = _MtxCols;// getOptimalDFTSize(_MtxCols);
    Mat z = Mat::zeros(1, optSize,CV_64FC1);
    for(auto iter=this->_pts.begin();iter!=this->_pts.end();iter++)
    {
        int k = iter->first;
      /*  int flip_K = k;
        if(k>0)
            flip_K = optSize - k;*/
        double v = iter->second;
        z.at<double>(0, k) = v;//
    }
    cv::dft(z,_eigen,DFT_COMPLEX_OUTPUT ,0);
}

Mat circulantMatrix::GetInverseEigen() {
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
 * @brief b \ a namely b.inverse * a
 */
circulantMatrix circulantMatrix::leftDivide(circulantMatrix b){
    if (_eigen.empty())
    {
        CaculateEigenValue();
    }
    Mat invEig;
    invEig = b.GetInverseEigen();
    Mat eig;
    mulSpectrums(_eigen,invEig,eig,0,false);
    int optSize = _MtxCols;// getOptimalDFTSize(eig.cols);
    Mat z;
    dft(eig,z,DFT_INVERSE + cv::DFT_SCALE,0);
    Mat planes[2];
    if(z.channels() >1)
    {
        split(z,planes);
    }
    circulantMatrix ret(_ImgWidth,_ImgHeight);
    Mat Re = planes[0];
    double* ptr = (double*) Re.data;
    for(int i = 0 ;i < Re.cols*Re.rows;i++)
    {
//        if(abs(*ptr) > 1e-4)
        ret.set(i,*ptr);
        ptr++;
    }
    ret.UpdateKernel();
    return ret;
}

