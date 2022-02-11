//
// Created by bytelai on 2021/12/6.
//

#include "SparseMatTools.h"
#include <Eigen/SparseCore>
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace std;
using namespace Eigen;
using namespace cv;
VectorXd Mat2Vec(Mat a)
{
    int numPix = a.cols * a.rows;
    VectorXd y(numPix);
    double *ptr = (double *)a.data;
    for (int i=0;i<numPix;i++)
    {
        y[i] = *ptr;
        ptr++;
    }
    return y;
}

RowVectorXd Mat2RowVec(Mat a)
{
    int numPix = a.cols * a.rows;
    RowVectorXd y(numPix);
    double *ptr = (double *)a.data;
    for (int i=0;i<numPix;i++)
    {
        y[i] = *ptr;
        ptr++;
    }
    return y;
}
void ReadMtx(string file, MatrixXd &mtx)
{
    ifstream fin(file,ios::binary);
    int rows = mtx.rows();
    int cols = mtx.cols();
    for (int i = 0; i< rows;i++)
    {
        for(int j = 0; j<cols; j++)
        {
            fin >> mtx(i,j);
        }
    }
}
Mat Vec2Mat(VectorXd v,int rows, int cols, int type)
{
    Mat tmp(rows,cols,type,v.data());
    Mat ret = tmp.clone();
    return ret;
}

Mat MatMulSp(SparseMatrix<double> sp, Mat u)
{
    VectorXd u_ = Mat2Vec(u);
    VectorXd tmp = sp*u_;
    Mat ret = Vec2Mat(tmp,u.rows,u.cols,u.type());
    return ret;
}

/*
 * @brief获取x差分算子的稀疏矩阵
 * @param isForward true 前向差分，false 后向差分
 * @param rows 图像的高度,注意不是稀疏矩阵
 */
SparseMatrix<double> GenDx(int rows, int cols, bool isForward)
{
    int size = rows * cols;
    SparseMatrix<double> D_x(size,size);
    vector<Triplet<double>> nonZeros;
    for(int j = 0;j<rows;j++){//dx
        int offset = j*cols;
        for(int i = 1-isForward; i < cols -isForward;i++)
        {
            nonZeros.emplace_back(offset + i, offset + i - 1 +isForward,-1.);
            nonZeros.emplace_back( offset + i, offset + i + isForward,1.);
        }
    }
    D_x.setFromTriplets(nonZeros.begin(),nonZeros.end());
    return D_x;

}

SparseMatrix<double> GenDy(int rows, int cols, bool isForward)
{
    int size = rows * cols;
    SparseMatrix<double> D_y(size,size);
    vector<Triplet<double>> nonZeros;
    for(int j = 1-isForward;j<rows - isForward;j++){//dy
        int offset = j*cols;
        for(int i = 0; i < cols;i++)
        {
            nonZeros.emplace_back( offset + i, offset + i - cols * (1-isForward),-1.);
            nonZeros.emplace_back( offset + i, offset + i + cols * isForward,1.);
        }
    }
    D_y.setFromTriplets(nonZeros.begin(),nonZeros.end());
    return D_y;

}