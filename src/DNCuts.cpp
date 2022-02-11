//
// Created by bytelai on 2021/11/30.
//

#include "DNCuts.h"
#include <opencv2/opencv.hpp>
#include <Eigen/SparseCore>

using namespace cv;
using namespace std;
using namespace Eigen;
#define IMG_DIAG_LENGHT 500
//使用matlab直接完成，c++目前没有很好的求解稀疏矩阵特征值的方法
//Mat GetStandardDeviation(Mat input, int winSize)
//{
//    Mat mean;
//    int borderType = BORDER_REPLICATE;
//    blur(input,mean,Size(winSize,winSize),Point(-1,-1),borderType);
//    Mat tmp = input - mean;
//    tmp = tmp.mul(tmp);
//    Mat stdDeviation ;
//    blur(tmp,stdDeviation,Size(winSize,winSize),Point(-1,-1),borderType);
//    sqrt(stdDeviation,stdDeviation);
//    return stdDeviation;
//}
//Eigen::SparseMatrix<double> GetAffinityMatrix(Mat rgb)
//{
//    Mat lab ;
//    double imgDiag = sqrt(rgb.rows * rgb.rows + rgb.cols * rgb.cols);
//    double scal = IMG_DIAG_LENGHT / imgDiag;
//    int rows = rgb.rows * scal;
//    int cols = rgb.cols * scal;
//
//    Mat rgbDownSampled;
//    resize(rgb,rgbDownSampled,Size(cols,rows));
//    cvtColor(rgbDownSampled, lab,COLOR_BGR2Lab);
//
//    int sigmaS = 1./125. * sqrt(rows*rows+cols*cols);
//    int winSzHalf = sigmaS * 3 / 2;
//    int winSize = winSzHalf * 2 + 1;
//    Mat stdDev = GetStandardDeviation(lab,winSize);
//    vector<Triplet<double>> nonZeros;
//    for(int i = 0; i<rows; i++)
//    {
//        int yStart = max(i - winSzHalf, 0);
//        int yEnd = min(i + winSzHalf, rows-1);
//        for(int j = 0; j < cols; j++)
//        {
//            Point p = Point(j,i);//point(x,y)
//            int xStart = max(j - winSzHalf,0);
//            int xEnd = min(j + winSzHalf, cols - 1);
//            for(int m = yStart; m++; m <=yEnd)
//            {
//                for(int n = xStart; n++; n <=xEnd)
//                {
//                    Point q = Point(n,m);
//                }
//            }
//
//        }
//    }
//
//}