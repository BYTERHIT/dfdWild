//
// Created by bytelai on 2021/12/6.
//

#ifndef DFDWILD_SPARSEMATTOOLS_H
#define DFDWILD_SPARSEMATTOOLS_H
#include <Eigen/SparseCore>
#include <opencv2/opencv.hpp>
#include <string>


Eigen::SparseMatrix<double> GenDx(int rows, int cols, bool isForward = true);
Eigen::SparseMatrix<double> GenDy(int rows, int cols, bool isForward = true);
Eigen::RowVectorXd Mat2RowVec(cv::Mat a);
Eigen::VectorXd Mat2Vec(cv::Mat a);
cv::Mat Vec2Mat(Eigen::VectorXd v,int rows, int cols, int type);
cv::Mat MatMulSp(Eigen::SparseMatrix<double> sp, cv::Mat u);
void ReadMtx(std::string file, Eigen::MatrixXd &mtx);


#endif //DFDWILD_SPARSEMATTOOLS_H
