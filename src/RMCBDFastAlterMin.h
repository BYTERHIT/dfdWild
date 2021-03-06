//
// Created by bytelai on 2021/12/6.
//

#ifndef DFDWILD_RMCBDFASTALTERMIN_H
#define DFDWILD_RMCBDFASTALTERMIN_H
#include "mat_vector.h"
#include <Eigen/Core>
#include <Eigen/SparseCore>
//#define DEBUG
/* [1] Sroubek F ,  Milanfar P .
 * Robust Multichannel Blind Deconvolution via Fast Alternating Minimization[J].
 * IEEE Transactions on Image Processing, 2012, 21(4):1687-1700.
 *
 */
class RMCBDFastAlterMin {
private:
    mat_vector _g;
    Eigen::MatrixXd _RDelta;
//    Eigen::MatrixXd _HStepDelta;
    cv::Mat _HStepDelta;
    cv::Mat _RDeltaMat;
    //Eigen::SparseMatrix<double> _Dx;
    //Eigen::SparseMatrix<double> _Dy;
    //Eigen::SparseMatrix<double> _Dx_t;
    //Eigen::SparseMatrix<double> _Dy_t;
    //Eigen::SparseMatrix<double> _SumDxyt;
    double _minTol = -1;
    int _winWidth, _winHeight,_maxLoop, _L;
    double _alpha, _gamma, _delta, _beta;

public:
    RMCBDFastAlterMin(int winWidth,int winHeight);
    mat_vector MCBlindDeconv(mat_vector g, int L, double alpha, double beta, double delta, double gamma, int maxLoop, double tol);
    cv::Mat u_step(mat_vector g, cv::Mat uInit, mat_vector h, double minTol, int maxLoop);
    cv::Mat u_step_tgv(mat_vector g, cv::Mat uInit, mat_vector h,double alpha_u, double alpha_w, double lambda, double minTol, int maxLoop);
    mat_vector h_step(cv::Mat u, mat_vector g, mat_vector hInit, double minTol, int maxLoop);
    void CaculateRdelta(const cv::Range &range);


};


#endif //DFDWILD_RMCBDFASTALTERMIN_H
