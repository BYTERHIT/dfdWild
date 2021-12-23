//

#include <opencv2/opencv.hpp>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include <io.h>
#include "mat_vector.h"
#include "RMCBDFastAlterMin.h"
#include "SparseMatTools.h"
//#include "circulantMatrix.h"
#include "BCCB.h"
using namespace cv;
using namespace std;
using namespace Eigen;

/*
 * @brief 获取H矩阵
 * @param h 模糊核函数，h1,h2
 * @param winSize 假设在winSize内深度是一致的
 */
SparseMatrix<double> GenH(mat_vector h,int winSize)
{
    assert(winSize>=h.width);
    Mat h1 = h[0];
    Mat h2 = h[1];
    int L = h1.rows;
    int N = winSize;
    int M = N-L+1;
    int rows = M*M;
    int cols = N*N;
    SparseMatrix<double> H(rows*2,cols);
    int size[2] = { rows * 2, cols};
    SparseMat Hmat(2, size, CV_64FC1);
    vector<Triplet<double>> nonZeros;
    for(int i = 0; i<L;i++)
    {
        int rowOffset = winSize * i;
        for(int j = 0; j < L; j++)
        {
            for(int k = 0; k < rows;k++)
            {
                nonZeros.emplace_back(k,rowOffset+j,h1.at<double>(i,j));
                *Hmat.ptr(k, rowOffset + j,1) = h1.at<double>(i, j);
                nonZeros.emplace_back(rows + k,rowOffset+j,h2.at<double>(i,j));
                *Hmat.ptr(rows + k, rowOffset + j,1) = h2.at<double>(i, j);
            }
        }
    }
    H.setFromTriplets(nonZeros.begin(),nonZeros.end());
    return H;
}

/*
 * Gk={vector(G(0,0)),vector(G(0,1)),...,vector(G(L-1,L-1))}
 * Gk_icol = G(icol/L,icol%L);
 * @param i 位于hi中的坐标，[0,L-1];
 * @param j 位于hi中的坐标，[0,L-1];
 *
 */
Mat GAt(int i,int j, Mat g, int L)
{
    int N = g.cols;
    int M = N-L+1;
    Rect validRoi = Rect(L/2,L/2,M,M);
    Mat gValid = g(validRoi).clone();
    int m = M-L+1;
    Rect roi = Rect(i,j,m,m);
    Mat ret = gValid(roi).clone();
    return ret;
}

Mat GColAt(int indx, Mat g, int L)
{
    int i = indx/L;
    int j = indx%L;
    return GAt(i,j,g,L);
}
/*
 * @brief return indx th col of the mtx [G2,-G1];
 * @param g laplacian of the blur image [g1,g2]
 * @param L width of the psf
 */
Mat GColAt(int indx, mat_vector g, int L)
{
    int i = indx/L;
    int j = indx%L;
    int N = g[1].cols;
    int M = N-L+1;
    int m = M-L+1;
    Mat Gmat;
    Rect validRoi = Rect(L/2,L/2,M,M);
    Mat gValid;
    if(i<L)
    {
        Gmat = g[1];
    }
    else
    {
        i -=L;
        Gmat = -g[0];
    }
    gValid = Gmat(validRoi).clone();
    Rect roi = Rect(i,j,m,m);
    Mat ret = gValid(roi).clone();
    return ret;
}

/*
 * U={vector(U(0,0)),vector(U(0,1)),...,vector(U(L-1,L-1))}
 * U_icol = U(Rect(icol/L,icol%L,M,M));
 * @param i 位于hi中的坐标，[0,L-1];
 * @param j 位于hi中的坐标，[0,L-1];
 *
 */
Mat UAt(int i,int j, Mat g, int L)
{
    int N = g.cols;
    int M = N-L+1;
    Rect roi = Rect(i,j,M,M);
    Mat ret = g(roi).clone();
    return ret;
}

Mat UColAt(int indx, Mat g, int L)
{
    int i = indx/L;
    int j = indx%L;
    return UAt(i,j,g,L);
}

void RMCBDFastAlterMin::CaculateRdelta(const cv::Range &range) {
    Mat g1, g2, lapG1, lapG2;
    _g[0].convertTo(g1,CV_64FC1);
    _g[1].convertTo(g2,CV_64FC1);
    Laplacian(g1,lapG1,CV_64FC1);//TODO,边缘条件需要重新考虑，此处用的是101reflect，需要考虑是否有影响
    Laplacian(g2,lapG2,CV_64FC1);
    mat_vector lapG;
    lapG.addItem(lapG1);
    lapG.addItem(lapG2);
    int LL = _L*_L;
    int step = 2*LL;
    _RDelta = MatrixXd(2*LL,2*LL);
    for(size_t idx = range.start; idx < range.end; idx++)
    {
        int i = idx / step;
        int j = idx % step;
        Mat G_i = GColAt(i,lapG,_L);
        Mat G_j = GColAt(j,lapG,_L);
        double r = sum(G_i.mul(G_j))[0];
        _RDelta(i,j) = r;
        _RDelta(j,i) = r;
    }
}
/*
 * @param L是psf核函数的宽度，psf的长宽相等
 */
MatrixXd GenRDelta(mat_vector g, int L)
{
    Mat g1, g2, lapG1, lapG2;
    g[0].convertTo(g1,CV_64FC1);
    g[1].convertTo(g2,CV_64FC1);
    Laplacian(g1,lapG1,CV_64FC1);//TODO,边缘条件需要重新考虑，此处用的是101reflect，需要考虑是否有影响
    Laplacian(g2,lapG2,CV_64FC1);
    mat_vector lapG;
    lapG.addItem(lapG1);
    lapG.addItem(lapG2);
    int LL = L*L;
    MatrixXd Rdelta(2*LL,2*LL);
    for(int i = 0; i<2*LL; i++)
    {
        Mat G_i = GColAt(i,lapG,L);
        for(int j = i; j<2*LL;j++)
        {
            Mat G_j = GColAt(j,lapG,L);
            double r = sum(G_i.mul(G_j))[0];
            Rdelta(i,j) = r;
            Rdelta(j,i) = r;
        }
    }
    return Rdelta;
}

MatrixXd GenU(Mat u, int L)
{
    assert(u.cols == u.rows);
    Mat uMat;//
    u.convertTo(uMat,CV_64FC1);
    int LL = L*L;
    int N = u.cols;
    int M = N - L + 1;
    int MM=M*M;
    MatrixXd Umtx(MM,2*LL);
    for(int i = 0; i<LL; i++)
    {
        Umtx.col(i) = Mat2Vec(UColAt(i,uMat,L));
    }
    Umtx.block(0,LL,LL,LL) = Umtx.block(0,0,LL,LL);//todo 需要验证是否是深拷贝
    return Umtx;
}

RMCBDFastAlterMin::RMCBDFastAlterMin(int winSize)
{
    _Dx = GenDx(winSize,winSize);
    _Dy = GenDy(winSize,winSize);
    _Dx_t = _Dx.transpose();
    _Dy_t = _Dy.transpose();
    _SumDxyt = _Dx_t * _Dx + _Dy_t * _Dy;
    _winSize = winSize;

}
typedef BCCB CMT;
Mat RMCBDFastAlterMin::u_step(mat_vector g, cv::Mat uInit, mat_vector h) {

    int rows = g[0].rows, cols = g[0].cols;
    int L = h[0].cols;
    assert(rows==cols);
    assert(rows==_winSize);
    assert(L%2);
    int N = _winSize + L -1;
    int NN = N*N;
    Mat u = uInit.clone();

    Mat Vx = Mat::zeros(N, N,CV_64FC1);
    Mat Vy = Mat::zeros(N, N,CV_64FC1);
    Mat Ax = Mat::zeros(N, N,CV_64FC1);
    Mat Ay = Mat::zeros(N, N,CV_64FC1);

    //Mat Vx = Mat::zeros(rows, cols, CV_64FC1);
    //Mat Vy = Mat::zeros(rows, cols, CV_64FC1);
    //Mat Ax = Mat::zeros(rows, cols, CV_64FC1);
    //Mat Ay = Mat::zeros(rows, cols, CV_64FC1);

    Mat g1 = g[0]; Mat g2 = g[1];
    Mat h1 = h[0]; Mat h2 = h[1];
    g1.convertTo(g1,CV_64FC1);
    g2.convertTo(g2,CV_64FC1);
    h1.convertTo(h1,CV_64FC1);
    h2.convertTo(h2,CV_64FC1);

    CMT H1(h1,N,N);
    CMT H2(h2,N,N);
    CMT H1_t = H1.transpose();
    CMT H2_t = H2.transpose();
    //前向差分
    int m = h1.rows;
    int n = h1.cols;
    Mat mDx = (Mat_<double>(1, 2) << -1, 1);//
    Mat mDy = (Mat_<double>(2, 1) << -1, 1);//
    //Mat mDx =  Mat::zeros(m, n, CV_64FC1);//
    //mDx.at<double>(m / 2, n / 2) = -1;
    //mDx.at<double>(m / 2, n / 2 +1) = 1;
    //Mat mDy =   Mat::zeros(m, n, CV_64FC1); ;//
    //mDy.at<double>(m / 2, n / 2 ) = -1;
    //mDy.at<double>(m / 2+1, n / 2) = 1;
    //Mat mLap = Mat::zeros(m, n, CV_64FC1);//
    //mLap.at<double>(m / 2, n / 2) = -4;
    //mLap.at<double>(m / 2, n / 2 + 1) = 1;
    //mLap.at<double>(m / 2, n / 2 - 1) = 1;
    //mLap.at<double>(m / 2 + 1, n / 2) = 1;
    //mLap.at<double>(m / 2 - 1, n / 2) = 1;
    //CMT Lap(mLap, N);
    CMT Dx(mDx,N,N);
    CMT Dy(mDy,N,N);
    CMT Dy_t = Dy.transpose();
    CMT Dx_t = Dx.transpose();
    CMT Lap = Dx_t * Dx + Dy_t * Dy;//拉普拉斯算子，带循环边界的
    //Mat kernel = Dy_t.GetKernel();
    //CMT LapTest(kernel, N, N);


    CMT Den = H1_t * H1 + H2_t * H2 + _alpha / _gamma * Lap;
    //u = H1DenInv
    Den.GetInverseEigen();
    CMT H1tDenInv = H1_t.leftDivide(Den);
    CMT H2tDenInv = H2_t.leftDivide(Den);
    CMT DxtDenInv =  Dx_t.leftDivide(Den);
    DxtDenInv =  DxtDenInv * (_alpha / _gamma);
    CMT DytDenInv = Dy_t.leftDivide(Den);
    DytDenInv = (_alpha / _gamma) * DytDenInv;
    //Mat g1Expand, g2Expand;
    //copyMakeBorder(g1,g1Expand,)
    Mat HtGvDenInv = H1tDenInv * g1 + H2tDenInv * g2;
    Mat uExpand;
    int margin[2] = { N - u.rows,N - u.cols };
    copyMakeBorder(u, uExpand, margin[0]/2, margin[0]/2, margin[1]/2,margin[1]/2,BORDER_REFLECT101/*,Scalar(0.)*/);
    //uExpand = u.clone();
    int i = 0;
    while(i<20)
    {
        Mat DxU = Dx * uExpand;// Expand;
        Mat DyU = Dy * uExpand;// Expand;
        Mat Sx = DxU - Ax;
        Mat Sy = DyU - Ay;
        Mat S;
        magnitude(Sx,Sy,S);
        Mat factor = 1-1./_alpha/S;
        threshold(factor,factor,0.,1,THRESH_TOZERO);
        Vx = Sx.mul(factor);
        Vy = Sy.mul(factor);

        Ax = Vx - Sx;
        Ay = Vy - Sy;
        Mat x = Vx + Ax;
        Mat y = Vy + Ay;
        Mat oldU = uExpand.clone();
        uExpand = HtGvDenInv + DxtDenInv*x + DytDenInv*y;
        //Mat bias1 = H1*u-g1;
        //Mat bias2 = H2*u-g2;
        double err = sum(abs(oldU - uExpand))[0];
        double L1 = sum(abs(uExpand))[0];
        double tol = err /L1;// +sum(bias2)[0];
        cout<<"ustep: i \t" <<i << "\t tol \t" <<tol<<endl;
        Mat uShow;
        uExpand.convertTo(uShow, CV_8UC1,255.);
        namedWindow("u", WINDOW_NORMAL);
        imshow("u", uShow);
        waitKey(1);
        if(tol < 1e-3)
            break;
        i++;
    }
    return uExpand(Rect(margin[0] / 2, margin[1] / 2, cols, rows)).clone();// 
}

typedef enum{
    SAME=0,
    VALID,
    FULL
} CONV_TYPE;

/*
 * 计算U.transpose * Gv 注意，采用full模式的输出，对计算进行简化，但是会对边界产生ring效应
 * U 是根据u生成的卷积矩阵,G是根据g生成的卷积向量,L 是卷积核尺度，卷积核的width和height相等
 */

Mat UtransMulGv(Mat u,Mat g, int L, CONV_TYPE convType = SAME, int borderType = BORDER_REFLECT101,double val = 0)
{
    Mat A, B;
    switch(convType){
        case FULL:
            copyMakeBorder(u,A, L - 1, L - 1, L - 1, L - 1,borderType,Scalar(val));
            copyMakeBorder(g, B, (L - 1)/2, (L - 1)/2, (L - 1)/2, (L - 1)/2, borderType, Scalar(val));
            break;
        case VALID:
            A = u;
            B = g(Rect((L-1)/2,(L-1)/2,g.cols-L+1,g.rows-L+1));
            break;
        case SAME:
            copyMakeBorder(u,A, (L - 1)/2, (L - 1)/2, (L - 1)/2, (L - 1)/2,borderType,Scalar(val));
            B = g;
            break;
        default:
            copyMakeBorder(u,A, (L - 1)/2, (L - 1)/2, (L - 1)/2, (L - 1)/2,borderType,Scalar(val));
            B = g;
            break;
    }
    A.convertTo(A,CV_32FC1);
    B.convertTo(B,CV_32FC1);
    Mat correlation;
    matchTemplate(A,B,correlation,TM_CCORR);
    correlation.convertTo(correlation,CV_64FC1);
    return correlation;
}

/*
 * 计算 U.transpose * G, U 是卷积定义的矩阵, G是g生成的卷积矩阵
 */
Mat UtransMulG(Mat u, Mat g, int L,CONV_TYPE convType = SAME, int borderType = BORDER_REFLECT101,double val = 0)
{
    Mat A, B;
    switch(convType){
        case FULL:
            copyMakeBorder(u, A, L - 1, L - 1, L - 1, L - 1,borderType,Scalar(val));
            copyMakeBorder(g, B, L - 1, L - 1, L - 1, L - 1, borderType, Scalar(val));
            break;
        case VALID:
            A = u;
            B = g;
            break;
        case SAME:
            copyMakeBorder(u,A, (L - 1)/2, (L - 1)/2, (L - 1)/2, (L - 1)/2,borderType,Scalar(val));
            copyMakeBorder(g,B, (L - 1)/2, (L - 1)/2, (L - 1)/2, (L - 1)/2,borderType,Scalar(val));
            break;
        default:
            copyMakeBorder(u,A, (L - 1)/2, (L - 1)/2, (L - 1)/2, (L - 1)/2,borderType,Scalar(val));
            copyMakeBorder(g,B, (L - 1)/2, (L - 1)/2, (L - 1)/2, (L - 1)/2,borderType,Scalar(val));
            break;
    }
    A.convertTo(A,CV_32FC1);
    B.convertTo(B,CV_32FC1);
    Size roiSize = Size(B.cols-L+1,B.rows-L+1);
    int LL = L*L;
    Mat ret = Mat::zeros(LL,LL,CV_64FC1);
    parallel_for_(Range(0,LL),[&](const Range& range){
    for(size_t idx = range.start; idx < range.end; idx++)
    {
        int i = idx / L;
        int j = idx % L;
        Point orig = Point(j, i);
        Rect roi = Rect(orig, roiSize);
        Mat Broi = B(roi);
        Mat correlation;
        matchTemplate(A, Broi, correlation, TM_CCORR);
        //int l = linOffset + j;
        float* corPtr = (float*)correlation.data;
        for (int k = 0; k < LL; k++)
        {
            ret.at<double>(k, (int)idx) = *(corPtr + k);
        }
    }

});
   /* for(int i = 0;i<L;i++)
    {
        int linOffset = i *L;
        for(int j = 0;j<L;j++) {
            Point orig = Point(j,i);
            Rect roi = Rect(orig,roiSize);
            Mat Broi = B(roi);
            Mat correlation;
            matchTemplate(A,Broi,correlation,TM_CCORR);
            int l = linOffset+j;
            float *corPtr = (float*)correlation.data;
            for(int k = 0 ;k<LL;k++)
            {
                ret.at<double>(k,l) = *(corPtr+k);
            }
        }
    }*/
    return ret;
}

mat_vector solve(Mat DenInv,mat_vector b)
{
    int L = b[0].rows;
    int LL = L*L;
    Mat e = Mat::zeros(LL*2,1,CV_64FC1);
    b[0].reshape(1,LL).copyTo(e(Rect(0,0,1,LL)));
    b[1].reshape(1,LL).copyTo(e(Rect(0,LL,1,LL)));
    Mat ans = DenInv * e;
    e.reshape(1,2*L);
    mat_vector ret;
    Mat ret1 = e(Rect(0,0,L,L)).clone();
    Mat ret2 = e(Rect(0,L,L,L)).clone();
    ret.addItem(ret1);
    ret.addItem(ret2);
    return ret;
}

mat_vector RMCBDFastAlterMin::h_step(cv::Mat u, mat_vector g, mat_vector hInit)
{
    int L = hInit[0].cols;
    assert(L%2);
    int LL = L*L;
    int rows = g[0].rows, cols = g[0].cols;
    assert(rows==cols);
    assert(rows==_winSize);
    int N = _winSize;
    int M = N-L+1;
    Mat g1 = g[0];
    Mat g2 = g[1];
    g1.convertTo(g1,CV_64FC1);
    g2.convertTo(g2,CV_64FC1);
    Mat h1 = hInit[0];
    Mat h2 = hInit[1];
    mat_vector h, w, b;
    h1.convertTo(h1,CV_64FC1);
    h2.convertTo(h2,CV_64FC1);
    Mat w1 = Mat::zeros(L,L,CV_64FC1);
    Mat w2 = Mat::zeros(L,L,CV_64FC1);
    Mat b1 = Mat::zeros(L,L,CV_64FC1);
    Mat b2 = Mat::zeros(L,L,CV_64FC1);

    h.addItem(h1);
    h.addItem(h2);
    w.addItem(w1);
    w.addItem(w2);
    b.addItem(b1);
    b.addItem(b2);

    // use ‘full’ output
    Mat UtU =Mat::zeros(2*LL,2*LL,CV_64FC1);
    Mat utu = UtransMulG(u,u,L, VALID);
    utu.copyTo(UtU(Rect(0,0,LL,LL)));
    utu.copyTo(UtU(Rect(LL,LL,LL,LL)));

    Mat Den = UtU + _HStepDelta;
    MatrixXd DenMtx(Den.rows,Den.cols);
    cv2eigen(Den,DenMtx);

    //LLT<MatrixXd> solver;
    LDLT<MatrixXd> solver;
    solver.compute(DenMtx);

    Mat UtG1 = UtransMulGv(u,g1,L, VALID);
    Mat UtG2 = UtransMulGv(u,g2,L, VALID);
    mat_vector UtG;
    UtG.addItem(UtG1);
    UtG.addItem(UtG2);
    int i = 0;
    while (i<40)
    {
        mat_vector s = h - b;
        s = s - 1./_beta;

        w = s.threshold(0,1,THRESH_TOZERO);
        b = b - h + w;
        mat_vector c = UtG + _beta/_gamma * (w + b);
        VectorXd c1 = Mat2Vec(c[0]);
        VectorXd c2 = Mat2Vec(c[1]);
        VectorXd cc(2*LL);
        cc<<c1,c2;
        VectorXd ans = solver.solve(cc);
        Mat ansMat;
        eigen2cv(ans,ansMat);
        ansMat = ansMat.reshape(1,2*L);
        Mat ansMat1 = ansMat(Rect(0,0,L,L)).clone();
        Mat ansMat2 = ansMat(Rect(0,L,L,L)).clone();
        Mat h1Old = h[0];
        Mat h2Old = h[1];
        h.clear();
        h.addItem(ansMat1);
        h.addItem(ansMat2);
        
        double tol = (sum(abs(h1Old - ansMat1))[0] + sum(abs(h2Old - ansMat2))[0] )/(sum(abs(ansMat2))[0] + sum(abs(ansMat1))[0]);
        cout<<"hstep: i \t" <<i << "\t tol \t" <<tol<<endl;
        Mat h1show, h2show;
        ansMat1.convertTo(h1show, CV_8UC1,10000);
        ansMat2.convertTo(h2show, CV_8UC1,10000);
        namedWindow("h1", WINDOW_NORMAL);
        namedWindow("h2", WINDOW_NORMAL);
        imshow("h1", h1show);
        imshow("h2", h2show);
        waitKey(1);
        if(tol < 1e-2)
            break;
        i++;
    }
    return h;
}
mat_vector RMCBDFastAlterMin::MCBlindDeconv(mat_vector g, int L, double alpha, double beta, double delta, double gamma, int maxLoop, double tol) {
    double sigBeta = -3;
    _g = g;
    _L = L;
    int LL = L*L;
    _alpha = alpha;
    _beta = beta;
    _delta = delta;
    _gamma = gamma;
    _minTol = tol;
    _maxLoop = 10;
    _RDelta = MatrixXd(2*LL,2*LL);
    Mat g1, g2, lapG1, lapG2;
    _g[0].convertTo(g1, CV_32FC1);
    _g[1].convertTo(g2, CV_32FC1);
    Laplacian(g1, lapG1, CV_32FC1, 1, 1, 0, BORDER_REFLECT101);//TODO,边缘条件需要重新考虑，此处用的是101reflect，需要考虑是否有影响
    Laplacian(g2, lapG2, CV_32FC1, 1, 1, 0, BORDER_REFLECT101);

    imwrite("g1.png", g1);
    imwrite("g2.png", g2);
    Mat R = Mat::zeros(2 * LL, 2 * LL, CV_64FC1);
    Mat R11 = UtransMulG(lapG2, lapG2, L, VALID);
    Mat R12 = UtransMulG(lapG2, -lapG1, L, VALID);
    Mat R21 = UtransMulG(lapG1,-lapG2,L, VALID);
    Mat R22 = UtransMulG(lapG1,lapG1,L, VALID);
    R11.copyTo(R(Rect(0,0,LL,LL)));
    R12.copyTo(R(Rect(LL,0,LL,LL)));
    R21.copyTo(R(Rect(0,LL,LL,LL)));
    R22.copyTo(R(Rect(LL,LL,LL,LL)));
    _RDeltaMat = R;
    Mat I = Mat::diag(Mat::ones(1,LL*2,CV_64FC1));
    _HStepDelta = _delta / _gamma * _RDeltaMat + _beta / _gamma * I;
    double sigma1 = double(L)/6.;
    double sigma2 = double(L)/6. + sigBeta;

    //Mat h1 =  getGaussianKernel(L, sigma1);//
    //h1 = h1 * h1.t();
    //
    //Mat h2 = getGaussianKernel(L, sigma2); //
    //h2 = h2 * h2.t();

    Mat h2 = Mat::zeros(L, L, CV_64FC1);//
    h2.at<double>(20, 20) = 1.;
    Mat h1 = Mat::zeros(L, L, CV_64FC1);//
    h1.at<double>(20, 20) = 1.;
    double a = sum(h1)[0];
    double b = sum(h2)[0];
    mat_vector h;
    h.addItem(h1);
    h.addItem(h2);
    g1.convertTo(g1, CV_64FC1);
    g2.convertTo(g2, CV_64FC1);
    Mat u = (g1 + g2) / 2.;// Mat::zeros(g[0].rows, g[0].cols, CV_64FC1);
    Mat uInit = u;
    int i = 0;
    while( i < maxLoop)
    {
        u = u_step(g,u,h);
        h = h_step(u,g,h);
        Mat h1 = h[0];
        Mat h2 = h[1];
        Mat tmp1, tmp2;
        filter2D(u,tmp1,CV_64FC1,h1);
        filter2D(u,tmp2,CV_64FC1,h2);
        double tol1 = sum(abs(tmp1-g[0]))[0];
        double tol2 = sum(abs(tmp2-g[1]))[0];
        cout<<"mcbd: i "<<i <<"\t tol1: " << tol1 <<"\t tol2 : " << tol2<<endl;
        i++;
    }
    h.addItem(u);
    return h;
}

