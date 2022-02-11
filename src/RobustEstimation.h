//
// Created by bytelai on 2022/1/5.
//

#ifndef DFDWILD_ROBUSTESTIMATION_H
#define DFDWILD_ROBUSTESTIMATION_H
#include "UDEFilters.h"
#include "algTemplate.h"
//#include "imageFeature.h"
#include "utility.h"
#include <opencv2/opencv.hpp>

#define USING_TBB
#define  LOAD_ROBUSTEST_PATH  "D:/lwj/projects/dfdWild/data/qpfilter/"
/*
 *
 */
template<typename T>
Mat UpdateDepAtPatch(Mat Yp, Mat sStar,Mat s, int patchWidth, int patchHeight, Mat sigmad2, Mat sigmaPm2, Mat depStar, Mat depAtPix)
{
    int N = CONTROL_NUMBER;
    int rows = sStar.rows;
    int cols = sStar.cols;
    int hh = (patchHeight-1)/2;
    int hm = (patchHeight-1)/2 + (patchHeight-1)%2 + 1;
    int wh = (patchWidth-1)/2;
    int wm = (patchWidth-1)/2 + (patchWidth-1)%2 + 1;
    int cvType = getCvType<T>();
    Mat P = Mat::zeros(depAtPix.size(),depAtPix.type());
    Mat b = Mat::zeros(depAtPix.size(),depAtPix.type());
    Mat depNew = Mat::zeros(depAtPix.size(),depAtPix.type());
    parallel_for_(Range(0, rows * cols), [&](const Range& range) {
        for (size_t idx = range.start; idx < range.end; idx++)
        {
            int i = idx / cols;
            int j = idx % cols;
            int y_begin = max(0, i - hh);
            int y_end = min(rows, i + hm);
            int x_begin = max(0, j - wh);
            int x_end = min(cols, j + wm);
            Point p = Point(j, i);
            T dpStar = depStar.at<T>(p);
            T sD2;
            if (sigmad2.size() == s.size())
                sD2 = abs(sigmad2.at<T>(p)) + EPSILON;
            else
                sD2 = sigmad2.at<double>(0, 0);
            if (sD2 == 0)
                sD2 = EPSILON;
            T sPm2 = sigmaPm2.at<T>(p);
            uchar yp = Yp.at<uchar>(p);
            T Mp = 0;
            T sumDepAtPix = 0;
            for (int y = y_begin; y < y_end; y++)
                for (int x = x_begin; x < x_end; x++) {
                    Point q = Point(x, y);
                    if (sStar.at<uchar>(p) == s.at<uchar>(q) /* || sStar.at<uchar>(p) == SEG_NUMBER for no oclusion*/) {
                        Mp++;
                        sumDepAtPix += depAtPix.at<T>(q);
                    }
                }
            T Pp = 2 * yp * sPm2 + Mp / sD2;
            T bp = 2 * yp * sPm2 * dpStar + sumDepAtPix / sD2;
            depNew.at<T>(p) = bp / Pp;
            if (Pp == 0)
                depNew.at<T>(p) = 0;

        }
        });
    return depNew;

}
template<typename T>
mat_vector GetA(int imgWidth, int imgHeight)
{
    mat_vector ret;
    //A13=X,A23=Y,在遍历的时候直接有遍历算子得到，不需要计算
    Mat A11 = Mat::zeros(imgHeight, imgWidth, getCvType<T>());
    Mat A12 = Mat::zeros(imgHeight, imgWidth, getCvType<T>());
    Mat A22 = Mat::zeros(imgHeight, imgWidth, getCvType<T>());
    T * a11Ptr = (T*)A11.data;
    T * a12Ptr = (T*)A12.data;
    T * a22Ptr = (T*)A22.data;

    for(int y = 0; y < imgHeight; y++)
    {
        for(int x = 0; x<imgWidth; x++) {
            *a11Ptr = x * x;
            *a12Ptr = x * y;
            *a22Ptr = y * y;
            a11Ptr++; a12Ptr++;
            a22Ptr++;
        }
    }
    ret.addItem(A11);
    ret.addItem(A12);
    ret.addItem(A22);
    return ret;
}

template<typename T>
Mat getG(W_TYPE w, Mat sStar, Mat s,  Mat depAtPatch, int patchWidth, int patchHeight,Mat sigmad2 = Mat())
{
    int N = CONTROL_NUMBER;
    int rows = sStar.rows;
    int cols = sStar.cols;
    int hh = (patchHeight-1)/2;
    int hm = (patchHeight-1)/2 + (patchHeight-1)%2 + 1;
    int wh = (patchWidth-1)/2;
    int wm = (patchWidth-1)/2 + (patchWidth-1)%2 + 1;
    int cvType = getCvType<T>();
    Mat G = Mat::zeros(3*N,1,cvType);
    mat_vector gv(rows,G);
//    mat_vector g(3,Mat::zeros(N,1,cvType));
#ifdef USING_TBB
    parallel_for_(Range(0, rows), [&](const Range& range) {
        for (size_t idx = range.start; idx < range.end; idx++)
        {
            int i = idx;
#else
        for(int i = 0; i< rows;i++){
#endif
            int y_begin = max(0, i - hh);
            int y_end = min(rows, i + hm);
            for (int j = 0; j < cols; j++) {
                int x_begin = max(0, j - wh);
                int x_end = min(cols, j + wm);
                Point p = Point(j, i);
                T dp = depAtPatch.at<T>(p);
                T sd2 = 1.;
                if (!sigmad2.empty())
                    sd2 = abs(sigmad2.at<T>(p)) + EPSILON;
                T factor = dp / sd2;
                for (int y = y_begin; y < y_end; y++)
                    for (int x = x_begin; x < x_end; x++) {
                        Point q = Point(x, y);
                        if (sStar.at<uchar>(p) == s.at<uchar>(q) /* || sStar.at<uchar>(p) == SEG_NUMBER for no oclusion*/) {
                            vector<weight_st<T>> wq = w[y][x];
                            for (auto itern = wq.begin(); itern != wq.end(); itern++) {//TODO 此处有比较多的冗余运算可以考虑优化
                                int n = itern->indx;
                                T wqn = itern->weight;
#ifdef USING_TBB
                                gv[idx].at<T>(n, 0) += x * factor * wqn;
                                gv[idx].at<T>(n + N, 0) += y * factor * wqn;
                                gv[idx].at<T>(n + 2 * N, 0) += factor * wqn;
#else
                                G.at<T>(n, 0) += x * factor * wqn;
                                G.at<T>(n + N, 0) += y * factor * wqn;
                                G.at<T>(n + 2 * N, 0) += factor * wqn;
#endif
                            }
                        }
                    }
            }
        }
#ifdef USING_TBB
        });
    for(int i = 0; i<rows;i++)
    {
        G += gv[i];
    }
#endif
    return G;
}
typedef struct h_element_st{
                                int m;
                                int n;
                                double xxWmn;
                                double xyWmn;
                                double xWmn;
                                double yyWmn;
                                double yWmn;
                                double wmn;

                            }H_ELEMENT;

template<typename T>
Mat getH(W_TYPE w, Mat sStar, Mat s, mat_vector A, int patchWidth, int patchHeight, Mat sigmad2 = Mat())
{
    Mat xx = A[0];
    Mat xy = A[1];
    Mat yy = A[2];
    int N = CONTROL_NUMBER;
    int rows = sStar.rows;
    int cols = sStar.cols;
    int hh = (patchHeight - 1) / 2;
    int hm = (patchHeight - 1) / 2 + (patchHeight - 1) % 2 + 1;
    int wh = (patchWidth - 1) / 2;
    int wm = (patchWidth - 1) / 2 + (patchWidth - 1) % 2 + 1;
    int cvType = getCvType<T>();
    int threadNum = 10;
    Mat H = Mat::zeros(3 * CONTROL_NUMBER, 3 * CONTROL_NUMBER, cvType);
    mat_vector h(6, Mat::zeros(CONTROL_NUMBER, CONTROL_NUMBER, cvType));
    mat_vector hv(6*threadNum, Mat::zeros(CONTROL_NUMBER, CONTROL_NUMBER, cvType));
    vector<vector<vector<H_ELEMENT>>>   h_i_j_k(rows, vector<vector<H_ELEMENT>>(cols));// m,n, xx*wqmn,xy*wqmn,j*wqmn,yy*wqmn,i*wqmn,wqmn
    int rowsPerThread = rows / threadNum;

#ifdef USING_TBB_
    parallel_for_(Range(0, threadNum), [&](const Range& range) {
        for (size_t idx = range.start; idx < range.end; idx++)
        {
            cout << idx <<"start"<< endl;
            for(int i = idx*rowsPerThread; i < (idx+1)*rowsPerThread;i++)
                for (int j = 0; j < cols; j++) {
#else
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
#endif
                    Point p = Point(j, i);
                    T sD2 = 1.;
                    if (!sigmad2.empty())
                        sD2 = abs(sigmad2.at<T>(p)) + EPSILON;
                    vector<weight_st<T>> wq = w[i][j];
                    vector<H_ELEMENT> h_element_atP;
                    for (auto iterm = wq.begin(); iterm != wq.end(); iterm++) {//TODO 此处有比较多的冗余运算可以考虑优化
                        int m = iterm->indx;
                        T wqm = iterm->weight;
                        for (auto itern = iterm; itern != wq.end(); itern++) {
                            H_ELEMENT h_element;
                            int n = itern->indx;
                            T wqn = itern->weight;
                            T Wqmn = wqm * wqn / sD2;
                            h_element.m = m;
                            h_element.n = n;
                            h_element.xxWmn = xx.at<T>(p) * Wqmn;
                            h_element.xyWmn = xy.at<T>(p) * Wqmn;
                            h_element.xWmn = j * Wqmn;
                            h_element.yyWmn = yy.at<T>(p) * Wqmn;
                            h_element.yWmn = i * Wqmn;
                            h_element.wmn = Wqmn;
                            h_element_atP.push_back(h_element);
                        }
                    }
                    h_i_j_k[i][j] = h_element_atP;
                }
#ifdef USING_TBB_

        cout << idx <<"end"<< endl;
        }
    });
#else
        }
#endif;
#ifdef USING_TBB
    parallel_for_(Range(0, threadNum), [&](const Range& range) {
        for (size_t idx = range.start; idx < range.end; idx++)
        {
            for(int i = idx*rowsPerThread; i < (idx+1)*rowsPerThread;i++)
                for (int j = 0; j < cols; j++) {
#else
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
#endif
                    int y_begin = max(0, i - hh);
                    int y_end = min(rows, i + hm);
                    int x_begin = max(0, j - wh);
                    int x_end = min(cols, j + wm);
                    Point p = Point(j, i);
                    vector<weight_st<T>> wq = w[i][j];
                    for (int y = y_begin; y < y_end; y++)
                        for (int x = x_begin; x < x_end; x++) {
                            Point q = Point(x, y);
                            if (sStar.at<uchar>(p) ==
                                s.at<uchar>(q)/* || sStar.at<uchar>(p)==SEG_NUMBER for no oclusion*/) {
                                vector<H_ELEMENT> hlist = h_i_j_k[y][x];
                                for (auto iter = hlist.begin(); iter != hlist.end(); iter++) {//TODO 此处有比较多的冗余运算可以考虑优化
                                    int m = iter->m;
                                    int n = iter->n;
#ifdef USING_TBB
                                    hv[0 + idx * 6].at<T>(m, n) = hv[0 + idx * 6].at<T>(m, n) + iter->xxWmn;//xx
                                    hv[1 + idx * 6].at<T>(m, n) = hv[1 + idx * 6].at<T>(m, n) + iter->xyWmn;//xy
                                    hv[2 + idx * 6].at<T>(m, n) = hv[2 + idx * 6].at<T>(m, n) + iter->xWmn;//x
                                    hv[3 + idx * 6].at<T>(m, n) = hv[3 + idx * 6].at<T>(m, n) + iter->yyWmn;//yy
                                    hv[4 + idx * 6].at<T>(m, n) = hv[4 + idx * 6].at<T>(m, n) + iter->yWmn;//y
                                    hv[5 + idx * 6].at<T>(m, n) = hv[5 + idx * 6].at<T>(m, n) + iter->wmn;//1
#else
                                    h[0].at<T>(m, n) = h[0].at<T>(m, n) + iter->xxWmn;//xx
                                    h[1].at<T>(m, n) = h[1].at<T>(m, n) + iter->xyWmn;//xy
                                    h[2].at<T>(m, n) = h[2].at<T>(m, n) + iter->xWmn;//x
                                    h[3].at<T>(m, n) = h[3].at<T>(m, n) + iter->yyWmn;//yy
                                    h[4].at<T>(m, n) = h[4].at<T>(m, n) + iter->yWmn;//y
                                    h[5].at<T>(m, n) = h[5].at<T>(m, n) + iter->wmn;//1
#endif
                                }
                            }
                        }

                }
#ifdef USING_TBB
        }
    });
#else
    }
#endif
#ifdef USING_TBB
    for(int i = 0; i<threadNum; i++)
    {
        h[0]+=hv[0+i*6];
        h[1]+=hv[1+i*6];
        h[2]+=hv[2+i*6];
        h[3]+=hv[3+i*6];
        h[4]+=hv[4+i*6];
        h[5]+=hv[5+i*6];
    }
#endif
    for(int m = 0; m<CONTROL_NUMBER;m++)
    {
        for(int n = m; n< CONTROL_NUMBER; n++)
        {

            h[0].at<T>(n,m) = h[0].at<T>(m,n);
            h[1].at<T>(n,m) = h[1].at<T>(m,n);
            h[2].at<T>(n,m) = h[2].at<T>(m,n);
            h[3].at<T>(n,m) = h[3].at<T>(m,n);
            h[4].at<T>(n,m) = h[4].at<T>(m,n);
            h[5].at<T>(n,m) = h[5].at<T>(m,n);//保证vect<T> w按照顺序排放的话，可以去掉赋值操作
        }
    }
    h[0].copyTo(H(Rect(0, 0,N,N)));
    h[1].copyTo(H(Rect(N, 0,N,N)));
    h[1].copyTo(H(Rect(0, N,N,N)));
    h[2].copyTo(H(Rect(2*N, 0,N,N)));
    h[2].copyTo(H(Rect(0, 2*N,N,N)));
    h[3].copyTo(H(Rect(N, N,N,N)));
    h[4].copyTo(H(Rect(2*N, N,N,N)));
    h[4].copyTo(H(Rect(N, 2*N,N,N)));
    h[5].copyTo(H(Rect(2*N, 2*N,N,N)));
    return H;
}
/*
 * A: xx,xy,x;
 *    xy,yy,y;
 *    x, y, 1;
 *
 * B11,B12,B13;
 *     B22,B23;
 *         B33;
 */

template<typename T>
mat_vector GetB(W_TYPE w, Mat delta, mat_vector A, PARAM par, Mat sigmad2 = Mat())
{
    Mat xx = A[0];
    Mat xy = A[1];
    Mat yy = A[2];
    int rows = xx.rows;
    mat_vector B(6, Mat::zeros(delta.size(), getCvType<T>()));
    mat_vector Bv(6*rows, Mat::zeros(delta.size(), getCvType<T>()));
#ifdef USING_TBB
    parallel_for_(Range(0, xx.size().height), [&](const Range& range) {
        for (size_t idx = range.start; idx < range.end; idx++)
        {
            int y = idx;
#else
    for (int y = 0; y < xx.rows; y++)
    {
#endif
            for (int x = 0; x < xx.cols; x++) {
                Point q = Point(x, y);
                vector<weight_st<T>> wq = w[y][x];
                T sD2 = 1.;
                if (!sigmad2.empty())
                    sD2 = sigmad2.at<T>(q) + EPSILON;
                T wArray[CONTROL_NUMBER] = {0.};
                for (auto iterm = wq.begin(); iterm != wq.end(); iterm++) {
                    int m = iterm->indx;
                    T wqm = iterm->weight;
                    wArray[m] = wqm;
                    for (auto itern = iterm; itern != wq.end(); itern++) {
                        int n = itern->indx;
                        T wqn = itern->weight;
                        T Wqmn = (par.lambda_f * wqm * wqn + par.lambda_s * T(delta.at<uchar>(m, n)) * (wqm + wqn)) /
                                 sD2;
#ifdef USING_TBB
                        Bv[0+idx*6].at<T>(m, n) += xx.at<T>(q) * Wqmn;//xx
                        Bv[1+idx*6].at<T>(m, n) += xy.at<T>(q) * Wqmn;//xy
                        Bv[2+idx*6].at<T>(m, n) += T(x) * Wqmn;//x
                        Bv[3+idx*6].at<T>(m, n) += yy.at<T>(q) * Wqmn;//yy
                        Bv[4+idx*6].at<T>(m, n) += y * Wqmn;//y
                        Bv[5+idx*6].at<T>(m, n) += Wqmn;//y
#else
                        B[0].at<T>(m, n) += xx.at<T>(q) * Wqmn;//xx
                        B[1].at<T>(m, n) += xy.at<T>(q) * Wqmn;//xy
                        B[2].at<T>(m, n) += T(x) * Wqmn;//x
                        B[3].at<T>(m, n) += yy.at<T>(q) * Wqmn;//yy
                        B[4].at<T>(m, n) += y * Wqmn;//y
                        B[5].at<T>(m, n) += Wqmn;//y
#endif

                    }
                }
                //get zeros points
                vector<int> zerosIndx;
                for (int i = 0; i < CONTROL_NUMBER; i++) {
                    if (wArray[i] == 0)
                        zerosIndx.push_back(i);
                }
                // amend zeros ones
                for (auto iter = wq.begin(); iter != wq.end(); iter++) {
                    int m, n;
                    T wqm;
                    m = iter->indx;
                    wqm = iter->weight;
                    for (auto iter2 = zerosIndx.begin(); iter2 != zerosIndx.end(); iter2++) {
                        n = *iter2;
                        T Wqmn = par.lambda_s * T(delta.at<uchar>(m, n)) * wqm;
#ifdef USING_TBB
                        if(n>m)
                        {
                            Bv[0+idx*6].at<T>(m, n) += xx.at<T>(q) * Wqmn;//xx
                            Bv[1+idx*6].at<T>(m, n) += xy.at<T>(q) * Wqmn;//xy
                            Bv[2+idx*6].at<T>(m, n) += T(x) * Wqmn;//x
                            Bv[3+idx*6].at<T>(m, n) += yy.at<T>(q) * Wqmn;//yy
                            Bv[4+idx*6].at<T>(m, n) += y * Wqmn;//y
                            Bv[5+idx*6].at<T>(m, n) += Wqmn;//y
                        }
                        else
                        {
                            Bv[0+idx*6].at<T>(n, m) += xx.at<T>(q) * Wqmn;//xx
                            Bv[1+idx*6].at<T>(n, m) += xy.at<T>(q) * Wqmn;//xy
                            Bv[2+idx*6].at<T>(n, m) += T(x) * Wqmn;//x
                            Bv[3+idx*6].at<T>(n, m) += yy.at<T>(q) * Wqmn;//yy
                            Bv[4+idx*6].at<T>(n, m) += y * Wqmn;//y
                            Bv[5+idx*6].at<T>(n, m) += Wqmn;//y

                        }
#else
                        if(n>m)
                        {
                            B[0].at<T>(m, n) += xx.at<T>(q) * Wqmn;//xx
                            B[1].at<T>(m, n) += xy.at<T>(q) * Wqmn;//xy
                            B[2].at<T>(m, n) += T(x) * Wqmn;//x
                            B[3].at<T>(m, n) += yy.at<T>(q) * Wqmn;//yy
                            B[4].at<T>(m, n) += y * Wqmn;//y
                            B[5].at<T>(m, n) += Wqmn;//y
                        }
                        else
                        {
                            B[0].at<T>(n, m) += xx.at<T>(q) * Wqmn;//xx
                            B[1].at<T>(n, m) += xy.at<T>(q) * Wqmn;//xy
                            B[2].at<T>(n, m) += T(x) * Wqmn;//x
                            B[3].at<T>(n, m) += yy.at<T>(q) * Wqmn;//yy
                            B[4].at<T>(n, m) += y * Wqmn;//y
                            B[5].at<T>(n, m) += Wqmn;//y
                        }
#endif
                    }
                }
            }
        }
#ifdef USING_TBB
        });
#endif
#ifdef USING_TBB
    for(int i = 0; i<rows;i++) {
        B[0] += Bv[0+i*6];
        B[1] += Bv[1+i*6];
        B[2] += Bv[2+i*6];
        B[3] += Bv[3+i*6];
        B[4] += Bv[4+i*6];
        B[5] += Bv[5+i*6];
    }
#endif
    for(int m=0;m<CONTROL_NUMBER;m++)
        for(int n = m; n<CONTROL_NUMBER;n++)
        {
            B[0].at<T>(n, m) = B[0].at<T>(m, n);
            B[1].at<T>(n, m) = B[1].at<T>(m, n);
            B[2].at<T>(n, m) = B[2].at<T>(m, n);
            B[3].at<T>(n, m) = B[3].at<T>(m, n);
            B[4].at<T>(n, m) = B[4].at<T>(m, n);
            B[5].at<T>(n, m) = B[5].at<T>(m, n);//保证vect<T> w按照顺序排放的话，可以去掉赋值操作
        }
    return B;
}

template<typename T>
class RobustEstimation {
private:
    QP_FITER_MAT _QpFilter;
    cv::Mat _depAtPatch;
    cv::Mat _D;
    cv::Mat _sigmad2Mat;
    double _sigmad2;
    cv::Mat _delta;
    cv::Mat _depAtPix;
    cv::Mat GetYp(cv::Mat dep,double tau_i);
    void load();
    void serialize();
public:
    RobustEstimation(cv::Mat i1, cv::Mat i2,  double sigmai, PARAM par);
    /*
     * z=ax+by+c; d=[a1,a2,...,an,b1,b2,...,bn,c1,c2,...,cn]
     */
    cv::Mat getPlanFitter();
    cv::Mat getDepAtPatch();
    cv::Mat getDelta();
    cv::Mat getSigmad2Mat();
    double getSigmad2();
    cv::Mat getDepAtPix();
    cv::Mat UpdatePlanFitter_DepAtPatch(W_TYPE w, std::vector<CONTROL_POINT_INFO> C, cv::Mat s, cv::Mat sStar, PARAM par);

};

template<typename T>
Mat RobustEstimation<T>::GetYp(Mat dep, double tau_i)
{
    Mat Qp = _QpFilter.Qp(dep);
    Mat Yp;
    threshold(Qp,Yp,tau_i,1,THRESH_BINARY_INV);
    Yp.convertTo(Yp,CV_8UC1);
    return Yp;
}
template<typename T>
void RobustEstimation<T>::load()
{
    string root_dir = LOAD_ROBUSTEST_PATH;
    string subPath = root_dir + "depStar.bin";

    _QpFilter.dpStar = matread(subPath);
    subPath = root_dir + "sigmaPm2";
    _QpFilter.sigmapm2 = matread(subPath);

    subPath = root_dir + "qp";
    _QpFilter.qp = matread(subPath);
    subPath = root_dir + "sigmad2";
    _sigmad2Mat = abs(matread(subPath));
    _depAtPatch = _QpFilter.dpStar.clone();
    subPath = root_dir + "_delta";
    _delta = matread(subPath);
    subPath = root_dir + "_depAtPix";
    _depAtPix = matread(subPath);
    subPath = root_dir + "_D";
    _D = matread(subPath);


}

template<typename T>
void RobustEstimation<T>::serialize() {
    string root_dir = LOAD_ROBUSTEST_PATH;
    if(createDirectory(root_dir) == 0) {
        string subPath = root_dir + "depStar.bin";
        matwrite(subPath,_QpFilter.dpStar);
        subPath = root_dir + "sigmaPm2";
        matwrite(subPath,_QpFilter.sigmapm2);

        subPath = root_dir + "qp";
        matwrite(subPath,_QpFilter.qp);
        subPath = root_dir + "sigmad2";
        matwrite(subPath,_sigmad2Mat);
        subPath = root_dir + "_delta";
        matwrite(subPath,_delta);
        subPath = root_dir + "_depAtPix";
        matwrite(subPath,_depAtPix);
        subPath = root_dir + "_D";
        matwrite(subPath,_D);
    }
}
template<typename T>
RobustEstimation<T>::RobustEstimation(cv::Mat i1, cv::Mat i2, double sigmai, PARAM par) {//i1,i2 should be normalize gray
    string root_dir = LOAD_ROBUSTEST_PATH;
    string subPath = root_dir + "depStar.bin";
  /*  if (_access(subPath.c_str(), 0) == 0)
    {
        load();
        _sigmad2 = GetVariance(_depAtPatch);
    }
    else
    {*/
        _QpFilter = GetQpFiter(i1,i2, par.patchWidth,sigmai,par.dep_max, par.dep_min);
        _depAtPatch = _QpFilter.dpStar.clone();
        _D = Mat::zeros(3*CONTROL_NUMBER,1, getCvType<T>());
        _sigmad2Mat = GetVariance(_depAtPatch, par.patchWidth,par.patchHeight);
        _sigmad2 = GetVariance(_depAtPatch);
        _delta = Mat::ones(CONTROL_NUMBER, CONTROL_NUMBER, CV_8UC1);
        _depAtPix = Mat::zeros(_depAtPatch.size(), _depAtPatch.type());
        serialize();

    //}
}
/*
 * alg3
 */
template<typename T>
cv::Mat RobustEstimation<T>::UpdatePlanFitter_DepAtPatch(W_TYPE w, vector<CONTROL_POINT_INFO> C, cv::Mat s,Mat sStar, PARAM par) {
    _delta = GetDelta(C, _sigmad2 * Mat::ones(1, 1, CV_64FC1), s.size(), par.tau_s, w);//
    _depAtPix = GetDepthAtPix(w, C, s.cols, s.rows);
    static mat_vector A = GetA<T>(s.cols,s.rows);
    Mat dep = _depAtPatch.clone();
    Mat oldDep =dep.clone();
    int iterCount = 0;
    Mat D = _D.clone();
    Mat oldD = D.clone();
    Mat y = Mat::ones(s.size(),CV_8UC1);
    Mat delta = Mat::ones(CONTROL_NUMBER,CONTROL_NUMBER,CV_8UC1);
//    Mat sigmaD2 = _sigmad2;//GetVariance(dep,par.patchWidth,par.patchHeight);
    double sigmaD2 = GetVariance(dep);//GetVariance(dep,par.patchWidth,par.patchHeight);
    double sigmaD2Old = sigmaD2;
    Mat depAtPatVariance = GetVariance(dep, par.patchWidth, par.patchHeight);
    Mat depAtPix = _depAtPix;//GetDepthAtPix(w,D,s.cols,s.rows);
    Mat depAtPixOld = depAtPix.clone();

    Mat H = getH(w, sStar, s, A, par.patchWidth, par.patchHeight);
    

   
    while(iterCount < par.alg3MaxLoop)
    {
        iterCount++;
        
        Mat g = getG(w, sStar, s, dep, par.patchWidth, par.patchHeight);
        //H = H * (sigmaD2Old / (sigmaD2 + DBL_EPSILON));
        oldD = D.clone();
        depAtPixOld = depAtPix.clone();
        mat_vector B = GetB(w, delta, A, par);
        Mat F = getF(B);
        Mat den = 2 * F + H;
        Mat denInv;
        invert(den,denInv,DECOMP_SVD);
        D = denInv*g;

        depAtPix = GetDepthAtPix(w,D,s.cols,s.rows);

        oldDep = dep.clone();
        dep = UpdateDepAtPatch<T>(y, sStar, s, par.patchWidth, par.patchHeight, depAtPatVariance/* sigmaD2 * Mat::ones(1, 1, CV_64FC1)*/, _QpFilter.sigmapm2, _QpFilter.dpStar, depAtPix);
        //sigmaD2 = GetVariance(dep);
        //depAtPatVariance = GetVariance(dep, par.patchWidth, par.patchHeight);

        y = GetYp(dep,par.tau_i);
        delta = GetDelta(C, sigmaD2*Mat::ones(1,1,CV_64FC1), sStar.size(), par.tau_s, w,D);
        T errDep = sum(abs(dep-oldDep) )[0] / (sum(abs(dep))[0]+EPSILON);
        T errD = sum(abs(D-oldD))[0] / (sum(abs(D))[0] + EPSILON);
        T errDepAtPix = sum(abs(depAtPixOld - depAtPix))[0] / (sum(abs(depAtPix))[0] + EPSILON);
        cout << "robust estimation " <<iterCount<< " err(dep) : " << errDep << " err(D) : " << errD << " err(depAtpix) "<< errDepAtPix<<endl;
        if(errDepAtPix <par.dep_tol)
            break;
    }
    _D = D;
    _sigmad2Mat = GetVariance(dep, par.patchWidth, par.patchHeight);
    _sigmad2 = GetVariance(dep);
    _delta = delta;
    _depAtPix = depAtPix;
    _depAtPatch = dep;
//    UpdateControlPoints<T>(C,D,delta);
    return D;
}

template<typename T>
cv::Mat RobustEstimation<T>::getPlanFitter() {
    return _D.clone();
}

template<typename T>
cv::Mat RobustEstimation<T>::getDepAtPatch(){
    return _depAtPatch.clone();
}

template<typename T>
cv::Mat RobustEstimation<T>::getDepAtPix(){
    return _depAtPix.clone();
}

template<typename T>
double RobustEstimation<T>::getSigmad2(){
    return _sigmad2;
}

template<typename T>
cv::Mat RobustEstimation<T>::getSigmad2Mat(){
    return _sigmad2Mat.clone();
}

template<typename T>
cv::Mat RobustEstimation<T>::getDelta() {
    return _delta.clone();
}


#endif //DFDWILD_ROBUSTESTIMATION_H
