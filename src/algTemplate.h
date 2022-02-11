//
// Created by bytelai on 2022/1/7.
//

#ifndef DFDWILD_ALGTEMPLATE_H
#define DFDWILD_ALGTEMPLATE_H
#include "AlgOperator.h"
#include <opencv2/opencv.hpp>
#include "utility.h"
#include <fstream>
#include "myTypes.h"

using namespace cv;
using namespace std;

template<typename T>
T integral_roi(T* ptr,int tr,int bl, int tl,int br) {
    return *(ptr + br) + *(ptr + tl) - *(ptr + tr) - *(ptr + bl);
}

template<typename T>
void splitWeightIndx(T wk,int& indx,T& w)
{
    indx = floor(wk/10);
    w = wk - indx*10;
}

template<typename T>
T mergeWeightIndx(int indx,T w)
{
    T ret = indx * 10 + w;
    return ret;
}
template<typename T>
PLANE_FITER parsePlanFitter(Mat D, int n)
{
    PLANE_FITER Dm;
    Dm.cx = D.at<T>(n, 0);
    Dm.cy = D.at<T>(CONTROL_NUMBER + n, 0);
    Dm.b = D.at<T>(2 * CONTROL_NUMBER + n, 0);
    return Dm;
}

template<typename T>
Mat  GetDelta(vector<CONTROL_POINT_INFO> C, Mat sigmad2, Size imgSz, double tau_s, W_TYPE w, Mat D = Mat())
{
    Mat delta = Mat::ones(CONTROL_NUMBER, CONTROL_NUMBER, CV_8UC1);
    static double disThresh = 0.01;//* (imgSz.height^2 + imgSz.width^2);
    bool deltamn = false;
    int rows = imgSz.height;
    mat_vector wSumV(rows,Mat::zeros(CONTROL_NUMBER,CONTROL_NUMBER, getCvType<T>()));
    mat_vector wPsiSumV(rows,Mat::zeros(CONTROL_NUMBER,CONTROL_NUMBER, getCvType<T>()));
    Mat wSum = Mat::zeros(CONTROL_NUMBER,CONTROL_NUMBER, getCvType<T>());
    Mat wPsiSum = Mat::zeros(CONTROL_NUMBER,CONTROL_NUMBER, getCvType<T>());
    parallel_for_(Range(0, imgSz.height), [&](const Range& range) {
        for (size_t i = range.start; i < range.end; i++) {
            for (int j = 0; j < imgSz.width; j++) {
                vector<weight_st<T>> wv = w[i][j];
                Point p = Point(j, i);
                double sigma;
                if (sigmad2.size() == imgSz)
                    sigma = sigmad2.at<T>(p);
                else
                    sigma = sigmad2.at<double>(0, 0);
                T wArray[CONTROL_NUMBER] = {0.};
                for (auto iter = wv.begin(); iter != wv.end(); iter++) {
                    int m, n;
                    T wm, wn;
                    m = iter->indx;
                    wm = iter->weight;
                    wArray[m] = wm;
                    for (auto iter2 = iter; iter2 != wv.end(); iter2++) {
                        n = iter2->indx;
                        wn = iter2->weight;
                        if (ControlPointDistanceP2(C[m], C[n]) < disThresh) {
                            PLANE_FITER Dm, Dn;// = parsePlanFitter<T>(D,m);
                            if (D.empty()) {
                                Dm = C[m].D;
                                Dn = C[n].D;
                            } else {
                                Dm = parsePlanFitter<T>(D, m);
                                Dn = parsePlanFitter<T>(D, n);
                            }
                            T wpsi = (wn + wm) * Psi(Dm, Dn, p, sigma);
                            wPsiSumV[i].at<T>(m, n) += wpsi;
                            wSumV[i].at<T>(m, n) += (wm + wn);
                        }
                    }
                }
                //get zeros points
                vector<int> zerosIndx;
                for (int i = 0; i < CONTROL_NUMBER; i++) {
                    if (wArray[i] == 0)
                        zerosIndx.push_back(i);
                }
                // amend zeros ones
                for (auto iter = wv.begin(); iter != wv.end(); iter++) {
                    int m, n;
                    T wm;
                    m = iter->indx;
                    wm = iter->weight;
                    for (auto iter2 = zerosIndx.begin(); iter2 != zerosIndx.end(); iter2++) {
                        n = *iter2;
                        PLANE_FITER Dm, Dn;// = parsePlanFitter<T>(D,m);
                        if (D.empty()) {
                            Dm = C[m].D;
                            Dn = C[n].D;
                        } else {
                            Dm = parsePlanFitter<T>(D, m);
                            Dn = parsePlanFitter<T>(D, n);
                        }
                        T wpsi = wm * Psi(Dm, Dn, p, sigma);
                        if (n > m) {
                            wPsiSumV[i].at<T>(m, n) = wpsi;
                            wSumV[i].at<T>(m, n) = wpsi;
                        } else {
                            wPsiSumV[i].at<T>(n, m) = wpsi;
                            wSumV[i].at<T>(n, m) = wpsi;
                        }
                    }
                }
            }
        }
    });
    for(int i = 0;i<rows;i++)
    {
        wSum += wSumV[i];
        wPsiSum += wPsiSumV[i];
    }

    for(int m = 0; m < CONTROL_NUMBER; m++)
        for(int n = m+1; n < CONTROL_NUMBER; n++)
        {
            if(ControlPointDistanceP2(C[m],C[n]) < disThresh)
            {
                double ls;
                ls= wPsiSum.at<T>(m,n)/wSum.at<T>(m,n);
                if(wSum.at<T>(m,n) == 0)
                    ls = 0;
                deltamn = ls < tau_s;
                delta.at<uchar>(m,n) = deltamn;
                delta.at<uchar>(n,m) = deltamn;
            }
            else
            {
                delta.at<uchar>(m,n) = 0;
                delta.at<uchar>(n,m) = 0;
            }
        }
    return delta;
}
/*
 * @note mask uchar type
 */
template<typename T>
Mat GetDq(PLANE_FITER D ,Size imgSz, Size patchSz={1,1})
{
    Size outSz = imgSz + patchSz -Size(1,1);
    int x_begin = -(patchSz.width-1)/2;
    int x_end = imgSz.width + (patchSz.width-1)/2 + (patchSz.width-1)%2 -1;
    int y_begin = -(patchSz.height-1)/2;
    int y_end = imgSz.height + (patchSz.height-1)/2 + (patchSz.height-1)%2 -1;
    int cvType = getCvType<T>();
    Mat Dq(outSz,cvType);
#ifdef USING_MAT
    static Mat X,Y;
    meshgrid(Range(x_begin,x_end),Range(y_begin,y_end),X,Y);//maybe faster?
#else
    for(int y = y_begin; y <= y_end; y++)
        for(int x = x_begin; x <= x_end; x++)
        {
            Dq.at<double>(y-y_begin,x-x_begin) = D.Dq(x,y);
        }
#endif
    return Dq;
}

/*
 * @brief d = w*Dp
 */
template <typename T>
Mat GetDepthAtPix(W_TYPE w, vector<CONTROL_POINT_INFO> C, int imgWidth, int imgHeight) {
    int cvType = getCvType<T>();
    Mat depAtPix(imgHeight, imgWidth, cvType);
    for (int i = 0; i < imgHeight; i++)
        for (int j = 0; j < imgWidth; j++) {
            Point2d p(j, i);
            double d=0;
            vector<weight_st<T>> wv = w[i][j];
            for (auto iter = wv.begin();iter!=wv.end();iter++) {
                int k = iter->indx;
                T wk = iter->weight;
//                splitWeightIndx(*iter,k,wk);
                d += C[k].D.Dq(p) * wk;
            }
            depAtPix.at<T>(p) = d;
        }
    return depAtPix;
}
/*
 * @brief d = w*Dp
 */
template <typename T>
Mat GetDepthAtPix(W_TYPE w, Mat D, int imgWidth, int imgHeight) {
    int cvType = getCvType<T>();
    Mat depAtPix(imgHeight, imgWidth,cvType);
    for (int i = 0; i < imgHeight; i++)
        for (int j = 0; j < imgWidth; j++) {
            Point2d p(j, i);
            double d=0;
            vector<weight_st<T>> wv = w[i][j];
            for (auto iter = wv.begin();iter!=wv.end();iter++) {
                int k = iter->indx;
                T wk = iter->weight;
//                splitWeightIndx(*iter,k,wk);
                T cx = D.at<T>(k,0);
                T cy = D.at<T>(k+CONTROL_NUMBER,0);
                T b = D.at<T>(k+2*CONTROL_NUMBER,0);
                d += (cx*T(j) + cy*T(i) + b) * wk;
            }
            depAtPix.at<T>(p) = d;
        }
    return depAtPix;
}
template<typename T>
void saveWLine(vector<vector<T>> wline,int lineNumber, int seg) {
    string root_dir = DATA_DIR;
    string path = root_dir + "/seg_" + to_string(seg) + "/";
    int cols = wline.size();
    if (createDirectory(path) == 0) {
        string subPath = path + "w_lin" + to_string(lineNumber) + ".txt";
        ofstream OutFile(subPath);
        OutFile.setf(ios::fixed, ios::floatfield);  // 设定为 fixed 模式，以小数点表示浮点数
        OutFile.precision(5);  // 设置精度 5
        for (int j = 0; j < cols; j++) {
            vector<T> wv = wline[j];
            for (auto iter = wv.begin(); iter != wv.end(); iter++) {
                OutFile << *iter << " ";
            }
            OutFile << endl;
        }
    }
}
/*
 * @note 将w按照每一行一个文件，每个文件每一行是一个wq来进行存储
 */
template<typename T>
void saveW(W_TYPE w, int seg)
{
    string root_dir = DATA_DIR;
    string path = root_dir + "/seg_"+to_string(seg)+"/";
    if(createDirectory(path) == 0)
    {
        int rows = w.size();
        int cols = w[0].size();
        for(int i = 0; i<rows; i++)
        {
            string subPath = path + "w_lin" +to_string(i)+".txt";
            ofstream OutFile(subPath);
            OutFile.setf(ios::fixed, ios::floatfield);  // 设定为 fixed 模式，以小数点表示浮点数
            OutFile.precision(5);  // 设置精度 5
            for(int j = 0; j< cols; j++)
            {
                vector<T> wv = w[i][j];
                for(auto iter = wv.begin(); iter != wv.end(); iter++)
                {
                    OutFile<<*iter<<" ";
                }
                OutFile<<endl;
            }
        }
    }

}

template<typename T>
vector<T> readW(int x, int y, int segId)
{
    string root_dir = DATA_DIR;
    string path = root_dir + "seg_"+to_string(segId)+"/";
    string subPath = path + "w_lin" +to_string(y)+".txt";
    string line;
    vector<T> ret;
    ifstream fin(subPath);
    if (!fin)
    {
        cout << "weight file open failed" << endl;
        return ret;
    }
    
    int i = -1;
    while(i<x)
    {
        i++;
        getline(fin,line);
    }
    T wk;
    stringstream ss(line);
    while (ss >> wk)
        ret.push_back(wk);
    return ret;
}

template <typename T>
cv::Mat GetSpStar(Mat depAtPatch, W_TYPE w, vector<CONTROL_POINT_INFO> C, int patchWidth, int patchHeight) {
    
    int ImgHeight = depAtPatch.rows;
    int ImgWidth = depAtPatch.cols;
    Mat label = Mat::zeros(ImgHeight,ImgWidth,CV_8UC1);
    mat_vector DqV;
    for (int k = 0; k < CONTROL_NUMBER; k++) {
        Mat Dqk = GetDq<T>(C[k].D, Size(ImgWidth, ImgHeight));
        DqV.addItem(Dqk);
    }
    mat_vector tmp = DqV - depAtPatch;
    int ht = (patchHeight-1)/2;
    int hb = (patchHeight-1)/2 + (patchHeight-1)%2 + 1;
    int wl = (patchWidth-1)/2;
    int wr = (patchWidth-1)/2 + (patchWidth-1)%2 + 1;
    for (int i = 0; i < ImgHeight; i++) {
        int y_begin = max(0, i - ht);
        int y_end = min(ImgHeight, i + hb);
        for (int j = 0; j < ImgWidth; j++) {
            int x_begin = max(0, j - wl);
            int x_end = min(ImgWidth, j + wr);
            double dist;
            T dp = depAtPatch.at<T>(i, j);
            Point p(j, i);
            T dis_pn = 1e5;
            for (int y = y_begin; y < y_end; y++)
                for (int x = x_begin; x < x_end; x++) {
                    Point q = Point(x, y);
                    vector<weight_st<T>> wqv = w[y][x];
                    for (auto iter = wqv.begin(); iter != wqv.end(); iter++) {
                        int n = iter->indx;
                        T wqn = iter->weight;
                        dist = abs(dp - DqV[n].at<T>(q));
                        if (dis_pn > dist) {
                            dis_pn = dist;
                            label.at<uchar>(p) = C[n].t;
                        }
                    }
                }
        }
    }
    return label;
}

template <typename T>
Mat GetSpStar(Mat oclusion, Mat s,PARAM par)
{
    Mat sStar = Mat::zeros(s.size(),CV_8UC1);
    int ht = (par.patchHeight-1)/2;
    int hb = (par.patchHeight-1)/2 + (par.patchHeight-1)%2 + 1;
    int wl = (par.patchWidth-1)/2;
    int wr = (par.patchWidth-1)/2 + (par.patchWidth-1)%2 + 1;
    for(int i = 0; i<s.rows; i++)
    {
        int y_begin = max(0, i - ht);
        int y_end = min(s.rows, i + hb);
        for(int j =0; j< s.cols; j++)
        {
            Point p = Point(j, i);
            uchar sp = s.at<uchar>(p);
            uchar frontSeg = sp;
            int x_begin = max(0, j - wl);
            int x_end = min(s.cols, j + wr);
            for (int y = y_begin; y < y_end; y++) {
                for (int x = x_begin; x < x_end; x++) {
                    Point q = Point(x, y);
                    uchar sq = s.at<uchar>(q);
                    if(oclusion.at<char>(frontSeg,sq)<0)
                    {
                        frontSeg = sq;
                    }
//                    if(oclusion.at<char>(frontSeg,sq)==0 && sq!=frontSeg)
//                    {
//                        frontSeg = sp;// SEG_NUMBER;//for all segs in front no oclusion
//                        break;
//                    }
                }
            }
            sStar.at<uchar>(p) = frontSeg;
        }
    }
    return sStar;
}
/*
 * @biref get Nq and dBar  and dBarvar in front most neighborhood(first part in the eq A36)
 * 使用积分图来进行优化
 */
template <typename T>
vector<AUX_TYPE> GetAux(Mat sStar, Mat depAtPatch, int patchWidth, int patchHeight, int segNum)
{
    int rows = sStar.rows;
    int cols = sStar.cols;
    int ht = (patchHeight-1)/2;
    int hb = (patchHeight-1)/2 + (patchHeight-1)%2 + 1;
    int wl = (patchWidth-1)/2;
    int wr = (patchWidth-1)/2 + (patchWidth-1)%2 + 1;
    int cvType = getCvType<T>();
    mat_vector nqV(segNum, Mat::zeros(rows, cols, CV_16UC1));
    mat_vector dBarV(segNum, Mat::zeros(rows, cols, cvType));
    mat_vector dBarVV(segNum, Mat::zeros(rows, cols, cvType));

    for(int i = 0; i < rows; i++)
    {
        for( int j = 0; j < cols ; j++)
        {
            Point p = Point(j,i);
            int labelInx = sStar.at<uchar>(p);
            assert(labelInx < segNum);
            nqV[labelInx].at<uint16_t>(p) = 1;
            dBarV[labelInx].at<T>(p) = depAtPatch.at<T>(p);
            dBarVV[labelInx].at<T>(p) = pow(depAtPatch.at<T>(p), 2);
        }
    }

  

    vector<AUX_TYPE> auxV(segNum);
    parallel_for_(Range(0, segNum), [&](const Range& range) {
        for (size_t idx = range.start; idx < range.end; idx++) {
            int i = idx;
//    for(int i = 0; i < segNum; i++)
//    {
            Mat nqI, dBarI, dBarVI;
            Mat sr1, sr2, sr3;
            copyMakeBorder(nqV[i], sr1, ht, hb, wl, wr, BORDER_CONSTANT, 0.);
            copyMakeBorder(dBarV[i], sr2, ht, hb, wl, wr, BORDER_CONSTANT, 0.);
            copyMakeBorder(dBarVV[i], sr3, ht, hb, wl, wr, BORDER_CONSTANT, 0.);

            integral(sr1, nqI, CV_64F);
            integral(sr2, dBarI, CV_64F);
            integral(sr3, dBarVI, CV_64F);
            int expandWidth = nqI.cols;
            int tl = -ht * expandWidth - wl;
            int tr = -ht * expandWidth + wr;
            int bl = hb * expandWidth - wl;
            int br = hb * expandWidth + wr;
            AUX_TYPE aux;
            aux.Nq = Mat::zeros(rows, cols, CV_16UC1);
            aux.dBar = Mat::zeros(rows, cols, cvType);
            aux.dBarVar = Mat::zeros(rows, cols, cvType);
            int offsetImg = 0;

            for (int i = ht; i < rows + ht; i++) {

                for (int j = wl; j < cols + wl; j++) {
                    int nq = 0;
                    double dbar = 0.;
                    double dbarVar = 0.;
                    uint16_t *nqPtr = (uint16_t *) aux.Nq.data;
                    T *dPtr = (T *) aux.dBar.data;
                    T *dVPtr = (T *) aux.dBarVar.data;
                    int offsetInteg = i * expandWidth + j;
                    double *dIntPtr = (double *) nqI.data + offsetInteg;
                    double *dBarIntPtr = (double *) dBarI.data + offsetInteg;
                    double *dBarVIntPtr = (double *) dBarVI.data + offsetInteg;
                    nq = integral_roi<double>(dIntPtr, tr, bl, tl, br);
                    if (nq < 0.5) {
                        dbar = 0;
                        dbarVar = 0;
                    } else {
                        dbar = integral_roi<double>(dBarIntPtr, tr, bl, tl, br);
                        dbar = dbar / nq;
                        dbarVar = integral_roi<double>(dBarVIntPtr, tr, bl, tl, br) / nq - pow(dbar, 2);
                    }

                    *(nqPtr + offsetImg) = nq;
                    *(dPtr + offsetImg) = dbar;
                    *(dVPtr + offsetImg) = dbarVar;
                    offsetImg++;
                }
            }
            auxV[i] = aux;
        }
    });
    return auxV;
}


template<typename T>
Mat UpdateO(W_TYPE w,vector<CONTROL_POINT_INFO> C,Mat s,Mat depAtPatch,Mat dAtPix, Mat sigmad2, PARAM par, int segNum)
{
    int width = s.cols;
    int height = s.rows;
    Mat o= Mat::zeros(segNum,segNum,CV_8SC1);
    Mat LijMat = Mat::zeros(segNum, segNum, CV_64FC3);

    int ht = (par.patchHeight-1)/2;
    int hb = (par.patchHeight-1)/2 + (par.patchHeight-1)%2 + 1;
    int wl = (par.patchWidth-1)/2;
    int wr = (par.patchWidth-1)/2 + (par.patchWidth-1)%2 + 1;

    for(int i = 0; i < height; i++) {
        int y_begin = max(0, i - ht);
        int y_end = min(height, i + hb);
        for (int j = 0; j < width; j++) {
            int x_begin = max(0, j - wl);
            int x_end = min(width, j + wr);
            Point p = Point(j, i);
            vector<int> segHist(segNum,0);
            vector<T> LqpHist(segNum,0.);
            uchar segQ;
            int total = 0;
            T totalLqp = 0;
            double sigma = 0;
            if(sigmad2.size() == s.size())
                sigma = abs(sigmad2.at<T>(p));
            else
                sigma = abs(sigmad2.at<double>(0,0));
            if(sigma == 0)
                sigma == EPSILON;

            for (int y = y_begin; y < y_end; y++)
            {
                for (int x = x_begin; x < x_end; x++) {
                    Point q = Point(x, y);
                    segQ = s.at<uchar>(q);
                    segHist[segQ] +=1;
                    total++;
                }
            }
            vector<uchar> segV;
            if(segHist[segQ]==total)
                continue;

            for (int y = y_begin; y < y_end; y++)
            {
                for (int x = x_begin; x < x_end; x++) {
                    Point q = Point(x, y);
                    segQ = s.at<uchar>(q);
                    T lqp = pow(depAtPatch.at<T>(p) - dAtPix.at<T>(q),2)/2/sigma;
                    LqpHist[segQ] += lqp;
                    totalLqp +=lqp;
                }
            }
            for(int segi =0; segi<segNum; segi++) {
                if (segHist[segi] != 0)
                {
                    segV.push_back(segi);
                }
            }

            //double n = segV.size() -1.;
            if (segV.size() > 2)//todo 3+ seg label should be checked.因为窗口比较小,可以忽略三个label的影响
                continue;
            int segi = segV[0];
            int segj = segV[1];
            T cost_Oij_1 = LqpHist[segi]  + segHist[segj] * par.tau_o;
            T cost_Oij_0 = LqpHist[segi] + LqpHist[segj];
            T cost_Oij__1 = LqpHist[segj] + segHist[segi] * par.tau_o;
            LijMat.at<Vec3d>(segi, segj)[0] += cost_Oij_0;
            LijMat.at<Vec3d>(segj, segi)[0] += cost_Oij_0;
            LijMat.at<Vec3d>(segi, segj)[1] += cost_Oij_1;
            LijMat.at<Vec3d>(segj, segi)[2] += cost_Oij_1;
            LijMat.at<Vec3d>(segi, segj)[2] += cost_Oij__1;
            LijMat.at<Vec3d>(segj, segi)[1] += cost_Oij__1;
        }
    }
    for(int i = 0; i < segNum; i++)
        for(int j = i+1; j<segNum;j++)
        {
//            int idx = oIdx(i,j);
            char oij = 0;
           
            Vec3d lij;
            lij = LijMat.at<Vec3d>(i, j);// Lij[idx];
            T min = lij[0];
            for(char k = 0; k<3;k++)
            {
                if(lij[k] < min)
                {
                    if (k == 0)
                        oij = k;
                    else if (k == 1)
                        oij = 1;
                    else
                        oij = -1;
                    min = lij[k];
                }
            }

            o.at<char>(i,j) = oij;
            o.at<char>(j,i) = -oij;
        }
    return o;
}
#endif //DFDWILD_ALGTEMPLATE_H