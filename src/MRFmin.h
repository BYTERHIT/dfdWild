//
// Created by bytelai on 2021/12/29.
//H. Tang, S. Cohen, B. Price, S. Schiller, and K. N.Kutulakos. Depth from defocus in the wild: supple-mentary material. 2017.
// ALG 2,
// Boykov Y, Veksler O, Zabih R. Fast approximate energy minimization via graph cuts[J]. IEEE Transactions on pattern analysis and machine intelligence, 2001, 23(11): 1222-1239.
//

#ifndef DFDWILD_MRFMIN_H
#define DFDWILD_MRFMIN_H
#include <opencv2/opencv.hpp>
#include <vector>
#include "mat_vector.h"
#include "imageFeature.h"
#include "algTemplate.h"
#include "utility.h"
#include "maxflow/graph.h"
#define INFTY 1E20

template <typename T>
class MRFmin {
private:
    void Init(int imgWidth, int imgHeight, int patchWidth, int patchHeight);
    cv:: Mat _s;
    W_TYPE _w;
    vector<W_TYPE> _wSeg;

public:
    MRFmin(int imgWidth, int imgHeight, int patchwidth, int patchHeight);
    void update_s_w_MRF(Mat depAtPatch, Mat sStar, Mat s,std::vector<CONTROL_POINT_INFO> C, mat_vector fnormv, cv::Mat delta, cv::Mat sigmad2, PARAM par, int segNum);
    mat_vector GetDataTerm(Mat depAtPatch, Mat sStar,Mat s, Mat sigmad2, mat_vector fnormv, vector<CONTROL_POINT_INFO> C, Mat delta, PARAM par, int segNum);
    W_TYPE GetWeights();
    cv::Mat GetSeg();
};

template<typename T>
vector<weight_st<T>> getAvgWeight(int num)
{
    vector<weight_st<T>> avgWeight(num);
    for(int i = 0; i<num; i++)
    {
        avgWeight[i].indx = i;
        avgWeight[i].weight = 1. / num;
    }
    return avgWeight;
}

template<typename T>
vector<weight_st<T>> getWeightByFeatureDistance(vector<T> fnorm)
{
    vector<weight_st<T>> weightV;
    double minDistance = INFTY;
    int minIdx = 0;
    double sum = 0;
    for(int i = 0; i<fnorm.size(); i++)
    {
        //sum +=exp(-fnorm[i]);

        if(minDistance > fnorm[i])
        {
            minIdx = i;
            minDistance = fnorm[i];
        }
    }
 /*   for(int i = 0; i<fnorm.size(); i++)
    {
        weight_st<T> w;
        w.indx = i;
        w.weight = exp(-fnorm[i])/sum;
        weightV.push_back(w);
    }*/
    weight_st<T> w;
    w.weight = 1.0;
    w.indx = minIdx;
    weightV.push_back(w);
    return weightV;
}

template<typename T>
T GetBias(vector<weight_st<T>> a, vector<weight_st<T>> b)
{
    if (a.size() != b.size())
        return 1e20;
    T err = 0;
    T norm = 0;
    for(int i = 0; i < a.size(); i++)
    {
        err += abs(a[i].indx - b[i].indx);
        err += abs(a[i].weight - b[i].weight);
        norm += abs(a[i].weight);
    }
    return err / (norm+EPSILON);
}

template<typename T>
T getSumWEqmn_2(vector<weight_st<T>> w, int n,Mat eqmn_2)
{
    T sumWEqmn_2 = 0;
    for(auto iter2 = w.begin(); iter2!=w.end(); iter2++)
    {
        int m = iter2->indx;
        T wm = iter2->weight;
//        splitWeightIndx(*iter2,m,wm);
        sumWEqmn_2 += wm*(eqmn_2.at<T>(m,n));
    }
    return sumWEqmn_2;
}

template<typename T>
Mat GetPsiMat(vector<CONTROL_POINT_INFO> C, T sigma, Point q)
{
    Mat psiMat = Mat::zeros(CONTROL_NUMBER,CONTROL_NUMBER, getCvType<T>());

    double psiMn, sumPsi = 0;
    for(int m =0; m<CONTROL_NUMBER; m++)
    {
        for(int n = m;n<CONTROL_NUMBER; n++)
        {
            psiMn = Psi(C[m].D, C[n].D,q,sigma);
            psiMat.at<T>(m,n) = psiMn;
            psiMat.at<T>(n,m) = psiMn;
        }
    }
    return psiMat;
}

template<typename T>
T GetSumDeltaPsiNorm(Mat psiMat,Mat delta,PARAM par,T fNormq, int n)
{
    T sumPsi = 0;
    for(int m = 0; m< CONTROL_NUMBER;m++)
    {
        T psiMn = psiMat.at<T>(m,n);
        T del = delta.at<uchar>(m,n);
        sumPsi = sumPsi + psiMn * del + (1-del)*par.tau_s;
    }
    return 2.0 * par.lambda_s * sumPsi + par.lambda_i*fNormq;
}
template<typename T>
T GetEqn1(int n,vector<CONTROL_POINT_INFO> C, Mat psiMat,Mat delta,PARAM par, T dBar, Point q, int nq,T sigmad2,T fNormq)
{
    T eqn_1;
    T dq = C[n].D.Dq(q);
    T dn2 = pow(dBar-dq,2);
    T sumPsi = 0;
    for(int m = 0; m< CONTROL_NUMBER;m++)
    {
        T psiMn = psiMat.at<T>(m,n);
        T del = delta.at<uchar>(m,n);
        sumPsi = sumPsi + psiMn * del + (1-del)*par.tau_s;
    }
    //A37
    double sigma = sigmad2;
    if(sigma == 0)
        sigma = EPSILON;
    eqn_1 = 0.5*nq/sigmad2*dn2 + 2.0 * par.lambda_s * sumPsi + par.lambda_i*fNormq;
    return eqn_1;
}

template<typename T>
mat_vector MRFmin<T>::GetDataTerm(Mat depAtPatch, Mat sStar,Mat s,Mat sigmad2, mat_vector fnormv, vector<CONTROL_POINT_INFO> C,  Mat delta, PARAM par, int segNum)
{
//    string root_dir = DATA_DIR;
    vector<AUX_TYPE> auxV = GetAux<T>(sStar, depAtPatch, par.patchWidth, par.patchHeight,segNum);//
    Size imgSz = auxV[0].Nq.size();
    mat_vector DataTerm;
    mat_vector eq_0 = CaculateEq_0(auxV,sigmad2,par.tau_o,par.patchWidth,par.patchHeight, segNum);
    mat_vector ret;
    mat_vector dataTerm(segNum,Mat::zeros(imgSz, getCvType<T>()));

    parallel_for_(Range(0, imgSz.area()), [&](const Range& range) {
        for (size_t idx = range.start; idx < range.end; idx++)
        {
            //for (int y = 0; y < imgSz.height; y++)
            //{
            //    vector<vector<vector<weight_st<T>>>> wlin(SEG_NUMBER);
            //    for (int x = 0; x < imgSz.width; x++)
            //    {
            int x = idx % imgSz.width;
            int y = idx / imgSz.width;
            Point q = Point(x, y);
            vector<weight_st<T>> wq = _w[y][x];
            T sigma;
            if (sigmad2.size() == imgSz)
                sigma = sigmad2.at<T>(q);
            else
                sigma = sigmad2.at<T>(0, 0);
            if (sigma == 0)
                sigma = EPSILON;
            vector<T> fNormq = fnormv.at<T>(q);
            vector<weight_st<T>> defaultWeight;//
            Mat psiMat = GetPsiMat(C, sigma, q);
            //Wq_hat_i
            T sumPsiDelta[CONTROL_NUMBER];
            for (int n = 0; n < CONTROL_NUMBER; n++)
            {
                sumPsiDelta[n] = GetSumDeltaPsiNorm<T>(psiMat, delta, par, fNormq[n], n);
            }
            for (int seg = 0; seg < segNum; seg++)
            {
                uint16_t nq = auxV[seg].Nq.at<uint16_t>(q);
                T d_ = auxV[seg].dBar.at<T>(q);
                vector<T> eqn_1(CONTROL_NUMBER, 0);

                for (int n = 0; n < CONTROL_NUMBER; n++)
                {
                    T dq = C[n].D.Dq(q);
                    T dn2 = pow(d_ - dq, 2);
                    eqn_1[n] = 0.5 * nq / sigma * dn2 + sumPsiDelta[n];

                }

                //A38
                Mat eqmn_2 = (par.lambda_f - nq * 0.5) * (psiMat);

                //CaculateDataTerm
                T d_default = INFTY;

                int iterN = 0;
                vector<weight_st<T>> wOld = wq;
                vector<weight_st<T>> wqNew;
                T sumWEqmn_2 = 0.;
                vector<weight_st<T>> wTmp;
                vector<weight_st<T>> wTmp1;
                while (iterN < par.alg2MaxLoop)
                {
                    iterN++;
                    wqNew.clear();
                    wTmp.clear();
                    wTmp1.clear();
                    T sumW = 0.;
                    T wn = 0.;
                    T maxPower = -1e100;//减少数据溢出；
                    for (int n = 0; n < CONTROL_NUMBER; n++)
                    {
                        if (C[n].t == seg /* || C[n].t >= SEG_NUMBER*/) {
                            sumWEqmn_2 = getSumWEqmn_2<T>(wOld, n, eqmn_2);
                            wn = -eqn_1[n] - 2 * sumWEqmn_2;
                            if (wn > maxPower)
                                maxPower = wn;
                            wTmp.push_back({ n,wn });
                            wTmp1.push_back({ n,wn });
                            //sumW += wn;
                        }
                    }
                    T powerOffset = maxPower;// powerSum / wTmp.size();
                    for (int i = 0; i < wTmp.size(); i++) {
                        T weight = exp(wTmp[i].weight - powerOffset);
                        sumW += weight;
                        wTmp[i].weight = weight;
                    }
                    for (int i = 0; i < wTmp.size(); i++) {
                        if (sumW == 0 || isnan(sumW) || !isfinite(sumW))
                        {
                            cout << "no a number" << endl;
                        }
                        T weight = wTmp[i].weight / (sumW);
                        if (weight == 0)//很小的数，直接忽略
                            continue;
                        wqNew.push_back({ wTmp[i].indx,weight });
                    }
                    T bias = GetBias(wOld, wqNew);

                    if (bias < par.w_tol || wqNew.empty() || iterN >= par.alg2MaxLoop)
                    {
                        break;
                    }
                    wOld = wqNew;
                }
                if (wqNew.empty())
                {
                    wqNew = defaultWeight;
                    dataTerm[seg].at<T>(q) = d_default;
                    if (!wTmp.empty())
                    {
                        cout << "error" << endl;
                    }
                }
                else
                {
                    T d = eq_0[seg].at<T>(q);

                    for (auto iter = wqNew.begin(); iter != wqNew.end(); iter++)
                    {
                        int indx = iter->indx;
                        T weight = iter->weight;
                        sumWEqmn_2 = getSumWEqmn_2<T>(wqNew, indx, eqmn_2);
                        d += weight * (log(weight) + eqn_1[indx] + sumWEqmn_2);
                        if (isnan(d)) {
                            cout << "no a number" << endl;
                        }
                    }
                    dataTerm[seg].at<T>(q) = d;
                }
                _wSeg[seg][y][x] = wqNew;
                //wlin[seg].push_back(wqNew);
            }
            //}
            /* for(int seg = 0; seg < SEG_NUMBER; seg++) {
                 _wSeg[seg][y]=wlin[seg];
             }*/
             //}
        }
        });
    return dataTerm;
}
template<typename T>
Mat findBestAExp(const Mat s, const uchar alpha, mat_vector dataTerm, PARAM par)
{
    int imgHeight = s.rows;
    int imgWidth = s.cols;
    int nbPix = imgWidth *imgHeight;
    int maxNodes = 3* nbPix;
    int maxArcs = 6 * nbPix;
    Graph* g = new Graph(maxNodes,maxArcs);
    Graph::node_id * nodes = new Graph::node_id[nbPix];
    for(int i = 0; i < imgHeight; i++)
    {
        for(int j = 0; j < imgWidth; j++)
        {
            int idx = i*imgWidth + j;
            uchar fp = s.at<uchar>(i,j);
            double linkToSource = dataTerm[alpha].at<T>(i,j);
            double linkToSink = (fp == alpha) ? INFTY : dataTerm[fp].at<T>(i,j);
            nodes[idx] = g->add_node();
            g->set_tweights(nodes[idx], linkToSource, linkToSink);

        }
    }
    for (int i = 0; i < imgHeight; ++i) {
        for (int j = 0; j < imgWidth; ++j) {
            if (i < imgHeight - 1) {
                unsigned index1 = i * imgWidth + j;
                unsigned index2 = (i + 1) * imgWidth + j;
                if (s.at<uchar>(i, j) == s.at<uchar>(i + 1, j)) {
                    uchar fp = s.at<uchar>(i, j);
                    uchar fq = alpha;
                    double weight = fp == fq ? 0 : par.lambda_b;//Vpq
                    g->add_edge(nodes[index1], nodes[index2],
                                weight, weight);
                } else {
                    Graph::node_id aux = g->add_node();
                    double weight = s.at<uchar>(i, j) == alpha ? 0 : par.lambda_b;
                    g->add_edge(nodes[index1], aux,
                                weight, weight);
                    weight = s.at<uchar>(i + 1, j) == alpha ? 0 : par.lambda_b;
                    g->add_edge(nodes[index1], aux,
                                weight, weight);
                    g->set_tweights(aux, 0,
                                    s.at<uchar>(i, j) == s.at<uchar>(i + 1, j) ?
                                    0 : par.lambda_b);
                }
            }
            if (j < imgWidth - 1) {
                unsigned index1 = i * imgWidth + j;
                unsigned index2 = i * imgWidth + j + 1;
                if (s.at<unsigned char>(i, j) == s.at<unsigned char>(i, j + 1)) {
                    double weight = s.at<uchar>(i, j) == alpha ? 0 : par.lambda_b;
                    g->add_edge(nodes[index1], nodes[index2],
                                weight, weight);
                } else {
                    Graph::node_id aux = g->add_node();
                    double weight = s.at<unsigned char>(i, j) == alpha ? 0 : par.lambda_b;
                    g->add_edge(nodes[index1], aux, weight, weight);
                    weight = s.at<unsigned char>(i, j + 1) == alpha ? 0 : par.lambda_b;
                    g->add_edge(nodes[index1], aux,
                                weight, weight);
                    g->add_tweights(aux, 0,
                                    s.at<unsigned char>(i, j) == s.at<unsigned char>(i, j + 1) ? 0 : par.lambda_b);
                }
            }
        }
    }
    g->maxflow();
    Mat sNew(imgHeight, imgWidth, s.type());
    for (int i = 0; i < imgHeight; ++i)
        for (int j = 0; j < imgWidth; ++j)
        {
            if (g->what_segment(nodes[i * imgWidth +j])
                != Graph::SOURCE)
                sNew.at<unsigned char>(i, j) = alpha;
            else
                sNew.at<unsigned char>(i, j) = s.at<unsigned char>(i, j);
        }

    delete g;
    delete[] nodes;
    return sNew;
}
template<typename T>
double computeEnergy(Mat input, mat_vector dt, double lambda_b)
{
    double Energy = 0;

    for (int i = 0; i < input.size().height; ++i)
    {
        for (int j = 0; j < input.size().width; ++j)
        {
            Energy += dt[input.at<uchar>(i, j)].at<T>(i,j);

            if (i < input.size().height - 1)
            {
                Energy += input.at<unsigned char>(i, j) == input.at<unsigned char>(i + 1, j) ? 0 : lambda_b;
            }
            if (j < input.size().width - 1)
            {
                Energy += input.at<unsigned char>(i, j) == input.at<unsigned char>(i, j + 1) ? 0 : lambda_b;
            }
        }
    }

    return Energy;

}

template<typename T>
Mat UpdateSegment(Mat seg, mat_vector dt,PARAM par, int segNum)
{
    cv::Mat currentSolution = seg;
    double currentSolutionEnergy = computeEnergy<T>(seg, dt,par.lambda_b);

    std::clog << "Original energy = " << currentSolutionEnergy << std::endl;

    bool MadeProgress;
    do
    {
        std::clog << "Start iteration." << std::endl;

        MadeProgress = false;

        for (uchar alpha = 0; alpha < segNum; alpha++)
        {
            //std::clog << "Find AExp..." << std::flush;
            cv::Mat contestant = findBestAExp<T>(currentSolution, alpha,dt,par);
            //std::clog << "Done." << std::endl;
            //std::clog << "Compute energy..." << std::flush;
            double contestantEnergy = computeEnergy<T>(contestant, dt, par.lambda_b);
            //std::clog << "Done." << std::endl;
            //std::clog << "Contestant (" << int(alpha) << ") = " << contestantEnergy << std::endl;
            if (contestantEnergy < currentSolutionEnergy)
            {
                currentSolution = contestant;
                currentSolutionEnergy = contestantEnergy;
                MadeProgress = true;
            }
        }

        if (MadeProgress)
        {
            std::clog << "Made progress, energy = " << currentSolutionEnergy
                      << std::endl;
        }

    } while (MadeProgress);

    std::clog << "Final energy = " << currentSolutionEnergy << std::endl;

    return currentSolution;
}
template<typename T>
W_TYPE UpdateW(Mat seg)
{
    W_TYPE w(seg.rows,vector<vector<T>>(seg.cols,vector<T>(0)));
    for (int i = 0; i < seg.size().height; ++i) {
        for (int j = 0; j < seg.size().width; ++j) {
            uchar segId = seg.at<uchar>(i,j);
            vector<T> wq = readW<T>(j,i,segId);
            w[i][j]=wq;
        }
    }
    return w;
}

template<typename T>
void MRFmin<T>::update_s_w_MRF(Mat depAtPatch, Mat sStar,Mat s,  vector<CONTROL_POINT_INFO> C, mat_vector fnormv, Mat delta, Mat sigmad2, PARAM par, int segNum) {

    double maxLabel = 0;
    minMaxLoc(s,0,&maxLabel);
    mat_vector dt = GetDataTerm(depAtPatch, sStar,s, sigmad2, fnormv, C,delta,par, max(maxLabel+1,double(segNum)));
    //Mat error = dt[0] - dt[1];
    _s = UpdateSegment<T>(_s,dt,par, segNum);
    //_w = UpdateW<T>(_s);
    for (int i = 0; i < _s.size().height; ++i) {
        for (int j = 0; j < _s.size().width; ++j) {
            uchar segId = _s.at<uchar>(i, j);
            _w[i][j] = _wSeg[segId][i][j];
        }
    }
}


template<typename T>
W_TYPE MRFmin<T>::GetWeights(){
    return _w;
}

template<typename T>
cv::Mat MRFmin<T>::GetSeg()
{
    return _s;
}

template<typename T>
void MRFmin<T>::Init(int imgWidth, int imgHeight, int patchWidth, int patchHeight) {
    _s = Mat::zeros(imgHeight,imgWidth,CV_8UC1);
    _w.clear();
    for(int i =0; i<imgHeight; i++)
    {
        vector<vector<weight_st<T>>> wline;
        for(int i =0; i< imgWidth; i++) {
            vector<weight_st<T>> wq;// = getAvgWeight<T>(CONTROL_NUMBER);
            wline.push_back(wq);
        }
        _w.push_back(wline);
    }
    for(int seg =0; seg<CONTROL_NUMBER;seg++)
    {
        W_TYPE w;
        for (int i = 0; i < imgHeight; i++)
        {
            vector<vector<weight_st<T>>> wline;
            for (int i = 0; i < imgWidth; i++) {
                vector<weight_st<T>> wq;// = getAvgWeight<T>(CONTROL_NUMBER);
                wline.push_back(wq);
            }
            w.push_back(wline);
        }
        _wSeg.push_back(w);
    }
}

template<typename T>
MRFmin<T>::MRFmin(int imgWidth, int imgHeight, int patchwidth, int patchHeight) {
    Init(imgWidth, imgHeight, patchwidth,patchHeight);
}

#endif //DFDWILD_MRFMIN_H
