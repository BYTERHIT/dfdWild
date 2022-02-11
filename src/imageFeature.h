//
// Created by bytelai on 2021/12/31.
//

#ifndef DFDWILD_IMAGEFEATURE_H
#define DFDWILD_IMAGEFEATURE_H
#include "AlgOperator.h"
#include "parameters.h"
#include "ugraph.h"
#include "utility.h"
#include <io.h>
#include <direct.h>
#include <fstream>
#include "algTemplate.h"
//#include <boost/archive/text_oarchive.hpp>
#define LOCATION_FACTOR 1
#define LAB_STANDARD_DEV 0.5
#define EIG_STANDARD_DEV 0.2
#define DEP_STANDARD_DEV 1./2.
using namespace std;
using namespace cv;
template<typename T>
class imageFeature {
private:
    IMAGE_FEATURE_MAT _feature;
    mat_vector _fnormv;
    std::vector<CONTROL_POINT_INFO> _controlPoints;
public:
    cv::Mat norm2(IMAGE_FEATURE offset);
    imageFeature(cv::Mat img, cv::Mat dep = Mat());
    void serialize();
    void load();
    void updateControlPointsFeaturs(W_TYPE w);
    int UpdateControlPointsPlanFitterAndLabel(cv::Mat D,cv::Mat delta,cv::Mat dep=Mat());
    std::vector<CONTROL_POINT_INFO> getControlPoints();
    IMAGE_FEATURE_MAT getImageFeature();
    mat_vector getFeatureNorm();


};
/*
 * @param img rgb
 */
template<typename T>
imageFeature<T>::imageFeature(cv::Mat img, Mat dep) {
    string root_dir = DATA_DIR;
    string path = root_dir + "/imagefeature/";
    string subPath = path + "lab.bin";
   /* if (_access(subPath.c_str(), 0) == 0)
    {
        load();
    }
    else
    {*/
        Mat lab;
        cvtColor(img,lab,COLOR_BGR2Lab);
        lab.convertTo(lab, CV_64FC1);
        //normalize(lab,_feature.lab,1,0,NORM_MINMAX);

        _feature.lab = standardize(lab, LAB_STANDARD_DEV);
        for(int i = 1; i<=EIG_NUMBER;i++)
        {
            string path = EIG_PATH + to_string(i) + ".png";
            Mat eig = imread(path, IMREAD_UNCHANGED);
            resize(eig,eig,img.size(),0,0,INTER_NEAREST);
            fastGuidedFilter(eig,eig,16,0.02*255*255,8);
            eig.convertTo(eig, CV_64FC1);
            //normalize(eig,_feature.spec_cluster_eigenv[i-1],1,0,NORM_MINMAX);
            _feature.spec_cluster_eigenv[i-1] = standardize(eig, EIG_STANDARD_DEV);
        }
        _feature.x = Mat::zeros(img.size(), getCvType<T>());
        _feature.y = Mat::zeros(img.size(), getCvType<T>());
        double imgDiag = sqrt(pow(img.rows,2)+pow(img.cols,2)) / LOCATION_FACTOR;
        for(int i = 0 ; i<img.rows;i++)
        {
            for(int j = 0; j<img.cols; j++)
            {
                _feature.x.at<T>(i,j) = double(j) / imgDiag ;
                _feature.y.at<T>(i,j) = double(i) / imgDiag ;
            }
        }
        if (dep.empty())
            _feature.dep = Mat::zeros(img.size(), getCvType<T>());
        else
            _feature.dep = standardize(dep, DEP_STANDARD_DEV);
        _controlPoints = GetInitControlPoints<T>(_feature);
        _fnormv.resize(CONTROL_NUMBER);
        for(int n = 0; n < CONTROL_NUMBER; n++)
        {
            Mat fnorm = norm2(_controlPoints[n].f);
            _fnormv[n]=fnorm;
        }
        serialize();
    //}
}
template<typename T>
void imageFeature<T>::load() {
    string root_dir = DATA_DIR;
    string path = root_dir + "/imagefeature/";
    string subPath = path + "lab.bin";

    _feature.lab = matread(subPath);

    for(int i = 0; i<EIG_NUMBER;i++) {
        subPath = path + "spec_cluster_eigenv_" + to_string(i) + ".bin";
        _feature.spec_cluster_eigenv[i] = matread(subPath);
    }
    subPath = path + "x.bin";
    _feature.x = matread(subPath);
    subPath = path + "y.bin";
    _feature.y = matread(subPath);
    _fnormv.resize(CONTROL_NUMBER);
    ifstream fin(path + "controlPoints.txt");
//    boost::archive::text_iarchive ia(fin);
    for(int i = 0; i<CONTROL_NUMBER; i++) {
        CONTROL_POINT_INFO c;
        fin >> c;
        _controlPoints.push_back(c);
        subPath = path + "_fnormv_"+ to_string(i)+".bin";
        _fnormv[i] = (matread(subPath));
    }
    fin.close();
    subPath = path + "dep.bin";
    _feature.dep = matread(subPath);
}

template<typename T>
void imageFeature<T>::serialize() {
    string root_dir = DATA_DIR;
    string path = root_dir + "/imagefeature/";
    if(createDirectory(path) == 0) {
        string subPath = path + "lab.bin";
        matwrite(subPath,_feature.lab);
        for(int i = 0; i<EIG_NUMBER;i++)
        {
            subPath = path + "spec_cluster_eigenv_"+ to_string(i)+".bin";
            matwrite(subPath,_feature.spec_cluster_eigenv[i]);
        }
        subPath = path + "x.bin";
        matwrite(subPath,_feature.x);
        subPath = path + "y.bin";
        matwrite(subPath,_feature.y);
        subPath = path + "controlPoints.txt";
        ofstream fout(subPath);//
        for(int i = 0; i<CONTROL_NUMBER; i++)
        {
            fout<<_controlPoints[i]<<endl;
            subPath = path + "_fnormv_"+ to_string(i)+".bin";
            matwrite(subPath,_fnormv[i]);
        }
        subPath = path + "dep.bin";
        matwrite(subPath, _feature.dep);
        fout.close();
    }
}

template<typename T>
Mat imageFeature<T>::norm2(IMAGE_FEATURE offset) {
    Mat norm = Mat::zeros(_feature.lab.rows,_feature.lab.cols,_feature.x.type());
    for(int i = 0 ; i<_feature.lab.rows;i++)
    {
        for(int j = 0; j<_feature.lab.cols; j++)
        {
            norm.at<T>(i,j) += pow(_feature.x.at<double>(i,j)- offset.pixLocation[0],2);
            norm.at<T>(i,j) += pow(_feature.y.at<double>(i,j) - offset.pixLocation[1],2);
        }
    }
    for(int i = 0; i < EIG_NUMBER; i++)
    {
        Mat eigoffset = _feature.spec_cluster_eigenv[i] - offset.spec_cluster_eigenv[i];
        pow(eigoffset,2,eigoffset);
        norm += eigoffset;
    }
    Mat lab[3];
    split(_feature.lab,lab);
    Mat ld2 = lab[0]-offset.lab[0];
    pow(ld2,2,ld2);
    norm +=ld2;
    Mat ad2 = lab[1]-offset.lab[1];
    pow(ad2,2,ad2);
    norm +=ad2;
    Mat bd2 = lab[2]-offset.lab[2];
    pow(bd2,2,bd2);
    norm +=bd2;
    Mat dep2 = _feature.dep - offset.dep;
    pow(dep2, 2, dep2);
    norm += dep2;
    return norm;
}

/*
 * using k-means
 */
template<typename T>
Mat sortCtrPoints(vector<CONTROL_POINT_INFO> C, int width, int height, int clusterNum )
{
    int dims = 3 + EIG_NUMBER + 3 + 3;
    double imgDiag = width * width + height * height;
    imgDiag = sqrt(imgDiag)/ LOCATION_FACTOR;

    // 初始化定义
    int sampleCount = C.size();
    int clusterCount = clusterNum;
    Mat points(sampleCount, dims, CV_32FC1, Scalar(10));
    Mat centers(clusterCount, 1, points.type());
    Mat labels;
    // RGB 数据类型转化到样本数据
    int index = 0;
    for (int idx = 0; idx < sampleCount;idx++)
    {
        // 多维转一维
        points.at<float>(idx, 0) = C[idx].f.pixLocation[0];
        points.at<float>(idx, 1) = C[idx].f.pixLocation[1];
        points.at<float>(idx, 2) = C[idx].f.lab[0];
        points.at<float>(idx, 3) = C[idx].f.lab[1];
        points.at<float>(idx, 4) = C[idx].f.lab[2];
        points.at<float>(idx, 5) = C[idx].f.dep;
        points.at<float>(idx, 6) = C[idx].D.cx;
        points.at<float>(idx, 7) = C[idx].D.cy;
        points.at<float>(idx, 8) = C[idx].D.b;
    }
    for(int i = 0; i<EIG_NUMBER;i++)
    {
        for (int idx = 0; idx < sampleCount;idx++) {
            points.at<float>(idx, 9+i) = C[idx].f.spec_cluster_eigenv[i];
        }
    }

    // KMeans
    TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1);
    kmeans(points, clusterCount, labels, criteria, 3, KMEANS_PP_CENTERS, centers);

    // 显示图像分割后的结果，一维转多维
    Mat result = Mat::zeros(height,width, CV_8UC3);
    // 中心点显示
    for (int i = 0; i < centers.rows; i++)
    {
        int x = centers.at<float>(i, 0)* imgDiag ;
        int y = centers.at<float>(i, 1)* imgDiag ;
        circle(result, Point(x, y), 10, Scalar(255,255,255), 1, LINE_AA);
        string str = to_string(i);
        putText(result, str, Point(x, y), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }

    namedWindow("result", WINDOW_NORMAL);
    imshow("result", result);

    waitKey(1);

    return labels;
}

typedef struct kmeans_result_st {
    Mat centers;
    Mat labels;
}KMEANS_SEG_RESULT;

template<typename T>
KMEANS_SEG_RESULT KmeansSeg(IMAGE_FEATURE_MAT f,int clusterNum)
{
    int width = f.lab.cols;
    int height = f.lab.rows;
    int dims = f.lab.channels() + EIG_NUMBER + 2 + 1;


    // 初始化定义
    int sampleCount = width * height;
    int clusterCount = clusterNum;
    Mat points(sampleCount, dims, CV_32F, Scalar(0));
    Mat labels;
    Mat centers(clusterCount, dims, points.type());
    Mat planes[3];
    split(f.lab, planes);

    // RGB 数据类型转化到样本数据
    int index = 0;
    T* xPtr = (T*)f.x.data;
    T* yPtr = (T*)f.y.data;
    T* lPtr = (T*)planes[0].data;
    T* aPtr = (T*)planes[1].data;
    T* bPtr = (T*)planes[2].data;
    T* dPtr = (T*)f.dep.data;
    for (int idx = 0; idx < sampleCount; idx++)
    {
        // 多维转一维
        points.at<float>(idx, 0) = *(xPtr + idx);
        points.at<float>(idx, 1) = *(yPtr + idx);
        points.at<float>(idx, 2) = *(lPtr + idx);
        points.at<float>(idx, 3) = *(aPtr + idx);
        points.at<float>(idx, 4) = *(bPtr + idx);
        points.at<float>(idx, 5) = *(dPtr + idx);
    }
    for (int i = 0; i < EIG_NUMBER; i++)
    {
        T* ePtr = (T*)f.spec_cluster_eigenv[i].data;
        for (int idx = 0; idx < sampleCount; idx++) {
            points.at<float>(idx, 6 + i) = *(ePtr + idx);
        }
    }


    // KMeans
    TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1);
    kmeans(points, clusterCount, labels, criteria, 3, KMEANS_PP_CENTERS, centers);

    // 显示图像分割后的结果，一维转多维
    Mat result = Mat::zeros(f.lab.size(), CV_8UC1);
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            index = row * width + col;
            int label = labels.at<int>(index, 0);
            result.at<uchar>(row, col) = label;
        }
    }
    KMEANS_SEG_RESULT ret;
    ret.centers = centers;
    ret.labels = result;
    return ret;
}
/*
 * using k-means
 */
template<typename T>
vector<CONTROL_POINT_INFO> GetInitControlPoints(IMAGE_FEATURE_MAT f)
{
    int width = f.lab.cols;
    int height = f.lab.rows;
    double imgDiag = width * width + height * height;
    imgDiag = sqrt(imgDiag) / LOCATION_FACTOR;
    KMEANS_SEG_RESULT segResult = KmeansSeg<T>(f, CONTROL_NUMBER);
    Mat centers = segResult.centers;
    Mat result = segResult.labels;
    vector<CONTROL_POINT_INFO> C;
    // 中心点显示
    for (int i = 0; i < centers.rows; i++)
    {
        int x = centers.at<float>(i, 0)* imgDiag;
        int y = centers.at<float>(i, 1)* imgDiag;
        circle(result, Point(x, y), 10, Scalar(10), 1, LINE_AA);
        string str = to_string(i);
        putText(result, str, Point(x, y), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(10), 1);
        CONTROL_POINT_INFO c;
        IMAGE_FEATURE cf;
        cf.pixLocation[0] = centers.at<float>(i, 0);
        cf.pixLocation[1] = centers.at<float>(i, 1);
        cf.lab[0] = centers.at<float>(i, 2);
        cf.lab[1] = centers.at<float>(i, 3);
        cf.lab[2] = centers.at<float>(i, 4);
        cf.dep = centers.at<float>(i, 5);
        for (int j = 0; j < EIG_NUMBER; j++)
        {
            cf.spec_cluster_eigenv[j]= centers.at<float>(i, 6+j);
        }
        
        c.f = cf;// f.at<T>(y, x);
        C.push_back(c);
    }

    namedWindow("result", WINDOW_NORMAL);
    imshow("result", result);

    waitKey(1);

    return C;
}

template<typename T>
void imageFeature<T>::updateControlPointsFeaturs(W_TYPE w)
{
    int width = _feature.lab.cols;
    int height = _feature.lab.rows;
    int N = _controlPoints.size();
    vector<IMAGE_FEATURE> sumWf(N,{0});
    vector<T> sumW(N,0);
    Mat l, a, b;
    Mat planes[3];
    split(_feature.lab, planes);
    l = planes[0];
    a = planes[1];
    b = planes[2];
    
    for(int y = 0; y<height; y++)
    {
        for(int x = 0; x<width; x++)
        {
            Point q = Point(x,y);
            vector<weight_st<T>> wq = w[y][x];
            for(auto iter = wq.begin(); iter!=wq.end();iter++)
            {
                int n = iter->indx;// = floor(*iter);
                T wqn = iter->weight;// = *iter - n;
//                splitWeightIndx<T>(*iter,n,wqn);
                sumWf[n].pixLocation[0] += _feature.x.at<T>(y,x)*wqn;
                sumWf[n].pixLocation[1] += _feature.y.at<T>(y,x)*wqn;
                sumWf[n].lab[0] += l.at<T>(y,x)*wqn;
                sumWf[n].lab[1] += a.at<T>(y,x)*wqn;
                sumWf[n].lab[2] += b.at<T>(y,x)*wqn;
                sumWf[n].dep += _feature.dep.at<T>(y, x) * wqn;
                for(int i = 0;i<EIG_NUMBER;i++)
                {
                    sumWf[n].spec_cluster_eigenv[i] += _feature.spec_cluster_eigenv[i].at<T>(y,x)*wqn;
                }
                sumW[n] +=wqn;
            }
        }
    }
    for(int i = 0 ; i< N;i++)
    {
        if (sumW[i] ==0)
            continue;
        _controlPoints[i].f = sumWf[i]/sumW[i];
    }

    for(int n = 0; n < CONTROL_NUMBER; n++)
    {
        if (sumW[n] ==0)
            continue;
        Mat fnorm = norm2(_controlPoints[n].f);
        _fnormv[n]=(fnorm);
    }
}
/*
 * @return 新的分类数目
 */
template<typename T>
int  imageFeature<T>::UpdateControlPointsPlanFitterAndLabel(Mat D,Mat delta,Mat dep)
{
    if(!dep.empty())
        _feature.dep = dep.clone();
    int N = CONTROL_NUMBER;
    for(int i =0;i<CONTROL_NUMBER;i++)
    {
        _controlPoints[i].D = parsePlanFitter<T>(D, i);// D.at<T>(i, 0);
        //_controlPoints[i].D.cy = D.at<T>(N+i,0);
        //_controlPoints[i].D.b = D.at<T>(2*N+i,0);
    }
    int seg = 0;
    int width = _feature.lab.cols;
    int height = _feature.lab.rows;
    double imgDiag = width * width + height * height;
    imgDiag = sqrt(imgDiag)/ LOCATION_FACTOR;
    //update t,根据邻接矩阵统计连通域，计算联通域中出现频率最高的label，作为连通域内所有control point的seg
    vector<vector<int>> connectedComps =ConnnectedRegionOfUndrrectedGraph(delta);
#if 0 
    //if control number>seg number use kmeans to sort the control points
    if (connectedComps.size() > SEG_NUMBER)
    {
#if 1
        vector<CONTROL_POINT_INFO> ctrSegList;

        for (auto iterc = connectedComps.begin(); iterc != connectedComps.end(); iterc++) {
            CONTROL_POINT_INFO ctr = { 0 };
            int n = 0;
            for (auto iterv = iterc->begin(); iterv != iterc->end(); iterv++) {
                ctr.f = ctr.f + _controlPoints[*iterv].f;
                ctr.D.b = ctr.D.b + _controlPoints[*iterv].D.b;
                ctr.D.cx = ctr.D.cx + _controlPoints[*iterv].D.cx;
                ctr.D.cy = ctr.D.cy + _controlPoints[*iterv].D.cy;
                n++;
            }
                
            ctr.f = ctr.f / n;
            ctr.D.b = ctr.D.b / n;
            ctr.D.cx = ctr.D.cx / n;
            ctr.D.cy = ctr.D.cy / n;
            ctrSegList.push_back(ctr);
        }

        Mat labels = sortCtrPoints<T>(ctrSegList, width, height,SEG_NUMBER);
        int n = 0;
        for (auto iterc = connectedComps.begin(); iterc != connectedComps.end(); iterc++) {
            int segKmeans = labels.at<int>(n, 0);
            for (auto iterv = iterc->begin(); iterv != iterc->end(); iterv++) {
                _controlPoints[*iterv].t = segKmeans;
            }
            n++;
        }
#else
        KMEANS_SEG_RESULT segResult = KmeansSeg<T>(_feature, SEG_NUMBER);
        for (int i = 0; i < CONTROL_NUMBER; i++)
        {
            Point ctrPos = Point(_controlPoints[i].f.pixLocation[0]* imgDiag, _controlPoints[i].f.pixLocation[1]* imgDiag);
            _controlPoints[i].t = segResult.labels.at<uchar>(ctrPos);
        }
#endif
    }else{
        for(auto iterc = connectedComps.begin();iterc!=connectedComps.end();iterc++)
        {
            for(auto iterv = iterc->begin();iterv!=iterc->end();iterv++)
            {
                _controlPoints[*iterv].t = seg;// maxSeg;
            }
            seg++;
        }
    }
#endif 
    for (auto iterc = connectedComps.begin(); iterc != connectedComps.end(); iterc++)
    {
        for (auto iterv = iterc->begin(); iterv != iterc->end(); iterv++)
        {
            _controlPoints[*iterv].t = seg;// maxSeg;
        }
        seg++;
    }
    seg = 0;
    int segNum = connectedComps.size();
    vector<int> segLabelLut(segNum,-1);
    for (int i = 0; i < CONTROL_NUMBER; i++)
    {
        int segOld = _controlPoints[i].t;
        if (segLabelLut[segOld] == -1)
        {
            segLabelLut[segOld] = seg;
            seg++;
        }
        _controlPoints[i].t = segLabelLut[segOld];
    }
    return segNum;
}
template<typename T>
std::vector<CONTROL_POINT_INFO> imageFeature<T>::getControlPoints()
{
    return _controlPoints;
}
template<typename T>
IMAGE_FEATURE_MAT imageFeature<T>::getImageFeature()
{
    return _feature;
}

template<typename T>
mat_vector imageFeature<T>::getFeatureNorm() {
    return _fnormv;
}

#endif //DFDWILD_IMAGEFEATURE_H
