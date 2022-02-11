//
// Created by bytelai on 2022/1/5.
//

#ifndef DFDWILD_ALGOPERATOR_H
#define DFDWILD_ALGOPERATOR_H
//#include "algTemplate.h"
#include "mat_vector.h"
#include "parameters.h"
#include "myTypes.h"
#include <fstream>
#include <iostream>
typedef struct image_feature_st{
    double pixLocation[2] = {0,0};//divided by img diagonal x,y
    double lab[3] = {0.,0.,0.};//l,a,b normalize each eigenvector standard deviation in each channel is 0.1
    double spec_cluster_eigenv[EIG_NUMBER]={0.};//spectral clustering eigenvectors. normalize each eigenvector so that ist standard deviation is 1/30
    double dep = 0.;
    template<typename datatype>
    image_feature_st operator *(datatype b){
        image_feature_st ret;
        ret.pixLocation[0] = b*pixLocation[0];
        ret.pixLocation[1] = b*pixLocation[1];
        ret.lab[0] = b*lab[0];
        ret.lab[1] = b*lab[1];
        ret.lab[2] = b*lab[2];
        for(int i = 0; i < EIG_NUMBER;i++)
        {
            ret.spec_cluster_eigenv[i] = spec_cluster_eigenv[i] * b;
        }
        ret.dep = b * dep;
        return ret;
    }

    template<typename datatype>
    image_feature_st operator /(datatype b){
        image_feature_st ret;
        ret.pixLocation[0] = pixLocation[0]/b;
        ret.pixLocation[1] = pixLocation[1]/ b;
        ret.lab[0] = lab[0]/ b;
        ret.lab[1] = lab[1]/ b;
        ret.lab[2] = lab[2]/ b;
        for(int i = 0; i < EIG_NUMBER;i++)
        {
            ret.spec_cluster_eigenv[i] = spec_cluster_eigenv[i] / b;
        }
        ret.dep = dep / b;
        return ret;
    }

    image_feature_st operator +(image_feature_st b){
        image_feature_st ret;
        ret.pixLocation[0] = b.pixLocation[0] + pixLocation[0];
        ret.pixLocation[1] = b.pixLocation[1] + pixLocation[1];
        ret.lab[0] = b.lab[0]+lab[0];
        ret.lab[1] = b.lab[1]+lab[1];
        ret.lab[2] = b.lab[2]+lab[2];
        for(int i = 0; i < EIG_NUMBER;i++)
        {
            ret.spec_cluster_eigenv[i] = spec_cluster_eigenv[i] * b.spec_cluster_eigenv[i];
        }
        ret.dep = b.dep + dep;
        return ret;
    }

} IMAGE_FEATURE;

typedef struct plane_fitter_st{
    double cx = 0;
    double cy = 0;
    double b = 0;
    double Dq(cv::Point2d q){return cx*q.x + cy* q.y +b;}
    double Dq(int x, int y ){return cx*double(x) + cy*double(y) +b;}
} PLANE_FITER;

typedef struct control_point_info_st{
    PLANE_FITER D;
    IMAGE_FEATURE f;
    int t = 0;
}CONTROL_POINT_INFO;
typedef struct image_feature_mat_st{
    cv::Mat x;
    cv::Mat y;
    cv::Mat lab;//l,a,b
    cv::Mat spec_cluster_eigenv[EIG_NUMBER];//spectral clustering eigenvectors
    cv::Mat dep;
    template<typename T>
    IMAGE_FEATURE at(int row , int col){
        IMAGE_FEATURE f;
        f.pixLocation[0] = x.at<T>(row,col);
        f.pixLocation[1] = y.at<T>(row,col);
        cv::Mat planes[3];
        cv::split(lab,planes);
        f.lab[0] = planes[0].at<T>(row,col);
        f.lab[1] = planes[1].at<T>(row,col);
        f.lab[2] = planes[2].at<T>(row,col);
        for(int i = 0; i<EIG_NUMBER;i++)
        {
            f.spec_cluster_eigenv[i] = spec_cluster_eigenv[i].at<T>(row,col);
        }
        f.dep = dep.at<T>(row, col);
        return f;
    }
} IMAGE_FEATURE_MAT;

std::ostream& operator<<(std::ostream& os,const PLANE_FITER & D ) ;

std::istream& operator>>(std::istream& is, PLANE_FITER & D ) ;
std::ostream& operator<<(std::ostream& os,const IMAGE_FEATURE & f );

std::istream& operator>>(std::istream& is, IMAGE_FEATURE & f );

std::ostream& operator<<(std::ostream& os,const CONTROL_POINT_INFO & controlP );

std::istream& operator>>(std::istream& is, CONTROL_POINT_INFO & controlP );

//int oIdx(int i,int j);

double ControlPointDistanceP2(CONTROL_POINT_INFO Cm, CONTROL_POINT_INFO Cn);

void meshgrid(const cv::Range &xgv, const cv::Range &ygv, cv::Mat &X, cv::Mat &Y);
cv::Mat GetVariance(cv::Mat depth, int winWidth, int winHeight);
double GetVariance(cv::Mat depth);
cv::Mat PsiMn(PLANE_FITER Dm, PLANE_FITER Dn, cv::Size imgSz, cv::Mat simgadm2);
double Psi(PLANE_FITER Dm, PLANE_FITER Dn, cv::Point p, double sigmadm2);
cv::Mat LapMatrix(cv::Mat Bij);
cv::Mat getF(mat_vector B);
cv::Mat fastGuidedFilter(cv::Mat I_org, cv::Mat p_org, int r, double eps, int s);
cv::Mat standardize(cv::Mat input, double stdev);
cv::Mat CaculateEq_0(std::vector<cv::Mat> aux, cv::Mat sigmaD2, double tau_0, int patchWidth, int patchHeight);
mat_vector CaculateEq_0(std::vector<AUX_TYPE> aux, cv::Mat sigmaD2, double tau_0, int patchWidth, int patchHeight, int segNum);

#endif //DFDWILD_ALGOPERATOR_H
