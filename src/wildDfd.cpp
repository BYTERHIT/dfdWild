//
// Created by bytelai on 2022/1/7.
//

//#include "wildDfd.h"
#include "MRFmin.h"
#include "RobustEstimation.h"
using namespace cv;
using namespace std;
/*
 * alg 1.
 */
template<typename T>
void wildDfd(Mat i1,Mat i2, PARAM par)
{
    T sigmai = 0.05;//todo 需要增加sigmai的标定逻辑

    RobustEstimation<T> depDRobustEsti(i1, i2, sigmai, par);
    Mat depAtPatch = depDRobustEsti.getDepAtPatch();

    imageFeature<T> imgFeature(i1);
    vector<CONTROL_POINT_INFO> C = imgFeature.getControlPoints();
    MRFmin<T> segMrf(i1.cols,i1.rows,par.patchWidth,par.patchHeight);
    W_TYPE w = segMrf.GetWeights();

    int iterN = 0;
    Mat s = segMrf.GetSeg();
    Mat D = depDRobustEsti.getPlanFitter();
    mat_vector fnorm = imgFeature.getFeatureNorm();
    //Mat delta = depDRobustEsti.getDelta();
    Mat delta = Mat::ones(CONTROL_NUMBER, CONTROL_NUMBER, CV_8UC1);
    double sigmad2 =depDRobustEsti.getSigmad2();

    Mat depAtPix = depDRobustEsti.getDepAtPix();
    Mat depOld = depAtPix.clone();
    namedWindow("depPatch",WINDOW_NORMAL);
    namedWindow("seg",WINDOW_NORMAL);
    namedWindow("depPix",WINDOW_NORMAL);
    Mat sStar = GetSpStar(depAtPatch, w, C, par.patchWidth, par.patchHeight);//todo 和 irls中涉及的front seg的计算是否相同
    int segNum = 1;
    while(iterN<par.alg1MaxLoop)
    {
        iterN++;
        //step3
        sStar = GetSpStar(depAtPatch, w, C, par.patchWidth, par.patchHeight);//todo 和 irls中涉及的front seg的计算是否相同
        segMrf.update_s_w_MRF(depAtPatch,sStar,s,C,fnorm,delta,Mat::ones(1,1,CV_64FC1)*sigmad2,par,segNum);
        w = segMrf.GetWeights();
        mat_vector wMatV(CONTROL_NUMBER, Mat::zeros(s.size(), CV_64FC1));
        for (int i = 0; i < s.rows; i++)
        {
            for (int j = 0; j < s.cols; j++)
            {
                vector<weight_st<T>> wqv = w[i][j];
                for (auto k = wqv.begin(); k != wqv.end(); k++)
                {
                    int n = k->indx;
                    T wq = k->weight;
                    wMatV[n].at<double>(i, j) = wq;
                }
            }
        }
        s = segMrf.GetSeg();
        //step 4
        Mat oclussion = UpdateO(w,C,s,depAtPatch,depAtPix,Mat::ones(1,1,CV_64FC1)*sigmad2,par,segNum);//todo 确定是否robust estimation 每次迭代都需要更新;
        sStar = GetSpStar<T>(oclussion, s, par);


        //step 5
        depDRobustEsti.UpdatePlanFitter_DepAtPatch(w,C,s, sStar,par);
        delta = depDRobustEsti.getDelta();
        depAtPatch = depDRobustEsti.getDepAtPatch();
        depAtPix = depDRobustEsti.getDepAtPix();
        sigmad2 = depDRobustEsti.getSigmad2();
        Mat show;
        D = depDRobustEsti.getPlanFitter();
        

        //step 6

        segNum = imgFeature.UpdateControlPointsPlanFitterAndLabel(D,delta);
        imgFeature.updateControlPointsFeaturs(w);

        C = imgFeature.getControlPoints();
        fnorm = imgFeature.getFeatureNorm();

        

        T err = sum(abs(depAtPix - depOld))[0] / (sum(abs(depAtPix))[0]+EPSILON);
        depOld = depAtPix.clone();
        cout << err << endl;
        imshow("depPatch", depAtPatch/4.2);
        imshow("depPix", depAtPix/4.2);
        s.convertTo(show, CV_8UC1, 255./double(segNum));
        Mat sDbg = s.clone();
        double imgDiag = sqrt(pow(depOld.rows, 2) + pow(depOld.cols, 2))/ LOCATION_FACTOR;
        for (int i = 0; i < CONTROL_NUMBER; i++)
        {
            int x = C[i].f.pixLocation[0] * imgDiag;
            int y = C[i].f.pixLocation[1] * imgDiag;
            circle(show, Point(x, y), 10, Scalar(255, 255, 255), 1, LINE_AA);
            circle(sDbg, Point(x, y), 10, Scalar(10), 1, LINE_AA);
            string str = to_string(C[i].t);
            putText(show, to_string(i), Point(x, y), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            putText(sDbg, str, Point(x, y), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(10), 1);
            putText(sDbg, to_string(i), Point(x+5, y+12), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(10), 1);
        }
        imshow("seg", show);
        waitKey(1);
      /*  if(err<par.alg1_tol)
            break;*/
    }
    matwrite("seg.bin", s);
    matwrite("depAtpix", depAtPix);
    matwrite("depAtPatch", depAtPatch);
}