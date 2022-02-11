#include <iostream>
#include "BlurEqualization.h"
#include "RMCBDFastAlterMin.h"
#include "wildDfd.h"
#include "test/unitTest.h"

using namespace cv;
using namespace std;
int main_RMCBD() {
    std::cout << "Hello, World!" << std::endl;
    string file_path = "D:/lwj/projects/DFD_QT/autufocus/cmake-build-release/2021_9_18_16_46_38/";// QFileDialog::getExistingDirectory(NULL, QStringLiteral("选择文件"), dirPre, (QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks));//| QFileDialog::DontUseNativeDialog
    Mat g1 = imread(file_path + "far.bmp",IMREAD_UNCHANGED);
    int kerSize = 45;
    cvtColor(g1,g1,COLOR_BGR2GRAY);
    GaussianBlur(g1,g1,{9,9},1.5, 0, BORDER_REFLECT101);
    Mat g2 = imread(file_path + "nolens.bmp", IMREAD_UNCHANGED);
    cvtColor(g2,g2,COLOR_BGR2GRAY);
    GaussianBlur(g2,g2,{9,9},1.5,0,BORDER_REFLECT101);

    mat_vector g;
    g.addItem(g1);
    g.addItem(g2);
    g.convertTo(g, CV_64FC1);
    g = g / 255.;
    int winWidth = 256;
    int winHeight = 256;
//    Rect roi = Rect(1905, 1233, winSize, winSize);
    Size outputSz = {400,300};
    int hstep = g.width / outputSz.width;
    int vstep = g.height / outputSz.height;
    winHeight = max(vstep,winHeight);
    winWidth = max(hstep,winWidth);
    //winWidth = 256;
    //winHeight = 256;
//    gRoi = gRoi.t();
//    double beta = -1.46;
//    double thresh = 2.;
//    mat_vector sigma = BlurEqualization(g1,g2,beta,thresh,kerSize);

#if 1
    
    int L = 41;
    double gamma = 1e2;
    double alpha = 1e-1*gamma;
    double beta = 1e4*gamma;
    double delta = 1e3*gamma;
    int maxLoop = 50;
    double tol = 1e-1;
    //vector<double> sigma;

    //int i = 9;// idx / outputSz.width;
    //int j = 0;// idx% outputSz.width;
    //int roiSize = min(min(g.width - j * hstep, g.height - i * vstep), winSize);
    //Rect roi = Rect(j * hstep, i * vstep, roiSize, roiSize);
    //mat_vector gRoi = g(roi);
    //RMCBDFastAlterMin* proc = new RMCBDFastAlterMin(roiSize);
    //mat_vector ret = proc->MCBlindDeconv(gRoi, L, alpha, beta, delta, gamma, maxLoop, tol);
    //imwrite("h1_" + to_string(i) + "_" + to_string(j) + ".jpg", ret[0] * 10000);
    //imwrite("h2_" + to_string(i) + "_" + to_string(j) + ".jpg", ret[1] * 10000);
    //imwrite("u_" + to_string(i) + "_" + to_string(j) + ".jpg", reweight<T> * 255);
#ifndef DEBUG
    parallel_for_(Range(0, outputSz.height), [&](const Range& range) {
        for (size_t idx = range.start; idx < range.end; idx++){
        //for (size_t idx = 0; idx < outputSz.area(); idx++)
        //{
            int i = idx;
    //for(int i = 0; i<outputSz.height;i++){

//            if (i < 28)
//                continue;

            for (int j = 0; j < outputSz.width; j++) {
        //int j = 142;
                int roiWidth = min(g.width - j * hstep, winWidth);
                int roiHeight = min(g.height - i * vstep, winHeight);
                Rect roi = Rect(j * hstep, i * vstep, roiWidth, roiHeight);
                mat_vector gRoi = g(roi);
                RMCBDFastAlterMin* proc = new RMCBDFastAlterMin(roiWidth, roiHeight);
                mat_vector ret = proc->MCBlindDeconv(gRoi, L, alpha, beta, delta, gamma, maxLoop, tol);
                imwrite("./result/h1_" + to_string(i) + "_" + to_string(j) + ".jpg", ret[0] * 10000);
                imwrite("./result/h2_" + to_string(i) + "_" + to_string(j) + ".jpg", ret[1] * 10000);
                imwrite("./result/u_" + to_string(i) + "_" + to_string(j) + ".jpg", ret[2] * 255);
                cout << "+++++++++++++++++++++++++++++++++++ " << i << "  :  " << j << "   ++++++++++++"<<endl;
                delete proc;
            }
        }

        });
#else

        Rect roi = Rect(1915, 1151, winWidth, winHeight);
        mat_vector gRoi = g(roi);
        RMCBDFastAlterMin* proc = new RMCBDFastAlterMin(winWidth, winHeight);
        mat_vector ret = proc->MCBlindDeconv(gRoi, L, alpha, beta, delta, gamma, maxLoop, tol);
#endif
#endif
    return 0;
}
int mainWild() {
    std::cout << "Hello, World!" << std::endl;
    string file_path = "D:/lwj/projects/DFD_QT/autufocus/cmake-build-release/2021_9_18_16_46_38/";// QFileDialog::getExistingDirectory(NULL, QStringLiteral("选择文件"), dirPre, (QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks));//| QFileDialog::DontUseNativeDialog
    Mat g1 = imread(file_path + "far.bmp",IMREAD_UNCHANGED);
    int kerSize = 45;
//    cvtColor(g1,g1,COLOR_BGR2GRAY);
    GaussianBlur(g1,g1,{9,9},1.5, 0, BORDER_REFLECT101);
    Mat g2 = imread(file_path + "nolens.bmp", IMREAD_UNCHANGED);
//    cvtColor(g2,g2,COLOR_BGR2GRAY);
    GaussianBlur(g2,g2,{9,9},1.5,0,BORDER_REFLECT101);

//    double scaleFactor = 2592. / 400.;
    resize(g1,g1,{OUTPUT_WIDTH,OUTPUT_HEIGHT});
    resize(g2,g2,{OUTPUT_WIDTH,OUTPUT_HEIGHT});

//    mat_vector g;
//    g.addItem(g1);
//    g.addItem(g2);
    //g.convertTo(g, CV_64FC1);
    //g = g / 255.;
    PARAM par;
    par.patchHeight = 41 * SCALE_FACTOR;
    par.patchWidth = 41 * SCALE_FACTOR;
    par.alg1MaxLoop=100;
    par.alg2MaxLoop =10;
    par.alg3MaxLoop =10;
    par.dep_max = 5.0;
    par.dep_min = 0.3;
    par.dep_tol = 1e-2;
    par.D_tol = 1e-2;
    par.w_tol = 1e-2;
    par.alg1_tol = 1e-3;
    par.tau_i = 30;
    par.tau_s = 1e-2;
    par.tau_o = 5e-3;
    par.lambda_s = 1e-5;
    par.lambda_f = 2;
    par.lambda_i = 1e3;
    par.lambda_b = 5;
    wildDfd<double>(g1,g2,par);

    return 0;
}

int main()
{
    main_RMCBD();
    //mainWild();
    //test();
}