#include <iostream>
#include "BlurEqualization.h"
#include "RMCBDFastAlterMin.h"

using namespace cv;
using namespace std;
int main() {
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
    int winSize = 256;
    Rect roi = Rect(1905, 1166, winSize, winSize);
    mat_vector gRoi = g(roi);
    gRoi.convertTo(gRoi, CV_64FC1);
    gRoi = gRoi / 255.;
//    double beta = -7;
//    double thresh = 2.;
//    mat_vector sigma = BlurEqualization(g1,g2,beta,thresh,kerSize);


    
    int L = 41;
    RMCBDFastAlterMin *proc = new RMCBDFastAlterMin(winSize);
    double gamma = 1e2;
    double alpha = 1e0*gamma;
    double beta = 1e4*gamma;
    double delta = 1e3*gamma;
    int maxLoop = 100;
    double tol = 1e-1;
    
    mat_vector ret = proc->MCBlindDeconv(gRoi,L,alpha,beta,delta,gamma,maxLoop,tol);
    return 0;
}
