//
// Created by bytelai on 2021/12/27.
//

#ifndef DFDWILD_PARAMETERS_H
#define DFDWILD_PARAMETERS_H
#define EPSILON DBL_EPSILON
#define DEPTH_PRE_CACULATE 32
#define OUTPUT_WIDTH 400
#define OUTPUT_HEIGHT 300
#define SCALE_FACTOR (OUTPUT_WIDTH / 2592.)
#define PIX_SIZE  2.2e-6 / SCALE_FACTOR
#define U0  0.3
#define PG  120
#define P1  122
#define P2  118
#define V0  1/(PG - 1/U0)
#define APERTURE_SIZE 1e-3
//#define SEG_NUMBER 10//segmentation labels size less than 256
#define W_TYPE std::vector<std::vector<std::vector<weight_st<T>>>>
#define CONTROL_NUMBER 80//control points num
#define EIG_NUMBER 30
#define EIG_PATH "D:/lwj/projects/matlab/dncuts/"
#define DATA_DIR "D:/lwj/projects/dfdWild/data/"

//z = cx*x + cy* y + b;

typedef struct param_st{
    double tau_i = 30.;
    double tau_o = 0.01;
    double tau_s = 0.01;
    double lambda_f = 2.;
    double lambda_s = 1e-5;
    double lambda_i = 1e3;
    double lambda_b = 5.;
    double w_tol = 1e-3;
    double D_tol = 1e-3;
    double alg1_tol = 1e-3;
    int alg2MaxLoop = 100;
    int alg3MaxLoop = 100;
    int alg1MaxLoop = 100;
    double dep_tol = 1e-3;
    int patchWidth = 9;
    int patchHeight = 9;
    double dep_max = 5.0;
    double dep_min = 0.3;
} PARAM;


#endif //DFDWILD_PARAMETERS_H
