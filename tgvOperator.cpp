//
// Created by laiwenjie on 2021/10/26.
//

#include "tgvOperator.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include "mat_vector.h"
#include <Eigen/Core>
#include <Eigen/SparseCore>
using namespace cv;
using namespace std;
using namespace Eigen;

MAX_MIN_NORM MaxMinNormalizeNoZero(Mat input)
{
    Mat tmp;
    if(input.type() != CV_64FC1)
    {
        input.convertTo(tmp,CV_64FC1);
    }
    else
    {
        tmp = input.clone();
    }
    Mat ret ;//= tmp.clone();//Mat::zeros(input.rows,input.cols,CV_64FC1);
    double * tPtr = (double *)tmp.data;
    double min = DBL_MAX, max = DBL_EPSILON;
    for(int i = 0; i < input.rows *input.cols;i++)
    {
        if(*tPtr >0)
        {
            if(*tPtr < min)
                min = *tPtr;
            if(*tPtr > max)
                max = *tPtr;
        }
        else
        {
            *tPtr = 0;
        }
        tPtr++;
    }
    ret =( tmp - min )/(max - min);
    double * rPtr = (double *)ret.data;
    for(int i = 0; i < input.rows *input.cols;i++)
    {
        if(*rPtr <0)
        {
            *rPtr = 0;
        }
        if(*rPtr > 1)
        {
            *rPtr = 1;
        }
        rPtr++;
    }
    MAX_MIN_NORM  norm;
    norm.max = max;
    norm.min = min;
    norm.norm = ret;
    return norm;
}

Point2d projL1(Point2d p2d, double alpha = 1.0)
{
    double norm = abs(p2d.x)+abs(p2d.y);
    norm = std::max(1.0,norm/alpha);
    return p2d/norm;
}
Point2d projL2(Point2d p2d, double alpha = 1.0)
{
    double norm = sqrt(p2d.x*p2d.x+p2d.y*p2d.y);
    norm = std::max(1.0,norm/alpha);
    return p2d/norm;
}
//前向差分,输入是一个通道的深度数据,注意边界的处理！
mat_vector  derivativeForward(Mat input,bool circulant)
{
    /*
    mat_vector difVec;
    Rect xRoi1 = Rect(0,0,input.cols-1,input.rows);
    Rect xRoi2 = Rect(1,0,input.cols-1,input.rows);
    Rect yRoi1 = Rect(0,0,input.cols,input.rows-1);
    Rect yRoi2 = Rect(0,1,input.cols,input.rows-1);

    Mat xDif = Mat::zeros(input.rows,input.cols,input.type());
    Mat yDif = Mat::zeros(input.rows,input.cols,input.type());
    xDif(xRoi1) = input(xRoi2) - input(xRoi1);
    yDif(yRoi1) = input(yRoi2) - input(yRoi1);
    difVec.addItem(xDif);
    difVec.addItem(yDif);
    return difVec;
     */
    Mat xDif = Mat::zeros(input.rows,input.cols,input.type());
    Mat yDif = Mat::zeros(input.rows,input.cols,input.type());
    int height = input.rows;
    int width = input.cols;
    double *xPtr = (double*)xDif.data;
    double *yPtr = (double*)yDif.data;
    double *dPtr = (double*)input.data;
    int lastLineOffset = width * (height - 1);
    int imgSize = height * width;
    for(int i = 0 ;i< imgSize; i++)
    {
        if(i%width == width - 1 )
        {
            if(circulant)
                *xPtr = *(dPtr-width + 1) - *dPtr;
            else
                *xPtr = 0;//*(dPtr-width + 1) - *dPtr;
        }
        else
        {
            *xPtr = *(dPtr+1) - *dPtr;
        }
        if(i>=lastLineOffset)
            if(circulant)
                *yPtr = *(dPtr-lastLineOffset) - *dPtr;
            else
                *yPtr = 0;//*(dPtr-lastLineOffset) - *dPtr;
        else
            *yPtr = *(dPtr+width) - *dPtr;
        xPtr++;yPtr++;dPtr++;
    }
    mat_vector difVec;
    difVec.addItem(xDif);
    difVec.addItem(yDif);
    return difVec;
}

//单边后向差分,输入是一个通道的深度数据
mat_vector  derivativeBackward(Mat input)
{
    Mat xDif = Mat::zeros(input.rows,input.cols,input.type());
    Mat yDif = Mat::zeros(input.rows,input.cols,input.type());
    int height = input.rows;
    int width = input.cols;
    double *xPtr = (double*)xDif.data;
    double *yPtr = (double*)yDif.data;
    double *dPtr = (double*)input.data;
    int lastLineOffset = width * (height - 1);
    int imgSize = height * width;
    for(int i = 0 ;i< imgSize; i++)
    {
        if(i%width == 0)
        {
            *xPtr = 0;//*dPtr - *(dPtr+width-1);
        }
        else
        {
            *xPtr = *dPtr - *(dPtr -1);
        }
        if(i<width)
            *yPtr = 0;//*dPtr - *(dPtr + lastLineOffset);
        else
            *yPtr = *dPtr - *(dPtr - width);
        xPtr++;yPtr++;dPtr++;
    }
    mat_vector difVec;
    difVec.addItem(xDif);
    difVec.addItem(yDif);
    return difVec;
}

//利用反向差分的负（和前向差分共轭）散度和差分是负共轭关系
//Mat divergenceForward(mat_vector grad )
//{
//    Mat xGrad = grad[0];
//    Mat yGrad = grad[1];
//    int height = xGrad.rows;
//    int width = xGrad.cols;
//    int type = xGrad.type();
//
//    Rect fistCol = Rect(0,0,1,height);
//    Rect back2Col = Rect(width-2,0,1,height);
//    Rect xRoi1 = Rect(0,0,width-2,height);
//    Rect xRoi2 = Rect(1,0,width-2,height);
//
//    Rect fistRow = Rect(0,0,width,1);
//    Rect back2Row = Rect(0,height-2,width,1);
//    Rect yRoi1 = Rect(0,0,width,height-2);
//    Rect yRoi2 = Rect(0,1,width,height-2);
//    Mat div = Mat::zeros(height,width,type);
//
//    div(fistCol)+=xGrad(fistCol);
//    div(xRoi2)+=xGrad(xRoi2)-xGrad(xRoi1);
//    div(back2Col)-=xGrad(back2Col);
//    div(fistRow)+=yGrad(fistRow);
//    div(yRoi2)+=yGrad(yRoi2)-yGrad(yRoi1);
//    div(back2Row)-=yGrad(back2Row);
//    return div;
//}
//利用反向差分的负（和前向差分共轭）散度和差分是负共轭关系
Mat divergenceForward(mat_vector grad,bool circulant )
{
    Mat xGrad = grad[0];
    Mat yGrad = grad[1];
    int height = xGrad.rows;
    int width = xGrad.cols;
    int type = xGrad.type();
    Mat diver = Mat::zeros(height,width,type);
    double *xPtr = (double*)xGrad.data;
    double *yPtr = (double*)yGrad.data;
    double *dPtr = (double*)diver.data;
    int lastLine = width * (height -1);
    int imgSize = height * width;
    for(int i = 0 ;i< height; i++)
    {
        for(int j = 0;j<width; j++)
        {
            if(j == 0)
            {
                if(circulant)
                    *dPtr += *(xPtr) - *(xPtr+width-1);
                else
                    *dPtr += *xPtr;//*(xPtr) - *(xPtr+width-1);
            }
            else if(j == width -1)
            {
                if(circulant)
                    *dPtr += *(xPtr) -*(xPtr-1);
                else
                    *dPtr += -*(xPtr-1);
            }
            else
            {
                *dPtr += *(xPtr) - *(xPtr-1);
            }
            if(i == 0)
            {
                if(circulant)
                    *dPtr += *yPtr - *(yPtr+lastLine);
                else
                    *dPtr += *yPtr;//*yPtr - *(yPtr+lastLine);
            }
            else if(i==width-1)
            {
                if(circulant)
                    *dPtr += *yPtr - *(yPtr-width);
                else
                    *dPtr += - *(yPtr-width);
            }
            else
            {
                *dPtr += *yPtr - *(yPtr-width);
            }
            xPtr++;yPtr++;dPtr++;
        }
    }
    return diver;
}
//单边循环后向差分的负共轭
Mat divergenceBackward(mat_vector grad)
{
    Mat xGrad = grad[0];
    Mat yGrad = grad[1];
    int height = xGrad.rows;
    int width = xGrad.cols;
    int type = xGrad.type();
    Mat diver = Mat::zeros(height,width,type);
    double *xPtr = (double*)xGrad.data;
    double *yPtr = (double*)yGrad.data;
    double *dPtr = (double*)diver.data;
    int lastLine = width * (height -1);
    int imgSize = height * width;
    for(int i = 0 ;i< height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            if(j == width-1)
            {
                *dPtr += -*xPtr;//*(xPtr - width -1) - *xPtr;
            }
            else if(j==0)
            {
                *dPtr += *(xPtr+1);//*(xPtr - width -1) - *xPtr;
            }
            else
            {
                *dPtr += *(xPtr+1) - *xPtr;
            }
            if(i == height-1)
            {
                *dPtr += -*yPtr;//*(yPtr-lastLine) - *yPtr;
            }
            else if(i == 0)
            {
                *dPtr += *(yPtr + width);//*(yPtr-lastLine) - *yPtr;
            }
            else
            {
                *dPtr += *(yPtr + width) - *yPtr;
            }
            xPtr++;yPtr++;dPtr++;
        }
    }
    return diver;
}

//单边循环边界后向差分实现对称差分算子
mat_vector symmetrizedSecondDerivativeBackward(mat_vector grad)
{
    mat_vector sym2ndDif;
    Mat xGrad = grad[0];
    Mat yGrad = grad[1];
    mat_vector difX = derivativeBackward(xGrad);
    mat_vector difY = derivativeBackward(yGrad);
    sym2ndDif.addItem(difX[0]);//xx
    sym2ndDif.addItem((difX[1] + difY[0])/2);//xy
    sym2ndDif.addItem((difX[1] + difY[0])/2);//yx
    sym2ndDif.addItem(difY[1]);//yy
    return sym2ndDif;
}
//单边循环边界后向差分实现对称差分算子的负共轭
mat_vector secondOrderDivergenceBackward(mat_vector grad)
{
    mat_vector vec;
    Mat xx = grad[0];
    Mat xy = grad[1], yx = grad[2], yy = grad[3];
    Mat xy_yx = (xy + yx) * 0.5;
    mat_vector xGrad, yGrad;
    xGrad.addItem(xx);
    xGrad.addItem(xy_yx);
    Mat xDiv = divergenceBackward(xGrad);
    yGrad.addItem(xy_yx);
    yGrad.addItem(yy);
    Mat yDiv = divergenceBackward(yGrad);
    vec.addItem(xDiv);
    vec.addItem(yDiv);
    return vec;
}

mat_vector symmetrizedSecondDerivativeForward(mat_vector grad,bool circulant)
{
    mat_vector sym2ndDif;
    Mat xGrad = grad[0];
    Mat yGrad = grad[1];
    mat_vector difX = derivativeForward(xGrad,circulant);
    mat_vector difY = derivativeForward(yGrad,circulant);
    sym2ndDif.addItem(difX[0]);//xx
//    sym2ndDif.addItem(difX[1]);//xy
//    sym2ndDif.addItem(difY[0]);//yx
    sym2ndDif.addItem((difX[1]+difY[0])/2);//(xy+yx)/2  xy
    sym2ndDif.addItem((difX[1]+difY[0])/2);//(xy+yx)/2  yx
    sym2ndDif.addItem(difY[1]);//yy
    return sym2ndDif;
}
//div2 symmetrizedSecondDerivative 的共轭算子
mat_vector secondOrderDivergenceForward(mat_vector second_order_derivative,bool circulant ){
    mat_vector vec;
    Mat xxGrad = second_order_derivative[0];
    Mat xyGrad = second_order_derivative[1];
    Mat yxGrad = second_order_derivative[2];
    Mat yyGrad = second_order_derivative[3];
    Mat xy_yx = (xyGrad + yxGrad)*0.5;

    mat_vector xGrad,yGrad;
    xGrad.addItem(xxGrad);
    xGrad.addItem(xy_yx);
    yGrad.addItem(xy_yx);
    yGrad.addItem(yyGrad);
    Mat xDiv = divergenceForward(xGrad,circulant);
    Mat yDiv = divergenceForward(yGrad,circulant);

    vec.addItem(xDiv);
    vec.addItem(yDiv);
    return vec;
}

//D*dU
//grad normerlized grad [dy*dy,-dx*dy,-dy*dx,dx*dx]
//edgePos pos = y*width + x;
mat_vector D_OPERATOR(mat_vector edgeGrad, mat_vector du)
{
    if(edgeGrad.empty())
    {
        return du;
    }
    Mat uDx = du[0];
    Mat uDy = du[1];
    Mat a = edgeGrad[0];
    Mat b = edgeGrad[1];
    Mat c = edgeGrad[2];
    mat_vector vec;
    Mat dx = uDx.mul(a) + uDy.mul(c);
    Mat dy = uDx.mul(c) + uDy.mul(b);
    vec.addItem(dx);
    vec.addItem(dy);
    return vec;
}
//D*dU
//grad normerlized grad [dy*dy,-dx*dy,-dy*dx,dx*dx]
//edgePos pos = y*width + x;
mat_vector D_OPERATOR(vector<EDGE_GRAD> edgeGrad, mat_vector du)
{
    if(edgeGrad.empty())
    {
        return du;
    }
    Mat uDx = du[0];
    Mat uDy = du[1];
    mat_vector vec;
    Mat dx = du[0].clone();
    Mat dy = du[1].clone();
    double* uDxPtr = (double*)uDx.data;
    double* uDyPtr = (double*)uDy.data;
    double* xPtr = (double*)dx.data;
    double* yPtr = (double*)dy.data;
    for(auto iter = edgeGrad.begin();iter!=edgeGrad.end();iter++)
    {
        int pos = iter->idx;
        double iDx2AtPos = iter->tGradProjMtx[1][1];
        double iDy2AtPos = iter->tGradProjMtx[0][0];
        double iDxDyAtPos = iter->tGradProjMtx[0][1];
        double uDxAtPos = *(uDxPtr+pos);
        double uDyAtPos = *(uDyPtr+pos);
        *(xPtr + pos) = uDxAtPos*iDy2AtPos + uDyAtPos * iDxDyAtPos;
        *(yPtr + pos) = uDxAtPos*iDxDyAtPos + uDyAtPos * iDx2AtPos;
    }
    vec.addItem(dx);
    vec.addItem(dy);
    return vec;
}
//(I+sigma*dF_star)^-1
// todo 需要确定是用L1还是L2范数，此处先用L1范数
mat_vector F_STAR_OPERATOR(mat_vector pBar, double alpha)
{
    int vecSize = pBar.size();
    mat_vector result;
    int width = pBar[0].cols, height = pBar[0].rows;
    Mat sum = Mat::zeros(height,width,pBar[0].type());
    for(auto iter = pBar.begin(); iter !=pBar.end();iter++)
    {
#ifdef USING_L1
        sum += abs(*iter);
#else
        sum += iter->mul(*iter);
#endif
    }
#ifndef USING_L1
    sqrt(sum,sum);
#endif
    sum /= alpha;
    double *ptr = (double*)sum.data;
    for(int i = 0 ;i <width*height;i++)
    {
        if(*ptr < 1.)
            *ptr = 1.;
        ptr++;
    }

    for(auto iter = pBar.begin(); iter !=pBar.end();iter++)
    {
        Mat item;
        divide(*iter,sum,item);
        result.addItem(item);
    }
    return result;
}

Mat G_OPERATOR(Mat g, Mat uBar, Mat to, double lambda, double thresh)
{
    Mat u = Mat::zeros(uBar.rows,uBar.cols,uBar.type());
    double *uPtr = (double*)u.data;
    double *uBarPtr = (double*)uBar.data;
    double *gPtr = (double*)g.data;
    double *toPtr = (double*)to.data;
    for(int i = 0;i < uBar.rows*uBar.cols;i++)
    {
        //lamba=0,when g==0
        if(*gPtr > thresh)
            *uPtr = (*uBarPtr + (*toPtr) * lambda * (*gPtr))/(1. + (*toPtr)*lambda);
        else
            *uPtr = *uBarPtr;
        uPtr++;
        uBarPtr++;
        gPtr++;
        toPtr++;
    }
    return u;

}

Mat G_OPERATOR(Mat g, Mat uBar, Mat to, Mat lambda, double thresh)
{
    Mat u = Mat::zeros(uBar.rows,uBar.cols,uBar.type());
    double *uPtr = (double*)u.data;
    double *uBarPtr = (double*)uBar.data;
    double *gPtr = (double*)g.data;
    double *toPtr = (double*)to.data;
    double *lambdaPtr = (double*)lambda.data;
    for(int i = 0;i < uBar.rows*uBar.cols;i++)
    {
        //lamba=0,when g==0
        *uPtr = (*uBarPtr + (*toPtr) * (*lambdaPtr) * (*gPtr))/(1. + (*toPtr)*(*lambdaPtr));
        uPtr++;
        uBarPtr++;
        gPtr++;
        toPtr++;
        lambdaPtr++;
    }
    return u;

}
//(I+to*dG)^-1
Mat G_OPERATOR(Mat g, Mat uBar,double to, double lambda)
{
    Mat u = Mat::zeros(uBar.rows, uBar.cols, uBar.type());
    double *uPtr = (double*)u.data;
    double *uBarPtr = (double*)uBar.data;
    double *gPtr = (double*)g.data;
    for(int i = 0;i < uBar.rows*uBar.cols;i++)
    {
        //lamba=0,when g==0
        if(*gPtr != 0)
            *uPtr = (*uBarPtr + to * lambda * (*gPtr))/(1. + to*lambda);
        else
            *uPtr = *uBarPtr;
        uPtr++;
        uBarPtr++;
        gPtr++;
    }
    return u;
}

double GetEnerge(Mat u,Mat g, mat_vector w, mat_vector edgeGrad, double lambda, double alpha_u, double alpha_w)
{
    double tgv = GetTgvCost(u,w,edgeGrad,alpha_u,alpha_w);
    double fidelity = GetFidelityCost(g,u,lambda);
    double energe = fidelity + tgv;//tgv;
    return energe;
}
double GetTgvCost(Mat u, mat_vector w, mat_vector edgeGrad, double alpha_u, double alpha_w)
{
#ifdef USING_BACKWARD
    mat_vector div = derivativeBackward(u);
#else
    mat_vector div = derivativeForward(u);
#endif
#ifndef d_u_w
    mat_vector divD = D_OPERATOR(edgeGrad,div) - w;
#else
    mat_vector divD = D_OPERATOR(edgeGrad,div - w);
#endif
#ifdef USING_BACKWARD
    mat_vector dif2 = symmetrizedSecondDerivativeBackward(w);
#else
    mat_vector dif2 = symmetrizedSecondDerivativeForward(w);
#endif
    double tv = div.norm1();
#ifdef USING_L1
    double tgv = alpha_u*divD.norm1() + alpha_w*dif2.norm1();
#else
    double tgv = alpha_u*divD.norm2() + alpha_w*dif2.norm2();
#endif
    return tgv;
}

double GetTgvCost(Mat u, mat_vector w, vector<EDGE_GRAD> edgeGrad, double alpha_u, double alpha_w)
{
#ifdef USING_BACKWARD
    mat_vector div = derivativeBackward(u);
#else
    mat_vector div = derivativeForward(u);
#endif
#ifndef d_u_w
    mat_vector divD = D_OPERATOR(edgeGrad,div) - w;
#else
    mat_vector divD = D_OPERATOR(edgeGrad,div - w);
#endif
#ifdef USING_BACKWARD
    mat_vector dif2 = symmetrizedSecondDerivativeBackward(w);
#else
    mat_vector dif2 = symmetrizedSecondDerivativeForward(w);
#endif
    double tv = div.norm1();
#ifdef USING_L1
    double tgv = alpha_u*divD.norm1() + alpha_w*dif2.norm1();
#else
    double tgv = alpha_u*divD.norm2() + alpha_w*dif2.norm2();
#endif
    return tgv;
}
double GetFidelityCost(Mat g, Mat u, double lambda)
{
    double minDep = 0, maxDep = 10;
    minMaxLoc(g,&minDep,&maxDep);
    Mat offset = u - g;
    Mat mask;
    threshold(g,mask,DBL_EPSILON + minDep,1.,THRESH_BINARY);
    Mat fidelityMat = offset.mul(offset).mul(mask);
    double energe = 0.5*lambda*sum(fidelityMat)[0];
    return energe;
}
double GetEnerge(Mat u,Mat g, mat_vector w, vector<EDGE_GRAD> edgeGrad, double lambda, double alpha_u, double alpha_w)
{
    double tgv = GetTgvCost(u,w,edgeGrad,alpha_u,alpha_w);
    double fidelity = GetFidelityCost(g,u,lambda);
    double energe = fidelity + tgv;//tgv;
    return energe;
}

Point GetCoor(int idx ,int rows, int cols)
{
    int i = idx / cols;
    int j = idx % rows;
    return Point(j,i);
}

/* 对于图像m行n列，前向差分算子[DX;DY]:
 * |-----------------------------m block----------------------------------------------------------|
 *  |--n cols--|
 *  -1  1      |
 *      .  .   |
 *        -1  1|
 *            0|
 * -------------------------
 *             |-1  1      |
 *             |    .  .   |
 *             |      -1  1|
 *             |          0|
 *             -------------
 *                               *
 *                                     *
 *                                           |------------
 *                                           |-1  1      |
 *                                           |    .  .   |
 *                                           |      -1  1|
 *                                           |          0|
 * -------------------------                 |------------
 *  -1         | 1         |
 *      .      |    .      |
 *         .   |       .   |
 *           -1|          1|
 * ------------------------------------------------------
 *                   *           *
 *                         *           *
 * ------------------------------|------------------------
 *                               |-1         | 1         |
 *                               |    .      |    .      |
 *                               |       .   |       .   |
 *                               |         -1|          1|
 *  ------------                             -------------
 *   0         |                             | -1        |
 *      .      |                             |    .      |
 *         .   |                             |       .   |
 *            0|                             |         -1|
 *  对于D_EDGE，D=[A11,A12;A21,A22]
 *  K:
 *  {alpha_u*D_edge*[dx,      -I,               0;
 *                   dy,       0,              -I]
 *                   0,    alpha_w*dx,          0;
 *                   0,    alpha_w/2*dy,  alpha_w/2*dx;
 *                   0,    alpha_w/2*dy,  alpha_w/2*dx;
 *                   0,        0,            alpha_w*dy}
 * */
mat_vector GetSteps(mat_vector edgeGrad, int rows, int cols, double alpha_u, double alpha_w, double alpha = 1.)
{
    int size = rows * cols;
    SparseMatrix<double> firstTwoRowOfBlocks(2*size,3*size);
    SparseMatrix<double> K(6*size,3*size);
    SparseMatrix<double> D_edge(2*size,2*size);
    vector<Triplet<double>> nonZeros;
    //K:
    //{alpha_u*D_edge*[dx,      -I,               0;
    //                 dy,       0,              -I]
    //                 0,    alpha_w*dx,          0;
    //                 0,    alpha_w/2*dy,  alpha_w/2*dx;
    //                 0,    alpha_w/2*dy,  alpha_w/2*dx;
    //                 0,        0,            alpha_w*dy}
    for(int j = 0;j<rows;j++){//dx
        int offset = j*cols;
        for(int i = 0; i < cols -1;i++)
        {
            nonZeros.emplace_back(offset + i, offset + i,-1.);
            nonZeros.emplace_back( offset + i, offset + i + 1,1.);
        }
    }
    for(int j = 0;j<rows -1;j++){//dy
        int offset = j*cols;
        for(int i = 0; i < cols;i++)
        {
            nonZeros.emplace_back(offset + size + i, offset + i,-1.);
            nonZeros.emplace_back( offset + size + i, offset + i + cols,1.);
        }
    }
#ifdef d_u_w
    for(int j = 0;j<rows;j++){//-I 0;0 -I;
        int offset = j*cols;
        for(int i = 0; i < cols;i++)
        {
            nonZeros.emplace_back(offset + i, offset + size + i,-1.);
            nonZeros.emplace_back( offset + i + size, offset + 2*size + i, -1.);
        }
    }
#endif
    firstTwoRowOfBlocks.setFromTriplets(nonZeros.begin(),nonZeros.end());

    vector<Triplet<double>> nonZerosDedge;
    if (!edgeGrad.empty())
    {
        double *edgeXXptr = (double*)edgeGrad[0].data;
        double *edgeYYptr = (double*)edgeGrad[1].data;
        double *edgeXYptr = (double*)edgeGrad[2].data;
        for (int j = 0; j < size; j++) {
            nonZerosDedge.emplace_back(j, j, *(edgeXXptr+j));//xx
            nonZerosDedge.emplace_back(j, j + size, *(edgeXYptr+j));//xy
            nonZerosDedge.emplace_back(j + size, j, *(edgeXYptr+j));//yx
            nonZerosDedge.emplace_back(j + size, j + size, *(edgeYYptr+j));//yy
        }
    }
    else
    {
        for (int j = 0; j < size; j++) {
            nonZerosDedge.emplace_back(j, j, 1.);//xx
            nonZerosDedge.emplace_back(j + size, j + size, 1.);//yy
        }
    }
    
    D_edge.setFromTriplets(nonZerosDedge.begin(),nonZerosDedge.end());

    firstTwoRowOfBlocks =  D_edge * firstTwoRowOfBlocks;
    firstTwoRowOfBlocks *= alpha_u;
    vector<Triplet<double>> nonZerosK;
    for (int k=0; k<firstTwoRowOfBlocks.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(firstTwoRowOfBlocks, k); it; ++it) {
            double Kij = it.value(); // 元素值
            int i = it.row();   // 行标row index
            int j = it.col();   // 列标（此处等于k）
            nonZerosK.emplace_back(i,j,Kij);
        }
    }

    //delta x
    for (int j = 0; j < rows; j++) {
        int offset = j * cols;
        for (int i = 0; i < cols - 1; i++) {
#ifdef USING_BACKWARD
            nonZerosK.emplace_back(size*2 + offset + i + 1, size + offset + i + 1,  alpha_w);
            nonZerosK.emplace_back(size*2 + offset + i + 1, size + offset + i    ,  -alpha_w);

            nonZerosK.emplace_back(size*3 + offset + i + 1, size*2 + offset + i + 1, 0.5 * alpha_w);
            nonZerosK.emplace_back(size*3 + offset + i + 1, size*2 + offset + i    , -0.5 * alpha_w);

            nonZerosK.emplace_back(size*4 + offset + i + 1, size*2 + offset + i + 1, 0.5 * alpha_w);
            nonZerosK.emplace_back(size*4 + offset + i + 1, size*2 + offset + i    , -0.5 * alpha_w);
#else

            nonZerosK.emplace_back(size*2 + offset + i, size + offset + i, -alpha_w);
            nonZerosK.emplace_back(size*2 + offset + i, size + offset + i + 1, alpha_w);

            nonZerosK.emplace_back(size*3 + offset + i, size*2 + offset + i  , -0.5 * alpha_w);
            nonZerosK.emplace_back(size*3 + offset + i, size*2 + offset + i   + 1, 0.5 * alpha_w);

            nonZerosK.emplace_back(size*4 + offset + i, size*2 + offset + i  , -0.5 * alpha_w);
            nonZerosK.emplace_back(size*4 + offset + i, size*2 + offset + i   + 1, 0.5 * alpha_w);
#endif
        }
    }
    //delta y
    for (int j = 0; j < rows - 1; j++) {
        int offset = j * cols;
        for (int i = 0; i < cols; i++) {
#ifdef USING_BACKWARD
            nonZerosK.emplace_back(offset + i + 3*size + cols, size + offset + i, -0.5 * alpha_w);
            nonZerosK.emplace_back(offset + i + 3*size + cols, size + offset + i + cols, 0.5 * alpha_w);

            nonZerosK.emplace_back(offset + i + 4*size + cols, size + offset + i, -0.5 * alpha_w);
            nonZerosK.emplace_back(offset + i + 4*size + cols, size + offset + i + cols, 0.5 * alpha_w);

            nonZerosK.emplace_back(offset + i + 5*size + cols, offset + 2*size + i, -alpha_w);
            nonZerosK.emplace_back(offset + i + 5*size + cols, offset + 2*size + i + cols, alpha_w);
#else
            nonZerosK.emplace_back(offset + i + 3*size, size + offset + i, -0.5 * alpha_w);
            nonZerosK.emplace_back(offset + i + 3*size, size + offset + i + cols, 0.5 * alpha_w);

            nonZerosK.emplace_back(offset + i + 4*size, size + offset + i, -0.5 * alpha_w);
            nonZerosK.emplace_back(offset + i + 4*size, size + offset + i + cols, 0.5 * alpha_w);

            nonZerosK.emplace_back(offset + i + 5*size, offset + 2*size + i, -alpha_w);
            nonZerosK.emplace_back(offset + i + 5*size, offset + 2*size + i + cols, alpha_w);
#endif
        }
    }
#ifndef d_u_w
    for(int j = 0;j<rows;j++){//-I 0;0 -I;
        int offset = j*cols;
        for(int i = 0; i < cols;i++)
        {
            nonZerosK.emplace_back(offset + i, offset + size + i,-alpha_u);
            nonZerosK.emplace_back( offset + i + size, offset + 2*size + i, -alpha_u);
        }
    }
#endif
    K.setFromTriplets(nonZerosK.begin(), nonZerosK.end());
//    cout << K << endl;
    //迭代访问稀疏矩阵
    mat_vector to(3,Mat::zeros(rows,cols,CV_64FC1));
    mat_vector sigma(6,Mat::zeros(rows,cols,CV_64FC1));
    for (int k=0; k<K.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(K, k); it; ++it) {
            double Kij = abs(it.value()); // 元素值
            int i = it.row();   // 行标row index
            int j = it.col();   // 列标（此处等于k）
            int toMatIdx = j / size;
            int toOffset = j % size;
            double *toPtr = (double *)to[toMatIdx].data;
            *(toPtr + toOffset) += pow(Kij,2-alpha);
            int sigmaMatIdx = i / size;
            int sigmaOffset = i % size;
            double *sigmaPtr = (double *)sigma[sigmaMatIdx].data;
            *(sigmaPtr + sigmaOffset) += pow(Kij,alpha);
        }
    }
    mat_vector params;
    Mat ones = Mat::ones(rows, cols,CV_64FC1);
    //Mat xoffset = sigmaX - sigmaX_;
    //Mat yoffset = sigmaY - sigmaY_;
    //Mat tooffset = to - to_;

    double delta = DBL_EPSILON;
    for(int i = 0; i<3;i++)
    {
        Mat toTmp;
        divide(ones,to[i] + delta,toTmp);
        params.addItem(toTmp);
    }
    for(int i = 0; i<6; i++)
    {
        Mat tmp;
        divide(ones,sigma[i] + delta,tmp);
        params.addItem(tmp);
    }
    return params;
}

mat_vector GetUstepsUsingMat(vector<EDGE_GRAD> edgeGrad, int rows, int cols)
{
    //debug code to seee grad
    Mat A11 = Mat::ones(rows, cols, CV_64FC1),
        A12 = Mat::zeros(rows, cols, CV_64FC1),
        A21 = A12.clone(),
        A22 = A11.clone();
    double* a11Ptr = (double*)A11.data;
    double* a12Ptr = (double*)A12.data;
    double* a21Ptr = (double*)A21.data;
    double* a22Ptr = (double*)A22.data;
    for (auto it = edgeGrad.begin(); it != edgeGrad.end(); it++)
    {
        *(a11Ptr + it->idx) = it->tGradProjMtx[0][0];
        *(a12Ptr + it->idx) = it->tGradProjMtx[0][1];
        *(a21Ptr + it->idx) = it->tGradProjMtx[1][0];
        *(a22Ptr + it->idx) = it->tGradProjMtx[1][1];
    }
    Mat A11Dx, A12Dy, A21Dx, A22Dy;
    //对m行n列的图像，其差分算子DX，DY是mn行mn列的稀疏矩阵，只有在以下位置(坐标序号为(row,col))有值
    Mat DXDiag; //对角线元素
    Mat DYDiag;//对角线
    Mat DYDiagURN;//(0,n)->(mn-n,mn)
    Mat DXDiagUR1 = Mat::ones(rows, cols, CV_64FC1);//(0,1)->(mn-1,mn)
    DYDiag = DXDiagUR1.clone();
    DYDiagURN = DYDiag.clone();
    DXDiag = -1 * Mat::ones(rows, cols, CV_64FC1);
    Rect lastCol = Rect(cols - 1, 0, 1, rows);
    Rect lastRow = Rect(0, rows - 1, cols, 1);
    Rect ur1Region = Rect(1, 0, cols - 1, rows);
    Rect ur1RegionAnchor = Rect(0, 0, cols - 1, rows);
    Rect urnRegion = Rect(0, 1, cols, rows - 1);
    Rect urnRegionAnchor = Rect(0, 0, cols, rows - 1);
    DXDiag(lastCol) *= 0;
    DXDiagUR1(lastCol) *= 0;
    DYDiag(lastRow) *= 0;
    DYDiag *= -1;
    DYDiagURN(lastRow) *= 0;
    Mat K11Diag, K11DiagUR1, K11DiagURN, K21Diag, K21DiagURN, K21DiagUR1;
    K11Diag = A11.mul(DXDiag) + A12.mul(DYDiag);
    K11DiagUR1 = A11.mul(DXDiagUR1);
    K11DiagURN = A12.mul(DYDiagURN);
    K21Diag = A21.mul(DXDiag) + A22.mul(DYDiag);
    K21DiagUR1 = A21.mul(DXDiagUR1);
    K21DiagURN = A22.mul(DYDiagURN);

    Mat sigmaX_ = abs(K11Diag) + abs(K11DiagUR1) + abs(K11DiagURN);
    Mat sigmaY_ = abs(K21Diag) + abs(K21DiagUR1) + abs(K21DiagURN);
    Mat to_ = abs(K11Diag) + abs(K21Diag);
    to_(ur1Region) += abs(K11DiagUR1(ur1RegionAnchor)) + abs(K21DiagUR1(ur1RegionAnchor));
    to_(urnRegion) += abs(K11DiagURN(urnRegionAnchor)) + abs(K21DiagURN(urnRegionAnchor));

    mat_vector params;
    Mat ones = Mat::ones(rows, cols, CV_64FC1);
    //Mat xoffset = sigmaX - sigmaX_;
    //Mat yoffset = sigmaY - sigmaY_;
    //Mat tooffset = to - to_;

    divide(ones, to_ + DBL_EPSILON, to_);
    divide(ones, sigmaX_ + DBL_EPSILON, sigmaX_);
    divide(ones, sigmaY_ + DBL_EPSILON, sigmaY_);
    params.addItem(to_);
    params.addItem(sigmaX_);
    params.addItem(sigmaY_);
    return params;
}
/*
 * w=[w1;w2],2mn*1;
 * eps(w) = [dx 0; dy/2 dx/2; dy/2 dx/2; 0 dy]*w
 */
mat_vector GetWSteps(int rows, int cols) {
    int size = rows * cols;
    SparseMatrix<double> EPSILON(4 * size, 2 * size), K(2 * size, size);
    vector<Triplet<double>> nonZeros;
    //delta x
    for (int j = 0; j < rows; j++) {
        int offset = j * cols;
        for (int i = 0; i < cols - 1; i++) {
            nonZeros.emplace_back(offset + i, offset + i, -1.);
            nonZeros.emplace_back(offset + i, offset + i + 1, 1.);

            nonZeros.emplace_back(offset + i + size, offset + i + size, -0.5);
            nonZeros.emplace_back(offset + i + size, offset + i + size + 1, 0.5);

            nonZeros.emplace_back(offset + i + 2*size, offset + i + size, -0.5);
            nonZeros.emplace_back(offset + i + 2*size, offset + i + size + 1, 0.5);
        }
    }
    //delta y
    for (int j = 0; j < rows - 1; j++) {
        int offset = j * cols;
        for (int i = 0; i < cols; i++) {
            nonZeros.emplace_back(offset + i + size, offset + i, -0.5);
            nonZeros.emplace_back(offset + i + size, offset + i + cols, 0.5);

            nonZeros.emplace_back(offset + i + 2*size, offset + i, -0.5);
            nonZeros.emplace_back(offset + i + 2*size, offset + i + cols, 0.5);

            nonZeros.emplace_back(offset + i + 3*size, offset + size + i, -1);
            nonZeros.emplace_back(offset + i + 3*size, offset + size + i + cols, 1);
        }
    }
    EPSILON.setFromTriplets(nonZeros.begin(),nonZeros.end());
//    cout <<"EPSILON:" << EPSILON << endl;
    cout <<"迭代访问稀疏矩阵的元素 "<<endl;
    Mat toX = Mat::zeros(rows,cols,CV_64FC1);
    Mat toY = Mat::zeros(rows,cols,CV_64FC1);
    Mat sigmaXX = Mat::zeros(rows,cols,CV_64FC1);
    Mat sigmaXY = Mat::zeros(rows,cols,CV_64FC1);
    Mat sigmaYX = Mat::zeros(rows,cols,CV_64FC1);
    Mat sigmaYY = Mat::zeros(rows,cols,CV_64FC1);
    double *toXPtr = (double*)toX.data;
    double *toYPtr = (double*)toY.data;
    double *sigmaXXPtr = (double*)sigmaXX.data;
    double *sigmaXYPtr = (double*)sigmaXY.data;
    double *sigmaYXPtr = (double*)sigmaYX.data;
    double *sigmaYYPtr = (double*)sigmaYY.data;
    double alpha = 1.;
    for (int k=0; k< EPSILON.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(EPSILON, k); it; ++it) {
            double Kij = abs(it.value()); // 元素值
            int i = it.row();   // 行标row index
            int j = it.col();   // 列标（此处等于k）
            if(j>=size)
                *(toYPtr + j - size) += pow(Kij, 2-alpha);
            else
                *(toXPtr + j) += pow(Kij,2-alpha);
            if(i >= 3*size)
                *(sigmaYYPtr + i - 3*size) += pow(Kij,alpha);
            else if(i>=2*size)
                *(sigmaYXPtr + i - 2*size) += pow(Kij,alpha);
            else if(i>=size)
                *(sigmaXYPtr + i - size) += pow(Kij,alpha);
            else
                *(sigmaXXPtr + i) += pow(Kij,alpha);
        }
    }
    mat_vector ret;
    Mat ones = Mat::ones(rows, cols,CV_64FC1);
    divide(ones,toX + DBL_EPSILON,toX);
    divide(ones,toY + DBL_EPSILON,toY);
    divide(ones,sigmaXX + DBL_EPSILON,sigmaXX);
    divide(ones,sigmaXY + DBL_EPSILON,sigmaXY);
    divide(ones,sigmaYX + DBL_EPSILON,sigmaYX);
    divide(ones,sigmaYY + DBL_EPSILON,sigmaYY);
    ret.addItem(toX);
    ret.addItem(toY);
    ret.addItem(sigmaXX);
    ret.addItem(sigmaXY);
    ret.addItem(sigmaYX);
    ret.addItem(sigmaYY);
    return ret;
}
#define MIN_NORM_VALUE 1e-8
#define MIN_TENSOR_VAL 1e-8
//前向差分
mat_vector  GetDGradMtx(Mat grayImg, double gama, double beta)
{
    Mat img = grayImg.clone();
    int width = img.cols, height = img.rows;
    Mat a = Mat::zeros(height,width,CV_64FC1);
    Mat b = Mat::zeros(height,width,CV_64FC1);
    Mat c = Mat::zeros(height,width,CV_64FC1);
    Mat gradX;
    Mat gradY;
//    Mat G_x = (Mat_<double>(3,3)<<1,0,-1,2,0,-2,1,0,-1);
    Mat G_x = (Mat_<double>(2,2)<<-1,1,-1,1);//0,-2,1,0,-1);
    Mat G_y = G_x.t();
    img.convertTo(img, CV_64FC1,1./255);
//    filter2D(img,gradX,CV_64FC1,G_x,Point(-1,-1),0,BORDER_REPLICATE);
//    filter2D(img,gradY,CV_64FC1,G_y,Point(-1,-1),0,BORDER_REPLICATE);
    mat_vector gradImg = derivativeForward(img);
    gradX = gradImg[0];
    gradY = gradImg[1];
    Mat gradNormL2 = gradX.mul(gradX) + gradY.mul(gradY);
    sqrt(gradNormL2,gradNormL2);

    divide(gradX,gradNormL2,gradX);
    divide(gradY,gradNormL2,gradY);

    Mat tmp;
    pow(gradNormL2,gama,tmp);
    tmp = -beta * tmp;
    Mat factor;
    exp(tmp,factor);

    double * normPtr = (double *) gradNormL2.data;
    double * dxPtr = (double *) gradX.data;
    double * dyPtr = (double *) gradY.data;
    double * ePtr = (double *) factor.data;
    double *aPtr = (double *)a.data;
    double *bPtr = (double *)b.data;
    double *cPtr = (double *)c.data;

    for (int i = 0; i < height * width; i++) {
        if(*normPtr < MIN_NORM_VALUE)
        {
            *dxPtr = 1;
            *dyPtr = 0;
        }
        if(*ePtr<MIN_TENSOR_VAL)
        {
            *ePtr = MIN_TENSOR_VAL;
        }

        double dxdx = (*dxPtr)*(*dxPtr);
        double dydx = (*dxPtr)*(*dyPtr);
        double dydy = (*dyPtr)*(*dyPtr);
        double e = *ePtr;
        //[a,c;c,b]
        *aPtr = e * dxdx +  dydy;
        *cPtr = (e-1)*dydx;
        *bPtr = e*dydy + dxdx;
        aPtr++;bPtr++;cPtr++;
        dxPtr++;dyPtr++;normPtr++;ePtr++;
    }
    mat_vector ret;
    ret.addItem(a);ret.addItem(b);ret.addItem(c);
    return ret;
}
//xx,yy,-xy
mat_vector GetTensor(Mat spMap, Mat grayImg,Mat depth)
{
    Mat img = grayImg.clone();
    int width = img.cols, height = img.rows;
//    Mat G_x = (Mat_<double>(3,3)<<1,0,-1,2,0,-2,1,0,-1);
    Mat G_x = (Mat_<double>(2,2)<<-1,1,-1,1);//0,-2,1,0,-1);
    Mat G_y = G_x.t();
    Mat edge;
    img.convertTo(img, CV_64FC1, 1.);
    spMap.convertTo(edge, CV_64FC1, 1.);
    Mat gradXImg, gradYImg, gradXSp, gradYSp;
//    filter2D(img,gradXImg,CV_64FC1,G_x,Point(-1,-1),0,BORDER_REPLICATE);
//    filter2D(img,gradYImg,CV_64FC1,G_y,Point(-1,-1),0,BORDER_REPLICATE);
    mat_vector gradImg = derivativeForward(img);
    gradXImg = gradImg[0];
    gradYImg = gradImg[1];
    Mat gradImgNormL2 = gradXImg.mul(gradXImg) + gradYImg.mul(gradYImg);
    sqrt(gradImgNormL2,gradImgNormL2);
    divide(gradXImg,gradImgNormL2,gradXImg);
    divide(gradYImg,gradImgNormL2,gradYImg);
//    filter2D(edge,gradXSp,CV_64FC1,G_x,Point(-1,-1),0,BORDER_REPLICATE);
//    filter2D(edge,gradYSp,CV_64FC1,G_y,Point(-1,-1),0,BORDER_REPLICATE);
    mat_vector gradSp = derivativeForward(edge);
    gradXSp = gradSp[0];
    gradYSp = gradSp[1];
    Mat gradSpNormL2 = gradXSp.mul(gradXSp) + gradYSp.mul(gradYSp);
    sqrt(gradSpNormL2,gradSpNormL2);
    divide(gradXSp,gradSpNormL2,gradXSp);
    divide(gradYSp,gradSpNormL2,gradYSp);
    Mat a = Mat::zeros(height,width,CV_64FC1);
    Mat b = Mat::zeros(height,width,CV_64FC1);
    Mat c = Mat::zeros(height,width,CV_64FC1);
    Mat convas;
    if(depth.empty())
        convas = grayImg.clone();
    else
        depth.convertTo(convas,CV_8UC3);
    for (int i = 0; i < edge.rows; i++) {
        for (int j = 0; j < edge.cols; j++) {

            if(gradSpNormL2.at<double>(i,j) > MIN_NORM_VALUE )//edge
            {
                double xGrad = gradXSp.at<double>(i,j);
                double yGrad = gradYSp.at<double>(i,j);
                a.at<double>(i,j) = yGrad * yGrad;
                b.at<double>(i,j) = xGrad * xGrad;
                c.at<double>(i,j) = -xGrad * yGrad;
                Point p(j, i);
                circle(convas, p, 0, Scalar(0, 0, 0), -1);
////                EDGE_GRAD edgeGrad;
//                if(gradImgNormL2.at<double>(i,j) < MIN_NORM_VALUE)
//                {
//                    gradXImg.at<double>(i,j) = 1;
//                    gradYImg.at<double>(i,j) = 0;
//                }
//                double xGrad = gradXImg.at<double>(i,j);
//                double yGrad = gradYImg.at<double>(i,j);
//                a.at<double>(i,j) = yGrad * yGrad;
//                b.at<double>(i,j) = xGrad * xGrad;
//                c.at<double>(i,j) = -xGrad * yGrad;
//                Point p(j, i);
//                circle(img, p, 0, Scalar(0, 0, 0), -1);
            }
            else
            {
                a.at<double>(i,j) = 1.;
                b.at<double>(i,j) = 1;
                c.at<double>(i,j) = 0;
            }
        }
    }
    imwrite("graySuperPix.jpg",convas);
    mat_vector ret;
    ret.addItem(a);ret.addItem(b);ret.addItem(c);
    return ret;
}

