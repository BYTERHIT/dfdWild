//
// Created by bytelai on 2022/1/7.
//

#include "ugraph.h"
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
typedef struct undirected_graph{
    vector<int> edge;
}UGRAPH;

void dfs(int x, vector<int> &visited,vector<UGRAPH> &ugraph, vector<int> &connnected)
{
    connnected.push_back(x);
    visited[x] = 1;    //遍历到了就置1
    for( int i = 0 ; i < ugraph[x].edge.size() ; i++ )       //遍历与x相邻的所有点
    {
        int node = ugraph[x].edge[i];
        if( !visited[node] ) dfs(node,visited,ugraph,connnected);      //如果这个点没被访问过，那么以它为起点搜索下去
    }
}

//std::vector<std::vector<int>> ConnnectedRegionOfUndrrectedGraph(cv::Mat ajaMat);
vector<vector<int>> ConnnectedRegionOfUndrrectedGraph(Mat ajaMat)
{
    int N = ajaMat.rows;
    vector<int> visited(N,0);    //记录每一个点是否被访问过
    vector<UGRAPH> ugraph(N);
    for( int i = 0 ; i < N ; i++ )   //常规建图
    {
        for(int j = i+1; j<N;j++)
        {
            if (ajaMat.at<uchar>(i, j) == 1)
            {
                ugraph[i].edge.push_back(j);
                ugraph[j].edge.push_back(i);
            }
        }
    }
    vector<vector<int>> connectComps;
    for( int i = 0; i < N ; i++ )     //遍历每一个点求联通分量
    {
        if( !visited[i] )
        {
            vector<int> component;
            dfs(i,visited,ugraph,component);
            connectComps.push_back(component);
        }
    }
    return connectComps;
}
