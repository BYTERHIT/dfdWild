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
    visited[x] = 1;    //�������˾���1
    for( int i = 0 ; i < ugraph[x].edge.size() ; i++ )       //������x���ڵ����е�
    {
        int node = ugraph[x].edge[i];
        if( !visited[node] ) dfs(node,visited,ugraph,connnected);      //��������û�����ʹ�����ô����Ϊ���������ȥ
    }
}

//std::vector<std::vector<int>> ConnnectedRegionOfUndrrectedGraph(cv::Mat ajaMat);
vector<vector<int>> ConnnectedRegionOfUndrrectedGraph(Mat ajaMat)
{
    int N = ajaMat.rows;
    vector<int> visited(N,0);    //��¼ÿһ�����Ƿ񱻷��ʹ�
    vector<UGRAPH> ugraph(N);
    for( int i = 0 ; i < N ; i++ )   //���潨ͼ
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
    for( int i = 0; i < N ; i++ )     //����ÿһ��������ͨ����
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
