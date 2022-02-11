//
// Created by bytelai on 2022/1/12.
//

#include "unitTest.h"
#include "../algTemplate.h"

void test()
{
    vector<vector<vector<double>>> w;
    for (int i = 0; i < 300; i++)
    {
        vector<vector<double>> w_lin;
        for (int j = 0; j < 400; j++)
        {

            vector<double> w_ij = readW<double>(j, i, 0);
            w_lin.push_back(w_ij);
        }
        w.push_back(w_lin);
    }

        

}