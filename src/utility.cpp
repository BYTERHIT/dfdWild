#include <io.h>
#include <direct.h>
#include <opencv2/opencv.hpp>
#include "utility.h"
#include <time.h>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;
int createDirectory(std::string path)
{
    int len = path.length();
    char tmpDirPath[256] = { 0 };
    for (int i = 0; i < len; i++)
    {
        tmpDirPath[i] = path[i];
        if (tmpDirPath[i] == '\\' || tmpDirPath[i] == '/')
        {
            if (_access(tmpDirPath, 0) == -1)
            {
                int ret = _mkdir(tmpDirPath);
                if (ret == -1) return ret;
            }
        }
    }
    return 0;
}

std::string GetTimeString()
{
    time_t rawtime;
    struct tm timeinfo;
    time(&rawtime);
    localtime_s(&timeinfo, &rawtime);
    std::string sfilename;
    sfilename = std::to_string(timeinfo.tm_year + 1900) + "_" + std::to_string(timeinfo.tm_mon + 1) + "_" \
        + std::to_string(timeinfo.tm_mday) + "_" + std::to_string(timeinfo.tm_hour) + "_" + std::to_string(timeinfo.tm_min) + "_" + \
        std::to_string(timeinfo.tm_sec);
    return sfilename;
}
void matwrite(const string& filename, const Mat& mat)
{
    ofstream fs(filename, fstream::binary);

    // Header
    int type = mat.type();
    int channels = mat.channels();
    fs.write((char*)&mat.rows, sizeof(int));    // rows
    fs.write((char*)&mat.cols, sizeof(int));    // cols
    fs.write((char*)&type, sizeof(int));        // type
    fs.write((char*)&channels, sizeof(int));    // channels

    // Data
    if (mat.isContinuous())
    {
        fs.write(mat.ptr<char>(0), (mat.dataend - mat.datastart));
    }
    else
    {
        int rowsz = CV_ELEM_SIZE(type) * mat.cols;
        for (int r = 0; r < mat.rows; ++r)
        {
            fs.write(mat.ptr<char>(r), rowsz);
        }
    }
}

Mat matread(const string& filename)
{
    ifstream fs(filename, fstream::binary);

    // Header
    int rows, cols, type, channels;
    fs.read((char*)&rows, sizeof(int));         // rows
    fs.read((char*)&cols, sizeof(int));         // cols
    fs.read((char*)&type, sizeof(int));         // type
    fs.read((char*)&channels, sizeof(int));     // channels

    // Data
    Mat mat(rows, cols, type);
    fs.read((char*)mat.data, CV_ELEM_SIZE(type) * rows * cols);

    return mat;
}
