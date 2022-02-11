//
// Created by bytelai on 2022/1/1.
//

#ifndef DFDWILD_COMPRESSEDWEIGHTARRAY_H
#define DFDWILD_COMPRESSEDWEIGHTARRAY_H
#include <vector>

template<typename T>
class compressedWeightArray {
private:
    std::vector<std::vector<std::vector<T>>> _weightArray;
public:
    T at(int x, int y, int z);

};


#endif //DFDWILD_COMPRESSEDWEIGHTARRAY_H
