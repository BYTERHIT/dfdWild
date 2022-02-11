cmake_minimum_required(VERSION 3.20)
set(CMAKE_TOOLCHAIN_FILE D:/lwj/data/vcpkg/scripts/buildsystems/vcpkg.cmake)
project(dfdWild)
set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV REQUIRED)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "opencv Not Found!")
endif (NOT OpenCV_FOUND)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(dfdWild
        ./src/main.cpp
        src/DNCuts.cpp
        src/DNCuts.h
        src/UnbiasedDefocusEqualizationFilters.cpp
        src/UnbiasedDefocusEqualizationFilters.h
        src/BlurEqualization.cpp
        src/BlurEqualization.h
        src/RMCBDFastAlterMin.cpp
        src/RMCBDFastAlterMin.h
        src/SparseMatTools.cpp
        src/SparseMatTools.h
        src/circulantMatrix.h
        src/circulantMatrix.cpp
        src/BCCB.cpp
        src/BCCB.h
        src/tgvOperator.cpp
        src/UDEFilters.cpp
        src/UDEFilters.h
        src/parameters.h
        src/maxflow/graph.cpp
        src/maxflow/block.h
        src/maxflow/maxflow.cpp
        src/MRFmin.h
        src/imageFeature.h
        src/compressedWeightArray.cpp
        src/utility.cpp
        src/compressedWeightArray.h
        src/RobustEstimation.h
        src/AlgOperator.h
        src/AlgOperator.cpp
        src/ugraph.cpp src/ugraph.h src/wildDfd.cpp src/wildDfd.h src/algTemplate.h src/test/unitTest.cpp src/test/unitTest.h src/myTypes.h)
target_link_libraries(dfdWild ${OpenCV_LIBS}
        )
