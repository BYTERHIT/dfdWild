cmake_minimum_required(VERSION 3.20)
set(CMAKE_TOOLCHAIN_FILE D:/lwj/data/vcpkg/scripts/buildsystems/vcpkg.cmake)
project(dfdWild)
set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV REQUIRED)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "opencv Not Found!")
endif (NOT OpenCV_FOUND)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(dfdWild main.cpp cmake-build-debug/DNCuts.cpp
        cmake-build-debug/DNCuts.h
        UnbiasedDefocusEqualizationFilters.cpp
        UnbiasedDefocusEqualizationFilters.h
        BlurEqualization.cpp
        BlurEqualization.h
        RMCBDFastAlterMin.cpp
        RMCBDFastAlterMin.h
        SparseMatTools.cpp
        SparseMatTools.h
        circulantMatrix.h
        circulantMatrix.cpp
        BCCB.cpp
        BCCB.h
        tgvOperator.cpp UDEFilters.cpp UDEFilters.h parameters.h)
target_link_libraries(dfdWild ${OpenCV_LIBS}
        )
