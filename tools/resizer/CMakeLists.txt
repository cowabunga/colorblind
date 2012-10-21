cmake_minimum_required(VERSION 2.8)

SET(OPENCV_MATCHER_REQUIRED_DEPS opencv_core opencv_highgui opencv_imgproc)

ocv_check_dependencies(${OPENCV_MATCHER_REQUIRED_DEPS})
ocv_include_modules(${OPENCV_MATCHER_REQUIRED_DEPS})
ocv_include_directories(../../include)

project(resizer)
add_executable(resizer main.cpp)
target_link_libraries(resizer ${OPENCV_LINKER_LIBS} ${OPENCV_MATCHER_REQUIRED_DEPS} )
