cmake_minimum_required(VERSION 2.8)

SET(OPENCV_MATCHER_REQUIRED_DEPS opencv_core opencv_highgui opencv_nonfree opencv_features2d opencv_flann opencv_imgproc opencv_photo opencv_video opencv_objdetect opencv_calib3d opencv_ml opencv_contrib)

ocv_check_dependencies(${OPENCV_MATCHER_REQUIRED_DEPS})
ocv_include_modules(${OPENCV_MATCHER_REQUIRED_DEPS})
ocv_include_directories(../include)

project( imdb )
add_library( imdb STATIC imdb.cpp )
