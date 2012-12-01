cmake_minimum_required(VERSION 2.8)

SET(OPENCV_MATCHER_REQUIRED_DEPS opencv_core opencv_highgui opencv_nonfree opencv_features2d opencv_flann opencv_imgproc opencv_photo opencv_video opencv_objdetect opencv_calib3d opencv_ml opencv_contrib)

ocv_check_dependencies(${OPENCV_MATCHER_REQUIRED_DEPS})
ocv_include_modules(${OPENCV_MATCHER_REQUIRED_DEPS})
ocv_include_directories(../include include)

project( brief_match )
add_executable( brief_match brief_match.cpp )
target_link_libraries(brief_match photo_match imgpathes ${OPENCV_LINKER_LIBS} ${OPENCV_MATCHER_REQUIRED_DEPS} )
