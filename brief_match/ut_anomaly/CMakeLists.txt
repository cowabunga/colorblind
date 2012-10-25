cmake_minimum_required(VERSION 2.8)

SET(OPENCV_MATCHER_REQUIRED_DEPS opencv_core opencv_highgui opencv_nonfree opencv_features2d opencv_flann opencv_imgproc opencv_photo opencv_video opencv_objdetect opencv_calib3d opencv_ml opencv_contrib)

ocv_check_dependencies(${OPENCV_MATCHER_REQUIRED_DEPS})
ocv_include_modules(${OPENCV_MATCHER_REQUIRED_DEPS})
ocv_include_directories(../include)

project( ut_anomaly )
add_executable( ut_anomaly main.cpp )
target_link_libraries(ut_anomaly ${OPENCV_LINKER_LIBS} ${OPENCV_MATCHER_REQUIRED_DEPS} )
