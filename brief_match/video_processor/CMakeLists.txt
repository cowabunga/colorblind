cmake_minimum_required(VERSION 2.8)

SET(OPENCV_MATCHER_REQUIRED_DEPS opencv_core opencv_highgui opencv_nonfree opencv_features2d opencv_flann opencv_imgproc opencv_photo opencv_video opencv_objdetect opencv_calib3d opencv_ml opencv_contrib)

ocv_check_dependencies(${OPENCV_MATCHER_REQUIRED_DEPS})
ocv_include_modules(${OPENCV_MATCHER_REQUIRED_DEPS})
ocv_include_directories(../include)

project( video_processor )
add_executable( video_processor video_processor.cpp )
target_link_libraries(video_processor photo_match ${OPENCV_LINKER_LIBS} ${OPENCV_MATCHER_REQUIRED_DEPS} )
