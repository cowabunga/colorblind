/*
 * Brief matcher
 *      Author: L
 */

#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "photo_match.hpp"

int main(int argc, const char ** argv)
{
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " image1 textimage1 image2" << std::endl;
    exit(3);
  }

  const char* im1_name = argv[1];
  const char* im1text_name = argv[2];
  const char* im2_name = argv[3];

  cv::Mat im1 = cv::imread(im1_name);
  cv::Mat im1text = cv::imread(im1text_name);
  cv::Mat im2 = cv::imread(im2_name);

  if (im1.empty()) {
    std::cout << "could not load image: " << im1_name;
    return 1;
  } else if (im1text.empty()) {
    std::cout << "could not load image: " << im1text_name;
    return 1;
  } else if (im2.empty()) {
    std::cout << "could not load image: " << im2_name;
    return 1;
  }

  cv::Mat im2_with_text;
  bool matched = matchImagesAndPutLabel(im1, im1text, im2, im2_with_text, true);
  return matched ? 0 : 1;
}
