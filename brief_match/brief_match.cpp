/*
 * matching_test.cpp
 *
 *  Created on: Oct 17, 2010
 *      Author: ethan
 */
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include <iostream>
#include <algorithm>
#include <memory>

using namespace cv;
using namespace std;

//Copy (x,y) location of descriptor matches found from KeyPoint data structures into Point2f vectors
static void matches2points(const vector<DMatch>& matches, const vector<KeyPoint>& kpts_train,
                    const vector<KeyPoint>& kpts_query, vector<Point2f>& pts_train, vector<Point2f>& pts_query)
{
  pts_train.clear();
  pts_query.clear();
  pts_train.reserve(matches.size());
  pts_query.reserve(matches.size());
  for (size_t i = 0; i < matches.size(); i++)
  {
    const DMatch& match = matches[i];
    pts_query.push_back(kpts_query[match.queryIdx].pt);
    pts_train.push_back(kpts_train[match.trainIdx].pt);
  }

}

static double match(const vector<KeyPoint>& /*kpts_train*/, const vector<KeyPoint>& /*kpts_query*/, DescriptorMatcher& matcher,
            const Mat& train, const Mat& query, vector<DMatch>& matches)
{

  double t = (double)getTickCount();
  matcher.match(query, train, matches); //Using features2d
  return ((double)getTickCount() - t) / getTickFrequency();
}

static void compute_distance_historgam(const vector<DMatch>& matches, const int histogramSize)
{
  if (matches.size() == 0)
      return;
  double minDist = matches.front().distance;
  double maxDist = matches.back().distance;
  double distStep = (maxDist - minDist)/histogramSize;

  cout << "Distance histogram:" << endl;
  cout << "Min distance: " << minDist << endl;
  cout << "Max distance: " << maxDist << endl;
  cout << "Histogram distance step: " << distStep << endl;

  if (distStep <= 0)
      return;
  vector<int> counts(histogramSize, 0);
  for (vector<DMatch>::const_iterator it = matches.begin(); it < matches.end(); ++it)
      counts[min(int((it->distance - minDist)/distStep), histogramSize-1)] += 1;
  for (int i=0; i < histogramSize; ++i)
      cout << minDist + (i+0.5)*distStep << ": "<< counts[i] << endl;
  cout << endl;
}

int main(int argc, const char ** argv)
{
  if (argc < 3) {
      std::cerr << "Usage: " << argv[0] << " image1 image2" << std::endl;
      exit(3);
  }
  string im1_name = argv[1];
  string im2_name = argv[2];

  int descSize = 32;
  BriefDescriptorExtractor desc_extractor(descSize);
  double maxDistance = descSize*8;

  Mat im1 = imread(im1_name, CV_LOAD_IMAGE_GRAYSCALE);
  Mat im2 = imread(im2_name, CV_LOAD_IMAGE_GRAYSCALE);

  if (im1.empty() || im2.empty())
  {
    std::cout << "could not load image: " << (im1.empty()?im1_name:im2_name) << std::endl;
    return 1;
  }

  vector<KeyPoint> kpts_1, kpts_2;
  double time_step1 = (double)getTickCount();
  FastFeatureDetector detector1(50);
  detector1.detect(im1, kpts_1);
  detector1.detect(im2, kpts_2);

  time_step1 = ((double)getTickCount() - time_step1) / getTickFrequency();
  cout << "corners:" << endl
       << "found " << kpts_1.size() << " keypoints in " << im1_name << endl
       << "found " << kpts_2.size() << " keypoints in " << im2_name << endl
       << "took " << time_step1 << " seconds." << endl << endl;

  vector<KeyPoint> kpts_21, kpts_22;
  /*double t2 = (double)getTickCount();
  SimpleBlobDetector detector2;
  detector2.detect(im1, kpts_21);
  detector2.detect(im2, kpts_22);

  t2 = ((double)getTickCount() - t2) / getTickFrequency();
  cout << "BLOBS" << endl
       << "found " << kpts_21.size() << " keypoints in " << im1_name << endl
       << "found " << kpts_22.size() << " keypoints in " << im2_name << endl
       << "took " << t2 << " seconds." << endl << endl;

  kpts_1.insert(kpts_1.end(), kpts_21.begin(), kpts_21.end());
  kpts_2.insert(kpts_2.end(), kpts_22.begin(), kpts_22.end());

  cout << "TOTAL" << endl << "found " << kpts_1.size() << " keypoints in " << im1_name << endl
       << "found " << kpts_2.size() << " keypoints in " << im2_name << endl
       << "took " << time_step1 + t2 << " seconds." << endl << endl;*/

  Mat desc_1, desc_2;
  cout << "computing descriptors...";
  double time_step2 = (double)getTickCount();

  desc_extractor.compute(im1, kpts_1, desc_1);
  desc_extractor.compute(im2, kpts_2, desc_2);

  time_step2 = ((double)getTickCount() - time_step2) / getTickFrequency();
  cout << " took " << time_step2 << " seconds" << endl << endl;

  //Do matching using features2d
  cout << "matching with BruteForceMatcher<Hamming>...";
  BFMatcher matcher(NORM_HAMMING);
  vector<DMatch> matches;
  double time_step3 = match(kpts_1, kpts_2, matcher, desc_1, desc_2, matches);
  cout << " took " << time_step3 << " seconds" << endl << endl;

  std::vector<DMatch> good_matches;
  for (vector<DMatch>::const_iterator it = matches.begin(); it < matches.end(); ++it)
      if (it->distance < 0.07*maxDistance)
          good_matches.push_back(*it);

  cout << "All steps took " << time_step1 + time_step2 + time_step3 << " seconds" << endl << endl;

  std::sort(good_matches.begin(), good_matches.end());

  // Compute and print distance historgam
  compute_distance_historgam(good_matches, 50);

  //vector<DMatch> top100_matches(good_matches.begin(), good_matches.begin() + min<int>(100,good_matches.size()));

  vector<char> outlier_mask;
  size_t chunk_size = 300;
  for( size_t i = 0; i < good_matches.size() ; i += chunk_size ) {
      vector<DMatch> top_matches(good_matches.begin() + i, good_matches.begin() + min<int>(i + chunk_size, good_matches.size()));
      Mat outimg;
      drawMatches(im2, kpts_2, im1, kpts_1, top_matches, outimg, Scalar::all(-1), Scalar::all(-1));
      imshow("matches outliers removed", outimg);
      waitKey();

      if (top_matches.size() > 3) {
          vector<Point2f> mpts_1, mpts_2;
          matches2points(top_matches, kpts_1, kpts_2, mpts_1, mpts_2);
          Mat H = findHomography(mpts_2, mpts_1, RANSAC, 1);
          if (H.total() < 2)
              cout << "could not find a homography" << endl;
          else {
              Mat warped;
              Mat diff;
              warpPerspective(im2, warped, H, im1.size());
              imshow("warped", warped);
              absdiff(im1,warped,diff);
              imshow("diff", diff);
              waitKey();
          }
      }
      break;
  }
  
  return 0;
}
