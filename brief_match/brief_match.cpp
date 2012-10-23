/*
 * Brief matcher
 *      Author: L
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
  vector<DMatch>::const_iterator iMin = std::min_element(matches.begin(), matches.end());
  vector<DMatch>::const_iterator iMax = std::max_element(matches.begin(), matches.end());
  if (iMin == matches.end() || iMax == matches.end())
      return;
  double minDist = iMin->distance;
  double maxDist = iMax->distance;
  double distStep = std::max(maxDist - minDist, 1e-30)/histogramSize;

  std::cout << "Distance histogram:" << std::endl;
  std::cout << "Min distance: " << minDist << std::endl;
  std::cout << "Max distance: " << maxDist << std::endl;
  std::cout << "Histogram distance step: " << distStep << std::endl;

  vector<int> counts(histogramSize, 0);
  for (vector<DMatch>::const_iterator it = matches.begin(); it < matches.end(); ++it)
      counts[min(int((it->distance - minDist)/distStep), histogramSize-1)] += 1;
  for (int i=0; i < histogramSize; ++i)
      std::cout << minDist + (i+0.5)*distStep << ": "<< counts[i] << std::endl;
  std::cout << std::endl;
}

static void filter_matches(const std::vector<DMatch>& matches, double max_distance,
                           std::vector<DMatch>& filtered)
{
  filtered.clear();
  for (vector<DMatch>::const_iterator it = matches.begin(); it < matches.end(); ++it)
      if (it->distance < max_distance)
          filtered.push_back(*it);
}

// see perspectiveTransform(pnts1, );
static void apply_perspective_trf(const Mat& mat, const Point2f& in, Point2f& out)
{
    double scale = mat.at<float>(2,0)*in.x + mat.at<float>(2,1)*in.y + mat.at<float>(2,2);
    if (fabs(scale) < 1e-32)
        out.x = out.y = 0.0;
    else {
        out.x = (mat.at<float>(0,0)*in.x + mat.at<float>(0,1)*in.y + mat.at<float>(0,2))/scale;
        out.y = (mat.at<float>(1,0)*in.x + mat.at<float>(1,1)*in.y + mat.at<float>(1,2))/scale;
    }
}

static double calc_trf_error(const vector<Point2f>& pnts1, const vector<Point2f>& pnts2, const Mat& trf )
{
    size_t size = pnts1.size();
    Point2f p;
    double error = 0;
    float dx, dy;
    Mat iTrf;
    cv::invert(trf, iTrf);
    for (size_t i = 0; i < size; ++i) {
        apply_perspective_trf(trf, pnts1[i], p);
        dx = pnts2[i].x - p.x;
        dy = pnts2[i].y - p.y;
        error += log(1.0+sqrt(dx*dx + dy*dy));

        apply_perspective_trf(iTrf, pnts2[i], p);
        dx = pnts1[i].x - p.x;
        dy = pnts1[i].y - p.y;
        error += log(1.0+sqrt(dx*dx + dy*dy));
    }
    return error;
}

static bool find_perspective_transform(const vector<Point2f>& pnts1, const vector<Point2f>& pnts2, Mat& best_trf )
{
    size_t size = pnts1.size();
    assert(size == pnts2.size());
    if (size > 30)
        std::cout << "WARNING: too many points to find perspective transform with brute force: " << size << std::endl;

    double err, min_err = -1;
    Mat trf;

    Point2f from[4], to[4];
    size_t i, j, k, l;
    for (i = 0; i < size; ++i) {
        from[0] = pnts1[i];
        to[0] = pnts2[i];
        for (j = i+1; j < size; ++j) {
            from[1] = pnts1[j];
            to[1] = pnts2[j];
            for (k = j+1; k < size; ++k) {
                from[2] = pnts1[k];
                to[2] = pnts2[k];
                for (l = k+1; l < size; ++l) {
                    from[3] = pnts1[l];
                    to[3] = pnts2[l];
                    trf = cv::getPerspectiveTransform(from, to);
                    err = calc_trf_error(pnts1, pnts2, trf);
                    std::cout << "err = " << err << std::endl;
                    if (min_err < 0 || err < min_err) {
                        min_err = err;
                        best_trf = trf;
                    }
                }
            }
        }
    }
    if (min_err < 0)
        return false;
    std::cout << "min error: " << min_err << std::endl;
    return true;
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
  std::cout << "corners:" << std::endl
       << "found " << kpts_1.size() << " keypoints in " << im1_name << std::endl
       << "found " << kpts_2.size() << " keypoints in " << im2_name << std::endl
       << "took " << time_step1 << " seconds." << std::endl << std::endl;

  vector<KeyPoint> kpts_21, kpts_22;
  /*double t2 = (double)getTickCount();
  SimpleBlobDetector detector2;
  detector2.detect(im1, kpts_21);
  detector2.detect(im2, kpts_22);

  t2 = ((double)getTickCount() - t2) / getTickFrequency();
  std::cout << "BLOBS" << std::endl
       << "found " << kpts_21.size() << " keypoints in " << im1_name << std::endl
       << "found " << kpts_22.size() << " keypoints in " << im2_name << std::endl
       << "took " << t2 << " seconds." << std::endl << std::endl;

  kpts_1.insert(kpts_1.end(), kpts_21.begin(), kpts_21.end());
  kpts_2.insert(kpts_2.end(), kpts_22.begin(), kpts_22.end());

  std::cout << "TOTAL" << std::endl << "found " << kpts_1.size() << " keypoints in " << im1_name << std::endl
       << "found " << kpts_2.size() << " keypoints in " << im2_name << std::endl
       << "took " << time_step1 + t2 << " seconds." << std::endl << std::endl;*/

  Mat desc_1, desc_2;
  std::cout << "computing descriptors...";
  double time_step2 = (double)getTickCount();

  desc_extractor.compute(im1, kpts_1, desc_1);
  desc_extractor.compute(im2, kpts_2, desc_2);

  time_step2 = ((double)getTickCount() - time_step2) / getTickFrequency();
  std::cout << " took " << time_step2 << " seconds" << std::endl << std::endl;

  //Do matching using features2d
  std::cout << "matching with BruteForceMatcher<Hamming>...";
  BFMatcher matcher(NORM_HAMMING);
  vector<DMatch> matches;
  double time_step3 = match(kpts_1, kpts_2, matcher, desc_1, desc_2, matches);
  std::cout << " took " << time_step3 << " seconds" << std::endl << std::endl;

  std::cout << "All steps took " << time_step1 + time_step2 + time_step3 << " seconds" << std::endl << std::endl;

  // Compute and print distance historgam
  compute_distance_historgam(matches, 20);

  // Get only relatively good matches
  std::vector<DMatch> good_matches;
  filter_matches(matches, 0.10*maxDistance, good_matches);

  std::sort(good_matches.begin(), good_matches.end());
  vector<DMatch> top10_matches(good_matches.begin(), good_matches.begin() + min<int>(30,good_matches.size()));

  vector<Point2f> mpts_1, mpts_2;
  matches2points(top10_matches, kpts_1, kpts_2, mpts_1, mpts_2);
  Mat trf;
  if( find_perspective_transform(mpts_1, mpts_2, trf) ) {
      Mat warped;
      Mat diff;
      warpPerspective(im1, warped, trf, im2.size());
      imshow("perspective", warped);
      absdiff(im2,warped,diff);
      imshow("perspective_diff", diff);
      waitKey();
  }

  vector<char> outlier_mask;
  size_t chunk_size = 300;
  for( size_t i = 0; i < good_matches.size() ; i += chunk_size ) {
      vector<DMatch> top_matches(good_matches.begin() + i, good_matches.begin() + min<int>(i + chunk_size, good_matches.size()));
      Mat outimg;
      drawMatches(im2, kpts_2, im1, kpts_1, top_matches, outimg, Scalar::all(-1), Scalar::all(-1));
      imshow("matches outliers removed", outimg);
      waitKey();

      if (top_matches.size() > 3) {
          matches2points(top_matches, kpts_1, kpts_2, mpts_1, mpts_2);
          Mat homographyTrf = findHomography(mpts_1, mpts_2, RANSAC, 1);
          if (homographyTrf.total() < 2)
              std::cout << "could not find a homography" << std::endl;
          else {
              Mat warped;
              Mat diff;
              warpPerspective(im1, warped, homographyTrf, im2.size());
              imshow("homography", warped);
              absdiff(im2,warped,diff);
              imshow("homography_diff", diff);
              waitKey();
          }
      }
      break;
  }
  
  return 0;
}
