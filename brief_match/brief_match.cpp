/*
 * Brief matcher
 *      Author: L
 */
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "brief_match.hpp"

#include <vector>
#include <iostream>
#include <algorithm>
#include <memory>

using namespace cv;

TimeMeter::TimeMeter( bool start_now, bool _verbos ):
    startTime(-1.0), totalTime(0.0), verbos(_verbos)
{
  if (start_now)
    start();
}

void TimeMeter::start()
{
  startTime = (double)getTickCount();
}

double TimeMeter::stop(const char* measure_name)
{
  double ms = 1000*((double)getTickCount() - startTime) / getTickFrequency();
  totalTime += ms;
  if (verbos)
    std::cerr << measure_name << " took " << ms << " ms." << std::endl;
  return ms;
}

//Copy (x,y) location of descriptor matches found from KeyPoint data structures into Point2f vectors
static void matches2points(
    const std::vector<DMatch>& matches,
    const std::vector<KeyPoint>& kpts_train,
    const std::vector<KeyPoint>& kpts_query,
    std::vector<Point2f>& pts_train,
    std::vector<Point2f>& pts_query)
{
  pts_train.clear();
  pts_query.clear();
  pts_train.reserve(matches.size());
  pts_query.reserve(matches.size());
  for (size_t i = 0; i < matches.size(); ++i) {
    const DMatch& match = matches[i];
    pts_query.push_back(kpts_query[match.queryIdx].pt);
    pts_train.push_back(kpts_train[match.trainIdx].pt);
  }

}

static void plot_hist_image(std::vector<DMatch> data,
                            size_t columns,
                            size_t scale = 1)
{
  std::sort(data.begin(), data.end());

  int minValue = data.front().distance;
  int maxValue = data.back().distance;

  std::vector<int> histData(columns, 0);

  int histDataMax = 0;
  for (std::vector<DMatch>::const_iterator it = data.begin()
      ; it != data.end(); ++it) {
    size_t rangeNum =
      (it->distance - minValue) * (columns - 1) / (maxValue - minValue);

    ++histData[rangeNum];
    if (histData[rangeNum] > histDataMax) {
      histDataMax = histData[rangeNum];
    }
  }

  cv::Mat histImg =
    cv::Mat::zeros(columns * scale, columns * scale, CV_8UC1);

  for (size_t i = 0; i < columns; ++i) {
    int columnSize = columns * scale * histData[i] / histDataMax;

    cv::Point leftBottom(i * scale, histImg.size().height);
    cv::Point rightTop((i + 1) * scale, histImg.size().height - columnSize);

    cv::rectangle(histImg, leftBottom, rightTop,
                  cv::Scalar::all(255), CV_FILLED);
  }

  cv::namedWindow("Histogram of all matches", 1);
  cv::imshow("Histogram of all matches", histImg);
}

static void compute_distance_historgam(
    const std::vector<DMatch>& matches,
    const int histogramSize)
{
  std::vector<DMatch>::const_iterator iMin =
      std::min_element(matches.begin(), matches.end());
  std::vector<DMatch>::const_iterator iMax =
      std::max_element(matches.begin(), matches.end());
  if (iMin == matches.end() || iMax == matches.end())
    return;
  double minDist = iMin->distance;
  double maxDist = iMax->distance;
  double distStep = std::max(maxDist - minDist, 1e-30)/histogramSize;

  std::cout << "Distance histogram:" << std::endl;
  std::cout << "Min distance: " << minDist << std::endl;
  std::cout << "Max distance: " << maxDist << std::endl;
  std::cout << "Histogram distance step: " << distStep << std::endl;

  std::vector<int> counts(histogramSize, 0);
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

static void apply_perspective_trf(const Mat& trf, const Point2f& in, Point2d& out)
{
  double scale = trf.at<double>(2,0)*in.x +
                 trf.at<double>(2,1)*in.y +
                 trf.at<double>(2,2);
  if (scale > 1e-32) {
    out.x = (trf.at<double>(0,0)*in.x +
             trf.at<double>(0,1)*in.y +
             trf.at<double>(0,2))/scale;
    out.y = (trf.at<double>(1,0)*in.x +
             trf.at<double>(1,1)*in.y +
             trf.at<double>(1,2)) / scale;
  }
  else
    out.x = out.y = 0;
}

static double calc_perspective_trf_error( const std::vector<Point2f>& pnts1,
    const std::vector<Point2f>& pnts2, Mat& trf )
{
  size_t size = pnts1.size();
  if (size == 0)
    return 0.0;
  Point2d pnt;
  double err = 0.0, variance = 0.0;
  double dx, dy;
  std::vector<double> errors(size);
  for (size_t i = 0; i < size; ++i) {
    apply_perspective_trf(trf, pnts1[i], pnt);
    dx = pnt.x-pnts2[i].x;
    dy = pnt.y-pnts2[i].y;
    err = sqrt(dx*dx + dy*dy);
    variance += err*err;
    errors[i] = err;
  }
  variance /= size;
  double sigma = sqrt(variance);

  std::sort(errors.begin(), errors.end());

  double total_err = 0.0;
  for (std::vector<double>::iterator it = errors.begin();
       it != errors.end() && *it <= sigma; ++it)
    total_err += *it;
  return total_err;
}

static bool find_perspective_transform(const std::vector<Point2f>& pnts1,
    const std::vector<Point2f>& pnts2, Mat& best_trf )
{
  size_t size = pnts1.size();
  assert(size == pnts2.size());
  if (size > 30)
    std::cout << "WARNING: too many points to find perspective transform with brute force: "
              << size << std::endl;

  double min_err = -1, err;
  Mat trf, iTrf;

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
          if (trf.total() == 9) {
            invert(trf, iTrf);
            err = calc_perspective_trf_error(pnts1, pnts2, trf) + 
              calc_perspective_trf_error(pnts2, pnts1, iTrf);
            if (min_err < 0 || err < min_err) {
              min_err = err;
              best_trf = trf;
              //std::cout << "Min perspective error: " << err << std::endl;
            }
          }

        }
      }
    }
  }
  if (min_err < 0)
    return false;
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
  TimeMeter time_meter(false, true);

  Mat im1 = imread(im1_name, CV_LOAD_IMAGE_GRAYSCALE);
  Mat im2 = imread(im2_name, CV_LOAD_IMAGE_GRAYSCALE);

  if (im1.empty() || im2.empty())
  {
    std::cout << "could not load image: " << (im1.empty()?im1_name:im2_name)
              << std::endl;
    return 1;
  }

  time_meter.start();
  std::vector<KeyPoint> kpts_1, kpts_2;
  FastFeatureDetector detector1(50);
  detector1.detect(im1, kpts_1);
  detector1.detect(im2, kpts_2);
  time_meter.stop("Feature detection");

  std::cout << "found " << kpts_1.size() << " keypoints in " << im1_name << std::endl
            << "found " << kpts_2.size() << " keypoints in " << im2_name << std::endl << std::endl;

  /*time_meter.start();
    std::vector<KeyPoint> kpts_21, kpts_22;
    SimpleBlobDetector detector2;
    detector2.detect(im1, kpts_21);
    detector2.detect(im2, kpts_22);

    time_meter.stop("simple blob detection");
    std::cout << "found " << kpts_21.size() << " keypoints in " << im1_name << std::endl
              << "found " << kpts_22.size() << " keypoints in " << im2_name << std::endl << std::endl;

    kpts_1.insert(kpts_1.end(), kpts_21.begin(), kpts_21.end());
    kpts_2.insert(kpts_2.end(), kpts_22.begin(), kpts_22.end());

    std::cout << "TOTAL" << std::endl
              << "found " << kpts_1.size() << " keypoints in " << im1_name << std::endl
              << "found " << kpts_2.size() << " keypoints in " << im2_name << std::endl; */

  time_meter.start();
  Mat desc_1, desc_2;
  desc_extractor.compute(im1, kpts_1, desc_1);
  desc_extractor.compute(im2, kpts_2, desc_2);
  time_meter.stop("Descriptors computation");

  //Do matching using features2d
  time_meter.start();
  BFMatcher matcher(NORM_HAMMING);
  std::vector<DMatch> matches;
  matcher.match(desc_2, desc_1, matches);
  time_meter.stop("matching with BruteForceMatcher<Hamming>");

  // Compute and print distance historgam
  //compute_distance_historgam(matches, 20);

  // Plot all matches histogram
  plot_hist_image(matches, 100, 5);

  // Get only relatively good matches
  time_meter.start();
  std::vector<DMatch> good_matches;
  filter_matches(matches, 0.1*maxDistance, good_matches);
  time_meter.stop("Good matches filtering");

  time_meter.start();
  std::sort(good_matches.begin(), good_matches.end());
  time_meter.stop("Matches sort");

  time_meter.start();
  std::vector<DMatch> top_matches(
      good_matches.begin(),
      good_matches.begin() + min<int>(30,good_matches.size()));
  time_meter.stop("getting top matches");

  time_meter.start();
  std::vector<Point2f> mpts_1, mpts_2;
  matches2points(top_matches, kpts_1, kpts_2, mpts_1, mpts_2);
  time_meter.stop("matches to points conversion");

  time_meter.start();
  Mat trf;
  bool found = find_perspective_transform(mpts_1, mpts_2, trf);
  time_meter.stop("perspective transform search");

  if (found) {
    Mat warped;
    Mat diff;
    warpPerspective(im1, warped, trf, im2.size());
    imshow("perspective", warped);
    absdiff(im2,warped,diff);
    imshow("perspective diff", diff);
    waitKey();
  }

  size_t chunk_size = 300;
  for( size_t i = 0; i < good_matches.size() ; i += chunk_size ) {
    top_matches.assign(
        good_matches.begin() + i,
        good_matches.begin() + min<int>(i + chunk_size, good_matches.size()));
    Mat outimg;
    drawMatches(im2, kpts_2, im1, kpts_1, top_matches, outimg,
                Scalar::all(-1), Scalar::all(-1));
    imshow("matches outliers removed", outimg);
    waitKey();

    if (top_matches.size() > 3) {
      time_meter.start();
      matches2points(top_matches, kpts_1, kpts_2, mpts_1, mpts_2);
      Mat homographyTrf = findHomography(mpts_1, mpts_2, RANSAC, 1);
      time_meter.stop("homography search");

      if (homographyTrf.total() < 2)
        std::cout << "could not find a homography" << std::endl;
      else {
        Mat warped;
        Mat diff;
        warpPerspective(im1, warped, homographyTrf, im2.size());
        imshow("homography", warped);
        absdiff(im2,warped,diff);
        imshow("homography diff", diff);
        waitKey();
      }
    }
    break;
  }

  std::cout << "All steps took " << time_meter.getTotalTime()
            << " ms" << std::endl << std::endl;

  return 0;
}
