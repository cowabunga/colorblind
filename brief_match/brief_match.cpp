/*
 * Brief matcher
 *      Author: L
 */
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "brief_match.hpp"

#include <vector>
#include <iostream>
#include <algorithm>
#include <memory>

using namespace cv;

const double MaxMatchDistance = 0.11; // max distance is 1

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

/*
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
*/

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

// Clear matches for which NN ratio is > than threshold
// return the number of removed points
// (corresponding entries being cleared,
// i.e. size will be 0)
static int remove_ambiguous_matches(
    std::vector<std::vector<cv::DMatch> >& matches,
    double ratio)
{
  int removed = 0;

  for (std::vector<std::vector<cv::DMatch> >::iterator it = matches.begin();
      it != matches.end(); ++it) {
    if (it->size() > 1) {
      if ((*it)[1].distance < 1e-32 || (*it)[0].distance / (*it)[1].distance > ratio) {
        it->clear(); // remove match
        removed++;
      }
    } else { // does not have 2 neighbours
      //it->clear(); // remove match
      //removed++;
    }
  }
  return removed;
}

// Insert symmetrical matches in symMatches vector
static void get_symmetrical_matches(
    const std::vector<std::vector<cv::DMatch> >& matches1,
    const std::vector<std::vector<cv::DMatch> >& matches2,
    std::vector<cv::DMatch>& symMatches)
{
  // for all matches image 1 -> image 2
  for (std::vector<std::vector<cv::DMatch> >::
      const_iterator matchIterator1= matches1.begin();
      matchIterator1 != matches1.end(); ++matchIterator1) {
    // ignore deleted matches
    if (matchIterator1->size() < 1)
      continue;
    // for all matches image 2 -> image 1
    for (std::vector<std::vector<cv::DMatch> >::
        const_iterator matchIterator2= matches2.begin();
        matchIterator2!= matches2.end();
        ++matchIterator2) {
      // ignore deleted matches
      if (matchIterator2->size() < 1)
        continue;
      // Match symmetry test
      if ((*matchIterator1)[0].queryIdx ==
          (*matchIterator2)[0].trainIdx &&
          (*matchIterator2)[0].queryIdx ==
          (*matchIterator1)[0].trainIdx) {
        // add symmetrical match
        symMatches.push_back(
            cv::DMatch((*matchIterator1)[0].queryIdx,
              (*matchIterator1)[0].trainIdx,
              (*matchIterator1)[0].distance));
        break; // next match in image 1 -> image 2
      }
    }
  }
}

static void detect_features(
    const Mat& im1,
    const Mat& im2,
    const char* im1_name,
    const char* im2_name,
    TimeMeter& time_meter,
    std::vector<KeyPoint>& kpts_1,
    std::vector<KeyPoint>& kpts_2 )
{
  time_meter.start();
  FastFeatureDetector detector1(50);
  //SiftFeatureDetector detector1;
  detector1.detect(im1, kpts_1);
  detector1.detect(im2, kpts_2);
  time_meter.stop("Feature detection");

  std::cout << "found " << kpts_1.size() << " keypoints in " << im1_name << std::endl
            << "found " << kpts_2.size() << " keypoints in " << im2_name << std::endl << std::endl;
  /*
  time_meter.start();
  std::vector<KeyPoint> kpts_21, kpts_22;
  SimpleBlobDetector detector2;
  detector2.detect(im1, kpts_21);
  detector2.detect(im2, kpts_22);
  time_meter.stop("simple blob detection");

  std::cout << "found " << kpts_21.size() << " keypoints in " << im1_name << std::endl
            << "found " << kpts_22.size() << " keypoints in " << im2_name << std::endl << std::endl;

  kpts_1.insert(kpts_1.end(), kpts_21.begin(), kpts_21.end());
  kpts_2.insert(kpts_2.end(), kpts_22.begin(), kpts_22.end());

  std::cout << "found " << kpts_1.size() << " keypoints total in " << im1_name << std::endl
            << "found " << kpts_2.size() << " keypoints total in " << im2_name << std::endl << std::endl;
  */
}

static void match_features(
  const Mat& desc_1,
  const Mat& desc_2,
  TimeMeter& time_meter,
  std::vector<DMatch>& matches)
{
  //Do matching using features2d
  time_meter.start();
  BFMatcher matcher(NORM_HAMMING);
  //FlannBasedMatcher matcher;

  std::vector<std::vector<DMatch> > matches12;
  std::vector<std::vector<DMatch> > matches21;
  matcher.knnMatch(desc_1, desc_2, matches12, 2);
  matcher.knnMatch(desc_2, desc_1, matches21, 2);
  std::cout << matches12.size() << " matches from 1st to 2nd image found" << std::endl;
  std::cout << matches21.size() << " matches from 2nd to 1st image found" << std::endl;
  time_meter.stop("matching");

  time_meter.start();
  int removed1 = remove_ambiguous_matches(matches12, 0.8);
  int removed2 = remove_ambiguous_matches(matches21, 0.8);
  std::cout << removed1 << " ambiguous matches removed" << std::endl;
  std::cout << removed2 << " ambiguous matches removed" << std::endl;
  time_meter.stop("removing ambiguous");

  //time_meter.start();
  //get_symmetrical_matches(matches21, matches12, matches);
  //time_meter.stop("getting symmetrical matches");
  for (std::vector<std::vector<cv::DMatch> >::const_iterator it = matches21.begin();
      it!= matches21.end(); ++it)
    if (it->size() > 1)
      matches.push_back((*it)[0]);

  //matcher.match(desc_2, desc_1, matches);
  //time_meter.stop("matching key points");
  std::cout << matches.size() << " matches found." << std::endl;
}

// Identify good matches using RANSAC
// Return fundemental matrix
static cv::Mat find_fundamental_transform(
    const std::vector<cv::DMatch>& matches,
    const std::vector<cv::KeyPoint>& keypoints1,
    const std::vector<cv::KeyPoint>& keypoints2,
    double distance,
    double confidence,
    bool refineF,
    std::vector<cv::DMatch>& outMatches)
{
  // Convert keypoints into Point2f
  std::vector<cv::Point2f> points1, points2;
  for (std::vector<cv::DMatch>:: const_iterator it= matches.begin();
      it!= matches.end(); ++it) {
    // Get the position of left keypoints
    float x = keypoints1[it->queryIdx].pt.x;
    float y = keypoints1[it->queryIdx].pt.y;
    points1.push_back(cv::Point2f(x,y));
    // Get the position of right keypoints
    x = keypoints2[it->trainIdx].pt.x;
    y = keypoints2[it->trainIdx].pt.y;
    points2.push_back(cv::Point2f(x,y));
  }
  // TODO use the following func instead:
  //matches2points(matches, keypoints1, keypoints2, points1, points2);

  // Compute F matrix using RANSAC
  std::vector<uchar> inliers(points1.size(),0);
  cv::Mat fundemental = cv::findFundamentalMat(
      cv::Mat(points1), cv::Mat(points2), // matching points
      inliers,
      // match status (inlier or outlier)
      CV_FM_RANSAC, // RANSAC method
      distance, // distance to epipolar line
      confidence); // confidence probability

  // extract the surviving (inliers) matches
  std::vector<uchar>::const_iterator itIn = inliers.begin();
  std::vector<cv::DMatch>::const_iterator itM = matches.begin();
  for ( ; itIn != inliers.end(); ++itIn, ++itM)
    if (*itIn) // it is a valid match
      outMatches.push_back(*itM);

  if (refineF) {
    // The F matrix will be recomputed with
    // all accepted matches
    // Convert keypoints into Point2f
    // for final F computation
    points1.clear();
    points2.clear();
    // TODO use matches2points()
    for (std::vector<cv::DMatch>:: const_iterator it = outMatches.begin();
        it != outMatches.end(); ++it) {
      // Get the position of left keypoints 
      float x = keypoints1[it->queryIdx].pt.x;
      float y = keypoints1[it->queryIdx].pt.y;
      points1.push_back(cv::Point2f(x,y));
      // Get the position of right keypoints
      x = keypoints2[it->trainIdx].pt.x;
      y = keypoints2[it->trainIdx].pt.y;
      points2.push_back(cv::Point2f(x,y));
    }
    // Compute 8-point F from all accepted matches
    fundemental = cv::findFundamentalMat(
        cv::Mat(points1), cv::Mat(points2), // matches
        CV_FM_8POINT); // 8-point method
  }
  return fundemental;
}

static void draw_epipolar_lines(const Mat& im1,
    const std::vector<Point2f>& mpts_1,
    const std::vector<Point2f>& mpts_2,
    const Mat& fundamental2to1,
    Mat& draw)
{
  RNG& rng = theRNG();

  cvtColor( im1, draw, CV_GRAY2BGR );

  Mat norm1, pnt2;
  Point2f pnt1;
  std::vector<float> pnt2_h(3);
  for (int i = 0; i < mpts_2.size(); ++i) {
    pnt2_h[0] = mpts_2[i].x;
    pnt2_h[1] = mpts_2[i].y;
    pnt2_h[2] = 1.0;
    Mat(pnt2_h).convertTo(pnt2, fundamental2to1.type());
    norm1 = fundamental2to1 * pnt2;
    pnt1.x = mpts_1[i].x + norm1.at<float>(1,0);
    pnt1.y = mpts_1[i].y - norm1.at<float>(0,0);

    Scalar color = Scalar( rng(256), rng(256), rng(256) );
    cv::line(draw, mpts_1[i], pnt1, color, 1, CV_AA, 0);
    cv::circle(draw, mpts_1[i], 3, color, 1, CV_AA, 0);
  }
}

int main(int argc, const char ** argv)
{
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " image1 image2" << std::endl;
    exit(3);
  }
  const char* im1_name = argv[1];
  const char* im2_name = argv[2];

  int descSize = 32;
  double maxDistance = descSize*8;
  TimeMeter time_meter(false, true);

  Mat im1 = imread(im1_name, CV_LOAD_IMAGE_GRAYSCALE);
  Mat im2 = imread(im2_name, CV_LOAD_IMAGE_GRAYSCALE);

  if (im1.empty() || im2.empty())
  {
    std::cout << "could not load image: "
              << (im1.empty()?im1_name:im2_name) << std::endl;
    return 1;
  }

  // Increase contrast
  time_meter.start();
  Mat im1_eq_hist, im2_eq_hist;
  equalizeHist(im1, im1_eq_hist);
  equalizeHist(im2, im2_eq_hist);
  im1 = im1_eq_hist;
  im2 = im2_eq_hist;
  time_meter.stop("Increasing contrast");

  std::vector<KeyPoint> kpts_1, kpts_2;
  detect_features(im1, im2, im1_name, im2_name, time_meter, kpts_1, kpts_2); 

  time_meter.start();
  Mat desc_1, desc_2;
  BriefDescriptorExtractor desc_extractor(descSize);
  //SurfDescriptorExtractor desc_extractor(50.0);
  desc_extractor.compute(im1, kpts_1, desc_1);
  desc_extractor.compute(im2, kpts_2, desc_2);
  time_meter.stop("Descriptors computation");

  std::vector<DMatch> matches;
  match_features(desc_1, desc_2, time_meter, matches);

  // Compute and print distance historgam
  //compute_distance_historgam(matches, 20);

  // Plot all matches histogram
  plot_hist_image(matches, 100, 5);

  // Get only relatively good matches
  time_meter.start();
  std::vector<DMatch> good_matches;
  filter_matches(matches, MaxMatchDistance*maxDistance, good_matches);
  time_meter.stop("Good matches filtering");
  std::cout << good_matches.size() << " good matches found." << std::endl;

  time_meter.start();
  std::vector<DMatch> accepted_matches;
  Mat fundamental = find_fundamental_transform(good_matches, kpts_1, kpts_2,
                                       20.0, 0.8, true,
                                       accepted_matches);
  time_meter.stop("fundamental transform search");
  std::cout << accepted_matches.size() << " matches accepted" << std::endl;

  if (fundamental.total() > 1) {
    Mat outimg;
    drawMatches(im2, kpts_2, im1, kpts_1, accepted_matches, outimg,
                Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imshow("accepted matches", outimg);

    Mat draw;
    std::vector<Point2f> mpts_1, mpts_2;
    matches2points(accepted_matches, kpts_1, kpts_2, mpts_1, mpts_2);
    draw_epipolar_lines(im1, mpts_1, mpts_2, fundamental, draw);
    imshow("epipolar lines", draw);

    waitKey();
  }
  //good_matches = accepted_matches;

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
    std::vector<DMatch> top_matches(
        good_matches.begin() + i,
        good_matches.begin() + min<int>(i + chunk_size, good_matches.size()));
    Mat outimg;
    drawMatches(im2, kpts_2, im1, kpts_1, top_matches, outimg,
                Scalar::all(-1), Scalar::all(-1));
    imshow("matches outliers removed", outimg);
    waitKey();

    if (top_matches.size() > 3) {
      time_meter.start();
      std::vector<Point2f> mpts_1, mpts_2;
      matches2points(top_matches, kpts_1, kpts_2, mpts_1, mpts_2);
      Mat homographyTrf = findHomography(mpts_1, mpts_2, RANSAC, 1);
      time_meter.stop("homography search");

      if (homographyTrf.total() < 2)
        std::cout << "could not find a homography" << std::endl;
      else {
        Mat warped, diff;
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
