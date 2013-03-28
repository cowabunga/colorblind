/*
 * Colorblind core
 *      Author: L
 */
#include "photo_match.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"

#include <vector>
#include <iostream>
#include <algorithm>
#include <memory>

using namespace cv;

const double MaxMatchDistance = 0.11; // max distance is 1
const double MaxDistanceToEpipolarLine = 20.0;
const double FundamentalTrfSearchConfidence = 0.8;

VerbosTimer::VerbosTimer( bool start_now, bool _verbos ):
    startTime(-1.0), totalTime(0.0), verbos(_verbos)
{
  if (start_now)
    start();
}

void VerbosTimer::start()
{
  startTime = (double)getTickCount();
}

double VerbosTimer::stop(const char* measure_name)
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
    const std::vector<KeyPoint>& kpts_query,
    const std::vector<KeyPoint>& kpts_train,
    std::vector<Point2f>& pts_query,
    std::vector<Point2f>& pts_train)
{
  pts_train.clear();
  pts_query.clear();
  pts_train.reserve(matches.size());
  pts_query.reserve(matches.size());
  for (size_t i = 0; i < matches.size(); ++i) {
    const DMatch& match = matches[i];
    pts_train.push_back(kpts_train[match.trainIdx].pt);
    pts_query.push_back(kpts_query[match.queryIdx].pt);
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
  if (size < 4)
    return false;

  double min_err = -1, err;
  Mat trf, iTrf;
  Point2f from[4], to[4];

  RandomSampleGenerator<4, true> rand_sample(0, size-1);
  AllSampleGenerator<int, 4> all_sample(0, size-1);
  SampleGenerator<int, 4>* sample = &rand_sample;
  int max_sample_num = 256;

  if (size < 10) {
    sample = &all_sample;
    max_sample_num = 100000;
  }

  for (int j, i = 0; i < max_sample_num && !sample->empty(); ++(*sample), ++i) {
    for (j = 0; j < 4; ++j) {
      from[j] = pnts1[(*sample)[j]];
      to[j] = pnts2[(*sample)[j]];
    }

    trf = cv::getPerspectiveTransform(from, to);
    if (trf.total() == 9) {
      invert(trf, iTrf);
      err = calc_perspective_trf_error(pnts1, pnts2, trf) + 
        calc_perspective_trf_error(pnts2, pnts1, iTrf);
      if (min_err < 0 || err < min_err) {
        min_err = err;
        best_trf = trf;
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
    VerbosTimer& timer,
    std::vector<KeyPoint>& kpts_1,
    std::vector<KeyPoint>& kpts_2 )
{
  timer.start();
  FastFeatureDetector detector1(50);
  //SiftFeatureDetector detector1;
  detector1.detect(im1, kpts_1);
  detector1.detect(im2, kpts_2);
  timer.stop("Feature detection");

  std::cout << "found " << kpts_1.size() << " keypoints in the first image" << std::endl
            << "found " << kpts_2.size() << " keypoints in the second image" << std::endl << std::endl;
  /*
  timer.start();
  std::vector<KeyPoint> kpts_21, kpts_22;
  SimpleBlobDetector detector2;
  detector2.detect(im1, kpts_21);
  detector2.detect(im2, kpts_22);
  timer.stop("simple blob detection");

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
  VerbosTimer& timer,
  std::vector<DMatch>& matches)
{
  //Do matching using features2d
  timer.start();
  BFMatcher matcher(NORM_HAMMING);
  //FlannBasedMatcher matcher;

  std::vector<std::vector<DMatch> > matches12;
  std::vector<std::vector<DMatch> > matches21;
  matcher.knnMatch(desc_1, desc_2, matches12, 2);
  matcher.knnMatch(desc_2, desc_1, matches21, 2);
  std::cout << matches12.size() << " matches from 1st to 2nd image found" << std::endl;
  std::cout << matches21.size() << " matches from 2nd to 1st image found" << std::endl;
  timer.stop("matching");

  timer.start();
  int removed1 = remove_ambiguous_matches(matches12, 0.8);
  int removed2 = remove_ambiguous_matches(matches21, 0.8);
  std::cout << removed1 << " ambiguous matches removed" << std::endl;
  std::cout << removed2 << " ambiguous matches removed" << std::endl;
  timer.stop("removing ambiguous");

  timer.start();
  get_symmetrical_matches(matches21, matches12, matches);
  timer.stop("getting symmetrical matches");

  for (std::vector<std::vector<cv::DMatch> >::const_iterator it = matches21.begin();
      it!= matches21.end(); ++it)
    if (it->size() > 1)
      matches.push_back((*it)[0]);

  //matcher.match(desc_2, desc_1, matches);
  //timer.stop("matching key points");
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
  cv::Mat fundemental;
  fundemental = Scalar(0);

  if (matches.size() < 7)
    return fundemental;

  // Convert keypoints into Point2f
  std::vector<cv::Point2f> points1, points2;
  matches2points(matches, keypoints1, keypoints2, points1, points2);

  // Compute F matrix using RANSAC
  std::vector<uchar> inliers(points1.size(),0);
  fundemental = cv::findFundamentalMat(
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
    matches2points(outMatches, keypoints1, keypoints2, points1, points2);
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
  for (size_t i = 0; i < mpts_2.size(); ++i) {
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

#if 0
static void overlay_image(const cv::Mat& src,
                          const cv::Mat& overlay,
                          cv::Mat& out)
{
  Size src_sz = src.size();
  Size over_sz = overlay.size();
  int row, col, i;
  double alpha;
  out = src;

/*
  const cv::CvMat m1 = src, m2 = overlay;
  cv::CVMat m3 = out;

  for( row = 0; row < m1.rows && row < m2.rows; ++row ) {
    for( col = 0; col < m1.cols && col < m2.cols; ++col ) {
      double v1 = cvmGet(m1, row, col);
      double v2 = cvmGet(m1, row, col);
      alpha = over.val[3]*1.0/255;
      for( i = 0; i < 3; ++i )
        merged.val[i] = int(alpha*source.val[i] + (1.0-alpha)*over.val[i]);
      merged.val[3] = source.val[3];
      // FIXME: not working :(
      mtfx.data.db[]
      out.at<CvScalar>(col, row) = merged;
    }
  }
  */

  for( int x = 0; x < over_sz.width && x < src_sz.width; ++x ) {
    for( int y = 0; y < over_sz.height && y < src_sz.height; ++y ) {
      CvScalar source = src.at<CvScalar>(y, x);
      CvScalar over = overlay.at<CvScalar>(y, x);
      CvScalar merged = source;
      double alpha1 = 0.01;//source.val[3];
      double alpha2 = 0.99;//over.val[3];

      if (*(unsigned long*)&over.val[3] < 10000000)
          continue;
      std::cout << "[" << *(unsigned long*)&over.val[0] << ", ";
      std::cout << *(unsigned long*)&over.val[1] << ", ";
      std::cout << *(unsigned long*)&over.val[2] << ", ";
      std::cout << *(unsigned long*)&over.val[3] << "]" << std::endl;

      for (unsigned long i = 0; i < 3; ++i)
          (*(unsigned long*)&merged.val[i]) = (*(unsigned long*)&source.val[i])*alpha1 + (*(unsigned long*)&over.val[i])*alpha2;
      merged.val[3] = 0;
      out.at<CvScalar>(y, x) = merged;
      //std::cout << over.val[3] << std::endl;
      //for( i = 0; i < 3; ++i )
        //merged.val[i] = int(alpha*source.val[i] + (1.0-alpha)*over.val[i]);
      //merged.val[3] = source.val[3];
      // FIXME: not working :(
      //mtfx.data.db[]
      //out.at<CvScalar>(y, x) = merged;
    }
  }
}
#endif

static void OverlayImage(cv::Mat& src, cv::Mat& overlay, cv::Mat& out)
{
  out = src;

  //cv::Mat mScalar(1, 1, overlay.type(), cvScalarAll(1000.0));
  //cv::multiply(overlay, mScalar, out, 1.0, overlay.type() );
  for(int i = 0; i < 3; ++i)
      cv::absdiff(out, overlay, out);
  cv::addWeighted(out, 1, overlay, 1, 0.0, out);
}

bool matchImagesAndPutLabel(
    const Mat& img1,
    const Mat& img1text,
    const Mat& img2,
    Mat& out,
    bool debug)
{
  bool images_matched = false;

  if (img1.empty() || img2.empty() || img1text.empty())
    return images_matched;

  cv::Mat im1 = img1, im1text = img1text, im2 = img2;
  cv::Mat im1_scaled, im1text_scaled, im2_scaled;

  double im1_scale = std::min(800.0/im1.size().width, 600.0/im1.size().height);
  cv::resize(im1, im1_scaled, Size(), im1_scale, im1_scale );
  im1 = im1_scaled;

  cv::resize(im1text, im1text_scaled, Size(), im1_scale, im1_scale );
  im1text = im1text_scaled;

  double im2_scale = std::min(800.0/im2.size().width, 600.0/im2.size().height);
  cv::resize(im2, im2_scaled, Size(), im2_scale, im2_scale );
  im2 = im2_scaled;

  if (img1.channels() != 3 || img1text.channels() != 3 || img2.channels() != 3)
    return images_matched;

  cv::cvtColor(im1, im1, CV_BGR2GRAY);
  cv::cvtColor(im1text, im1text, CV_BGR2GRAY);
  cv::cvtColor(im2, im2, CV_BGR2GRAY);

  int descSize = 32;
  double maxDistance = descSize*8;
  VerbosTimer timer(false, debug);

  // Increase contrast
  timer.start();
  Mat im1_eq_hist, im2_eq_hist;
  equalizeHist(im1, im1_eq_hist);
  equalizeHist(im2, im2_eq_hist);
  im1 = im1_eq_hist;
  im2 = im2_eq_hist;
  timer.stop("Increasing contrast");

  std::vector<KeyPoint> kpts_1, kpts_2;
  detect_features(im1, im2, timer, kpts_1, kpts_2); 

  if (kpts_1.empty() || kpts_2.empty())
    return false;

  timer.start();
  Mat desc_1, desc_2;
  BriefDescriptorExtractor desc_extractor(descSize);
  //SurfDescriptorExtractor desc_extractor(50.0);
  desc_extractor.compute(im1, kpts_1, desc_1);
  desc_extractor.compute(im2, kpts_2, desc_2);
  timer.stop("Descriptors computation");

  std::vector<DMatch> matches;
  match_features(desc_1, desc_2, timer, matches);

  if (debug) {
    Mat outimg;
    drawMatches(im2, kpts_2, im1, kpts_1, matches, outimg,
                Scalar::all(-1), Scalar::all(-1), std::vector<char>(),
                DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imshow("all matches", outimg);
    waitKey();
  }

  // Compute and print distance historgam
  //compute_distance_historgam(matches, 20);

  // Plot all matches histogram
  if (debug)
    plot_hist_image(matches, 100, 5);

  // Get only relatively good matches
  timer.start();
  std::vector<DMatch> good_matches;
  filter_matches(matches, MaxMatchDistance*maxDistance, good_matches);
  timer.stop("Good matches filtering");
  std::cout << good_matches.size() << " good matches found." << std::endl;

  timer.start();
  std::vector<DMatch> accepted_matches;
  Mat fundamental = find_fundamental_transform(good_matches, kpts_2, kpts_1,
                                       MaxDistanceToEpipolarLine,
                                       FundamentalTrfSearchConfidence,
                                       true, accepted_matches);
  timer.stop("fundamental transform search");

  if (fundamental.total() > 1 && accepted_matches.size() > 0) {
    if (debug) {
      Mat outimg;
      drawMatches(im2, kpts_2, im1, kpts_1, accepted_matches, outimg,
                  Scalar::all(-1), Scalar::all(-1), std::vector<char>(),
                  DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
      imshow("accepted matches", outimg);
      /*Mat draw;
      std::vector<Point2f> mpts_1, mpts_2;
      matches2points(accepted_matches, kpts_2, kpts_1, mpts_2, mpts_1);
      draw_epipolar_lines(im1, mpts_1, mpts_2, fundamental, draw);
      imshow("epipolar lines", draw);
      waitKey();*/
    }
    good_matches = accepted_matches;
    std::cout << accepted_matches.size() << " matches accepted" << std::endl;
  }

  timer.start();
  std::sort(good_matches.begin(), good_matches.end());
  timer.stop("Matches sort");

  timer.start( );
  std::vector<Point2f> mpts_1, mpts_2;
  matches2points(good_matches, kpts_2, kpts_1, mpts_2, mpts_1);
  timer.stop("matches to points conversion");

  timer.start();
  Mat perspective_trf;
  bool perspective_trf_found = find_perspective_transform(mpts_1, mpts_2, perspective_trf);
  timer.stop("perspective transform search");

  Mat homography_trf;
  bool homography_trf_found = false;
  if (good_matches.size() > 3) {
    timer.start();
    homography_trf = findHomography(mpts_1, mpts_2, RANSAC, 1);
    homography_trf_found = homography_trf.total() >= 9;
    timer.stop("homography search");
  }

  Mat trf;
  bool trf_found = perspective_trf_found || homography_trf_found;
  if (perspective_trf_found && homography_trf_found) {
    Mat i_trf;
    invert(perspective_trf, i_trf);
    double err1 = calc_perspective_trf_error(mpts_1, mpts_2, perspective_trf) +
                  calc_perspective_trf_error(mpts_2, mpts_1, i_trf);
    invert(homography_trf, i_trf);
    double err2 = calc_perspective_trf_error(mpts_1, mpts_2, homography_trf) +
                  calc_perspective_trf_error(mpts_2, mpts_1, i_trf);
    std::cout << "perspective trf error: " << err1 << std::endl;
    std::cout << "homography trf error: " << err2 << std::endl;
    if (err1 < err2)
      trf = perspective_trf;
    else
      trf = homography_trf;
  } else if (perspective_trf_found) {
    trf = perspective_trf;
  } else if (homography_trf_found) {
    trf = homography_trf;
  }

  if (debug) {
    Mat im1_with_text;
    //absdiff(im1_scaled, im1text_scaled, im1_with_text);
    OverlayImage(im1_scaled, im1text_scaled, im1_with_text);
    //overlay_image(im1_scaled, im1text_scaled, im1_with_text);
    imshow("original", im1_with_text);
  }

  if (trf_found) {
    Mat im1text_warped;
    cv::warpPerspective(im1text_scaled, im1text_warped, trf, im2.size());
    if (debug) {
      cv::Mat im2_w_text;
      OverlayImage(im2_scaled, im1text_warped, im2_w_text);
      imshow("im2_w_text", im2_w_text);
      waitKey();
    }
    cv::resize(im1text_warped, im1text_warped, Size(), 1.0/im2_scale, 1.0/im2_scale );
    cv::absdiff(img2, im1text_warped, out);
    images_matched = true;
  }

  if (debug)
    std::cout << "All steps took " << timer.getTotalTime() << " ms" << std::endl << std::endl;

  return images_matched;
}
