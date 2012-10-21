#include <stdio.h>
#include <algorithm>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>


using namespace cv;

static void help()
{
    printf("\nThis program demonstrates using features2d detector, descriptor extractor and simple matcher\n"
            "Using the SURF desriptor:\n"
            "\n"
            "Usage:\n matcher_simple <image1> <image2>\n");
}

int main(int argc, char** argv)
{
    if(argc != 3)
    {
        help();
        return -1;
    }

    Mat img1, img2;
    img1 = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    img2 = cv::imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    if(img1.empty() || img2.empty())
    {
        printf("Can't read one of the images\n");
        return -1;
    }

    printf("images loaded\n");

    // detecting keypoints
    printf("Detecting keypoints...\n");
    FastFeatureDetector detector(50);
    vector<KeyPoint> keypoints1, keypoints2;
    detector.detect(img1, keypoints1);
    detector.detect(img2, keypoints2);

    // computing descriptors
    printf("Computing descriptors...\n");
    SiftDescriptorExtractor extractor;
    Mat descriptors1, descriptors2;
    extractor.compute(img1, keypoints1, descriptors1);
    extractor.compute(img2, keypoints2, descriptors2);

    // matching descriptors
    printf("Matching descriptors...\n");
    BFMatcher matcher(NORM_L2);
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);
    std::sort(matches.begin(), matches.end());

    // drawing the results
    printf("Drawing the results...\n");
    namedWindow("matches", 1);
    Mat img_matches;

    size_t chunk_size = 10;
    for( int i = 0; i < matches.size() ; i += chunk_size ) {
        vector<DMatch> top_matches(matches.begin() + i, matches.begin() + min<int>(i + chunk_size, matches.size()));
        drawMatches(img1, keypoints1, img2, keypoints2, top_matches, img_matches);
        imshow("matches", img_matches);
        waitKey();
    }

    return 0;
}
