#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cstdio>

static void help() {
    printf("./resizer inputImage outputImage width height\n");
}

int main(int argc, char** argv) {
    if (argc < 5) {
        help();
        return 1;
    }


    printf("Loading %s image...", argv[1]);
    cv::Mat source = cv::imread(argv[1]);

    if (source.data == NULL) {
        printf(" error: Can't read image!\n");
        return 1;
    } else {
        printf(" ok\n");
    }



    size_t width = atoi(argv[3]);
    size_t height = atoi(argv[4]);
    cv::Mat resizedSource;



    printf("Resizing image to %ld x %ld...\n", width, height);
    cv::resize(source, resizedSource, cv::Size(width, height));



    printf("Saving image to %s...", argv[2]);
    imwrite(argv[2], resizedSource);
    printf(" ok\n");



    source.release();
    resizedSource.release();

    return 0;
}
