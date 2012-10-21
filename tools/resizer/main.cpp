#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>

#include <cstdio>

static void help() {
    printf("./resizer inputImage outputImage width height\n");
}

int main(int argc, char** argv) {
    if (argc < 5) {
        help();
        return 1;
    }


    IplImage * source = cvLoadImage(argv[1]);

    size_t width = atoi(argv[3]);
    size_t height = atoi(argv[4]);

    IplImage * resultImage =
        cvCreateImage(cvSize(width, height), source->depth, source->nChannels);


    cvResize(source, resultImage);

    cvSaveImage(argv[2], resultImage);

    // TODO
    // Fix mem leaks.

    return 0;
}
