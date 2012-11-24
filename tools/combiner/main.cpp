#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <imdb.hpp>

void usage() {
    printf("Usage:\n");
    printf("./combiner imdb_base_file.db\n");
}

static const size_t WIDTH = 640;
static const size_t HEIGHT = 480;

static const size_t VERTICAL = 4;
static const size_t HORIZONTAL = 4;

int main(int argc, char** argv) {
    std::srand(time(0));

    if (argc < 2) {
        usage();
        exit(1);
    }

    std::string baseFileName = argv[1];

    Imdb base;
    if (base.load(baseFileName) < 0) {
        fprintf(stderr, "Unable to load db from file '%s'.\n", baseFileName.c_str());
        exit(2);
    }

    printf("Base from '%s' loaded.\n", baseFileName.c_str());
    printf("Dump:\n");
    printf("%s\n", base.dump().c_str());

    cv::Mat img(HEIGHT, WIDTH, CV_8UC3);


    for (size_t pic = 0; pic < VERTICAL * HORIZONTAL; ++pic) {
        size_t x = (pic * WIDTH / HORIZONTAL) % WIDTH;
        size_t y = (pic / VERTICAL) * (HEIGHT / VERTICAL);
        cv::Mat roi(
            img,
            cv::Rect(x, y, WIDTH / VERTICAL, HEIGHT / HORIZONTAL)
        );
        cv::resize(
            base.getImage(rand() % base.size()),
            roi,
            cv::Size(WIDTH / HORIZONTAL, HEIGHT / VERTICAL)
        );
    }

    cv::imshow("Combination", img);
    cv::waitKey();

    return 0;
}
