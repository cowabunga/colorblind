#include <iostream>
#include <fstream>
#include <sstream>

#include <stdio.h>
#include <cstdlib>
#include <cassert>
#include <sys/stat.h>
#include <unistd.h>

#include <imdb.hpp>
#include "../assert.h"

std::string getTestData();
void createTestImdbFile();
void removeTestImdbFile();
void test_imdbLoad();




static const std::string testFileName = "test_file_for_ut_imdb.txt";
static const std::string testDirName = "Test_dir_for_ut_imdb";
static const std::string fullTestFileName = testDirName + "/" + testFileName;




std::string getTestData() {
    std::stringstream data(std::stringstream::out);

    data << "resized12.jpg resized12_text.png" << std::endl
         << "resized14.jpg resized14_text.png" << std::endl
         << "resized190.jpg resized190_text.png" << std::endl
         << "resized191.jpg resized191_text.png";

    return data.str();
}





void createTestImdbFile() {
    UTASSERT(mkdir(testDirName.c_str(), S_IRWXU) == 0);

    std::ofstream file;
    file.open(fullTestFileName.c_str());

    UTASSERT(file.is_open());

    file << getTestData();

    file.close();
}





void removeTestImdbFile() {
    UTASSERT(remove(fullTestFileName.c_str()) == 0);
    UTASSERT(rmdir(testDirName.c_str()) == 0)
}





cv::Mat getRandomMatrix(size_t rows = 480, size_t cols = 640) {
    cv::Mat result(rows, cols, CV_8UC1);

    for (int x = 0; x < cols; ++x) {
        for (int y = 0; y < rows; ++y) {
            result.at<unsigned>(y,x) = rand() % 200 + 55;
        }
    }

    return result;
}





void test_imdbLoad() {
    Imdb db;

    UTASSERT(db.load(fullTestFileName) != -1);

    std::string pathPrefix = testDirName + "/";

    std::stringstream dataStream(std::stringstream::in | std::stringstream::out);

    dataStream << getTestData();


    ImdbRecord record;
    size_t i = 0;
    while (dataStream >> record.pathToSourceImage >> record.pathToLabelImage) {
        UTASSERT(
            pathPrefix + record.pathToSourceImage == db[i].pathToSourceImage
        );
        UTASSERT(
            pathPrefix + record.pathToLabelImage == db[i].pathToLabelImage
        );
        ++i;
    }
}




void test_imdbGetImage() {
    Imdb db;

    UTASSERT(db.load(fullTestFileName) != -1);

}

int main() {
    std::srand(360);

    createTestImdbFile();
    test_imdbLoad();
    removeTestImdbFile();

    cv::Mat mat = getRandomMatrix();
    cv::imwrite("test_result.jpg", mat);

    return 0;
}
