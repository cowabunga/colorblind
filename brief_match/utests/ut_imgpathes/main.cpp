#include <iostream>
#include <fstream>
#include <sstream>

#include <stdio.h>
#include <cstdlib>
#include <sys/stat.h>
#include <unistd.h>

#include <imgpathes.hpp>
#include <brief_match/utests/MiniCppUnit.hxx>




class ImgPathesTests : public TestFixture<ImgPathesTests> {
public:
    TEST_FIXTURE(ImgPathesTests) {
        TEST_CASE(testImgPathesLoadFromFile);
        TEST_CASE(testImgPathesGetImage);
    }


    virtual void setUp() {
        createTestImgPathesFile();
    }


    virtual void tearDown() {
        removeTestImgPathesFile();
    }


    void testImgPathesLoadFromFile() {
        ImgPathes db;
        ASSERT_MESSAGE(db.load(fullTestFileName) != -1,
            "Can't load pathes from file");

        std::string pathPrefix = testDirName + "/";

        std::stringstream dataStream(std::stringstream::in | std::stringstream::out);
        dataStream << getTestData();


        ImgPathesRecord record;
        size_t i = 0;
        while (dataStream >> record.pathToSourceImage >> record.pathToLabelImage) {
            ASSERT_EQUALS(
                pathPrefix + record.pathToSourceImage,
                db[i].pathToSourceImage);
            ASSERT_EQUALS(
                pathPrefix + record.pathToLabelImage,
                db[i].pathToLabelImage);
            ++i;
        }
    }


    void testImgPathesGetImage() {
        ImgPathes db;

        ASSERT_MESSAGE(db.load(fullTestFileName) != -1,
            "Can't load pathes from file.");
    }



private:
    static const std::string testFileName;
    static const std::string testDirName;
    static const std::string fullTestFileName;


    std::string getTestData() {
        std::stringstream data(std::stringstream::out);

        data << "resized12.jpg resized12_text.png" << std::endl
             << "resized14.jpg resized14_text.png" << std::endl
             << "resized190.jpg resized190_text.png" << std::endl
             << "resized191.jpg resized191_text.png";

        return data.str();
    }


    void createTestImgPathesFile() {
        mkdir(testDirName.c_str(), S_IRWXU);

        std::ofstream file;
        file.open(fullTestFileName.c_str());

        file << getTestData();
        file.close();
    }


    void removeTestImgPathesFile() {
        remove(fullTestFileName.c_str());
        rmdir(testDirName.c_str()) != 0;
    }


    cv::Mat getRandomMatrix(size_t rows = 480, size_t cols = 640) {
        cv::Mat result(rows, cols, CV_8UC1);

        for (size_t x = 0; x < cols; ++x) {
            for (size_t y = 0; y < rows; ++y) {
                result.at<unsigned>(y,x) = rand() % 200 + 55;
            }
        }

        return result;
    }
};



const std::string ImgPathesTests::testFileName = "test_file_for_ut_imgpathes.txt";
const std::string ImgPathesTests::testDirName = "Test_dir_for_ut_imgpathes";
const std::string ImgPathesTests::fullTestFileName = testDirName + "/" + testFileName;



REGISTER_FIXTURE(ImgPathesTests);


int main() {
    std::srand(360);
    return TestFixtureFactory::theInstance().runTests() ? 0 : 1;
}
