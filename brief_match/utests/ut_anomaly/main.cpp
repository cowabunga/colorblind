#include <iostream>
#include <algorithm>

#include <cmath>

#include <anomaly.hpp>
#include <brief_match/utests/MiniCppUnit.hxx>

class AnomalyTests : public TestFixture<AnomalyTests> {
public:
    TEST_FIXTURE(AnomalyTests)
    {
        TEST_CASE(testCalcMean);
        TEST_CASE(testCalcMeanOnEmptyVector);
        TEST_CASE(testCalcVariance);
        TEST_CASE(testCalcDensity);
        TEST_CASE(testAnomalyDetection);
    }


    typedef cv::Vec2f Vecf;
    typedef cv::Mat_<float> Matf;


    virtual void setUp() {
        testData[0] = Vecf(8.710867, 11.523142);
        testData[1] = Vecf(8.097444, 6.503540);
        testData[2] = Vecf(13.866495, 8.814358);
        testData[3] = Vecf(13.889156, 7.560562);
        testData[4] = Vecf(8.442556, 5.409871);
        testData[5] = Vecf(14.403722, 11.449097);
        testData[6] = Vecf(1.426250, 11.697567);
        testData[7] = Vecf(12.359932, 12.446966);
        testData[8] = Vecf(14.937492, 5.528084);
        testData[9] = Vecf(12.733541, 20.837173);
    }


    void testCalcMean() {
        std::vector<Vecf> vecs(testData, testData + 10);

        Vecf mean = distribution.calcMean(vecs);

        ASSERT_EQUALS_EPSILON(mean[0], 10.8867, EPSILON);
        ASSERT_EQUALS_EPSILON(mean[1], 10.177, EPSILON);
    }


    void testCalcMeanOnEmptyVector() {
        std::vector<Vecf> vecs;

        Vecf mean = distribution.calcMean(vecs);

        ASSERT_EQUALS(mean, Vecf(0, 0));
    }


    void testCalcVariance() {
        std::vector<Vecf> vecs(testData, testData + 10);

        Matf var = distribution.calcVariance(vecs);

        ASSERT_EQUALS_EPSILON(var(0, 0), 16.02417, EPSILON);
        ASSERT_EQUALS_EPSILON(var(0, 1), 0.13414, EPSILON);
        ASSERT_EQUALS_EPSILON(var(1, 0), 0.13414, EPSILON);
        ASSERT_EQUALS_EPSILON(var(1, 1), 19.10693, EPSILON);
    }


    void testCalcDensity() {
        std::vector<Vecf> vecs(testData, testData + 10);

        Matf var = distribution.calcVariance(vecs);
        Vecf mean = distribution.calcMean(vecs);
        float sigma = sqrt(cv::determinant(var));

        Matf iVar;

        cv::invert(var, iVar);

        float density =
            distribution.calcDensity(Vecf(10, 10), mean, iVar, sigma);

        ASSERT_EQUALS_EPSILON(density, 0.0088688, EPSILON);
    }


    void testAnomalyDetection() {
        TAnomalyDetector<Vecf> detector;
        std::vector<Vecf> vecs(testData, testData + 10);

        detector.init(vecs);

        std::vector<size_t> indexes = detector.getFilteredIndexes(vecs);

        float maxErrorOfGoodVecs = 0;
        for (std::vector<size_t>::const_iterator it = indexes.begin();
             it != indexes.end(); ++it) {
            float curError = detector.calcError(vecs[*it]);

            maxErrorOfGoodVecs = std::max(maxErrorOfGoodVecs, curError);
        }

        for (size_t i = 0; i < vecs.size(); ++i) {
            float curError = detector.calcError(vecs[i]);

            ASSERT_MESSAGE(!(curError <= maxErrorOfGoodVecs &&
                std::find(indexes.begin(), indexes.end(), i) == indexes.end()),
                "Element is not anomaly but it wasn't append to non anomaly indexes.");

            ASSERT_MESSAGE(!(curError > maxErrorOfGoodVecs &&
                std::find(indexes.begin(), indexes.end(), i) != indexes.end()),
                "Element is anomaly but it was append to non anomaly indexes.");
        }
    }


private:
    static const double EPSILON;
    Vecf testData[10];
    TDistribution distribution;
};

const double AnomalyTests::EPSILON = 0.0001;

REGISTER_FIXTURE(AnomalyTests);




int main() {
    return TestFixtureFactory::theInstance().runTests() ? 0 : 1;
}
