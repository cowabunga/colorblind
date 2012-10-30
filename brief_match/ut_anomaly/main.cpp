#include "../brief_match/anomaly.h"

#include <iostream>
#include <cmath>


typedef cv::Mat_<float> Matf;


void test_TDistribution();
void test_TDistribution_calcMean();
void test_TDistribution_calcVariance();
void test_TDistribution_calcDensity();

void test_TAnomalyDetector();






void assertEqual(const Matf &, const Matf &, float);
void assertEqual(float, float, float);
void assertTrue(bool);


template <typename A, int B>
void assertEqual(const cv::Vec<A, B> & v, const cv::Vec<A, B> q
        , double EPSILON = 0.0001) {
    for (int i = 0; i < B; ++i) {
        if (std::abs(v[i] - q[i]) > EPSILON) {
            assertionFailed(v, q);
        }
    }
}


template <typename A, int B>
std::ostream & operator << (std::ostream & out, const cv::Vec<A, B> & vec) {
    out << "[ ";
    for (int x = 0; x < B; ++x) {
        out << vec[x] << " ";
    }

    out << "]";

    return out;
}


template <typename T>
void assertionFailed(const T & a, const T & b) {
        std::cerr << std::endl << std::endl
                << "ASSERTION FAILED!" << std::endl
                << "Arg1: " << a << std::endl
                << "Arg2: " << b << std::endl
                << std::endl;
        throw 1;
}


void assertEqual(const Matf & A, const Matf & B, float EPSILON = 0.0001) {
    if (A.rows != B.rows || A.cols != B.cols) {
        assertionFailed(A, B);
    }

    Matf::const_iterator itB = B.begin();
    for (Matf::const_iterator itA = A.begin(); itA != A.end()
            ; ++itA, ++itB) {
        if (std::fabs(*itA - *itB) > EPSILON) {
            assertionFailed(A, B);
        }
    }
}



void assertEqual(float a, float b, float EPSILON = 0.0001) {
    if (std::fabs(a - b) > EPSILON) {
        assertionFailed(a, b);
    }
}






void test_TDistribution_calcMean() {
    typedef cv::Vec2f P;

    TDistribution distribution;


    {
        P a[10] = {
            P(8.710867, 11.523142), P(8.097444, 6.503540),
            P(13.866495, 8.814358), P(13.889156, 7.560562),
            P(8.442556, 5.409871), P(14.403722, 11.449097),
            P(1.426250, 11.697567), P(12.359932, 12.446966),
            P(14.937492, 5.528084), P(12.733541, 20.837173)
        };

        std::vector<P> vecs(a, a + 10);

        P mean = distribution.calcMean(vecs);

        assertEqual(mean, P(10.8867, 10.177));
    }

    {
        std::vector<P> vecs;

        P mean = distribution.calcMean(vecs);

        assertEqual(mean, P(0, 0));
    }

}


void test_TDistribution_calcVariance() {
    typedef cv::Vec2f P;

    TDistribution distribution;

    {
        P a[10] = {
            P(8.710867, 11.523142), P(8.097444, 6.503540),
            P(13.866495, 8.814358), P(13.889156, 7.560562),
            P(8.442556, 5.409871), P(14.403722, 11.449097),
            P(1.426250, 11.697567), P(12.359932, 12.446966),
            P(14.937492, 5.528084), P(12.733541, 20.837173)
        };

        std::vector<P> vecs(a, a + 10);

        Matf var = distribution.calcVariance(vecs);

        Matf expect(2, 2);
        expect(0, 0) = 16.02417;
        expect(0, 1) = 0.13414;
        expect(1, 0) = 0.13414;
        expect(1, 1) = 19.10693;

        assertEqual(var, expect);
    }
}




void test_TDistribution_calcDensity() {
    typedef cv::Vec2f P;

    TDistribution distribution;

    {
        P a[10] = {
            P(8.710867, 11.523142), P(8.097444, 6.503540),
            P(13.866495, 8.814358), P(13.889156, 7.560562),
            P(8.442556, 5.409871), P(14.403722, 11.449097),
            P(1.426250, 11.697567), P(12.359932, 12.446966),
            P(14.937492, 5.528084), P(12.733541, 20.837173)
        };

        std::vector<P> vecs(a, a + 10);


        Matf var = distribution.calcVariance(vecs);
        P mean = distribution.calcMean(vecs);
        float sigma = sqrt(cv::determinant(var));

        Matf iVar;

        cv::invert(var, iVar);


        float density = distribution.calcDensity(P(10, 10), mean, iVar, sigma);

        assertEqual(density, 0.0088688);
    }
}





void test_TDistribution() {
    test_TDistribution_calcMean();
    test_TDistribution_calcVariance();
    test_TDistribution_calcDensity();
}





void test_TAnomalyDetector() {
    typedef cv::Vec2f P;

    P a[10] = {
        P(8.710867, 11.523142), P(8.097444, 6.503540),
        P(13.866495, 8.814358), P(13.889156, 7.560562),
        P(8.442556, 5.409871), P(14.403722, 11.449097),
        P(1.426250, 11.697567), P(12.359932, 12.446966),
        P(14.937492, 5.528084), P(12.733541, 20.837173)
    };

    TAnomalyDetector<P> detector;
    std::vector<P> vecs(a, a + 10);


    detector.init(vecs);

    {

    }
}


int main() {
    test_TDistribution();

    test_TAnomalyDetector();

    return 0;
}
