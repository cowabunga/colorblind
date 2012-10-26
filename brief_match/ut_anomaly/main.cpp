#include "../brief_match/anomaly.h"

#include <iostream>
#include <cmath>


void test_meanOfVectors();
void test_varianceOfVectors();
void test_TDistribution_calcMean();
void test_TDistribution_calcVariance();
void test_computeDensity();

void assertEqual(const cv::Point2f &, const cv::Point2f &, float);
void assertEqual(const Matf &, const Matf &, float);
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



template <typename T>
void assertEqual(const T & a, const T & b) {
    if (a != b) {
        assertionFailed(a, b);
    }
}



void assertEqual(const cv::Point2f & a
        , const cv::Point2f & b
        , float EPSILON = 0.0001) {
    if (std::fabs(a.x - b.x) > EPSILON || std::fabs(a.y - b.y) > EPSILON) {
        assertionFailed(a, b);
    }
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




void assertTrue(bool expr) {
    assertEqual(expr, true);
}





void test_meanOfVectors() {
    typedef cv::Point2f P;

    {
        P a[10] = {
            P(8.710867, 11.523142), P(8.097444, 6.503540),
            P(13.866495, 8.814358), P(13.889156, 7.560562),
            P(8.442556, 5.409871), P(14.403722, 11.449097),
            P(1.426250, 11.697567), P(12.359932, 12.446966),
            P(14.937492, 5.528084), P(12.733541, 20.837173)
        };

        std::vector<P> vecs(a, a + 10);

        P mean = meanOfVectors(vecs);

        assertEqual(mean, P(10.8867, 10.177));
    }

    {
        std::vector<P> vecs;

        P mean = meanOfVectors(vecs);

        assertEqual(mean, P(0, 0));
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
    }
}


void test_varianceOfVectors() {
    typedef cv::Point2f P;

    {
        P a[10] = {
            P(8.710867, 11.523142), P(8.097444, 6.503540),
            P(13.866495, 8.814358), P(13.889156, 7.560562),
            P(8.442556, 5.409871), P(14.403722, 11.449097),
            P(1.426250, 11.697567), P(12.359932, 12.446966),
            P(14.937492, 5.528084), P(12.733541, 20.837173)
        };

        std::vector<P> vecs(a, a + 10);


        Matf var = varianceOfVectors(vecs);


        // Octave result
        Matf expect(2, 2);
        expect(0, 0) = 16.02417;
        expect(0, 1) = 0.13414;
        expect(1, 0) = 0.13414;
        expect(1, 1) = 19.10693;


        assertEqual(var, expect);

        // det(expect) = 306.15;
        assertTrue(det(var) - float(306.15) < 0.01);
    }
}

void test_computeDensity() {
    typedef cv::Point2f P;

    {
        P a[10] = {
            P(8.710867, 11.523142), P(8.097444, 6.503540),
            P(13.866495, 8.814358), P(13.889156, 7.560562),
            P(8.442556, 5.409871), P(14.403722, 11.449097),
            P(1.426250, 11.697567), P(12.359932, 12.446966),
            P(14.937492, 5.528084), P(12.733541, 20.837173)
        };

        std::vector<P> vecs(a, a + 10);


        Matf var = varianceOfVectors(vecs);
        P mean = meanOfVectors(vecs);

        Matf ivar;

        cv::invert(var, ivar);

        float density = computeDensity(1, mean, ivar, P(10, 10));

        assertTrue(std::fabs(density - 0.97503) < 0.0001);
    }

}




int main() {
    test_meanOfVectors();

    test_varianceOfVectors();

    test_computeDensity();


    test_TDistribution_calcMean();

    test_TDistribution_calcVariance();

    return 0;
}
