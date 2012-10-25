#ifndef COLORBLIND_BRIEF_MATCH_ANOMALY_H
#define COLORBLIND_BRIEF_MATCH_ANOMALY_H


#include <opencv2/core/core.hpp>

#include <vector>

#include <cmath>





typedef cv::Mat_<float> Matf;





cv::Point2f meanOfVectors(const std::vector<cv::Point2f> & vecs);
Matf varianceOfVectors(const std::vector<cv::Point2f> & vecs);
float det(const Matf & mat);

std::vector<size_t> getBestVecIndexes(std::vector<cv::Point2f>);





cv::Point2f meanOfVectors(const std::vector<cv::Point2f> & vecs) {
    cv::Point2f mean(0, 0);


    for (std::vector<cv::Point2f>::const_iterator it = vecs.begin()
            ; it != vecs.end(); ++it) {
        mean += *it;
    }


    if (!vecs.empty()) {
        mean.x /= vecs.size();
        mean.y /= vecs.size();
    }


    return mean;
}






Matf varianceOfVectors(const std::vector<cv::Point2f> & vecs) {
    cv::Point2f mean = meanOfVectors(vecs);

    Matf var(2, 2, float(0));


    for (std::vector<cv::Point2f>::const_iterator it = vecs.begin()
            ; it != vecs.end(); ++it) {
        cv::Point2f norm = *it - mean;

        var(0, 0) += norm.x * norm.x;
        var(1, 1) += norm.y * norm.y;
        var(1, 0) += norm.x * norm.y;
        var(0, 1) += norm.x * norm.y;
    }

    if (!vecs.empty()) {
        var(0, 0) /= vecs.size();
        var(1, 0) /= vecs.size();
        var(0, 1) /= vecs.size();
        var(1, 1) /= vecs.size();
    }

    return var;
}





float det(const Matf & mat) {
    return mat(0, 0) * mat(1, 1) - mat(1, 0) * mat(0, 1);
}


float computeDensity(float factor, const cv::Point2f & mean
        , const Matf & ivar, const cv::Point2f vec) {
    cv::Point2f norm = vec - mean;


    // compute matrix multiplication
    // vec(1, 2) * mat(2, 2) * vec(2, 1)
    float x = norm.x * ivar(0, 0) + norm.y * ivar(1, 0);
    float y = norm.x * ivar(0, 1) + norm.y * ivar(1, 1);
    float power = -0.5;
    power *= x * norm.x + y * norm.y;



    return factor * exp(power);
}


// clean the vectors from anomaly using:
// http://en.wikipedia.org/wiki/Multivariate_normal_distribution
//
// Return vector of indexes.
std::vector<size_t> getBestVecIndexes(const std::vector<cv::Point2f> & vecs) {
    cv::Point2f mean = meanOfVectors(vecs);
    Matf var = varianceOfVectors(vecs);

    const float pi = 3.1415926535897932;

    std::vector<size_t> goodIndexes;


    float sigma = sqrt(det(var));

    float factor =  pow(2*pi, -float(vecs.size())/2) * sigma;

    var.inv();




    for (size_t i = 0; i < vecs.size(); ++i) {
        if (computeDensity(factor, mean, var, vecs[i]) < sigma) {
            goodIndexes.push_back(i);
        }
    }


    return goodIndexes;
}


#endif // COLORBLIND_BRIEF_MATCH_ANOMALY_H
