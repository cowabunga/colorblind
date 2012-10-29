#ifndef COLORBLIND_BRIEF_MATCH_ANOMALY_H
#define COLORBLIND_BRIEF_MATCH_ANOMALY_H


#include <opencv2/core/core.hpp>

#include <vector>

#include <cmath>



typedef cv::Mat_<float> Matf;





cv::Point2f meanOfVectors(const std::vector<cv::Point2f> & vecs);

Matf varianceOfVectors(const std::vector<cv::Point2f> & vecs);

// Don't use this function
// Use cv::determinant instead of
float det(const Matf & mat);

float computeDensity(float, const cv::Point2f &, const Matf &
        , const cv::Point2f);

std::vector<size_t> getBestVecIndexes(const std::vector<cv::Point2f> &);




class TDistribution {
    public:
    TDistribution() {
    }

    template <typename TVec>
    TVec calcMean(const std::vector<TVec> &) const;

    template <typename TVec>
    cv::Mat_<typename TVec::value_type> calcVariance(
            const std::vector<TVec> &) const;



    private:
    template <typename TVec>
    TVec vecDiff(const TVec &, const TVec &) const;

    template <typename TVec>
    cv::Mat_<typename TVec::value_type> vecMult(
            const TVec &, const TVec &) const;
};




template <typename TVec>
TVec TDistribution::calcMean(const std::vector<TVec> & evec) const {
    TVec mean;


    for (typename std::vector<TVec>::const_iterator example = evec.begin()
            ; example != evec.end(); ++example) {
        for (int feature = 0; feature < TVec::rows; ++feature) {
            mean[feature] += (*example)[feature];
        }
    }


    if (!evec.empty()) {
        for (int feature = 0; feature < TVec::rows; ++feature) {
            mean[feature] /= evec.size();
        }
    }


    return mean;
}



template <typename TVec>
cv::Mat_<typename TVec::value_type> TDistribution::calcVariance(
        const std::vector<TVec> & vec) const {
    cv::Mat_<typename TVec::value_type> variance(TVec::rows, TVec::rows
            , typename TVec::value_type(0));

    TVec mean = calcMean(vec);

    for (typename std::vector<TVec>::const_iterator it = vec.begin()
            ; it != vec.end(); ++it) {
        TVec normalized = vecDiff(*it, mean);

        variance += vecMult(normalized, normalized);
    }

    if (!vec.empty()) {
        variance /= vec.size();
    }


    return variance;
}



template <typename TVec>
TVec TDistribution::vecDiff(const TVec & a, const TVec & b) const {
    TVec result;

    for (int i = 0; i < a.rows; ++i) {
        result[i] = a[i] - b[i];
    }

    return result;
}



template <typename TVec>
cv::Mat_<typename TVec::value_type> TDistribution::vecMult(
        const TVec & a, const TVec & b) const {
    cv::Mat_<typename TVec::value_type> mat(TVec::rows, TVec::rows);

    for (int row = 0; row < mat.rows; ++row) {
        for (int col = 0; col < mat.cols; ++col) {
            mat(row, col) = a[row] * b[col];
        }
    }

    return mat;
}



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

    Matf ivar(2, 2, float(0));
    cv::invert(var, ivar);




    for (size_t i = 0; i < vecs.size(); ++i) {
        if (computeDensity(factor, mean, ivar, vecs[i]) < sigma) {
            goodIndexes.push_back(i);
        }
    }


    return goodIndexes;
}


#endif // COLORBLIND_BRIEF_MATCH_ANOMALY_H
