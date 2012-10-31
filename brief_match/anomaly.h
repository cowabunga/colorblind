#ifndef COLORBLIND_BRIEF_MATCH_ANOMALY_H
#define COLORBLIND_BRIEF_MATCH_ANOMALY_H


#include <opencv2/core/core.hpp>

#include <vector>

#include <cmath>

struct TDistribution;

template <typename TVec>
class TAnomalyDetector;







// http://en.wikipedia.org/wiki/Multivariate_normal_distribution
struct TDistribution {
    TDistribution() {
    }

    template <typename TVec>
    TVec calcMean(const std::vector<TVec> &) const;

    template <typename TVec>
    cv::Mat_<typename TVec::value_type> calcVariance(
            const std::vector<TVec> &) const;

    template <typename TVec>
    typename TVec::value_type calcHi(const TVec & vec, const TVec & mean
            , const cv::Mat_<typename TVec::value_type> & invertedVariance) const;

    template <typename TVec>
    typename TVec::value_type calcDensity(const TVec & vec
            , const TVec & mean
            , const cv::Mat_<typename TVec::value_type> & invertedVariance
            , const typename TVec::value_type & sigma);

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
        const std::vector<TVec> & evec) const {
    cv::Mat_<typename TVec::value_type> variance(TVec::rows, TVec::rows
            , typename TVec::value_type(0));

    TVec mean = calcMean(evec);

    for (typename std::vector<TVec>::const_iterator it = evec.begin()
            ; it != evec.end(); ++it) {
        TVec normalized = *it - mean;

        cv::Mat_<typename TVec::value_type> normalizedT;
        cv::transpose(normalized, normalizedT);

        variance +=
                cv::Mat_<typename TVec::value_type>(normalized) * normalizedT;
    }

    if (!evec.empty()) {
        variance /= evec.size();
    }


    return variance;
}



template <typename TVec>
typename TVec::value_type TDistribution::calcHi(const TVec & vec
        , const TVec & mean
        , const cv::Mat_<typename TVec::value_type> & invertedVariance) const {
    TVec normalized = vec - mean;

    cv::Mat_<typename TVec::value_type> normalizedT;
    cv::transpose(normalized, normalizedT);

    normalizedT *=
            invertedVariance * cv::Mat_<typename TVec::value_type>(normalized);

    return normalizedT(0, 0);
}



template <typename TVec>
typename TVec::value_type TDistribution::calcDensity(const TVec & vec
        , const TVec & mean
        , const cv::Mat_<typename TVec::value_type> & invertedVariance
        , const typename TVec::value_type & sigma) {
    typename TVec::value_type factor =
            sqrt(pow(2 * CV_PI, TVec::rows)) * sigma;

    typename TVec::value_type power =
            calcHi(vec, mean, invertedVariance) / -2;


    return exp(power)/factor;
}











template <typename TVec>
class TAnomalyDetector {
    public:
    TAnomalyDetector() {
    }

    void init(const std::vector<TVec> & evec);

    typename TVec::value_type getStdDiviation() const;

    const TVec & getMean() const;

    const cv::Mat_<typename TVec::value_type> & getVariance() const;

    std::vector<size_t> getFilteredIndexes(const std::vector<TVec> & evec) const;

    std::vector<size_t> getFilteredIndexes(const std::vector<TVec> & evec
            , typename TVec::value_type thresh) const;

    typename TVec::value_type calcError(const TVec & vec) const;


    private:
    TDistribution _distrib;
    TVec _mean;
    cv::Mat_<typename TVec::value_type> _variance;
    cv::Mat_<typename TVec::value_type> _iVariance;
    typename TVec::value_type _sigma;


    typename TVec::value_type measure(TVec a, TVec b);
};



template <typename TVec>
void TAnomalyDetector<TVec>::init(const std::vector<TVec> & evec) {
    _mean = _distrib.calcMean(evec);

    _variance = _distrib.calcVariance(evec);

    cv::invert(_variance, _iVariance);

    _sigma = sqrt(cv::determinant(_variance));
}



template <typename TVec>
typename TVec::value_type TAnomalyDetector<TVec>::getStdDiviation() const {
    return _sigma;
}



template <typename TVec>
const TVec & TAnomalyDetector<TVec>::getMean() const {
    return _mean;
}



template <typename TVec>
const cv::Mat_<typename TVec::value_type> & TAnomalyDetector<TVec>::getVariance() const {
    return _variance;
}


template <typename TVec>
std::vector<size_t> TAnomalyDetector<TVec>::getFilteredIndexes(
        const std::vector<TVec> & evec) const {
    return this->getFilteredIndexes(evec, typename TVec::value_type(TVec::rows));
}



// http://en.wikipedia.org/wiki/Multivariate_normal_distribution#Tolerance_region
template <typename TVec>
std::vector<size_t> TAnomalyDetector<TVec>::getFilteredIndexes(
        const std::vector<TVec> & evec
        , typename TVec::value_type thresh) const {
    std::vector<size_t> goodIndexes;


    for (size_t i = 0; i < evec.size(); ++i) {
        typename TVec::value_type hi =
                _distrib.calcHi(evec[i], _mean, _iVariance);

        if (hi <= thresh) {
            goodIndexes.push_back(i);
        }
    }


    return goodIndexes;
}



template <typename TVec>
typename TVec::value_type TAnomalyDetector<TVec>::calcError(
        const TVec & vec) const {
    typename TVec::value_type error = 0;


    for (int i = 0; i < TVec::rows; ++i) {
        error += (vec[i] - _mean[i]) * (vec[i] - _mean[i]);
    }


    return error;
}

#endif // COLORBLIND_BRIEF_MATCH_ANOMALY_H
