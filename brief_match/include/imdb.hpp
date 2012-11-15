#ifndef _COLORBLIND_BRIEF_MATCH_IMDB_H_
#define _COLORBLIND_BRIEF_MATCH_IMDB_H_

#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>




struct ImdbRecord {
    std::string pathToSourceImage;
    std::string pathToLabelImage;
};




class Imdb {
    public:
    Imdb();

    int load(const std::string & filename);
    cv::Mat getImage(size_t index);
    cv::Mat getLabel(size_t index);

    std::string dump();

    private:
    std::vector<ImdbRecord> records;

};





#endif // _COLORBLIND_BRIEF_MATCH_IMDB_H_
