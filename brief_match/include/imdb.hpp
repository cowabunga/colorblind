#ifndef _COLORBLIND_BRIEF_MATCH_IMDB_H_
#define _COLORBLIND_BRIEF_MATCH_IMDB_H_

#include <vector>
#include <string>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>




struct ImdbRecord {
    std::string pathToSourceImage;
    std::string pathToLabelImage;
};

std::ostream& operator << (std::ostream & out, const ImdbRecord & record);
bool operator == (const ImdbRecord & left, const ImdbRecord & right);



class Imdb {
    public:
    Imdb();

    int load(const std::string & filename);
    cv::Mat getImage(size_t index) const;
    cv::Mat getLabel(size_t index) const;

    std::string dump() const;

    const ImdbRecord & operator [] (size_t index) const;
    ImdbRecord & operator [] (size_t index);

    private:
    std::vector<ImdbRecord> records;
};





#endif // _COLORBLIND_BRIEF_MATCH_IMDB_H_
