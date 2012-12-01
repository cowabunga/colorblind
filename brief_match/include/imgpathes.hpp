#ifndef _COLORBLIND_BRIEF_MATCH_IMDB_H_
#define _COLORBLIND_BRIEF_MATCH_IMDB_H_

#include <vector>
#include <string>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>




struct ImgPathesRecord {
    std::string pathToSourceImage;
    std::string pathToLabelImage;
};

std::ostream& operator << (std::ostream & out, const ImgPathesRecord & record);
bool operator == (const ImgPathesRecord & left, const ImgPathesRecord & right);



class ImgPathes {
    public:
    ImgPathes();

    int load(const std::string & filename);
    cv::Mat getImage(size_t index) const;
    cv::Mat getLabel(size_t index) const;

    std::string dump() const;

    const ImgPathesRecord & operator [] (size_t index) const;
    ImgPathesRecord & operator [] (size_t index);

    size_t size() const;

    private:
    std::vector<ImgPathesRecord> records;
};





#endif // _COLORBLIND_BRIEF_MATCH_IMDB_H_
