#include <fstream>
#include <sstream>
#include <iostream>


#include <memory.h>


#include <imgpathes.hpp>


std::ostream & operator << (std::ostream & out, const ImgPathesRecord & record) {
    out << record.pathToSourceImage << " " << record.pathToLabelImage << std::endl;

    return out;
}



bool operator == (const ImgPathesRecord & left, const ImgPathesRecord & right) {
    return left.pathToSourceImage == right.pathToSourceImage &&
           left.pathToLabelImage == right.pathToLabelImage;
}



ImgPathes::ImgPathes() {
}



int ImgPathes::load(const std::string & filename) {
    std::ifstream in(filename.c_str());


    if (!in.good()) {
        return -1;
    }

    records.clear();

    std::string pathPrefix = "";
    {
        size_t slash = filename.rfind('/', static_cast<size_t>(-1));
        if (slash != static_cast<size_t>(-1)) {
            pathPrefix = filename.substr(0, slash);
        }
    }

    ImgPathesRecord record;
    while (in >> record.pathToSourceImage >> record.pathToLabelImage) {
        if (pathPrefix.size()) {
            record.pathToSourceImage.insert(0, pathPrefix + std::string("/"));
            record.pathToLabelImage.insert(0, pathPrefix + std::string("/"));
        }
        records.push_back(record);
    }

    return records.size();
}



cv::Mat ImgPathes::getImage(size_t index) const {
    return cv::imread(records.at(index).pathToSourceImage);
}



cv::Mat ImgPathes::getLabel(size_t index) const {
    return cv::imread(records.at(index).pathToLabelImage);
}



const ImgPathesRecord & ImgPathes::operator[] (size_t index) const {
    return records.at(index);
}



ImgPathesRecord & ImgPathes::operator[] (size_t index) {
    return records.at(index);
}



size_t ImgPathes::size() const {
    return records.size();
}



std::string ImgPathes::dump() const {
    std::stringstream out;

    out << "{" << std::endl;
    for (std::vector<ImgPathesRecord>::const_iterator it = records.begin();
         it != records.end(); ++it) {
        out << "    [" << it->pathToSourceImage << ", " <<
            it->pathToLabelImage << "]" << std::endl;
    }

    out << "}";

    return out.str();
}

