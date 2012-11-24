#include <fstream>
#include <sstream>
#include <iostream>


#include <memory.h>


#include <imdb.hpp>


std::ostream & operator << (std::ostream & out, const ImdbRecord & record) {
    out << record.pathToSourceImage << " " << record.pathToLabelImage << std::endl;

    return out;
}



bool operator == (const ImdbRecord & left, const ImdbRecord & right) {
    return left.pathToSourceImage == right.pathToSourceImage &&
           left.pathToLabelImage == right.pathToLabelImage;
}



Imdb::Imdb() {
}



int Imdb::load(const std::string & filename) {
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

    ImdbRecord record;
    while (in >> record.pathToSourceImage >> record.pathToLabelImage) {
        if (pathPrefix.size()) {
            record.pathToSourceImage.insert(0, pathPrefix + std::string("/"));
            record.pathToLabelImage.insert(0, pathPrefix + std::string("/"));
        }
        records.push_back(record);
    }

    return records.size();
}



cv::Mat Imdb::getImage(size_t index) const {
    return cv::imread(records.at(index).pathToSourceImage);
}



cv::Mat Imdb::getLabel(size_t index) const {
    return cv::imread(records.at(index).pathToLabelImage);
}



const ImdbRecord & Imdb::operator[] (size_t index) const {
    return records.at(index);
}



ImdbRecord & Imdb::operator[] (size_t index) {
    return records.at(index);
}



size_t Imdb::size() const {
    return records.size();
}



std::string Imdb::dump() const {
    std::stringstream out;

    out << "{" << std::endl;
    for (std::vector<ImdbRecord>::const_iterator it = records.begin();
         it != records.end(); ++it) {
        out << "    [" << it->pathToSourceImage << ", " <<
            it->pathToLabelImage << "]" << std::endl;
    }

    out << "}";

    return out.str();
}

