#include <fstream>
#include <sstream>
#include <iostream>


#include <memory.h>


#include <imdb.hpp>




Imdb::Imdb() {
}



int Imdb::load(const std::string & filename) {
    std::ifstream in(filename.c_str());

    if (!in.good()) {
        return -1;
    }

    records.clear();

    ImdbRecord record;
    while (in >> record.pathToSourceImage >> record.pathToLabelImage) {
        records.push_back(record);
    }

    return records.size();
}



cv::Mat Imdb::getImage(size_t index) {
    return cv::imread(records.at(index).pathToSourceImage);
}



cv::Mat Imdb::getLabel(size_t index) {
    return cv::imread(records.at(index).pathToLabelImage);
}



std::string Imdb::dump() {
    std::stringstream out;

    for (std::vector<ImdbRecord>::const_iterator it = records.begin();
         it != records.end(); ++it) {
        out << it->pathToLabelImage << " " <<
            it->pathToSourceImage << std::endl;
    }

    return out.str();
}

