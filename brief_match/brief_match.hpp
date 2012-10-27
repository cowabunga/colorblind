#ifndef __OPENCV_BRIEF_MATCH_HPP__
#define __OPENCV_BRIEF_MATCH_HPP__

#include "opencv2/core/core.hpp"

class TimeMeter
{    
    private:
             double startTime;
             double totalTime;
               bool verbos;

    public:

           TimeMeter( bool start_now = false, bool verbos = false );

      void start();
    double stop(const char *measure_name = "time measure");
    double getTotalTime() { return totalTime; }
};

#endif //__OPENCV_BRIEF_MATCH_HPP__
