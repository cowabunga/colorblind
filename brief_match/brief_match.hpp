#ifndef __OPENCV_BRIEF_MATCH_HPP__
#define __OPENCV_BRIEF_MATCH_HPP__

#include <stdlib.h>

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

template<class T, int N>
class RandomSampleGenerator
{
  public:
    RandomSampleGenerator(const T& min_val, const T& max_val, bool _different = false):
      minValue(min_val), maxValue(max_val), different(_different) {
        ++(*this);
    }

    const RandomSampleGenerator& operator ++ () {
      T v = minValue;
      bool found = false;

      for (int j = 0, i = 0; i < N;) {
        v = (T)(minValue + rand()*(maxValue*1.0/RAND_MAX));
        found = false;
        if (different)
          for (j = i-1; j>=0; --j)
            if (v == values[j]) {
              found = true;
              break;
            }
        if (!found)
          values[i++] = v;
      }

      return *this;
    }

    const T& operator [] (int n) { return values[n]; }

  private:
    const T minValue, maxValue;
    T values[N];
    bool different;
};

#endif //__OPENCV_BRIEF_MATCH_HPP__
