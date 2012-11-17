#ifndef __OPENCV_BRIEF_MATCH_HPP__
#define __OPENCV_BRIEF_MATCH_HPP__

#include <stdlib.h>
#include "opencv2/core/core.hpp"

class VerbosTimer
{    
    private:
             double startTime;
             double totalTime;
               bool verbos;

    public:

           VerbosTimer( bool start_now = false, bool verbos = false );

      void start();
    double stop(const char *measure_name = "time measure");
    double getTotalTime() { return totalTime; }
};

template<class T, int N>
class SampleGenerator
{
  public:
    SampleGenerator(const T& min_val, const T& max_val):
      minValue(min_val), maxValue(max_val), _empty(true) {
    }

    virtual bool operator ++ () = 0;
    const T& operator [] (int n) const { return values[n]; }
    bool empty() const { return _empty; }

  protected:
    const T minValue, maxValue;
    T values[N];
    bool _empty;
};

template<int N, bool different>
class RandomSampleGenerator: public SampleGenerator<int, N>
{
  typedef SampleGenerator<int, N> TParent;

  public:
    RandomSampleGenerator(const int min_val, const int max_val):
      TParent(min_val, max_val) {
        int num_variants = max_val - min_val + 1;
        TParent::_empty = (different && num_variants < N);
        ++(*this);
    }

    virtual bool operator ++ () {
      if (TParent::_empty)
        return false;

      int v, num_variants = TParent::maxValue - TParent::minValue + 1;
      bool found = false;

      for (int j = 0, i = 0; i < N;) {
        v = TParent::minValue + rand() % num_variants;
        found = false;
        if (different)
          for (j = i-1; j>=0; --j)
            if (v == TParent::values[j]) {
              found = true;
              break;
            }
        if (!found)
          TParent::values[i++] = v;
      }
      return true;
    }
};

template<class T, int N>
class AllSampleGenerator: public SampleGenerator<T, N>
{
  typedef SampleGenerator<T, N> TParent;

  public:
    AllSampleGenerator(const T& min_val, const T& max_val):
        TParent(min_val, max_val) {
      Init();
    }

    void Init() {
      if (TParent::maxValue - TParent::minValue >= N-1) {
        for (int i = 0; i < N; ++i)
          TParent::values[i] = TParent::minValue + i;
        TParent::_empty = false;
      } else
        TParent::_empty = true;
    }

    virtual bool operator ++ () {
      if (TParent::_empty)
        return false;
      for (int i = N-1; i >= 0; --i) {
        if (TParent::values[i] <= TParent::maxValue - N + i) {
          T value = ++TParent::values[i];
          for (int j = i+1; j < N; ++j)
            TParent::values[j] = ++value;
          return true;
        }
      }
      TParent::_empty = true;
      return false;
    }
};

extern bool matchImagesAndPutLabel(const cv::Mat& img1,
                                   const cv::Mat& img1text,
                                   const cv::Mat& img2,
                                   cv::Mat& out,
                                   bool debug = false);

#endif //__OPENCV_BRIEF_MATCH_HPP__
