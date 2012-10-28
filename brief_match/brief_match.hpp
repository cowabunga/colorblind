#ifndef __OPENCV_BRIEF_MATCH_HPP__
#define __OPENCV_BRIEF_MATCH_HPP__

class TimeMeter {    
 public:
  TimeMeter(bool start_now = false, bool verbos = false);

  void Start();
  double Stop(const char *measure_name = "time measure");
  double GetTotalTime() { return total_time_; }

private:
  double start_time_;
  double total_time_;
  bool verbos_;
};

#endif //__OPENCV_BRIEF_MATCH_HPP__
