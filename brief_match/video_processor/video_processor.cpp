/*
 * Brief matcher
 *      Author: CommanderDuck
 */
#include "photo_match.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <vector>
#include <iostream>
#include <algorithm>
#include <memory>

using namespace cv;

class VideoProcessor {
private:
  // the OpenCV video capture object
  cv::VideoCapture capture;
  // the callback function to be called
  // for the processing of each frame
  void (*process)(cv::Mat&, cv::Mat&);
  // a bool to determine if the
  // process callback will be called
  bool callIt;
  // Input display window name
  std::string windowNameInput;
  // Output display window name
  std::string windowNameOutput;
  // delay between each frame processing
  int delay;
  // number of processed frames
  long fnumber;
  // stop at this frame number
  long frameToStop;
  // to stop the processing
  bool stop;

public:
  VideoProcessor() : callIt(true), delay(0),
    fnumber(0), stop(false), frameToStop(-1) {}

  // set the callback function that
  // will be called for each frame
  void setFrameProcessor(
    void (*frameProcessingCallback)
    (cv::Mat&, cv::Mat&)) {
      process= frameProcessingCallback;
  }

  // set the name of the video file
  bool setInputFile(std::string filename) {
    fnumber= 0;
    // In case a resource was already
    // associated with the VideoCapture instance
    capture.release();
    //images.clear();
    // Open the video file
    return capture.open(filename);
    if (!capture.isOpened())
      return 1;
  }

  // set the name of the video file
  bool setInputCamera(int device) {
    fnumber= 0;
    // In case a resource was already
    // associated with the VideoCapture instance
    capture.release();
    //images.clear();
    // Open the video file
    return capture.open(device);
    if (!capture.isOpened())
      return 1;
  }

  // to display the processed frames
  void displayInput(std::string wn) {
    windowNameInput= wn;
    cv::namedWindow(windowNameInput);
  }

  // to display the processed frames
  void displayOutput(std::string wn) {
    windowNameOutput= wn;
    cv::namedWindow(windowNameOutput);
  }

  // do not display the processed frames
  void dontDisplay() {
    cv::destroyWindow(windowNameInput);
    cv::destroyWindow(windowNameOutput);
    windowNameInput.clear();
    windowNameOutput.clear();
  }

  // to grab (and process) the frames of the sequence
  void run() {
    // current frame
    cv::Mat frame;
    // output frame
    cv::Mat output;
    // if no capture device has been set
    if (!isOpened())
      return;
    stop= false;
    while (!isStopped()) {
      // read next frame if any
      if (!readNextFrame(frame))
        break;
      // display input frame
      if (windowNameInput.length()!=0)
        cv::imshow(windowNameInput,frame);
      // calling the process function
      if (callIt) {
        // process the frame
        process(frame, output);
        // increment frame number
        fnumber++;
      } else {
        output= frame;
      }
      // display output frame
      if (windowNameOutput.length()!=0)
        cv::imshow(windowNameOutput,output);
      // introduce a delay
      if (delay>=0 && cv::waitKey(delay)>=0)
        stopIt();
      // check if we should stop
      if (frameToStop>=0 &&
        getFrameNumber()==frameToStop)
        stopIt();
    }
  }
  // Stop the processing
  void stopIt() {
    stop= true;
  }
  // Is the process stopped?
  bool isStopped() {
    return stop;
  }
  // Is a capture device opened?
  bool isOpened() {
    return capture.isOpened();
  }
  // set a delay between each frame
  // 0 means wait at each frame
  // negative means no delay
  void setDelay(int d) {
    delay= d;
  }

  // to get the next frame
  // could be: video file or camera
  bool readNextFrame(cv::Mat& frame) {
    return capture.read(frame);
  }

  // process callback to be called
  void callProcess() {
    callIt= true;
  }

  // do not call process callback
  void dontCallProcess() {
    callIt= false;
  }

  void stopAtFrameNo(long frame) {
    frameToStop= frame;
  }

  // return the frame number of the next frame
  long getFrameNumber() {
    // get info of from the capture device
    long fnumber= static_cast<long>(
      capture.get(CV_CAP_PROP_POS_FRAMES));
    return fnumber;
  }
};

void canny(cv::Mat& img, cv::Mat& out) {
  // Convert to gray
  if (img.channels()==3)
    cv::cvtColor(img,out,CV_BGR2GRAY);
  // Compute Canny edges
  cv::Canny(out,out,100,200);
  // Invert the image
  cv::threshold(out,out,128,255,cv::THRESH_BINARY_INV);
}

cv::Mat image;
cv::Mat image_label;

void panny(cv::Mat& img, cv::Mat& out) {
  if (!matchImagesAndPutLabel(image, image_label, img, out)) {
    out = img;
  } else {
    std::cout << "Matched!" << std::endl;
  }
  //cv::Canny(out,out,100,200);
  //cv::threshold(out,out,128,255,cv::THRESH_BINARY_INV);
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " image image_label" << std::endl;
    return 1;
  }
  const char* image_fname = argv[1];
  const char* image_label_fname = argv[2];

  image = cv::imread(image_fname);
  image_label = cv::imread(image_label_fname);

  // Create instance
  VideoProcessor processor;
  // Open video file
  processor.setInputCamera(0);
  /*processor.setInputFile(argv[1]);*/
  // Declare a window to display the video
  processor.displayInput("Current Frame");
  processor.displayOutput("Output Frame");
  // Play the video at the original frame rate
  processor.setDelay(1000/24);
  // Set the frame processor callback function
  processor.setFrameProcessor(panny);
  // Start the process
  processor.run();

  return 0;
}
