#include "opencv2/opencv.hpp"

int main()
{
	cv::VideoCapture cap(1);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
	if (!cap.isOpened())
		return -1;

	cv::Mat frame;
	for (;;) {
		cap >> frame;
		cv::flip(frame, frame, 1);
		cv::imshow("frame", frame);
		if (cv::waitKey(30) == 27) 
			break;
	}
	return 0;
}