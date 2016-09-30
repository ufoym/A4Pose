#include "opencv2/opencv.hpp"
#include "charuco.hpp"


int main()
{
	// ------------------------------------------------------------------------
	// settings

	cv::Size frame_size(1280, 720);
	const int board_pad = 100;
	const int squares_x = 5;
	const int squares_y = 8;
	const float square_length = 0.04f;
	const float marker_length = 0.02f;
	const int dictionary_id = cv::aruco::DICT_6X6_250;

	// ------------------------------------------------------------------------
	// make board

	cv::Ptr<cv::aruco::Dictionary> dictionary =
		cv::aruco::getPredefinedDictionary(
		cv::aruco::PREDEFINED_DICTIONARY_NAME(dictionary_id));

	cv::Ptr<cv::aruco::CharucoBoard> board =
		cv::aruco::CharucoBoard::create(
		squares_x, squares_y, square_length, marker_length, dictionary);

	// ------------------------------------------------------------------------
	// make board image for print

	cv::Mat board_img;
	board->draw(cv::Size(2480, 3508), board_img, board_pad);
	cv::imwrite("board.png", board_img);

	// ------------------------------------------------------------------------
	// setup camera

	cv::VideoCapture cap(1);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, frame_size.width);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, frame_size.height);
	if (!cap.isOpened())
		return -1;

	// ------------------------------------------------------------------------
	// main loop

	std::vector< std::vector< std::vector<cv::Point2f> > > all_corners;
	std::vector< std::vector<int> > all_ids;
	std::vector< cv::Mat > all_imgs;
	cv::Mat frame;

	for (;;) {
		cap >> frame;

		std::vector< int > ids;
		std::vector< std::vector< cv::Point2f > > corners, rejected;
		cv::aruco::detectMarkers(frame, dictionary, corners, ids,
			cv::aruco::DetectorParameters::create(), rejected);
		cv::aruco::refineDetectedMarkers(frame,
			board.staticCast<cv::aruco::Board>(), corners, ids, rejected);

		cv::Mat vis;
		frame.copyTo(vis);
		if (ids.size() > 0) {
			cv::aruco::drawDetectedMarkers(
				vis, corners, ids, cv::Scalar(0, 255, 255));
		}
		cv::putText(vis,
			"Press 'c' to add current frame. 'ESC' to finish and calibrate",
			cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,
			cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
		cv::imshow("vis", vis);

		const char key = cv::waitKey(30);
		if (key == 27) {
			break;
		}
		else if (key == 'c' && ids.size() > 0) {
			std::cout << "Frame captured" << std::endl;
			all_corners.push_back(corners);
			all_ids.push_back(ids);
			all_imgs.push_back(frame);
		}
	}
	return 0;
}