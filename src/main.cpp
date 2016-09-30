#include "opencv2/opencv.hpp"
#include "charuco.hpp"

cv::Ptr<cv::aruco::CharucoBoard> make_board(
	const bool write = false,
	const int board_pad = 100,
	const int squares_x = 5,
	const int squares_y = 8,
	const float square_length = 0.04f,
	const float marker_length = 0.02f,
	const int dictionary_id = cv::aruco::DICT_6X6_250)
{
	cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(
		cv::aruco::PREDEFINED_DICTIONARY_NAME(dictionary_id));

	cv::Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(
		squares_x, squares_y, square_length, marker_length, dictionary);

	if (write) {
		cv::Mat board_img;
		board->draw(cv::Size(2480, 3508), board_img, board_pad);
		cv::imwrite("board.png", board_img);
	}

	return board;
}

int main()
{
	cv::Ptr<cv::aruco::CharucoBoard> board = make_board(true);

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