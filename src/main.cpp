#include "opencv2/opencv.hpp"
#include "charuco.hpp"


bool calibrate(
	const std::string filename,
	const cv::Size frame_size = cv::Size(1280, 720),
	const int frame_margin = 10,
	const float aspect_ratio = 1.0f,
	const int board_pad = 10,
	const int squares_x = 5,
	const int squares_y = 8,
	const float square_length = 0.04f,
	const float marker_length = 0.02f,
	const int dictionary_id = cv::aruco::DICT_6X6_250)
{
	// ------------------------------------------------------------------------
	// make board

	cv::Ptr<cv::aruco::Dictionary> dictionary =
		cv::aruco::getPredefinedDictionary(
		cv::aruco::PREDEFINED_DICTIONARY_NAME(dictionary_id));

	cv::Ptr<cv::aruco::CharucoBoard> ch_board =
		cv::aruco::CharucoBoard::create(
		squares_x, squares_y, square_length, marker_length, dictionary);
	cv::Ptr<cv::aruco::Board> board = ch_board.staticCast<cv::aruco::Board>();

	// ------------------------------------------------------------------------
	// make board image for print

	cv::Mat board_img;
	ch_board->draw(cv::Size(2480, 3508), board_img, board_pad);
	cv::imwrite("board.png", board_img);

	// ------------------------------------------------------------------------
	// setup camera

	cv::VideoCapture cap(1);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, frame_size.width);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, frame_size.height);
	if (!cap.isOpened())
		return false;

	// ------------------------------------------------------------------------
	// main loop

	std::vector< std::vector< std::vector<cv::Point2f> > > all_corners;
	std::vector< std::vector<int> > all_ids;
	std::vector< cv::Mat > all_imgs;
	cv::Mat frame;

	for (int i = 0, prev_i = -frame_margin;; ++i) {
		cap >> frame;

		std::vector< int > ids;
		std::vector< std::vector< cv::Point2f > > corners, rejected;
		cv::aruco::detectMarkers(frame, dictionary, corners, ids,
			cv::aruco::DetectorParameters::create(), rejected);
		cv::aruco::refineDetectedMarkers(frame, board, corners, ids, rejected);

		cv::Mat vis;
		frame.copyTo(vis);
		if (ids.size() > 0) {
			cv::aruco::drawDetectedMarkers(
				vis, corners, ids, cv::Scalar(0, 255, 255));
		}
		cv::putText(vis,
			"Press 'ESC' to finish and calibrate",
			cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.4,
			cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
		cv::imshow("vis", vis);

		const char key = cv::waitKey(30);
		if (key == 27) {
			break;
		}
		else if (ids.size() > 0 && i - prev_i >= frame_margin) {
			all_corners.push_back(corners);
			all_ids.push_back(ids);
			all_imgs.push_back(frame);
			prev_i = i;
			std::cout << "Frame captured #" << all_imgs.size() << std::endl;
		}
	}

	// ------------------------------------------------------------------------
	// camera calibration

	if (all_ids.size() < 1) {
		std::cerr << "Not enough captures for calibration" << std::endl;
		return false;
	}
	cv::Mat camera_matrix, dist_coeffs;
	std::vector< cv::Mat > rvecs, tvecs;

	// prepare data for calibration
	std::vector< std::vector< cv::Point2f > > allCornersConcatenated;
	std::vector< int > allIdsConcatenated;
	std::vector< int > markerCounterPerFrame;
	markerCounterPerFrame.reserve(all_corners.size());
	for (unsigned int i = 0; i < all_corners.size(); i++) {
		markerCounterPerFrame.push_back((int)all_corners[i].size());
		for (unsigned int j = 0; j < all_corners[i].size(); j++) {
			allCornersConcatenated.push_back(all_corners[i][j]);
			allIdsConcatenated.push_back(all_ids[i][j]);
		}
	}

	double arucoRepErr = cv::aruco::calibrateCameraAruco(
		allCornersConcatenated, allIdsConcatenated,
		markerCounterPerFrame, board, frame_size,
		camera_matrix, dist_coeffs);

	int nFrames = (int)all_corners.size();
	std::vector< cv::Mat > allCharucoCorners;
	std::vector< cv::Mat > allCharucoIds;
	allCharucoCorners.reserve(nFrames);
	allCharucoIds.reserve(nFrames);

	for (int i = 0; i < nFrames; i++) {
		// interpolate using camera parameters
		cv::Mat currentCharucoCorners, currentCharucoIds;
		cv::aruco::interpolateCornersCharuco(
			all_corners[i], all_ids[i], all_imgs[i], ch_board,
			currentCharucoCorners, currentCharucoIds,
			camera_matrix, dist_coeffs);

		const int num = currentCharucoCorners.size().height;
		if (num > 4) {
			allCharucoCorners.push_back(currentCharucoCorners);
			allCharucoIds.push_back(currentCharucoIds);
		}
	}

	if (allCharucoCorners.size() < 4) {
		std::cerr << "Not enough corners for calibration" << std::endl;
		return false;
	}

	double rep_error = cv::aruco::calibrateCameraCharuco(
		allCharucoCorners, allCharucoIds, ch_board, frame_size,
		camera_matrix, dist_coeffs, rvecs, tvecs, 0);

	// ------------------------------------------------------------------------
	// save camera parameters

	cv::FileStorage fs(filename, cv::FileStorage::WRITE);
	if (!fs.isOpened()) {
		std::cerr << "Cannot save output file" << std::endl;
		return false;
	}
	fs << "image_width" << frame_size.width;
	fs << "image_height" << frame_size.height;
	fs << "camera_matrix" << camera_matrix;
	fs << "distortion_coefficients" << dist_coeffs;
	fs << "avg_reprojection_error" << rep_error;
	fs.release();
	return true;
}


bool estimate(
	const cv::Mat & camera_matrix, 
	const cv::Mat & dist_coeffs,
	const cv::Size frame_size = cv::Size(1280, 720),
	const int frame_margin = 10,
	const float aspect_ratio = 1.0f,
	const int board_pad = 10,
	const int squares_x = 5,
	const int squares_y = 8,
	const float square_length = 0.04f,
	const float marker_length = 0.02f,
	const int dictionary_id = cv::aruco::DICT_6X6_250)
{
	float axisLength = 0.5f * (std::min(squares_x, squares_y) * (square_length));

	// ------------------------------------------------------------------------
	// make board

	cv::Ptr<cv::aruco::Dictionary> dictionary =
		cv::aruco::getPredefinedDictionary(
		cv::aruco::PREDEFINED_DICTIONARY_NAME(dictionary_id));

	cv::Ptr<cv::aruco::CharucoBoard> ch_board =
		cv::aruco::CharucoBoard::create(
		squares_x, squares_y, square_length, marker_length, dictionary);
	cv::Ptr<cv::aruco::Board> board = ch_board.staticCast<cv::aruco::Board>();

	// ------------------------------------------------------------------------
	// setup camera

	cv::VideoCapture cap(1);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, frame_size.width);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, frame_size.height);
	if (!cap.isOpened())
		return false;

	// ------------------------------------------------------------------------
	// main loop

	cv::Mat frame;

	for (int i = 0, prev_i = -frame_margin;; ++i) {
		cap >> frame;

		std::vector< int > ids, charucoIds;
		std::vector< std::vector< cv::Point2f > > corners, rejected;
		std::vector< cv::Point2f > charucoCorners;
		cv::aruco::detectMarkers(frame, dictionary, corners, ids,
			cv::aruco::DetectorParameters::create(), rejected);
		cv::aruco::refineDetectedMarkers(frame, board, corners, ids, rejected);

		int num_corners = 0;
		if (ids.size() > 0)
			num_corners = cv::aruco::interpolateCornersCharuco(
			corners, ids, frame, ch_board, charucoCorners,
			charucoIds, camera_matrix, dist_coeffs);

		cv::Vec3d rvec, tvec;
		bool validPose = false;
		if (camera_matrix.total() != 0) {
			validPose = cv::aruco::estimatePoseCharucoBoard(
				charucoCorners, charucoIds, ch_board,
				camera_matrix, dist_coeffs, rvec, tvec);
		}

		cv::Mat vis;
		frame.copyTo(vis);
		if (num_corners > 0)
			cv::aruco::drawDetectedCornersCharuco(
				vis, charucoCorners, charucoIds, cv::Scalar(0, 255, 255));
		if (validPose)
			cv::aruco::drawAxis(
			vis, camera_matrix, dist_coeffs, rvec, tvec, axisLength);
		cv::imshow("vis", vis);
		cv::waitKey(10);
	}

	return true;
}


int main()
{
	std::string filename = "camera.yml";
	if (!std::ifstream(filename.c_str()).good()) {
		if (!calibrate(filename)) {
			std::cerr << "Cannot calibrate the camera" << std::endl;
			return -1;
		}
	}

	cv::FileStorage fs(filename, cv::FileStorage::READ);
	if (!fs.isOpened()) {
		std::cerr << "Cannot open calibration file" << std::endl;
		return -1;
	}
	cv::Size frame_size;
	cv::Mat camera_matrix, dist_coeffs;
	fs["camera_matrix"] >> camera_matrix;
	fs["distortion_coefficients"] >> dist_coeffs;
	fs.release();

	std::cout << "camera_matrix: " << camera_matrix << std::endl;
	std::cout << "dist_coeffs: " << dist_coeffs << std::endl;

	estimate(camera_matrix, dist_coeffs);

	return 0;
}