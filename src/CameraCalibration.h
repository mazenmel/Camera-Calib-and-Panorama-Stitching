/*
 * [2020] Computer vision course - lab 04
 * All Rights Reserved.
 *
 * @Author Mazen Mel 1237873
*/

#ifndef H_CAMERACALIBRATION
#define H_CAMERACALIBRATION
#include <opencv2/opencv.hpp>

class CameraCalibration
{
public:
	CameraCalibration(const std::string& Path, int cols,int rows, float square_size);
	int CalibrateCamera();
    cv::Mat GetCameraMatrix();
    cv::Mat GetRotation();
    cv::Mat GetTranslation();
    cv::Mat GetDistCoeffs();

private:
    std::string imagesPath;
    // Checker Board spesifications
    int cornersPerCol ;   // Number of corners per column
    int cornersPerRow;   // Number of corners per row
    float squareSize;   // square size in meters
     // Corner positions in the 3D world coordinates
    
    // Calibration output matrices
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    cv::Mat R;
    cv::Mat T;
};
#endif //H_CAMERACALIBRATION
