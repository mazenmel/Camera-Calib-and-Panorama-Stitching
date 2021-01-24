/*
 * [2020] Computer vision course - lab 04
 * All Rights Reserved.
 *
 * @Author Mazen Mel 1237873
*/

#include "CameraCalibration.h"

CameraCalibration::CameraCalibration(const std::string& Path, int cols, int rows, float square_size) :imagesPath{ Path },
                                                                                                cornersPerRow{ cols },
                                                                                                cornersPerCol{ rows }, 
                                                                                                squareSize{ square_size }{};

int CameraCalibration::CalibrateCamera() {

    int G_ROWS;
    int G_COLS;
    std::vector<std::vector<cv::Point2f>> corners_2d;
    std::vector<cv::Point2f> corners;
    std::vector < std::string > images;
    cv::Mat gray_img;
    std::vector<std::vector<cv::Point3f>> corners_3d;
    cv::glob(imagesPath, images);
   
    for (int i = 0; i <  images.size(); ++i) {
        std::vector<cv::Point3f> temp_vec;
        for (int c = 0; c < cornersPerRow; ++c) {
            for (int r = 0; r < cornersPerCol; ++r) {
                cv::Point3f temp_sub_vec{ c * squareSize, r * squareSize, 0.0f };
                temp_vec.push_back(temp_sub_vec);
            }

        }
        corners_3d.push_back(temp_vec);
    }
   
    for (int i = 0; i < images.size(); i++) {
        cv::Mat image = cv::imread(images[i]);
        if (image.empty()) {
            std::cout << "ERROR cannot read the image" << std::endl;
            return 0;
        }
        // image dimensions
        G_ROWS = image.rows;
        G_COLS = image.cols;

        cv::cvtColor(image, gray_img, cv::COLOR_RGB2GRAY);

        bool is_found = cv::findChessboardCorners(gray_img, cv::Size(cornersPerCol, cornersPerRow), corners, cv::CALIB_CB_FAST_CHECK);

        if (is_found) {

            //sub pixel refinment
            cv::TermCriteria criteria(cv::TermCriteria::Type::EPS + cv::TermCriteria::Type::MAX_ITER, 30, 0.0001);
            cv::cornerSubPix(gray_img, corners, cv::Size(5, 5), cv::Size(-1, -1), criteria);
            corners_2d.push_back(corners);
            
           
        }
        else {
            std::cout << "ERROR could not find corners" << std::endl;
            return 0;
        }
        
    }
        std::cout << "Calibrating camera... Please wait, it might take few minutes." << std::endl;
        double reprojection_error = cv::calibrateCamera(corners_3d,corners_2d, cv::Size(G_COLS, G_ROWS),
                                                cameraMatrix,distCoeffs, R, T);
        std::cout << "Reprojection_error: " << reprojection_error << std::endl;
        return 1;
}

cv::Mat CameraCalibration::GetCameraMatrix() { return cameraMatrix; }

cv::Mat CameraCalibration::GetRotation() { return R; }

cv::Mat CameraCalibration::GetTranslation() { return T; }

cv::Mat CameraCalibration::GetDistCoeffs() { return distCoeffs; }