/*
 * [2020] Computer vision course - lab 04
 * All Rights Reserved.
 *
 * @Author Mazen Mel 1237873
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include "CameraCalibration.h"
#include "PanoramicImage.h"

int main(int argc, char* argv[])
{
    char c;
    std::cout << "Calibrate camera? [y/n]: "; 
    std::cin >> c; 
    if (c == 'y') {
        std::string path;
        int col, row; 
        float dim;
        float sensorWidth;
        int imageWidth;

        std::cout << "Enter checker board images path: ";
        std::cin >> path;
        std::cout << "Enter camera sensor width in mm: " ;
        std::cin >> sensorWidth;
        std::cout << "Enter image width in pixels: " ;
        std::cin >> imageWidth;
        std::cout << "Enter number of squares in columns (x): " ;
        std::cin >> col;
        std::cout << "Enter number of squares in rows (y): " ;
        std::cin >> row;
        std::cout << "Enter the dimension of the square in m: " ;
        std::cin >> dim;

        CameraCalibration calibrator(path, col, row, dim);
        calibrator.CalibrateCamera();
        cv::Mat cameraMatrix = calibrator.GetCameraMatrix();
        std::cout << "CAMERA MATRIX: " << cameraMatrix << std::endl;
        std::cout << "DISTORTION PARAMS: " << calibrator.GetDistCoeffs() << std::endl;
        
        double fmm = cameraMatrix.at<double>(0, 0) * sensorWidth / imageWidth;
        double FOV = ((2*std::atan((sensorWidth/2)/ fmm))*180)/CV_PI;
        std::cout << "FOV in degrees: " << FOV << std::endl;
    }
    else {
        std::string path;
        std::string descriptor;
        double FOV;
        std::cout << "Enter images path: ";
        std::cin >> path;
        std::cout << "Enter feature extractor [sift/surf/orb/fast]: ";
        std::cin>> descriptor;
        std::cout << "Enter FOV in degrees: ";
        std::cin >> FOV;
        PanoramicImage pan_img{ path,descriptor };
        pan_img.loadImages();
        pan_img.cylindricalProj(FOV/2);
        pan_img.getCylindricalPanorama();
        pan_img.getHomographyPanorama();
    }
        
/*Calibrate camera? [y/n]: y
Enter checker board images path: ./
Enter camera sensor width in mm: 4.82
Enter image width in pixels: 800
Enter number of squares in columns (x): 8
Enter number of squares in rows (y): 6
Enter the dimension of the square in m: 0.25
Calibrating camera... Please wait, it might take few minutes.
Reprojection_error: 0.216351
CAMERA MATRIX: [633.0763988394009, 0, 409.8696820336026;
 0, 635.606648562585, 285.4572865183902;
 0, 0, 1]
DISTORTION PARAMS: [0.1484416334674381, -0.3237594309015891, -0.00758885884979011, 0.005171430486838837, 0.2310585451637548]
FOV in degrees: 64.5723*/
}
