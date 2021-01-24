/*
 * [2020] Computer vision course - lab 04
 * All Rights Reserved.
 *
 * @Author Mazen Mel 1237873
*/

#ifndef H_PANORAMICIMAGE
#define H_PANORAMICIMAGE
#include <opencv2/opencv.hpp>


class PanoramicImage
{
    /* Class PanoramicImage performs panorama stitching of a set of  input images */
public:
    // Class constructor
    PanoramicImage(const std::string& path, const std::string& extractor);

    // Class methods
    cv::Mat cropPano(cv::Mat ref);
    void loadImages();
    void cylindricalProj(const double angle);
    void displayImages(bool projected);
    std::vector<std::pair<std::vector<cv::KeyPoint>, cv::Mat>>  extractFeatures(std::vector<cv::Mat> listImg);

    std::pair< std::vector<cv::Point2f>, std::vector<cv::Point2f>> matchImages(cv::Mat descriptor1, 
                                                                                cv::Mat descriptor2, 
                                                                                std::vector<cv::KeyPoint> keyPoints1, 
                                                                                std::vector<cv::KeyPoint> keyPoints2);
    int getCylindricalPanorama();
    int getHomographyPanorama();
    cv::Mat blendImages( cv::Mat img1,  cv::Mat img2,cv::Mat H);
private:
    std::string imagesPath;
    std::vector<cv::Mat> images;
    std::string featurExtractor;
    std::vector<cv::Mat> projectedImages;
};
#endif //H_PANORAMICIMAGE
