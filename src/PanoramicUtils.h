
#ifndef LAB4__PANORAMIC__UTILS__H
#define LAB4__PANORAMIC__UTILS__H

#include <memory>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/stitching.hpp>
#include "opencv2/xfeatures2d.hpp"

class PanoramicUtils
{
public:
    static cv::Mat cylindricalProj(const cv::Mat& image, const double angle)
    {
        cv::Mat tmp, result;
        cv::Mat results[3];
        cv::Mat channels[3];
        cv::split(image, channels);
        int j = 0;
        for (auto tmp : channels) {
            result = tmp.clone();

            double alpha(angle / 180 * CV_PI);
            double d((image.cols / 2.0) / tan(alpha));
            double r(d / cos(alpha));
            double d_by_r(d / r);
            int half_height_image(image.rows / 2);
            int half_width_image(image.cols / 2);

            for (int x = -half_width_image + 1,
                x_end = half_width_image; x < x_end; ++x)
            {
                for (int y = -half_height_image + 1,
                    y_end = half_height_image; y < y_end; ++y)
                {
                    double x1(d * tan(x / r));
                    double y1(y * d_by_r / cos(x / r));

                    if (x1 < half_width_image &&
                        x1 > -half_width_image + 1 &&
                        y1 < half_height_image &&
                        y1 > -half_height_image + 1)
                    {
                        result.at<uchar>(y + half_height_image, x + half_width_image)
                            = tmp.at<uchar>(round(y1 + half_height_image),
                                round(x1 + half_width_image));
                    }
                }
            }
            results[j] = result; j++;
        }

        cv::merge(results,3,result);
        return result;
    }
};

#endif // LAB4__PANORAMIC__UTILS__H
