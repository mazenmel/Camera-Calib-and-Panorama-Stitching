/*
 * [2020] Computer vision course - lab 04
 * All Rights Reserved.
 *
 * @Author Mazen Mel 1237873
*/

/* Note: User defined variables may affect the program execution.
	If errors occur that's likely a descriptor that doesn't work
	properly with the defined thresholds.*/ 

#include "PanoramicImage.h"
#include "PanoramicUtils.h"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/stitching/detail/exposure_compensate.hpp>
#include <chrono>
#include <opencv2/flann.hpp>

PanoramicImage::PanoramicImage(const std::string& path, const std::string& extractor) : imagesPath{ path }, featurExtractor{extractor}{};

cv::Mat PanoramicImage::cropPano(cv::Mat ref) {
	//trim image
	cv::Mat refGray;
	cv::cvtColor(ref, refGray, cv::COLOR_BGR2GRAY);

	// remove the black region from the image (the biggest contour)
	cv::threshold(refGray, refGray, 25, 255, cv::THRESH_BINARY); //Threshold the gray
	std::vector<std::vector<cv::Point> > contours; // Vector for storing contour
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(refGray, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE); // Find the contours in the image
	int largest_area = 0;
	int largest_contour_index = 0;
	cv::Rect bounding_rect;

	for (int i = 0; i < contours.size(); i++) // iterate through each contour. 
	{
		double a = contourArea(contours[i], false);  //  Find the area of contour
		if (a > largest_area) {
			largest_area = a;
			largest_contour_index = i;                //Store the index of largest contour
			bounding_rect = boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
		}

	}

	
	ref = ref(cv::Rect(bounding_rect.x, bounding_rect.y, bounding_rect.width, bounding_rect.height));
	return ref;
}

void PanoramicImage::loadImages() {
	std::vector<std::string> paths;
	cv::glob(imagesPath, paths);
	for (auto i : paths) {
		cv::Mat img;
		img = cv::imread(i);
		//resize images for faster computation
		//cv::resize(img, img, cv::Size(),0.5,0.5);
		images.push_back(img);
	}
	
}

std::pair< std::vector<cv::Point2f>, std::vector<cv::Point2f>> PanoramicImage::matchImages(cv::Mat descriptor1,cv::Mat descriptor2, std::vector<cv::KeyPoint> keyPoints1, std::vector<cv::KeyPoint> keyPoints2) {
	// create the knn matcher
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
	std::vector<std::vector<cv::DMatch>> tmpMatches;
	// find matches between two sets of descriptors
	matcher->knnMatch(descriptor1, descriptor2, tmpMatches, 2);
	// filter matches using the Lowe's ratio test
	const float ratio_thresh = 0.7f; // user defined ratio
	std::vector<cv::DMatch> good_matches;
	for (size_t i = 0; i < tmpMatches.size(); i++)
	{
		if (tmpMatches[i][0].distance < ratio_thresh * tmpMatches[i][1].distance)
		{
			good_matches.push_back(tmpMatches[i][0]);
		}
	}

	//update key points
	std::vector<cv::Point2f> updatedKeypoints1;
	std::vector<cv::Point2f> updatedKeypoints2;
	for (int k = 0; k < good_matches.size(); ++k) {
		updatedKeypoints1.push_back(keyPoints1[good_matches[k].queryIdx].pt);
		updatedKeypoints2.push_back(keyPoints2[good_matches[k].trainIdx].pt);
	}


	// detect inliers in tmpMatches using RANSAC
	std::vector<int> mask;
	std::vector<cv::DMatch> tmpInliers{ good_matches.size() };

	// Compute the homography matrix to get inliers stored in mask
	cv::Mat tmpHomography = cv::findHomography(updatedKeypoints1, updatedKeypoints2, cv::RANSAC, 3, mask);

	for (int l = 0; l < mask.size(); ++l) {
		if (mask[l] == 1) {
			tmpInliers[l] = good_matches[l];
		}

	}
	// update keypoints
	std::vector<cv::Point2f> fupdatedKeypoints1;
	std::vector<cv::Point2f> fupdatedKeypoints2;
	for (int k = 0; k < good_matches.size(); ++k) {
		if (tmpInliers[k].queryIdx >= 0) {
			fupdatedKeypoints1.push_back(keyPoints1[tmpInliers[k].queryIdx].pt);
			fupdatedKeypoints2.push_back(keyPoints2[tmpInliers[k].trainIdx].pt);
		}
	}
	return std::make_pair(fupdatedKeypoints1, fupdatedKeypoints2);
}

void PanoramicImage::cylindricalProj(const double angle) {
	
	for (auto i : images) {
		cv::Mat projection = PanoramicUtils::cylindricalProj(i,angle);
		projectedImages.push_back(projection);


	}
	
}

void PanoramicImage::displayImages(bool projected=true) {
	// debug function to verify projection is correct
	if (projected==true) {
		
		for (auto i : projectedImages) {
			cv::namedWindow("Projected images");
			cv::imshow("Projected images", i);
			cv::waitKey(0);
		}
		
	}
	else {
		
		for (auto i : images) {
			cv::namedWindow(" images");
			cv::imshow(" images", i);
			cv::waitKey(0);
		}
	}

}

cv::Mat PanoramicImage::blendImages( cv::Mat img1,  cv::Mat img2, cv::Mat H) {
	// Laplacian Pyramid Merging technique to seamlessly merge images
	// generate gaussian pyramid for img1 and img2
	cv::Mat g_img1, g_img2;
	img1.copyTo(g_img1);
	img2.copyTo(g_img2);
	std::vector<cv::Mat> gp_img1, gp_img2;
	gp_img1.push_back(g_img1);
	gp_img2.push_back(g_img2);
	for (int i = 0; i < 6; ++i) {
		cv::pyrDown(g_img1, g_img1);
		gp_img1.push_back(g_img1);
		cv::pyrDown(g_img2, g_img2);
		gp_img2.push_back(g_img2);

	}
	// generate Laplacian pyramid for img1 and img2
	std::vector<cv::Mat> lp_img1, lp_img2;
	lp_img1.push_back(gp_img1[5]);
	lp_img2.push_back(gp_img2[5]);
	for (int i = 5; i >0; --i) {
		cv::Mat dst1, sub1, dst2, sub2;
		cv::pyrUp(gp_img1[i], dst1,cv::Size(gp_img1[i - 1].cols, gp_img1[i - 1].rows));
		cv::pyrUp(gp_img2[i], dst2, cv::Size(gp_img2[i - 1].cols, gp_img2[i - 1].rows));
		cv::subtract(gp_img1[i - 1], dst1, sub1);
		cv::subtract(gp_img2[i - 1], dst2, sub2);
		lp_img1.push_back(sub1);
		lp_img2.push_back(sub2);
	}
	// add left and right halves of images in each level
	std::vector<cv::Mat> LS;
	cv::Mat result;
	for (int i = 0; i < lp_img1.size(); ++i) {
		result = lp_img1[i];
		cv::Mat half(result, cv::Rect(0, 0, lp_img2[i].cols, lp_img2[i].rows));
		lp_img2[i].copyTo(half);
		LS.push_back(result);
	}
	// reconstruct input
	cv::Mat ls_= LS[0];
	for (int i = 1; i < 6; ++i) {
		cv::pyrUp(ls_, ls_);
		cv::Rect rct(0, 0, LS[i].cols, LS[i].rows);
		cv::add(ls_(rct), LS[i], ls_);	
	}
	return ls_;

}

std::vector<std::pair<std::vector<cv::KeyPoint>, cv::Mat>> PanoramicImage::extractFeatures(std::vector<cv::Mat> listImg) {
	// feature extraction using SIFT SURF ORB FAST
	std::vector<std::pair<std::vector<cv::KeyPoint>, cv::Mat>> keypointsAndDescriptors;

	if (featurExtractor == "sift") {
		cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create();

		for (auto i : listImg) {
			cv::Mat grayI;
			cv::cvtColor(i, grayI, cv::COLOR_RGB2GRAY);
			std::vector<cv::KeyPoint> keypoints;
			cv::Mat descriptors;
			sift->detect(grayI, keypoints);
			sift->compute(grayI, keypoints, descriptors);
			keypointsAndDescriptors.push_back(std::make_pair(keypoints, descriptors));

		}

	}
	else if (featurExtractor == "orb") {
		cv::Ptr<cv::ORB> orb = cv::ORB::create();
		for (auto i : listImg) {
			cv::Mat grayI;
			cv::cvtColor(i, grayI, cv::COLOR_RGB2GRAY);
			std::vector<cv::KeyPoint> keypoints;
			cv::Mat descriptors;
			orb->detectAndCompute(grayI, cv::Mat(), keypoints, descriptors);
			keypointsAndDescriptors.push_back(std::make_pair(keypoints, descriptors));

		}

	}
	else if (featurExtractor == "surf") {
		cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create();
		for (auto i : listImg) {
			cv::Mat grayI;
			cv::cvtColor(i, grayI, cv::COLOR_RGB2GRAY);
			std::vector<cv::KeyPoint> keypoints;
			cv::Mat descriptors;
			surf->detectAndCompute(grayI, cv::Mat(), keypoints, descriptors);
			keypointsAndDescriptors.push_back(std::make_pair(keypoints, descriptors));
			

		}

	}else if (featurExtractor == "fast") {
		cv::Ptr<cv::xfeatures2d::StarDetector> fast = cv::xfeatures2d::StarDetector::create();
		cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief = cv::xfeatures2d::BriefDescriptorExtractor::create();
		for (auto i : listImg) {
			cv::Mat grayI;
			cv::cvtColor(i, grayI, cv::COLOR_RGB2GRAY);
			std::vector<cv::KeyPoint> keypoints;
			cv::Mat descriptors;
			fast->detect(grayI, keypoints);
			brief->compute(grayI, keypoints, descriptors);
			keypointsAndDescriptors.push_back(std::make_pair(keypoints, descriptors));
			

		}

	}
	return keypointsAndDescriptors;
}

int PanoramicImage::getCylindricalPanorama() {
	// build panorama with cylindrical projection
	auto t1 = std::chrono::high_resolution_clock::now(); // tic
	// extract features
	std::vector<std::pair<std::vector<cv::KeyPoint>, cv::Mat>> featureImages = extractFeatures(projectedImages);

	
	std::vector<std::vector<cv::DMatch>> matches;
	// Initialize the panorama frame to hold images with row margin of 150 and copy image 0 in the left
	cv::Mat finalStitch = cv::Mat(projectedImages[0].rows+150,projectedImages.size()* projectedImages[0].cols,CV_8UC3);
	projectedImages[0].copyTo(finalStitch(cv::Range(75, projectedImages[0].rows+75), cv::Range(0, projectedImages[0].cols)));

	bool tripod = true; // if using tripod
	float tmpAvrgDstx = 0;
	float tmpAvrgDsty = 75;

	// start looping over all images
	for (int i = 0; i < featureImages.size() - 1; ++i) {
		std::vector<cv::KeyPoint> keyPoints1 = featureImages[i].first;
		std::vector<cv::KeyPoint> keyPoints2 = featureImages[i + 1].first;
		cv::Mat descriptor1 = featureImages[i].second;
		cv::Mat descriptor2 = featureImages[i+1].second;
		// specify L2_NORM or HAMMING depending on chosen descriptor
		int norm_type ;
		if (featurExtractor == "orb" || featurExtractor == "fast") {
			norm_type = 6; //hamming
		}
		else {
			norm_type = 4; //l2
		} 
		// create the brute force matcher
		cv::BFMatcher matcher(norm_type);
		std::vector<cv::DMatch> tmpMatches;
		matcher.match(descriptor1, descriptor2, tmpMatches);

		//keep only good matches
		std::vector<float> matchDistances;
		// get match distances
		for (int j = 0; j < tmpMatches.size(); ++j) {
			matchDistances.push_back(tmpMatches[j].distance);	
		}
		// get min distance
		auto minmax = std::minmax_element(matchDistances.begin(), matchDistances.end());
		// specify criteria for discarding matches based on chosen descriptor
		if (featurExtractor == "orb" || featurExtractor == "fast") {
			tmpMatches.erase(std::remove_if(tmpMatches.begin(), tmpMatches.end(),
				[=](cv::DMatch x) {
					return x.distance - float(*minmax.first) > 10; //10 is a user defined variable
				}), tmpMatches.end());
		}
		else {
			tmpMatches.erase(std::remove_if(tmpMatches.begin(), tmpMatches.end(),
				[=](cv::DMatch x) {
					return x.distance  >  5*float(*minmax.first); //5 is a user defined variable
				}), tmpMatches.end());
		}

		//update key points
		std::vector<cv::Point2f> updatedKeypoints1;
		std::vector<cv::Point2f> updatedKeypoints2;
		for (int k = 0; k < tmpMatches.size(); ++k) {
			updatedKeypoints1.push_back(keyPoints1[tmpMatches[k].queryIdx].pt);
			updatedKeypoints2.push_back(keyPoints2[tmpMatches[k].trainIdx].pt);
		}

		matches.push_back(tmpMatches);
		// detect inliers in tmpMatches using RANSAC
		std::vector<int> mask;
		std::vector<std::vector<cv::DMatch>> inliers{ matches.size() };
		std::vector<cv::DMatch> tmpInliers{ tmpMatches.size() };
		
		cv::Mat tmpHomography = cv::findHomography(updatedKeypoints1, updatedKeypoints2, cv::RANSAC,3,mask);

		for (int l = 0; l < mask.size(); ++l) {
			if (mask[l]==1) {
				tmpInliers[l] = tmpMatches[l];
			}
			
		}
		//estimate avrg translation
		int sz = tmpInliers.size();

		std::vector<cv::Point2f> dst(sz);
		for (int l = 0; l < tmpInliers.size(); ++l) {
			if (tmpInliers[l].queryIdx >= 0) {
				dst[l].x = updatedKeypoints1[l].x - updatedKeypoints2[l].x;
				dst[l].y = updatedKeypoints1[l].y - updatedKeypoints2[l].y;

			}
			else {
				sz--;
				dst[l].x = 0.0f;
				dst[l].y = 0.0f;

				
			}
		}
		float avrgDstx = 0;
		float avrgDsty = 0;
		for (int l = 0; l < dst.size();++l)
		{
			avrgDstx += dst[l].x;
			avrgDsty += dst[l].y;
			
			
		}
		avrgDstx /= sz;
		avrgDsty /= sz;
		

	
		tmpAvrgDstx += avrgDstx;
		if(!tripod){ // if not tripod then account for vertical shift
			tmpAvrgDsty += avrgDsty;
		}
		
		projectedImages[i + 1].copyTo(finalStitch(cv::Range(tmpAvrgDsty, tmpAvrgDsty+ projectedImages[i + 1].rows ), cv::Range(tmpAvrgDstx, tmpAvrgDstx + projectedImages[i + 1].cols)));

		// remove seams and color correction
		
		if (tripod) {
			// avrg the pixel values in the overlapping region
			cv::Mat rightOverLap = cv::Mat(projectedImages[i].rows, projectedImages[i].cols - avrgDstx, CV_8UC3);
			rightOverLap = cv::Scalar::all(0);
			for (int y = 75; y < 75 + projectedImages[i].rows; ++y) {
				for (int x = avrgDstx; x < projectedImages[i].cols; ++x) {
					for (int c = 0; c < projectedImages[i].channels(); ++c) {
						rightOverLap.at<cv::Vec3b>(y - 75, x - avrgDstx)[c] = projectedImages[i].at<cv::Vec3b>(y - 75, x)[c];
						+projectedImages[i + 1].at<cv::Vec3b>(y - 75, x - avrgDstx)[c];
					}
				}
			}
			// copy result in the main image
			cv::Mat overlap(finalStitch, cv::Rect(tmpAvrgDstx, 75, rightOverLap.cols, rightOverLap.rows));
			rightOverLap.copyTo(overlap);



		}

		// remove seam line
		for (int y = 0; y < finalStitch.rows; ++y) {
			for (int c = 0; c < finalStitch.channels(); ++c) {
				finalStitch.at<cv::Vec3b>(y, tmpAvrgDstx)[c] = 0.5 * finalStitch.at<cv::Vec3b>(y, tmpAvrgDstx - 1)[c] +
					0.5 * finalStitch.at<cv::Vec3b>(y, tmpAvrgDstx + 1)[c];
			}
		}
		
	
	}



	auto t2 = std::chrono::high_resolution_clock::now(); //toc
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
	std::cout << "panorama built, it took: "<< duration;
	// crop panorama
	cv::Rect resizedFinalStitch(0,75, tmpAvrgDstx +1+ projectedImages[0].cols,finalStitch.rows-100 );

	// gamma correction
	cv::Mat lut_matrix(1, 256, CV_8UC1);
	uchar* ptr = lut_matrix.ptr();
	for (int i = 0; i < 256; i++)
		ptr[i] = (int)(pow((double)i / 255.0, 1/1.2) * 255.0);

	cv::Mat result;
	cv::LUT(finalStitch(resizedFinalStitch), lut_matrix, result);
	cv::namedWindow("Panorama", cv::WindowFlags::WINDOW_NORMAL);
	cv::imshow("Panorama", finalStitch(resizedFinalStitch));
	cv::waitKey(0);
	cv::imwrite("/Panorama.jpg", finalStitch(resizedFinalStitch));
	return 0;
}

int PanoramicImage::getHomographyPanorama() {
	// build panorama with homography matrix estimation
	auto t1 = std::chrono::high_resolution_clock::now();//tic
	
	if (images.size() == 2) { // if only 2 images are present the stitch them directly 
		auto t1 = std::chrono::high_resolution_clock::now();
		// extract features 
		auto features = extractFeatures(projectedImages);
		
		std::vector<cv::KeyPoint> keyPoints1 = features[0].first;
		std::vector<cv::KeyPoint> keyPoints2 = features[1].first;
		cv::Mat descriptor1 = features[0].second;
		cv::Mat descriptor2 = features[1].second;
		// match features and keep only good ones
		auto fupdatedKeypoints = matchImages(descriptor1, descriptor2, keyPoints1, keyPoints2);
		auto fupdatedKeypoints1 = fupdatedKeypoints.first;
		auto fupdatedKeypoints2 = fupdatedKeypoints.second;
		// estimate homography matrix
		cv::Mat tmpHomography = cv::findHomography(fupdatedKeypoints2, fupdatedKeypoints1, cv::RANSAC, 3);
		cv::Mat invHomography = tmpHomography.inv();
		cv::Mat perspective;
		// warp target image
		cv::warpPerspective(projectedImages[1], perspective, tmpHomography, cv::Size(projectedImages[0].cols + projectedImages[1].cols, projectedImages[1].rows));
		// copy warped image with the first one to form the panorama
		cv::Mat half(perspective, cv::Rect(0, 0, projectedImages[0].cols, projectedImages[0].rows));
		projectedImages[0].copyTo(half);
		// crop black region
		perspective = cropPano(perspective);
		auto t2 = std::chrono::high_resolution_clock::now(); //toc
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		std::cout << "panorama built, it took: " << duration;

		cv::namedWindow("Panorama", cv::WindowFlags::WINDOW_NORMAL);
		cv::imshow("Panorama", perspective);
		cv::waitKey(0);
		cv::imwrite("/Panorama.jpg", perspective);
		return 0;

	}
	// if more than 2 images are provided
		// extract features
		auto featureImages = extractFeatures(projectedImages);
		int centerIdx;
		// fix reference image which is the middle one
		if (images.size() % 2 == 0) {
			centerIdx = images.size() / 2;
			centerIdx--;
		}
		else if (images.size() % 2 != 0) {
			centerIdx = std::floor(images.size() / 2);

		}

		cv::Mat ref;
		std::pair<std::vector<cv::KeyPoint>, cv::Mat > refFeatureImage;

		// Start from the central image and stitch hierarchically from left to right

		// get sub set of images to be stitched
		std::vector<cv::Mat> storStitches(projectedImages.begin() + centerIdx, projectedImages.end());
		int areStitched = 0; // # of stitched images
		ref = storStitches[0];
		std::vector<cv::Mat> coupleImages{ ref,storStitches[1] };// stitch a couple at a time
		auto subfeatureImages = extractFeatures(coupleImages);
		int i = 0;
		int size = storStitches.size();
		while (areStitched < size && size > 1 && i < storStitches.size()) {// while not all images are stitched
			// extract and match keypoints
			
			refFeatureImage = subfeatureImages[0];
			std::vector<cv::KeyPoint> keyPoints1 = refFeatureImage.first;
			std::vector<cv::KeyPoint> keyPoints2 = subfeatureImages[1].first;
			cv::Mat descriptor1 = refFeatureImage.second;
			cv::Mat descriptor2 = subfeatureImages[1].second;

			auto fupdatedKeypoints = matchImages(descriptor1, descriptor2, keyPoints1, keyPoints2);
			auto fupdatedKeypoints1 = fupdatedKeypoints.first;
			auto fupdatedKeypoints2 = fupdatedKeypoints.second;
			cv::Mat tmpHomography = cv::findHomography(fupdatedKeypoints2, fupdatedKeypoints1, cv::RANSAC, 3);

			// warp perspective of target image
			cv::Mat perspective;
			cv::warpPerspective(coupleImages[1], perspective, tmpHomography, cv::Size(coupleImages[0].cols + coupleImages[1].cols, coupleImages[1].rows));
			// copy warped image with ref 
			cv::Mat half(perspective, cv::Rect(0, 0, coupleImages[0].cols, coupleImages[0].rows));
			coupleImages[0].copyTo(half);
			// crop result
			perspective = cropPano(perspective);
			// additional crop along cols to remove black region at the end 
			// here the # of pixels to remove (20) depend on the image
			perspective = perspective((cv::Rect(0, 0, perspective.cols - 20, perspective.rows)));
			//insert result at the begining of storStitches
			storStitches.insert(storStitches.begin() + areStitched, perspective);
			areStitched++; // icrmnt # of stitched images
			
			if (areStitched == size - 1 && size > 2) { //if we stitch all images at the previous level move to next one
				// keep only stitched images from first level
				storStitches.erase(storStitches.begin() + areStitched, storStitches.end());
				areStitched = 0; // 0 because new sets of images are to be stitched
				size = storStitches.size();
				i = 0; // 0 because new sets of images are to be stitched
				coupleImages[0] = storStitches[i]; // ref image 
				coupleImages[1] = storStitches[i + 1]; // next image 
				subfeatureImages = extractFeatures(coupleImages);// extract new features 
			}
			else if (areStitched == size - 1 && size >= 2) { // if stitching complete then terminate
				break;
			}
			else { // resume stitching of the current stage
				i++;
				coupleImages.erase(coupleImages.begin(), coupleImages.end());
				coupleImages.insert(coupleImages.begin(), storStitches[i + areStitched]);
				coupleImages.insert(coupleImages.begin() + 1, storStitches[i + areStitched + 1]);
				subfeatureImages = extractFeatures(coupleImages);

			}

		}
		// obtained right half
		cv::Mat rightImg = storStitches[0];



		// Start from the central image and stitch hierarchically from right to left
		// get sub set of images to be stitched
		std::vector<cv::Mat> storLeftStitches(projectedImages.begin(), projectedImages.begin() + centerIdx + 1);
		areStitched = 0;
		ref = storLeftStitches[storLeftStitches.size() - 1];
		std::vector<cv::Mat> coupleLeftImages{ ref,storLeftStitches[storLeftStitches.size() - 2] };// stitch a couple at a time
		subfeatureImages = extractFeatures(coupleLeftImages);
		i = storLeftStitches.size();
		size = storLeftStitches.size();
		while (areStitched <= size && size > 1 && i >= storLeftStitches.size()) {
			// feature extraction and matching
			refFeatureImage = subfeatureImages[0];

			std::vector<cv::KeyPoint> keyPoints1 = refFeatureImage.first;
			std::vector<cv::KeyPoint> keyPoints2 = subfeatureImages[1].first;
			cv::Mat descriptor1 = refFeatureImage.second;
			cv::Mat descriptor2 = subfeatureImages[1].second;

			auto fupdatedKeypoints = matchImages(descriptor1, descriptor2, keyPoints1, keyPoints2);
			auto fupdatedKeypoints1 = fupdatedKeypoints.first;
			auto fupdatedKeypoints2 = fupdatedKeypoints.second;
			cv::Mat tmpHomography = cv::findHomography(fupdatedKeypoints2, fupdatedKeypoints1, cv::RANSAC, 3);

			// warp perspective of target image
			cv::Mat invHomography = tmpHomography.inv();
			cv::Mat perspective;
			cv::warpPerspective(coupleLeftImages[0], perspective, invHomography, cv::Size(coupleLeftImages[0].cols + coupleLeftImages[1].cols, coupleLeftImages[1].rows));
			// copy warped image with ref 
			cv::Mat half(perspective, cv::Rect(0, 0, coupleLeftImages[1].cols, coupleLeftImages[1].rows));
			coupleLeftImages[1].copyTo(half);
			//crop result
			perspective = cropPano(perspective);
			// additional crop along cols to remove black region at the end 
			// here the # of pixels to remove (20) depend on the image
			perspective = perspective((cv::Rect(0, 0, perspective.cols - 20, perspective.rows)));
			//insert result at the begining of storStitches
			storLeftStitches.insert(storLeftStitches.begin(), perspective);
			areStitched++; // incrmt # of stitched images
			if (areStitched == size - 1 && size > 2) {//if we stitch all images at the previous level move to next one
				// keep only stitched images from first level
				storLeftStitches.erase(storLeftStitches.begin() + areStitched, storLeftStitches.end());
				areStitched = 0;
				size = storLeftStitches.size();
				i = storLeftStitches.size();
				coupleLeftImages.erase(coupleLeftImages.begin(), coupleLeftImages.end());
				// specify new ref image and next one
				coupleLeftImages.insert(coupleLeftImages.begin(), storLeftStitches[i - 1]);
				coupleLeftImages.insert(coupleLeftImages.begin() + 1, storLeftStitches[i - 2]);
				// extract new features
				subfeatureImages = extractFeatures(coupleLeftImages);
			}
			else if (areStitched == size - 1 && size <= 2) {// if stitching complete then terminate
				break;
			}
			else { // resume stitching of the current stage

				coupleLeftImages.erase(coupleLeftImages.begin(), coupleLeftImages.end());
				coupleLeftImages.insert(coupleLeftImages.begin(), storLeftStitches[i - areStitched]);
				coupleLeftImages.insert(coupleLeftImages.begin() + 1, storLeftStitches[i - areStitched - 1]);
				subfeatureImages = extractFeatures(coupleLeftImages);

				i++;

			}

		}
	
	// obtained right half
	cv::Mat leftImg = storLeftStitches[0];

	
	// Merge final 2 halves
	std::vector<cv::Mat> finalImages;
	finalImages.push_back(rightImg);
	finalImages.push_back(leftImg);

	featureImages = extractFeatures(finalImages);
	std::vector<cv::KeyPoint> keyPoints1 = featureImages[0].first;
	std::vector<cv::KeyPoint> keyPoints2 = featureImages[1].first;
	cv::Mat descriptor1 = featureImages[0].second;
	cv::Mat descriptor2 = featureImages[1].second;

	auto fupdatedKeypoints = matchImages(descriptor1, descriptor2, keyPoints1, keyPoints2);
	auto fupdatedKeypoints1 = fupdatedKeypoints.first;
	auto fupdatedKeypoints2 = fupdatedKeypoints.second;

	cv::Mat Homography = cv::findHomography(fupdatedKeypoints2, fupdatedKeypoints1, cv::RANSAC, 3);


	cv::Mat invHomography = Homography.inv();
	cv::Mat perspectiveR;
	cv::warpPerspective(rightImg, perspectiveR, invHomography, cv::Size(rightImg.cols+ leftImg.cols, rightImg.rows));

	cv::Mat half(perspectiveR, cv::Rect(0, 0, leftImg.cols, leftImg.rows));
	leftImg.copyTo(half);

	cv::Mat merged = cropPano(perspectiveR);

	auto t2 = std::chrono::high_resolution_clock::now(); //toc 
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
	std::cout << "panorama built, it took: " << duration;

	cv::namedWindow("Panorama", cv::WindowFlags::WINDOW_NORMAL);
	cv::imshow("Panorama", merged);
	cv::waitKey(0);
	cv::imwrite("/Panorama.jpg", merged);
	return 0;

}
