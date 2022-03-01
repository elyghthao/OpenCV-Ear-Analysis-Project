// ---------------------------------------------------------------------------
// main.cpp
// Author: Diego Guadarrama, Ely, Fred
// Created: 11/22/2021
// Last Modified: 12/07/2021
// ---------------------------------------------------------------------------
// Purpose: The purpose of Program4 is to use material learned in class and 
// develop a project to showcase what we've learned this quarter. This program
// takes two images "anchor" and a second image named "test" and through 
// several image processing techniques we attempt to first locate the ear in the 
// the test image and second determine whether the ear found in anchor image is the same 
// ear as in the test image. We use several techniques used in object/feature
// recognition and build upon those techniques focusing on key contrainst and 
// are unique to each persons ear. Program4 features include: 
// - allows for image input
// - allows for canny edge detection to be applied to image passed in.
// - allows for k-means algorithm to be applied to image passed in as well as passing
//   custom K values for k-means algorithm. Returns new image with K-Means algo. applied to it.
// - allows for custom cropping at given x,y coordinates of image and 
//   with specified width and height..
// - allows for image to be enhanced (change settings in img: brightness, contrast,
//   blur).
// - allows for image to be displayed in window when program is ran
// - allows for image to be saved to disk (folder)
// - allows for creation of 3D matrix to keep track of where most sift features
//   are found in the test image
// - allows for creation of 3D matrix to keep track of most common color in image 
// - allows for sift feature detection to be performed on image
// - allows for matchTemplate to be performed on given image 
// - allows for grabCut to be perofrmed on passed in image
// - allows for passed in image to be rotated as specified by user
// - allows for image to be shows with shorthand method that takes in name (string) and img file
//   to make debuggin easier and code looking cleaner
// - allows for image calibration to better detect ears at an angle
// Assumptions: 
// - all data is assumed to be correct and correctly formated
// - image is expected to be a jpg with the exception of jpeg for calibration method
// - testing images including "anchor" and "test" must be correctly located in the 
//   same place as main.cpp 
// - images from each category S (close up ear image), M(ear and head visibly shown), and 
//	 L (far away from ear) should be fairly close in image size.
// - OpenCV is expected to be installed correctly and working in program
// - methods called from OpenCv must have valid parameters
// ---------------------------------------------------------------------------

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

#include <stdio.h>
#include <iostream>
using namespace std;
using namespace cv;
 
// store cropped test image not affected by k-means
Mat testCropNoEffects;

// Defining the dimensions of checkerboard
// int CHECKERBOARD[2]{6 ,9};
int CHECKERBOARD[2]{ 9,7 };


// cannyEdgeDetection - performs canny edge detection using OpenCV
// preconditions: Takes valid image type
// postconditions: returns new image of same size with canny edge 
// detection performed
Mat cannyEdgeDetection(const Mat& input) {

	Mat imgGray, imgBlur, imgCanny, imgDil, imgErode;

	// converts image to grayscale  
	cvtColor(input, imgGray, COLOR_BGR2GRAY);

	// applies gaussian blur towards image  
	GaussianBlur(imgGray, imgBlur, Size(5, 5), 2);
	
	// applies canny edge detection to image  
	Canny(imgBlur, imgCanny, 20, 60);

	// return image with canny applied
	return imgCanny;
}

// K_Means - applies k-means algo. to input IMG passed in compatible
// with both color and gray scale image
// preconditions: Valid image and number of desired clusters
// postconditions: returns new copy of image passed in 
// with k-means applied to it with specific K value. 
Mat K_Means(Mat input, int K) {
	// https://www.youtube.com/watch?v=dCZASegkY9g
	// converts iamge data into float data
	Mat samples(input.rows * input.cols, input.channels(), CV_32F); 
	// loop through image values
	for (int y = 0; y < input.rows; y++) {
		for (int x = 0; x < input.cols; x++) {
			// check each channel
			for (int z = 0; z < input.channels(); z++) {
				// if image is colored convert each channel 
				if (input.channels() == 3) {
					samples.at<float>(y + x * input.rows, z) = input.at<Vec3b>(y,x)[z]; 
				}
				// grayscale only has one channel
				else {
					samples.at<float>(y + x * input.rows, z) = input.at<uchar>(y, x); 
				}
			}
		}
	}
	// output variables
	Mat labels; 
	int attempts = 5; 
	Mat centeres; 
	// built-in function from OpenCV for image clustering
	kmeans(samples, K, labels, TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 10, 1.0), attempts, KMEANS_PP_CENTERS, centeres);

	// convert our float data type to unsigned integer type and put in New_Image
	Mat New_Image(input.size(), input.type());
	for (int y = 0; y < input.rows; y++) {
		for (int x = 0; x < input.cols; x++) {
			int cluster_idx = labels.at<int>(y + x * input.rows, 0); 
			if (input.channels() == 3) {
				// color img
				for (int z = 0; z < input.channels(); z++) {
					New_Image.at<Vec3b>(y, x)[z] = centeres.at<float>(cluster_idx, z);
				}
			}
			else {
				// grayscale img
				New_Image.at<uchar>(y , x) = centeres.at<float>(cluster_idx, 0);
			} 
		}
	}
	// return input image with applied k-means to it
	return New_Image; 

}

// cropAt - crops input image and returns new cropped image
// preconditions: valid image with in range x and y coordinates that are
// within image range and passed in width and height from x,y coord. must 
// also not exceed range of input image rows and cols.
// postconditions: returns cropped image with indicated specifications
// from passed in variables
Mat cropAt(const Mat& input, int x, int y, int width, int height) {

	// x y width height
	Rect roi(x, y, width, height); // creates rectangle
	// crops image with rectangle and saves to imgCrop
	Mat imgCrop = input(roi);
	// return smaller cropped image
	return imgCrop;
}

// enhanceImg - applied gaussian blur and changes 
// brightness/contrast if specified or passed in
// preconditions: valid image, with desired blur, brightness, and  
// contrast to be applied.
// postconditions: returns new image with applied gaussian blur, brightness,
// and contrast levels if any were passed in by user 
Mat enhanceImg(const Mat& input, int blur, int brightness, int contrast) {
	
	Mat output = input.clone();
	// brightness levels 1-100
	int iBrightness = brightness - 50;
	// contrast levels 1-100
	double dContrast = contrast / 50.0;
	// apply new contrast and brightness settings (if any)
	input.convertTo(output, -1, dContrast, iBrightness);
	// apply gaussian blur
	GaussianBlur(output, output, Size(blur, blur), 0);
	// return new image with applied features
	return output;
}

// histogramCalcMostVotes_COLOR - gets the most common color in passed in image
// returning coordinates of where the most common bucket resides in the 3D matrix
// preconditions: all passed in values exist and are related to each other
// postconditions: updates passed by ref values to indicate in which bucket in
// the 3D matrix created the most common color resides in
void histogramCalcMostVotes_COLOR(const Mat& input, int &input_r, int &input_g, int &input_b) {

	// create an array of the histogram dimensions
	// size is a constant - the # of buckets in each dimension
	const int size = 4;
	int dims[] = { size, size, size };
	// create 3D histogram of integers initialized to zero	
	Mat hist(3, dims, CV_32S, Scalar::all(0));

	int red, green, blue;
	int bucketSize = 256 / size;

	for (int row = 0; row < input.rows; row++) {
		for (int col = 0; col < input.cols; col++) {

			// get rgb int values from pixel at row,col
			red = input.at<Vec3b>(row, col)[2];		// r
			green = input.at<Vec3b>(row, col)[1];	// g
			blue = input.at<Vec3b>(row, col)[0];	// b

			// calculate correct historgram bucket coordinate 0-3
			bucketSize = 256 / size;
			int r = red / bucketSize;
			int g = green / bucketSize;
			int b = blue / bucketSize;

			// increment votes at historgram bucket
			hist.at<int>(r, g, b)++;
		}
	}

	// store common color rgb value
	int cRed = 0;
	int cGreen = 0;
	int cBlue = 0;

	// store current max to locate common color bin
	int currMax = 0;

	// store common color coordinate for random access later
	int rBucket = 0;
	int gBucket = 0;
	int bBucket = 0;

	// loop through historgram to find bin with the most votes 
	for (int r = 0; r < size; r++) {
		for (int g = 0; g < size; g++) {
			for (int b = 0; b < size; b++) {

				// update variables for new common color
				if (hist.at<int>(r, g, b) > currMax) {
					// update common color approximate values
					cRed = r * bucketSize + bucketSize / 2;
					cGreen = g * bucketSize + bucketSize / 2;
					cBlue = b * bucketSize + bucketSize / 2;
					// update common color coordinates
					rBucket = r;
					gBucket = g;
					bBucket = b;
					// update currMax with current bin with most votes
					currMax = hist.at<int>(r, g, b);
				}
			}
		}
	}

	// update common color coordinates
	input_r = rBucket;
	input_g = gBucket;
	input_b = bBucket;

}

// histogramMostVotes_SIFT
// preconditions: all passed in values exist and are related to each 
// other, to be used in conjuction with siftTest
// postconditions: finds the center of a 3x3 int the 5x5 with the 
// most matches from good_matches
void histogramMostVotes_SIFT(const Mat& hayStack, vector<KeyPoint> kp2, int size, vector<DMatch> good_matches, int & row_match, int& col_match) {

	// size is a constant - the # of buckets in each dimension 
	int dims[] = { size, size };
	// create 2D histogram of integers initialized to zero	
	Mat commonCluster(2, dims, CV_32S, Scalar::all(0));

	// creates histogram
	for (size_t i = 0; i < good_matches.size(); i++) {
		Point2f haystackPts = kp2[good_matches[i].trainIdx].pt;

		// calculate correct historgram bucket coordinate 0-3
		int bucketSizeRow = hayStack.cols / size;
		int bucketSizeCol = hayStack.rows / size;
		int row = static_cast<int>(haystackPts.x / bucketSizeRow);
		int col = static_cast<int>(haystackPts.y / bucketSizeCol);

		// increment votes at historgram bucket
		commonCluster.at<int>(row, col)++;
	}

	//ints to keep track of current matches, and most matches
	int maxMatches = 0;
	int currMatches = 0;

	//loop through the 5x5
	for (int row = 1; row < size - 1; row++) {
		for (int col = 1; col < size - 1; col++) {
			currMatches = commonCluster.at<int>(row, col)//3x3 grid
				+ commonCluster.at<int>(row, col + 1)
				+ commonCluster.at<int>(row, col - 1)
				+ commonCluster.at<int>(row + 1, col + 1)
				+ commonCluster.at<int>(row + 1, col - 1)
				+ commonCluster.at<int>(row - 1, col + 1)
				+ commonCluster.at<int>(row - 1, col - 1)
				+ commonCluster.at<int>(row - 1, col)
				+ commonCluster.at<int>(row + 1, col);

			// compare num of matches at each position replace variables if true
			if (currMatches > maxMatches) {
				row_match = row;
				col_match = col;
				maxMatches = currMatches;
			}
		}
	}
}

// siftTest Identifies keypoint features using SIFT and makes matches using flann match detector, then makes crops on most matches
// preconditions: All passed in Mat images exist and float threshold is a reasonable threshold (no absurd value),
//					histogramMostVotes_SIFT() works correctly, cropAt() works correctly
// postconditions: a new cropped image of haystack is returned where the most amount of matches are found between 
//					needle an hayStack images
Mat siftTest(const Mat& needle, const Mat& hayStack, float float_threshold) {

	//initialize sift tool
	Ptr<SIFT> sift = SIFT::create();

	// needle Mat
	Mat des1, test1;
	vector<KeyPoint> kp1;

	//detect keypoint and draw them from needle image
	sift->detectAndCompute(needle, Mat(), kp1, des1);
	drawKeypoints(needle, kp1, test1);

	// hayStack Mat
	Mat des2, test2;
	vector<KeyPoint> keyPoints2;

	//detect keypoints and draw them from hayStack image
	sift->detectAndCompute(hayStack, Mat(), keyPoints2, des2);
	drawKeypoints(hayStack, keyPoints2, test2);

	//flann matcher to detect matches between needle an hayStack
	//https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	vector<vector<DMatch>> knn_matches; // create another vect to keep track of matches
	matcher->knnMatch(des1, des2, knn_matches, 2);
	//-- Filter matches using the Lowe's ratio test
	const float ratio_thresh = float_threshold; //originally .7f
	vector<DMatch> good_matches;
	for (size_t i = 0; i < knn_matches.size(); i++) {
		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
			good_matches.push_back(knn_matches[i][0]);
		}
	}
	//-- Draw matches
	Mat img_matches;
	drawMatches(needle, kp1, hayStack, keyPoints2, good_matches, img_matches, Scalar::all(-1),
		Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//used for scaling
	int row_match = 0;
	int col_match = 0;
	const int size = 5;

	//get the position in 5x5 of haystack where there are the most matches
	histogramMostVotes_SIFT(hayStack, keyPoints2, size, good_matches, row_match, col_match);

	//scale haystack image
	int x = static_cast<int>((row_match - 1) * (hayStack.cols / size));
	int y = static_cast<int>((col_match - 1) * (hayStack.rows / size));
	int width = static_cast<int>((hayStack.cols / size) * 3);
	int height = static_cast<int>((hayStack.rows / size) * 3);

	//uncomment below to show the sift keypoints and match detection 
	//			namedWindow("(SIFT) Good Matches", WINDOW_NORMAL);
	//			imshow("(SIFT) Good Matches", img_matches);

	//make a crop of the 3x3 in the 5x5 where the most matches are
	Mat croppedImg = cropAt(hayStack, x, y, width, height);
	//do the same crop to an vanilla image of haystacks
	testCropNoEffects = cropAt(testCropNoEffects, x, y, width, height);

	//returns cropped image
	return croppedImg; // croppedImg;
}


// matchTemplateTest
// preconditions: valid image's passed in, and ref variable "match"
// postconditions: returns new image with all matches found using the 6 template matching
// methods, also updates ref variable to indicate whether match was found 
Mat matchTemplateTest(const Mat& input1, const Mat& input2, bool &match) {

	// all the different methods of doing template matching 
	int methods[] = {TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_SQDIFF, TM_SQDIFF_NORMED};
	int index = 0;
	int const size = 6;
	// array to store x,y values from rectangles 
	int overlappingMatches_x[size];
	int overlappingMatches_y[size];
	// how far away one rectangle can be from one another to be counted as a match
	int acceptedDifference = 15; 

	Mat needle = input1.clone();
	Mat hayStack = input2.clone();
 
	Mat result,res;  
	vector<Point> points; 
	double minval, maxval;
	Point minloc, maxloc;

	// goes through each method of template match and runs OpenCV matchTemplate func drawing 
	// a rectangle at each maxloc
	for (int method : methods) {
		matchTemplate(hayStack, needle, result, method); // originally only used TM_CCOEFF_NORMED
		// calculate matches 
		minMaxLoc(result, &minval, &maxval, &minloc, &maxloc);
		// create rectangle at match with highest confidence level
		rectangle(hayStack, maxloc, Point(maxloc.x + needle.cols, maxloc.y + needle.rows), CV_RGB(0, 255, 0), 2);
		// prints coordinates of best match
		cout << "Best match top left position: " << maxloc << endl;
		// prints confidence level at coordinate
		cout << "Best match confidence: " << maxval << endl;
		
		// assign x, y coordinate to appropriate array and move index
		overlappingMatches_x[index] = maxloc.x;
		overlappingMatches_y[index] = maxloc.y;
		index++;

		// display
		cout << hayStack.rows << " |||||||||||||||||||||| " << hayStack.cols << endl;
		imshow("result", hayStack);
	}
	// deterministic variable to see if matches were found
	int Matches = 0;

	// goes through overlappingMatches_x and see if any
	for (int i = 0; i < size; i++) {
		int xleftRange = overlappingMatches_x[i] - acceptedDifference;
		int xrightRange = overlappingMatches_x[i] + acceptedDifference;
		int yUpRange = overlappingMatches_y[i] - acceptedDifference;
		int yDownRange = overlappingMatches_y[i] + acceptedDifference;

		if (xleftRange < 0) {
			xleftRange = 0;
		}
		if (hayStack.rows - 1 < xrightRange) {
			xrightRange = hayStack.rows - 1;
		}
		if (yUpRange < 0) {
			yUpRange = 0;
		}
		if (hayStack.cols - 1 < yDownRange) {
			yDownRange = hayStack.cols - 1;
		}
		if (xleftRange < overlappingMatches_x[i] && overlappingMatches_x[i] < xrightRange &&
			yUpRange < overlappingMatches_y[i] && overlappingMatches_y[i] < yDownRange) {
			Matches++;
		}
	}

	if (2 < Matches) {
		match = true; 
	}
	
	return result;
}

// grabCutTest - grabCut OpenCV method exclusive for anchor image
// to eliminate noise around ear (aka hair)
// preconditions: anchor image
// postconditions: returns grab cut on anchor image 
Mat grabCutTest(const Mat& input1) {
	Mat image = input1.clone();

	Mat image2 = image.clone();

	// define bounding rectangle
	// lef top right bot 
	Rect rectangle(40, 0, image.cols - 60, image.rows - 70);

	Mat result; // segmentation result (4 possible values)
	Mat bgModel, fgModel; // the models (internally used)

	// GrabCut segmentation
	grabCut(image,    // input image
		result,   // segmentation result
		rectangle,// rectangle containing foreground
		bgModel, fgModel, // models
		1,        // number of iterations
		GC_INIT_WITH_RECT); // use rectangle

	// Get the pixels marked as likely foreground
	compare(result, GC_PR_FGD, result, CMP_EQ);
	// Generate output image
	Mat foreground(image.size(), CV_8UC3, Scalar(255, 255, 255));
	image.copyTo(foreground, result); // bg pixels not copied

	// draw rectangle on original image
	cv::rectangle(image, rectangle, Scalar(255, 255, 255), 1);
	imwrite("img_1.jpg", image);

	imwrite("Foreground.jpg", foreground);

	return foreground;
}

// rotateImg
// preconditions: valid image and degree's the user wants the 
// program to rotate the image from accepts both neg and pos values
// for counter clockwise and clockwise rotations.
// postconditions: returns new image with rotation performed 
// on passed in image
Mat rotateImg(const Mat& input, int degreeToRotate) {
	Mat rotatedImg = input.clone();

	// rotates image
	Point2f pc(rotatedImg.cols / 2., rotatedImg.rows / 2.);
	// OpenCV rotate function
	Mat r = getRotationMatrix2D(pc, degreeToRotate, 1.0);
	warpAffine(rotatedImg, rotatedImg, r, rotatedImg.size());

	return rotatedImg; 
}

// showImg - method created to avoid having to making testing and 
// seeing images faster.
// preconditions: valid input image and desired window title (String)
// postconditions: returns image in window name dictated by string var 'name'
void showImg(const Mat& input, String name) {
	// sets window title to name
	namedWindow(name, WINDOW_NORMAL);
	// keep proportions on input img
	resizeWindow(name, input.cols, input.rows);
	// display
	imshow(name, input);
	//cout << name << ".cols: " << input.cols << " | "<<name<<".rows : " << input.rows << endl;
}

// calibration calibrates the camera by giving us values for camera matrix, distance coefficients, 
// rotation vecotor and translation vector to use in a later method
// preconditions: checkerboard int array is created and set to correct dimensions
// postconditions: Values related to camera calibration is printed and saved to be used later
void calibration() {

	// Creating vector to store vectors of 3D points for each checkerboard image
	vector<vector<Point3f> > objpoints;

	// Creating vector to store vectors of 2D points for each checkerboard image
	vector<vector<Point2f> > imgpoints;

	// Defining the world coordinates for 3D points
	vector<Point3f> objp;

	for (int i = 0; i < CHECKERBOARD[1]; i++) {
		for (int j = 0; j < CHECKERBOARD[0]; j++) {
			objp.push_back(cv::Point3f(j, i, 0));
		}
	}
	// Extracting path of individual image stored in a given directory
	vector<String> images;

	// Path of the folder containing checkerboard images
	  // string path = "./*.jpg";
	string path = "./*.jpeg";
	glob(path, images);

	Mat frame, gray;

	// vector to store the pixel coordinates of detected checker board corners
	vector<Point2f> corner_pts;
	bool success;

	// Looping over all the images in the directory
	for (int i = 0; i < images.size(); i++) {

		frame = imread(images[i]);
		cvtColor(frame, gray, COLOR_BGR2GRAY);

		// Finding checker board corners
		// If desired number of corners are found in the image then success = true 
		success = findChessboardCorners(gray, Size(CHECKERBOARD[0], CHECKERBOARD[1]),
			corner_pts, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);


		/*
		* If desired number of corner are detected,
		* we refine the pixel coordinates and display
		* them on the images of checker board
		*/
		if (success) {
			// TermCriteria criteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.001);

			// // refining pixel coordinates for given 2d points.
			// cornerSubPix(gray,corner_pts, Size(11,11), Size(-1,-1),criteria);

			// Displaying the detected corner points on the checker board
			drawChessboardCorners(frame, Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);

			objpoints.push_back(objp);
			imgpoints.push_back(corner_pts);
		}

		resize(frame, frame, Size(), .3, .3);
		imshow("Image", frame);
		waitKey();
	}

	destroyAllWindows();
	Mat cameraMatrix, distCoeffs, R, T;

	/*
	 * Performing camera calibration by
	 * passing the value of known 3D points (objpoints)
	 * and corresponding pixel coordinates of the
	 * detected corners (imgpoints)
	*/
	calibrateCamera(objpoints, imgpoints, Size(gray.rows, gray.cols), cameraMatrix, distCoeffs, R, T);

	cout << "cameraMatrix : " << cameraMatrix << endl;
	cout << "distCoeffs : " << distCoeffs << endl;
	cout << "Rotation vector : " << R << endl;
	cout << "Translation vector : " << T << endl;

}

// main
// preconditions: valid batch file values in order String char String
// postconditions: determine whether anchor ear img exists in test img and
// whether they're the same ear
int main(int argc, char* argv[]) {
	String anchorFile = "anchor.jpg";
	String testFile = " ";
	char sizeOfTestImg = ' ';
	bool camera = false;
	//*******************************************************************
	//**********************EDIT THIS PART ******************************
	//small image
	//testFile = "S_anchor2.jpg";
	//sizeOfTestImg = 'S';


	//medium image
	//testFile = "M_topDownView.jpg";
	//sizeOfTestImg = 'M';


	//large image
	testFile = "L_withOtherPeople.jpg";
	sizeOfTestImg = 'L';



	//camera calibration
	/*calibration();
	camera = true;*/
	//********************************************************************
	//********************************************************************


	if (!camera) {
		bool match = false;

		// read in images 
		Mat anchor = imread(anchorFile);
		Mat anchorOriginal = imread(anchorFile);
		Mat test = imread(testFile);
		Mat testOriginal = imread(testFile);

		// set values
		float resizeAnchor = 0;
		float resizeTest = 0;
		int anchorKValue = 0;
		int iterations = 0;
		double threshold = 0;

		// update values determined by image category
		if (sizeOfTestImg == 'S') {
			resizeAnchor = 1;
			resizeTest = .5;
			anchorKValue = 3;
			iterations = 2;
			threshold = 0.9;
		}
		else if (sizeOfTestImg == 'M') {
			resizeAnchor = .5;
			resizeTest = 0.5;
			anchorKValue = 4;
			iterations = 3;
			threshold = 0.7;
		}
		else {
			resizeAnchor = 1;
			resizeTest = 0.5;
			anchorKValue = 4;
			iterations = 3;
			threshold = 0.9;
		}

		// resize both anchor and test img to make program run more efficiently
		resize(anchor, anchor, Size(), resizeAnchor, resizeAnchor);
		resize(test, test, Size(), resizeTest, resizeTest);
		// update var to latest version of test img
		testCropNoEffects = test.clone();

		//----------------- apply k-means -----------------
		anchor = K_Means(anchor, anchorKValue);
		test = K_Means(test, 5);

		// ----------------- sift Iterator -----------------
		// detect ear in test image from anchor sift features
		for (int i = 0; i < iterations; i++) {
			test = siftTest(anchor, test, threshold);
			//showImg(test, testFile + string(1, i));
		}
		// update anchor and test images 
		// we did this for better code readability
		anchor = anchorOriginal.clone();
		test = testCropNoEffects.clone();

		// determine whether anchor and test ear are the same 
		// method may change based on char value
		if (sizeOfTestImg == 'S') {
			// set coordinate variable that will correspond to 3d matrix
			int anchor_r = 0;
			int anchor_g = 0;
			int anchor_b = 0;
			// find most common color in anchor
			histogramCalcMostVotes_COLOR(anchor, anchor_r, anchor_g, anchor_b);
			//cout << "anchor_r: " << anchor_r << " | anchor_g: "<< anchor_g << " | anchor_b: " << anchor_b << endl;
			// set coordinate variable that will correspond to 3d matrix
			int test_r = 0;
			int test_g = 0;
			int test_b = 0;
			// find most common color in test
			histogramCalcMostVotes_COLOR(test, test_r, test_g, test_b);
			//cout << "test_r: " << test_r << " | test_g: " << test_g << " | test_b: " << test_b << endl;
			// resize both test and anchorOriginal so they're proportional in size and ear features match
			// and would be of same scale
			resize(test, test, Size(), .8 + resizeTest, .8 + resizeTest);
			resize(anchorOriginal, anchorOriginal, Size(), resizeTest, resizeTest);
			showImg(test, "test");
			// match template test to see if ear feature from test resides within anchor img
			Mat matchTemp = matchTemplateTest(test, anchorOriginal, match);
			// compare whether 3D Coordinates are the same meaning same color of ear
			// print results
			cout << "Based on color: " << endl;
			if (anchor_r == test_r && anchor_g == test_g && anchor_b == test_b) {
				cout << anchorFile << " and " << testFile << " are the same ear: TRUE" << endl;
			}
			else {
				cout << anchorFile << " and " << testFile << " are the same ear: FALSE" << endl;
			}
			cout << "Based on matchTemplate: " << endl;
			cout << "A common match was found? (1-True & 0-False): " << match << endl;

		}

		else if (sizeOfTestImg == 'M') {
			resize(test, test, Size(), 1 + resizeTest, 1 + resizeTest);
			showImg(test, "test");
			// match template test to see if ear feature from test resides within anchor img
			Mat matchTemp = matchTemplateTest(test, anchorOriginal, match);
			// print results
			cout << "Based on matchTemplate: " << endl;
			cout << "A common match was found? (1-True & 0-False): " << match << endl;
		}

		else {
			resize(test, test, Size(), 1 + resizeTest, 1 + resizeTest);
			resize(anchorOriginal, anchorOriginal, Size(), .2, .2);
			showImg(anchorOriginal, "anchorOriginal");
			// match template test to see if ear feature from anchor resides within test img
			// was switched b
			Mat matchTemp = matchTemplateTest(anchorOriginal, test, match);
			// print results
			cout << "Based on matchTemplate: " << endl;
			cout << "A common match was found? (1-True & 0-False): " << match << endl;
		}

		// perform calibration on grid didn't get promising results when testing on ear
		// intended to be used 
		//calibration();

		waitKey();
	}
	return 0;
}
