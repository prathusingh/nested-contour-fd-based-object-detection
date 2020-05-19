#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;


class LaplacianEdgeDetector {

public:
	Mat applyLaplacianEdgeDetector(Mat src, int kernel_size,int scale,int delta,int ddepth)
	{
		Mat src_gray, dst;
		//int kernel_size = 3;
		//int scale = 1;
		//int delta = 0;
		//int ddepth = CV_16S;
		//char* window_name = "Laplace Demo";

		int c;

		/// Load an image
		//src = imread(argv[1]);

		

		/// Remove noise by blurring with a Gaussian filter
		GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);

		/// Convert the image to grayscale
		cvtColor(src, src_gray, CV_BGR2GRAY);

		/// Create window
	//	namedWindow(window_name, CV_WINDOW_AUTOSIZE);

		/// Apply Laplace function
		Mat abs_dst;

		Laplacian(src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
		convertScaleAbs(dst, abs_dst);

		/// Show what you got
		//imshow(window_name, abs_dst);

		return abs_dst;
	}
};