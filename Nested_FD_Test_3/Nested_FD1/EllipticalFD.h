
//#define RENDER
#define VERIFY_ANGLE
//#define USE_COMPLEX_COORDINATES
#define USE_CENTROID_DISTANCE
//#define USE_FIT_ERROR
#ifdef USE_COMPLEX_COORDINATES
#undef USE_CENTROID_DISTANCE
#undef USE_FIT_ERROR
#endif
#ifdef USE_CENTROID_DISTANCE
#undef USE_COMPLEX_COORDINATES
#undef USE_FIT_ERROR
#endif
#ifdef USE_FIT_ERROR
#undef USE_COMPLEX_COORDINATES
#undef USE_CENTROID_DISTANCE
#endif

#ifndef USE COMMANDLINE
#define NUM_TRAINING_IMAGES (66)
#endif
#define PI 3.1415926535
//#define DFT_LENGTH (1024)

#include <opencv2/opencv.hpp> 
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <math.h>
using namespace cv;
using namespace std;

FileStorage fs("testdata.yml", FileStorage::WRITE);
/*
#ifdef USE_COMPLEX_COORDINATES
Mat feature_vector(DFT_LENGTH − 1, NUM_TRAINING_IMAGES, CV_32FC1);
Mat final_feature_vector(NUM_TRAINING_IMAGES, DFT_LENGTH − 1, CV_32FC1);
#endif
#ifdef USE_CENTROID_DISTANCE
Mat feature_vector(DFT_LENGTH − 1, NUM_TRAINING_IMAGES, CV_32FC1);
Mat final_feature_vector(NUM_TRAINING_IMAGES, DFT_LENGTH − 1, CV_32FC1);
#endif
//#ifdef USE_FIT_ERROR
//Mat feature_vector(DFT_LENGTH - 1, 1, CV_32FC1);
//Mat final_feature_vector(1, DFT_LENGTH - 1, CV_32FC1);
//#endif
*/

class EllipticalFD {
public:
//	char filename[100] = { 0 };
	int valid_feature_vects[NUM_TRAINING_IMAGES];

	Mat generateFDHarmonics(vector<vector<Point>> contours1, int contourIndex, int noOfPoints)
	{
		Mat feature_vector(noOfPoints - 1, 1, CV_32FC1);
		Mat final_feature_vector(1, noOfPoints - 1, CV_32FC1);

		vector<Point> contour1= contours1[contourIndex];
		// I n i t i a l i z e boundary p o i n t s
		int contourLength = contour1.size();
		CvMat* BoundaryPoints_x = cvCreateMat(contourLength, 1, CV_32FC1);
		CvMat* BoundaryPoints_y = cvCreateMat(contourLength, 1, CV_32FC1);
		CvMat* BoundaryPoints_cart = cvCreateMat(contourLength, 2, CV_32FC1);

		// I n i t i a l i z e s t a t i s t i c a l v a r i a b l e s
		CvMat* mean_arr = cvCreateMat(contourLength, 2, CV_32FC1);
		CvMat* mean = cvCreateMat(1, 2, CV_32FC1);
		CvMat* mean_x = cvCreateMat(1, 1, CV_32FC1);
		CvMat* mean_y = cvCreateMat(1, 1, CV_32FC1);
		CvMat* mean_x_arr = cvCreateMat(contourLength, 1, CV_32FC1);
		CvMat* mean_y_arr = cvCreateMat(contourLength, 1, CV_32FC1);

		// Copy t h e c a r t e s i a n c o o r d i n a t e s i n t o t h e p r e v i o u s l y i n i t i a l i z e d v a r i a b l e s

		CvMemStorage* storage = cvCreateMemStorage(0);
			// By default the flag 0 is 64K 
			// but myKeypointVector.size()*(sizeof(KeyPoint)+sizeof(CvSeq)) should work
			// it may be more efficient but be careful there may have seg fault 
			// (not sure about the size

		CvSeq* myPointSeq = cvCreateSeq(0, sizeof(CvSeq), sizeof(Point), storage);
		// Create the seq at the location storage

			for (size_t i = 0; i<contourLength; i++) {
				int* added = (int*)cvSeqPush(myPointSeq, &(contour1[i]));
			// Should add the KeyPoint in the Seq
		}


			for (int i = 0; i<myPointSeq->total; ++i) {
				CvPoint* p = (CvPoint *) cvGetSeqElem(myPointSeq, i);
				cvmSet(BoundaryPoints_x, i, 0, p->x);
				cvmSet(BoundaryPoints_y, i, 0, p->y);
				cvmSet(BoundaryPoints_cart, i, 0, p->x);
				cvmSet(BoundaryPoints_cart, i, 1, p->y);
			}

			cvClearMemStorage(storage);
			cvReleaseMemStorage(&storage);

			/*
		for (int i = 0; i < contourLength; i++)
		{
			Point p = contour1[i];
			//cout << "\n" << p.x << " " << p.y;
			cvmSet(BoundaryPoints_x, i, 0, p.x);
			cvmSet(BoundaryPoints_y, i, 0, p.y);
			cvmSet(BoundaryPoints_cart, i, 0, p.x);
			cvmSet(BoundaryPoints_cart, i, 1, p.y);

		}
		*/

		// C a l c u l a t e mean( c e n t r o i d ) f o r t h e c on t our
		cvReduce(BoundaryPoints_cart, mean, 0, CV_REDUCE_AVG);
		cvReduce(BoundaryPoints_x, mean_x, 0, CV_REDUCE_AVG);
		cvReduce(BoundaryPoints_y, mean_y, 0, CV_REDUCE_AVG);
		cvRepeat(mean, mean_arr);
		cvRepeat(mean_x, mean_x_arr);
		cvRepeat(mean_y, mean_y_arr);

		// Remove e f f e c t o f c e n t r o i d from t h e c on t our
		cvSub(BoundaryPoints_cart, mean_arr, BoundaryPoints_cart);
		cvSub(BoundaryPoints_x, mean_x_arr, BoundaryPoints_x);
		cvSub(BoundaryPoints_y, mean_y_arr, BoundaryPoints_y);

		// Prepare t o send t h e co−o r d i n a t e s f o r f i t t i n g an e l l i p s e
		Mat points = cvarrToMat(BoundaryPoints_cart);
#ifdef VERIFY_ANGLE
		// Conver t t o Polar f o r sampl ing
		Mat polar(contourLength, 2, CV_32FC1);
		Mat M1 = polar.col(0);
		Mat M2 = polar.col(1);
		cartToPolar(points.col(0), points.col(1), M1, M2);
#endif

		Mat sampled_points(noOfPoints, 2, CV_32FC1);
		int counter = 0;
		for (float angle = 0; angle <= 2 * PI; angle = angle + PI / 180)
		{

			for (int go_through = 0; go_through < polar.rows; go_through++)
			{
				if (polar.at<float >(go_through, 1) >= angle
					&&polar.at<float>(go_through, 1) <= angle + PI / 90)
				{
					sampled_points.at<float >(counter, 0) = polar.at<float >(go_through, 0);
					sampled_points.at<float >(counter, 1) = polar.at<float >(go_through, 1);
					break;
				}
			}
			counter++;
		}

		Mat sampled_points_cart(noOfPoints, 2, CV_32FC1);
		M1 = sampled_points_cart.col(0);
		M2 = sampled_points_cart.col(1);
		// Conver t back t o c a r t e s i a n t o f i n d DFT
		polarToCart(sampled_points_cart.col(0), sampled_points_cart.col(1), M1, M2);


#ifdef USE_COMPLEX_COORDINATES
		sampled_points_cart = sampled_points_cart.reshape(2, sampled_points.rows);
		// Compute t h e DFT o f t h e e r r o r v e c t o r
		Mat dft_vect(sampled_points_cart);
		// C a l c u l a t e DFT o f e r r o r v e c t o r
		dft(sampled_points_cart, dft_vect, DFT_COMPLEX_OUTPUT);
		// Find Magnitude o f DFT
		dft_vect = dft_vect.reshape(1, dft_vect.rows);
		Mat final_dft(dft_vect.rows, dft_vect.cols, CV_32FC1);
		M1 = final_dft.col(0);
		M2 = final_dft.col(1);

		// Get F ou r ie r D e s c r i p t o r
		cartToPolar(dft_vect.col(0), dft_vect.col(1), M1, M2);
		M1 = M1 / M1.at<float>(0, 0);
		// Save F ou r ie r D e s c r i p t o r as f e a t u r e
		Mat temp = feature_vector.col(0); // j was the outer loop running for each contour !!
		M1.rowRange(1, M1.rows).copyTo(temp);
#endif

#ifdef USE_CENTROID_DISTANCE
		Mat centroid_distance = sampled_points.col(0);
		// Compute t h e DFT o f t h e e r r o r v e c t o r
		Mat dft_vect(centroid_distance);
		// C a l c u l a t e DFT o f e r r o r v e c t o r
		dft(centroid_distance, dft_vect, DFT_ROWS | DFT_COMPLEX_OUTPUT);
		// Find Magnitude of DFT
		dft_vect = dft_vect.reshape(1, dft_vect.rows);
		Mat final_dft(dft_vect.rows, dft_vect.cols, CV_32FC1);
		M1 = final_dft.col(0);
		M2 = final_dft.col(1);
		// Get F ou r ie r D e s c r i p t o r
		cartToPolar(dft_vect.col(0), dft_vect.col(1), M1, M2);
		cout << "\n dc-component " << M1.at<float>(0, 0);
		M1 = M1 / M1.at<float>(0, 0);
		// Save F ou r ie r D e s c r i p t o r as f e a t u r e
		Mat temp = feature_vector.col(0); //j was the outer looping through each contour
		M1.rowRange(1, M1.rows).copyTo(temp);
		//feature_vector.col(0) = temp;
#endif



#ifdef USE_FIT_ERROR
		sampled_points_cart = sampled_points_cart.reshape(2, sampled_points_cart.rows);
		CvBox2D ellipse = fitEllipse(sampled_points_cart);
		sampled_points_cart = sampled_points_cart.reshape(1, sampled_points_cart.rows);
		M1 = sampled_points_cart.col(0);
		M2 = sampled_points_cart.col(1);
		Mat fit_error;
		fit_error = (M1.mul(M1) / ((double)ellipse.size.height * ellipse.size.height))
			+ (M2.mul(M2) / ((double)ellipse.size.height * ellipse.size.height)) - 1;
		Mat dft_vect(fit_error);
		dft(fit_error, dft_vect, DFT_COMPLEX_OUTPUT);
		// Find Magnitude o f DFT
		dft_vect = dft_vect.reshape(1, dft_vect.rows);
		Mat final_dft(dft_vect.rows, dft_vect.cols, CV_32FC1);
		M1 = final_dft.col(0);
		M2 = final_dft.col(1);
		// Get F ou r ie r D e s c r i p t o r
		cartToPolar(dft_vect.col(0), dft_vect.col(1), M1, M2);
		M1 = M1 / M1.at<float >(0, 0);
		// Save F ou r ie r D e s c r i p t o r as f e a t u r e
		Mat temp = feature_vector.col(0);
		M1.rowRange(1, M1.rows).copyTo(temp);
#endif

#ifdef RENDER
		Mat I_threshold = I_thresholded;
		imshow(” Th re sh olded Image ”, I t h r e s h o l d);
		imshow(” Contours ”, LoadedImg);
		waitKey();
#endif


		// Prepare and save f e a t u r e v e c t o r
		transpose(feature_vector, final_feature_vector);

#ifdef USE_CENTROID_DISTANCE
		//Use onl y h a l f o f t h e f e a t u r e v e c t o r t o d e s c r i b e t h e o b j e c t
		final_feature_vector = final_feature_vector.colRange(0, noOfPoints / 2);
#endif

#ifdef USE_FIT_ERROR
		//Use onl y h a l f o f t h e f e a t u r e v e c t o r t o d e s c r i b e t h e o b j e c t
		final_feature_vector = final_feature_vector.colRange(0, DFT_LENGTH / 2);
#endif
		//fs << "FIT ERROR FD" << final_feature_vector;
		cout << "\n Feature vector length \n " << final_feature_vector.cols;
		//cout << final_feature_vector << endl;
		

#ifdef DEBUG
		cout << final_feature_vector << endl;
#endif


		// Create vector from matrix data (data with data copying)
		//vector<float> V;
		//V.assign((float*)final_feature_vector.datastart, (float*)final_feature_vector.dataend);


		return final_feature_vector;

	}










};