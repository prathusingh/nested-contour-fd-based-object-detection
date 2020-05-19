#include <opencv2\opencv.hpp>
#include <opencv2\imgproc\imgproc.hpp>
//#include <vector>
//#include <numeric>

#include "FVCompare.h"
#include "watershed.h"
#include "laplacianEdgeDetector.h"
#include "ZhangSuenEdgeThinning.h"
#include "IPToolbox.h"
#include "ParallelCompute.h"
//#include "FDPersoonFu.h"
#include "Deriche_Edge_Detector.h"
#include "EllipticalFD.h"
#include "RoadSignModels.h"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudastereo.hpp>
#include <omp.h>

#include<thread>
#include <future>  

#include <opencv2/ml/ml.hpp>


#define pi  3.141592653
using namespace cv;
using namespace std;

vector<int> getChildContoursClosed(vector<vector<Point>> contours, vector<Vec4i> hierarchy, int parentCntrIdx, float perimeterThresh, int level);
bool parentChildContoursPerimeterThresh(vector<vector<Point>> contours, int parentCntrIdx, int childCntrIndx, float perimeterThresh);
bool isConcentricContour(vector<vector<Point>> contours, int parentCntrIdx, int childCntrIndx, float perimeterThresh);
int getNextNonConcentricContour(vector<vector<Point>> contours, vector<Vec4i> hierarchy, int concentricContrIdx, int parentCntrIdx, float perimeterThresh);
int findNoOfChildren(vector<vector<Point>> contours, vector<Vec4i> hierarchy, int parentCntrIdx);
bool doContoursIntersect(vector<vector<Point>> contours, int parentCntrIdx, int childCntrIndx);
bool doContoursIntersect1(vector<vector<Point>> contours, int parentCntrIdx, int childCntrIndx);
bool isContourClosed(vector<vector<Point>> contours, int contourIdx);
void getForDiffFDCoeff(vector<vector<Point>> contours, int coeff, int cntrID);
//vector<int> getChildContoursClosed(vector<vector<Point>> contours, vector<Vec4i> hierarchy, int parentCntrIdx, float perimeterThresh, vector<int> childContourClosedIndexes);
void helloThread(int i);
vector<vector<Point>> preProcessContour(String imgName);
void appendFVToFileEFD(String fileName, String imagePath, String fvName, int numberOfCoeff);
Mat queryFvEFD(String imagePath, int numberOfCoeff);

vector<Point2d> queryFvPFu(String imagePath, int numberOfCoeff);
void appendFVToFilePFu(String fileName, String imagePath, String fvName, int numberOfCoeff);

void writeVectorOfVector(FileStorage &fs, string name, vector<vector<Point2d>> &vov);
void readVectorOfVector(FileStorage &fs, string name, vector<vector<Point2d>> &vov);

vector<Point2d> queryFvPFu_contour(vector<vector<Point>> contoursTopLevel1, int cntrIdx, int numberOfCoeff);
double getOrientation(const vector<Point> &pts);
double parentChildCentroidDist(vector<vector<Point>> contours, int parentCntrIdx, int childCntrIndx);
Point2f getContourCentroid(vector<vector<Point>> contours, int cntrIdx);
double angleBetween(const Point2f &v1, const Point2f &v2);
double parentChildAspectRatio(vector<vector<Point>> contours, int parentCntrIdx, int childCntrIndx);

//struct to sort the vector of pairs <int,double> based on the second double value
struct sort_pred {
	bool operator()(const std::pair<int, double> &left, const std::pair<int, double> &right) {
		return left.second < right.second;
	}
};
double rad2Deg(double rad){ return rad*(180 / pi); }
void ProccTimePrint(unsigned long Atime, string msg)
{
	unsigned long Btime = 0;
	float sec, fps;
	Btime = getTickCount();
	sec = (Btime - Atime) / getTickFrequency();
	fps = 1 / sec;
	printf("%s %.4lf(sec) / %.4lf(fps) \n", msg.c_str(), sec, fps);
}

string getFileName(const string& s) {

	char sep = '/';

#ifdef _WIN32
	sep = '\\';
#endif

	size_t i = s.rfind(sep, s.length());
	if (i != string::npos) {
		return(s.substr(i + 1, s.length() - i));
	}

	return("");
}

//split the first word in  string based on delimitter
string splitStringDelim(string str, char delimiter) {
	vector<string> internal;
	stringstream ss(str); // Turn the string into a stream.
	string tok;

	while (getline(ss, tok, delimiter)) {
		internal.push_back(tok);
	}

	return internal[0];
}



Mat imageTest, emptImage;

Mat globalFV;
vector<vector<Point2d>>globalFvPFu;



int main(int argc, char* argv[])
{
	//setUseOptimized(true);
	bool useCommandLine = false;
	Mat img;
	Mat img_closed_contours;
	float chilParentAreaRatio;
	int maxTopLevelComponents;
	int parentFvSize;
	int childFvSize;
	int matchCoeffParent, matchCoeffChild;
	int deserializeParent, deserializeChild;
	int level;
	string imageName = "stop1.jpg";

	if (useCommandLine)
	{

		if (argc < 11) {
			// Tell the user how to run the program
			std::cerr << "Usage: " << argv[0] << " IMAGE NAME/PATH <space> CHILD PARENT AREA RATIO <space> Max Top Level Contours <sapce> Child Contour Level <sapce> Parent FV size (1024) <space> Child FV size (1024/512/256) <space> Parent Coeff to match <space> Child Coeff to match <sapce> Parent FV to Deserialize <sapce> Child FV to Deserialize <sapce>  " << std::endl;
			/* "Usage messages" are a conventional way of telling the user
			* how to run a program if they enter the command incorrectly.
			*/
			return 1;
		}



		img = imread(argv[1]);
		img_closed_contours = img.clone();
		chilParentAreaRatio = stoi(argv[2]);
		maxTopLevelComponents = stoi(argv[3]);
		level = stoi(argv[4]);
		//FV size -- contours are re-sampled for the following number of fourier coeffs/descriptors  - 1024,512,256
		parentFvSize = stoi(argv[5]); // only 1024 right now
		childFvSize = stoi(argv[6]); //1024,512,256
		matchCoeffParent = stoi(argv[7]), matchCoeffChild = stoi(argv[8]);
		deserializeParent = stoi(argv[9]), deserializeChild = stoi(argv[10]);
	}
	else
	{

		img = imread(imageName);
		img_closed_contours = img.clone();
		chilParentAreaRatio = 5;
		maxTopLevelComponents = 1;
		//FV size -- contours are re-sampled for the following number of fourier coeffs/descriptors  - 1024,512,256
		parentFvSize = 1024; // only 1024 right now
		childFvSize = 256; //1024,512,256
		matchCoeffParent = 30, matchCoeffChild = 50;
		deserializeParent = 100, deserializeChild = 100;
		level = 1;
	}


	//empty copy of original image
	Mat3b emptyImg = Mat3b(img.rows, img.cols, CV_8UC3);
	emptImage = emptyImg.clone();

	Mat meanShiftFil;


	//Convert to Gray
	Mat gray, gray_quantized, gray_histo_equalized;
	//Use the watershed segmented image
	//cvtColor(waterShed1, gray, COLOR_BGR2GRAY);
	cvtColor(img, gray, COLOR_BGR2GRAY);
	imshow("Gray", gray);


	GaussianBlur(gray, gray, Size(3, 3), 1.5, 1.5);
	imshow("Filtered", gray);


	//Calculate the Otsu threshold values
	Mat1b opimg = Mat(gray.rows, gray.cols, CV_8UC1);
	double otsu_thresh_val = threshold(
		gray, opimg, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU
		);

	//imshow("otsu", opimg);

	double high_thresh_val = otsu_thresh_val,
		lower_thresh_val = otsu_thresh_val * 0.5;

	cout << lower_thresh_val << "   " << high_thresh_val << "\n \n";

	Mat edges = emptyImg.clone();
	Canny(gray, edges, lower_thresh_val, high_thresh_val, 3);
	//namedWindow("Canny Edges", 0);
	imshow("Canny Edges", edges);


	imageTest = edges;


	Mat contourImg = emptyImg.clone();
	vector<vector<Point>> contours, contours1;
	vector<Vec4i> hierarchy, hierarchy1;

	//findContours(edges.clone(), contours1, hierarchy1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	// gets the outer most contour ...
	// findContours(edges.clone(), contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE); 
	//retrieves all of the contours and organizes them into a two-level hierarchy. At the top level, there are external boundaries of the components. 
	//At the second level, there are boundaries of the holes. 
	// If there is another contour inside a hole of a connected component, it is still put at the top level.

	findContours(edges.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
	//retrieves all of the contours and reconstructs a full hierarchy of nested contours.

	//The hierarchy returned by findContours has the following form: hierarchy[idx][{0,1,2,3}]={next contour (same level), previous contour (same level), child contour, parent contour}
	//Sorting the contours at 0-level first 
	pair <int, double> areaPair;
	vector<pair <int, double>> areas1;

	vector<Point> approx;
	for (int i = 0; i < contours.size(); i++)
	{
		if ((hierarchy[i][2] >= 0) && (hierarchy[i][3] < 1) /*hierarchy[i][3] < 0 && isContourClosed(contours,i)*/)// We first see at 0-level hierarchy, if there are contour's that have child nodes, then only we go ahead 
		{
			approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);
			//if (approx.size()<=4)
			//{ 
			areaPair.first = i;
			areaPair.second = contourArea(contours[i]);
			areas1.push_back(areaPair);
			//}
		}
	}


	//sort in descending
	std::sort(areas1.rbegin(), areas1.rend(), sort_pred());



	//Printing the largest 5 areas ...
	for (int i = 0; i < (maxTopLevelComponents <= areas1.size() ? maxTopLevelComponents : areas1.size()); i++) {
		cout << "\n";
		cout << areas1.at(i).first << "  " << areas1.at(i).second;
	}
	cout << "\n Areas size " << areas1.size();

	pair<int, vector<int>> apair;
	vector<pair<int, vector<int>>> allTopLevelContours; // At heirarchy 0

	//parallel tbb


	//  cv::setNumThreads(50);
	//  cv::parallel_for_(cv::Range(0, maxTopLevelComponents), ParallelCompute::ParallelCompute(contours, hierarchy, 5, areas1, maxTopLevelComponents, allTopLevelContours));


	int nthreads = cvGetNumThreads();
	cout << "\n  threads :: " << nthreads;
	std::vector<std::future<std::vector<int>>> fut;

	//1st level - Tree
	std::vector<std::thread> vecThread;

	for (int i = 0; i < (maxTopLevelComponents <= areas1.size() ? maxTopLevelComponents : areas1.size()); i++) // Basically scans all contours at heirarchy zero ... as hierarchy[cntIdx][0] options looks for all next contours at same level ...
	{
		//if ((hierarchy[i][2]>0) && (hierarchy[i][3]<0) && isContourClosed(contours, i))// We first see at 0-level hierarchy, if there are contour's that have child nodes, then only we go ahead 
		//{
		//getChildContours(),getChildContoursClosed(),getChildContoursDontIntersectParent,getChildContoursAreaThresh()
		//apair.first = areas1.at(i).first;
		fut.push_back(std::async(std::launch::async, getChildContoursClosed, contours, hierarchy, areas1.at(i).first, chilParentAreaRatio, level));
		//fut.at(i) = std::async(getChildContoursClosed, contours, hierarchy, areas1.at(i).first, 5);
		//tt[i] = std::thread(getChildContoursClosed, contours, hierarchy, areas1.at(i).first, 5);
		//apair.second = getChildContoursClosed(contours, hierarchy, areas1.at(i).first, 5); //getChildContoursClosed(contours,hierarchy,childContourIndex,PercentRatio)
		//allTopLevelContours.push_back(apair);
		//drawContours(heirarch1, contours, i, CV_RGB(128, 255, 128), 1, 8, hierarchy, 0);
		//drawContours(heirarch2, contours, i, CV_RGB(255, 133, 222), 1, 8, hierarchy, 4);
		//}
		//if (i <= maxTopLevelComponents - 1) break;
	}


	for (int i = 0; i < (maxTopLevelComponents <= areas1.size() ? maxTopLevelComponents : areas1.size()); i++) // Basically scans all contours at heirarchy zero ... as hierarchy[cntIdx][0] options looks for all next contours at same level ...
	{
		apair.first = areas1.at(i).first;
		apair.second = fut.at(i).get();
		allTopLevelContours.push_back(apair);
	}

	Mat outer_hierarchy0_img = emptyImg.clone();
	Mat childContours_img = emptyImg.clone();
	//vector<Mat> img_child_closed=img.clone();
	//vector<String> windowNames, windowNames2;

	//size of allTopLevelConturs 
	cout << "\n";
	cout << "size of allTopLevelContours  " << allTopLevelContours.size() << "\n";

	//Storing the shape prediction results in <vector<vector<string>> ... 
	vector< pair<String, tuple< vector<String>, vector<double>, vector<double>, vector<int>> > > shapePred;



	//Instantiating the FVCompare class
	FVCompare fvC;

	//De-searilize the FV's from yml's
	//Argument inside the deserializeFvPFu* function indicate the number of FV's you want to randonly pick from each class of shapes's .. 
	// lesser the number faster the speed of comaprision , but there is chance for a faulty match ....

	//fvC.deserializeFvEFDParent();
	fvC.deserializeFvPFuParent(deserializeParent);
	//fvC.deserializeFvEFDChild();
	fvC.deserializeFvPFuChild(deserializeChild);

	Mat oimg = img.clone();

	for (int k = 0; k < allTopLevelContours.size(); k++)
	{
		vector<String> childLabeltemp;
		vector<double> childParentCentroidDist;
		vector<double> childOrientation;
		vector<int> cntrIndices;

		//draw parents
		Scalar color(rand() & 255, rand() & 255, rand() & 255);
		drawContours(outer_hierarchy0_img, contours, allTopLevelContours.at(k).first, color, 1, 8, hierarchy, 0); //0-level
		cout << "index - top level parent contours  " << allTopLevelContours.at(k).first << "\n";
		vector<Point2d>  queryParent;
		try{
			queryParent = queryFvPFu_contour(contours, allTopLevelContours.at(k).first, parentFvSize);
		}
		catch (string std)
		{
			cout << std;
			continue;
		}
		cntrIndices.push_back(allTopLevelContours.at(k).first);
		String labelParent = fvC.usePFu_Parent(queryParent, parentFvSize, matchCoeffParent);
		Rect recP = boundingRect(contours[allTopLevelContours.at(k).first]);
		rectangle(oimg, Point(recP.x - 5, recP.y - 5), Point(recP.x + recP.width + 5, recP.y + recP.height + 5), Scalar(0, 255, 0), 2, 8, 0); //closed contour - draw a green rectangle
		putText(oimg, labelParent, Point(recP.x, recP.y - 10), FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(0, 255, 0), 1, CV_AA);

		double orientationP = getOrientation(contours[allTopLevelContours.at(k).first]);
		//outer_hierarchy0.push_back(outer_hierarchy0_img);
		//windowNames.push_back(" Parent");
		//namedWindow("Parent", 0);
		imshow("Parent", outer_hierarchy0_img);
		//	outer_hierarchy0_img = emptyImg.clone(); //Reset 
		//waitKey(0);

		//draw child for each parent
		Rect boundParent = boundingRect(contours[allTopLevelContours.at(k).first]);
		double parentPerm = 2 * ((double)boundParent.width + (double)boundParent.height);
		//Point2f parentCentroid = Point( (boundParent.x + boundParent.width) / 2, (boundParent.y + boundParent.height) / 2 );

		Point2f parentCentroid = getContourCentroid(contours, allTopLevelContours.at(k).first);

		#pragma omp parallel for num_threads(allTopLevelContours.at(k).second.size())
		for (int childIdx = 0; childIdx < allTopLevelContours.at(k).second.size(); childIdx++)
		{
			drawContours(childContours_img, contours, allTopLevelContours.at(k).second.at(childIdx), color, 1, 8, hierarchy, 0);
			cout << "index - child contours  " << allTopLevelContours.at(k).second.at(childIdx) << "\n";

			vector<Point2d>  queryChild;
			try{
				queryChild = queryFvPFu_contour(contours, allTopLevelContours.at(k).second.at(childIdx), childFvSize);
			}
			catch (string std)
			{
				cout << std;
				continue;
			}
			cntrIndices.push_back(allTopLevelContours.at(k).second.at(childIdx));
			String labelChild = fvC.usePFu_Child(queryChild, childFvSize, matchCoeffChild);
			//String labelChild = fvC.usePFu_ChildFLANN(queryChild, childFvSize, matchCoeffChild);
			Rect recC = boundingRect(contours[allTopLevelContours.at(k).second.at(childIdx)]);
			rectangle(oimg, Point(recC.x - 5, recC.y - 5), Point(recC.x + recC.width + 5, recC.y + recC.height + 5), Scalar(255, 0, 0), 2, 8, 0); //closed contour - draw a green rectangle
			putText(oimg, labelChild, Point(recC.x, recC.y - 10), FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(255, 0, 0), 1, CV_AA);


			childLabeltemp.push_back(labelChild);
			Point2f childCentroid = getContourCentroid(contours, allTopLevelContours.at(k).second.at(childIdx));

			double orientationC = getOrientation(contours[allTopLevelContours.at(k).second.at(childIdx)]);

			childOrientation.push_back(fabs(angleBetween(parentCentroid, childCentroid)));


			double parentchildcenterDist = cv::norm(childCentroid - parentCentroid);
			childParentCentroidDist.push_back(parentchildcenterDist / parentPerm);


			//double centroidDist
			//childContours.push_back(childContours_img);
			//windowNames2.push_back(" Child");
			//namedWindow("Child", 0);
			imshow("Child", childContours_img);
			//childContours_img = emptyImg.clone();
			//waitKey(0);
		}


		pair<String, tuple<vector<String>, vector<double>, vector<double>, vector<int>> > parentChildPair;
		parentChildPair.first = labelParent;
		parentChildPair.second = std::make_tuple(childLabeltemp, childParentCentroidDist, childOrientation, cntrIndices);
		shapePred.push_back(parentChildPair);

		childLabeltemp.clear();
		childParentCentroidDist.clear();
		childOrientation.clear();
		cntrIndices.clear();

		//drawContours(heirarch1, contours, childContourClosedIndexes.at(k), CV_RGB(138, 43, 226), 1, 8, hierarchy,1);
		//Rect r1 = boundingRect(contours[childContourClosedIndexes.at(k)]);
		//rectangle(img_child_closed, Point(r1.x - 10, r1.y - 10), Point(r1.x + r1.width + 10, r1.y + r1.height + 10), Scalar(138, 43, 226), 1, 8, 0); //closed contour - draw a green rectangle
	}


	imshow("Image", img);

	stringstream ss;
	std::string f1;
	if (useCommandLine)
	{
		f1 = argv[1];
	}
	else
	{
		f1 = imageName;
	}
	//ss << argv[1];
	//ss >> f1;

	//cout << "\n argv1  " << argv[1];
	//cout << "\n filename " << splitStringDelim(f1,'.');
	string oimgFileName = "./FinalOutModels/" + splitStringDelim(f1, '.') + "_Labeled.jpg";
	imwrite(oimgFileName, oimg);

	imshow("Labeled Image", oimg);


	cout << "\n Tuple size ..." << shapePred[0].second._Mysize << "\n";

	//Shape we got :
	for (int i = 0; i < shapePred.size(); i++)
	{
		cout << "\n Parent is a : " << shapePred[i].first;
		cout << "\n Children are : ";
		tuple<vector<String>, vector<double>, vector<double>, vector<int>> localTuple;
		localTuple = shapePred[i].second;
		vector<String> localStringVec;
		vector<double> localCenterDistVec;
		vector<double> localOrientVec;
		vector<int> localIndex;
		localStringVec = std::get<0>(localTuple);
		localCenterDistVec = std::get<1>(localTuple);
		localOrientVec = std::get<2>(localTuple);
		localIndex = std::get<3>(localTuple);
		for (int j = 0; j < localStringVec.size(); j++)
		{
			cout << "\n" << localStringVec[j] << "\t" << localCenterDistVec[j] << "\t" << localOrientVec[j] << "\t" << localIndex[j + 1];
		}

	}

	RoadSignModels rm;
	//rm.createModel(shapePred, "YIELD_SIGN");

	rm.readModels();


	vector<tuple<String, tuple<vector<String>, vector<double>, vector<double>, vector<int>>>> resultCompModels;

	resultCompModels = rm.compareModels(shapePred);

	Mat oimg1 = img.clone();

	for (int i = 0; i < resultCompModels.size(); i++)
	{
		tuple<String, tuple<vector<String>, vector<double>, vector<double>, vector<int>>> localT1;
		localT1 = resultCompModels[i];
		String modelPred = std::get<0>(localT1);

		tuple<vector<String>, vector<double>, vector<double>, vector<int>> lt2;
		lt2 = std::get<1>(localT1);
		vector<String> retChild;
		vector<double> retCentroidDist;
		vector<double> retOrient;
		vector<int> retCntrIdx;

		retChild = std::get<0>(lt2);
		retCentroidDist = std::get<1>(lt2);
		retOrient = std::get<2>(lt2);
		retCntrIdx = std::get<3>(lt2);

		//Bound the parent 
		Rect recRetP = boundingRect(contours[retCntrIdx[retCntrIdx.size() - 1]]);
		rectangle(oimg1, Point(recRetP.x - 5, recRetP.y - 5), Point(recRetP.x + recRetP.width + 5, recRetP.y + recRetP.height + 5), Scalar(0, 255, 0), 2, 8, 0); //closed contour - draw a green rectangle
		putText(oimg1, modelPred, Point(recRetP.x, recRetP.y - 10), FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(0, 255, 0), 1, CV_AA);


		for (int j = 0; j < retChild.size(); j++)
		{
			String labelRetC = retChild[j];// +" " + to_string(retCentroidDist[j]) + " " + to_string(retOrient[j]);
			Rect recRetC = boundingRect(contours[retCntrIdx[j]]);
			rectangle(oimg1, Point(recRetC.x - 5, recRetC.y - 5), Point(recRetC.x + recRetC.width + 5, recRetC.y + recRetC.height + 5), Scalar(255, 0, 0), 2, 8, 0); //closed contour - draw a green rectangle
			putText(oimg1, labelRetC, Point(recRetC.x, recRetC.y - 10), FONT_HERSHEY_COMPLEX_SMALL, 0.6, Scalar(255, 0, 0), 1, CV_AA);
		}

	}

	string oimgFileName1 = "./FinalOutModels/" + splitStringDelim(f1, '.') + "_Analysis.jpg";
	imwrite(oimgFileName1, oimg1);

	imshow("Analysis", oimg1);


	//	Mat imageTest11 = emptyImg.clone();
	//drawContours(imageTest11, contours, 5, CV_RGB(0, 128, 128), 6, 8, hierarchy, 0); /// IT WAS NOISE !
	//imshow("contour 66", imageTest11);

	//drawContours(imageTest11, contours, 3167, CV_RGB(255, 255, 255), 1, 8, hierarchy, 0);
	//imshow("contour 66", imageTest11);

	waitKey();

	if (useCommandLine)
	{
		return 1;
	}
	else
	{
		return 0;
	}


}

double parentChildAspectRatio(vector<vector<Point>> contours, int parentCntrIdx, int childCntrIndx)
{
	Rect bRectParent, bRectChild;
	//get bounding rect ... parent and child
	//also experimenting with the contour areas ... 

	bRectParent = boundingRect(contours[parentCntrIdx]);
	bRectChild = boundingRect(contours[childCntrIndx]);



	//double cntrP, cntrC;
	//cntrP = contourArea(contours[parentCntrIdx]);
	//cntrC = contourArea(contours[childCntrIndx]);

	//Aspect Ratio - ratio of width to height of bounding rect 
	// Aspect Ratio Child / Aspect Ratio Parent

	double parentChildAspectR = (((double)bRectChild.width / (double)bRectChild.height));// / ((double)bRectParent.width / (double)bRectParent.height));

	return parentChildAspectR;

}

Point2f getContourCentroid(vector<vector<Point>> contours, int cntrIdx)
{
	vector<Moments> mu(1);
	//get moments
	mu[0] = moments(contours[cntrIdx], false);
	//get mass centers
	vector<Point2f> mc(1);

	return Point2f(mu[0].m10 / mu[0].m00, mu[0].m01 / mu[0].m00); ////  center of mass Point 
}

double parentChildCentroidDist(vector<vector<Point>> contours, int parentCntrIdx, int childCntrIndx)
{
	vector<Moments> mu(2);
	//get moments
	mu[0] = moments(contours[parentCntrIdx], false);
	mu[1] = moments(contours[childCntrIndx], false);
	//parentChildContoursAreaThresh(contours, parentCntrIdx, childCntrIndx, areaThresh);
	//get mass centers
	vector<Point2f> mc(2);

	mc[1] = Point2f(mu[1].m10 / mu[1].m00, mu[1].m01 / mu[1].m00); //// child center of mass Point 
	mc[0] = Point2f(mu[0].m10 / mu[0].m00, mu[0].m01 / mu[0].m00); // parent center of mass Point 

	double res = cv::norm(mc[1] - mc[0]); //distance between center of mass
	return res;
}

double angleBetween(const Point2f &v1, const Point2f &v2)
{

	double delta_x = v2.x - v1.x;
	double delta_y = v2.y - v1.y;
	double theta_radians = atan2(delta_y, delta_x);

	/*
	float len1 = sqrt(v1.x * v1.x + v1.y * v1.y);
	float len2 = sqrt(v2.x * v2.x + v2.y * v2.y);

	float dot = v1.x * v2.x + v1.y * v2.y;

	float a = dot / (len1 * len2);

	if (a >= 1.0)
		return 0.0;
	else if (a <= -1.0)
		return PI;
	else
		return acos(a); // 0..PI
		*/
	
	return theta_radians;
   
}

double getOrientation(const vector<Point> &pts)
{
	//Construct a buffer used by the pca analysis
	int sz = static_cast<int>(pts.size());
	Mat data_pts = Mat(sz, 2, CV_64FC1);
	for (int i = 0; i < data_pts.rows; ++i)
	{
		data_pts.at<double>(i, 0) = pts[i].x;
		data_pts.at<double>(i, 1) = pts[i].y;
	}

	//Perform PCA analysis
	PCA pca_analysis(data_pts, Mat(), CV_PCA_DATA_AS_ROW);

	//Store the center of the object
	Point cntr = Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)),
		static_cast<int>(pca_analysis.mean.at<double>(0, 1)));

	//Store the eigenvalues and eigenvectors
	vector<Point2d> eigen_vecs(2);
	vector<double> eigen_val(2);
	for (int i = 0; i < 2; ++i)
	{
		eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
			pca_analysis.eigenvectors.at<double>(i, 1));
		eigen_val[i] = pca_analysis.eigenvalues.at<double>(0, i);
	}

	// Draw the principal components
	//Point p1 = cntr + 0.02 * Point(static_cast<int>(eigen_vecs[0].x * eigen_val[0]), static_cast<int>(eigen_vecs[0].y * eigen_val[0]));
	//Point p2 = cntr - 0.02 * Point(static_cast<int>(eigen_vecs[1].x * eigen_val[1]), static_cast<int>(eigen_vecs[1].y * eigen_val[1]));

	double angle = atan2(eigen_vecs[0].y, eigen_vecs[0].x); // orientation in radians

	//convert to degrees
	//angle = rad2Deg(angle);
	
	return angle;
}

Mat queryFvEFD(String imagePath,  int numberOfCoeff)
{
	EllipticalFD efd1;
	Mat efdV1;

	vector<vector<Point>> contoursTopLevel1;
	contoursTopLevel1 = preProcessContour(imagePath);
	if (contoursTopLevel1.empty())
	{
		throw string("Error at " + imagePath);
	}
	else
	{
		efdV1 = efd1.generateFDHarmonics(contoursTopLevel1, 0, numberOfCoeff);

		return efdV1;
	}
}

void appendFVToFileEFD(String fileName, String imagePath, String fvName, int numberOfCoeff)
{
	FileStorage fs(fileName, FileStorage::WRITE);

	EllipticalFD efd1;
	Mat efdV1;

	vector<vector<Point>> contoursTopLevel1;
	contoursTopLevel1 = preProcessContour(imagePath);
	if (contoursTopLevel1.empty())
	{
		throw string("Error at : " + imagePath);
	}
	else
	{

		efdV1 = efd1.generateFDHarmonics(contoursTopLevel1, 0, numberOfCoeff);
		globalFV.push_back(efdV1);
		fs << fvName << globalFV;
		fs.release();
	}
}


vector<Point2d> queryFvPFu(String imagePath, int numberOfCoeff)
{
	FDPersoonFu fd1;
	vector<Point2d>  dft1;

	vector<vector<Point>> contoursTopLevel1;
	contoursTopLevel1 = preProcessContour(imagePath);

	if (contoursTopLevel1.empty())
	{
		throw string("Error at : " + imagePath);
	}
	else
	{

		dft1 = fd1.generateFDHarmonicsPFu(contoursTopLevel1, 0, numberOfCoeff);

		return dft1;
	}
}


vector<Point2d> queryFvPFu_contour(vector<vector<Point>> contoursTopLevel1, int cntrIdx, int numberOfCoeff)
{
	FDPersoonFu fd1;
	vector<Point2d>  dft1;

	if (contoursTopLevel1.empty())
	{
		throw string("No Contours Exist - Parent Level");
	}
	else
	{
		if (contoursTopLevel1[cntrIdx].empty())
		{
			throw string("No Contours Exist - Child Level - for the Parent with contour index : " +cntrIdx);
		}
		else
		{ 
		dft1 = fd1.generateFDHarmonicsPFu(contoursTopLevel1, cntrIdx, numberOfCoeff);

		return dft1;
		}
	}
}





void appendFVToFilePFu(String fileName, String imagePath, String fvName, int numberOfCoeff)
{
	FileStorage fs1(fileName, FileStorage::WRITE);

	FDPersoonFu fd2;
	vector<Point2d>  dft2;

	vector<vector<Point>> contoursTopLevel1;
	contoursTopLevel1 = preProcessContour(imagePath);
	if (contoursTopLevel1.empty())
	{
		throw string("Error at : " + fileName);
	}
	else
	{ 

	dft2 = fd2.generateFDHarmonicsPFu(contoursTopLevel1, 0, numberOfCoeff);
		
	globalFvPFu.push_back(dft2);
	}
	//fs1 << fvName << globalFvPFu;
	//fs1.release();
}

void writeVectorOfVector(FileStorage &fs, string name, vector<vector<Point2d>> &vov)
{
	fs << name;
	fs << "{";
	for (int i = 0; i < vov.size(); i++)
	{
		fs << name + "_" + to_string(i);
		vector<Point2d> tmp = vov[i];
		fs << tmp;
	}
	fs << "}";
}

void readVectorOfVector(FileStorage &fs, string name, vector<vector<Point2d>> &vov)
{
	vov.clear();
	FileNode fn = fs[name];
	if (fn.empty()){
		return;
	}

	FileNodeIterator current = fn.begin(), it_end = fn.end(); // Go through the node
	for (; current != it_end; ++current)
	{
		vector<Point2d> tmp;
		FileNode item = *current;
		item >> tmp;
		vov.push_back(tmp);
	}
}





vector<vector<Point>> preProcessContour(String imgName)
{


	//setUseOptimized(true);
	Mat img = imread(imgName);
	Mat img_closed_contours = img.clone();


	//empty copy of original image
	Mat3b emptyImg = Mat3b(img.rows, img.cols, CV_8UC3);
	emptImage = emptyImg.clone();



	//Convert to Gray
	Mat gray, gray_quantized, gray_histo_equalized;

	cvtColor(img, gray, COLOR_BGR2GRAY);
	//imshow("Gray", gray);

	GaussianBlur(gray, gray, Size(3, 3), 1.5, 1.5);
	//imshow("Filtered", gray);

	//Calculate the Otsu threshold values
	Mat1b opimg = Mat(gray.rows, gray.cols, CV_8UC1);
	double otsu_thresh_val = threshold(
		gray, opimg, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU
		);

	//imshow("otsu", opimg);

	double high_thresh_val = otsu_thresh_val,
		lower_thresh_val = otsu_thresh_val * 0.5;

	Mat edges = emptyImg.clone();
	Canny(gray, edges, lower_thresh_val, high_thresh_val, 3);
	//namedWindow("Canny Edges", 0);

	//imshow("Canny Edges", edges);

	Mat contourImg = emptyImg.clone();
	vector<vector<Point>> contours, contoursTopLevel;
	vector<Vec4i> hierarchy, hierarchy1;


	findContours(edges.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

	//Find the largest closed contour at top level !

	//Sorting the contours at 0-level first 
	pair <int, double> areaPair;
	vector<pair <int, double>> areas1;

	for (int i = 0; i < contours.size(); i++)
	{
		if ((hierarchy[i][2] >= 0) && (hierarchy[i][3] < 1) /*hierarchy[i][3] < 0 && isContourClosed(contours,i)*/)// We first see at 0-level hierarchy, if there are contour's that have child nodes, then only we go ahead 
		{
			//if (approx.size()<=4)
			//{ 
			areaPair.first = i; // contour index
			areaPair.second = contourArea(contours[i]);
			areas1.push_back(areaPair);
			//}
		}
	}

	//struct to sort the vector of pairs <int,double> based on the second double value
	struct sort_pred {
		bool operator()(const std::pair<int, double> &left, const std::pair<int, double> &right) {
			return left.second < right.second;
		}
	};

	//sort in descending order, based in contour areas !
	std::sort(areas1.rbegin(), areas1.rend(), sort_pred());
	if (areas1.empty())
	{
		return contoursTopLevel;
	}
	else
	{ 

	//pushing the contour index of the largest area contour found at the top level !
	contoursTopLevel.push_back(contours[areas1[0].first]);



	Mat drwCntr1 = emptImage.clone();
	drawContours(drwCntr1, contoursTopLevel, 0, CV_RGB(128, 255, 128), 1, 8);

	 
	imshow(imgName, drwCntr1);
	
	//String folder = "./Child_Test/" + getFileName(imgName);
	//imwrite(folder, drwCntr1);

	//cout << "\n Contour size : " + imgName << contoursTopLevel.size();

	return contoursTopLevel;
	}

}



void getForDiffFDCoeff(vector<vector<Point>> contours, int coeff, int cntrID)
{
	Mat image = emptImage.clone();
	String imageName = to_string(coeff);

	vector<vector<Point>> contoursFD;
	vector<Point> cntrReconstructed;
	FDPersoonFu fd1;

	//cntrReconstructed = fd1.generateFDHarmonics(contours, cntrID, coeff);
	contoursFD.push_back(cntrReconstructed);

	drawContours(image, contoursFD, 0, CV_RGB(0, 128, 128), 1, 8);
	imshow(imageName, image);

}

void helloThread(int i)
{

	cout << " i am from thread \n" << i;


}


//getChildContoursClosed(),getChildContoursClosedDontIntersectParent, {parentChildContoursAreaThresh() -->  this get used in both the latter methods (to discard small noisy contours)}

//Need to re-write as Recursive !!!
//vector<int> childContourClosedIndexes;
vector<int> getChildContoursClosed(vector<vector<Point>> contours, vector<Vec4i> hierarchy, int parentCntrIdx, float perimeterThresh, int level)
{
	//	Mat childContours_img;
	vector<int> childContourClosedIndexes;
	int nextNonConcentricCntrIdx = 0;
	for (int i = hierarchy[parentCntrIdx][2]; i < contours.size(); i = hierarchy[i][0]) //Starting at child of parent and looking from there onwards
	{
		if ((hierarchy[i][2] >= 0) && parentChildContoursPerimeterThresh(contours, parentCntrIdx, i, perimeterThresh))
		{
			  if (isConcentricContour(contours, parentCntrIdx, i, 80))
				{
					cout << "\n \n concentric contour " << i;
					//drawContours(imageTest, contours, i, CV_RGB(128, 128, 255), 2, 8, hierarchy, 0);
					//imshow("Concentric Contour Detected", imageTest);
					nextNonConcentricCntrIdx = getNextNonConcentricContour(contours, hierarchy, i, parentCntrIdx, 95);
					cout << "\n nextNonConcentricCntrIdx" << nextNonConcentricCntrIdx;
					//return getChildContoursClosed(contours, hierarchy, nextNonConcentricCntrIdx, perimeterThresh);
					//drawContours(imageTest, contours, nextNonConcentricCntrIdx, CV_RGB(138, 222, 14), 2, 8, hierarchy, 0);
					//imshow("nextNonConcentricCntrIdx", imageTest);
					
					for (int k = nextNonConcentricCntrIdx; k < contours.size(); k = hierarchy[k][0])
					{
						if ((hierarchy[k][2]>=0) && (parentChildContoursPerimeterThresh(contours, parentCntrIdx, k, perimeterThresh)) )
						{
							childContourClosedIndexes.push_back(k);

							if (level==2)
							{ 
							//Add all closed children of these parents ..... second level .... The inner two loops can be commented if this is not desired !
							for (int y = k; y < contours.size(); y = hierarchy[y][2])
							{

								for (int d = y; d < contours.size(); d = hierarchy[d][0])
								{
									if ((hierarchy[d][2] >= 0))// && (parentChildContoursPerimeterThresh(contours, k, y, perimeterThresh)))
									{

										childContourClosedIndexes.push_back(d);
									}
								}

							} // for the second level !
							}

							//drawContours(imageTest, contours, k, CV_RGB(0, 128, 255), 2, 8, hierarchy, 0);
							//imshow("nextNonConcentricCntrIdx", imageTest);
						}
					}
					
				}//end if isConcentricContour
				else
				{
					// cout << "\n bingo!";
					if (parentChildContoursPerimeterThresh(contours, parentCntrIdx, i, perimeterThresh))
					{
						//  cout << "\n bingo! 1";
						childContourClosedIndexes.push_back(i);
					}
				}//end else isConcentricContour

			//} // end if doContoursIntersect
			//else
			//return getChildContoursClosed(contours,hierarchy, hierarchy[i][2], perimeterThresh);

		} //end if (hierarchy[i][2] > 0)
	} //end for

	//Sorting All child of parents as per area --Write a method doing this !!
	pair <int, double> childAreaPair;
	vector<pair <int, double>> childAreas;
	for (int i = 0; i < childContourClosedIndexes.size(); i++)
	{
		childAreaPair.first = childContourClosedIndexes.at(i);
		childAreaPair.second = contourArea(Mat(contours[childContourClosedIndexes.at(i)]));
		childAreas.push_back(childAreaPair);
	}

	//sort in descending
	std::sort(childAreas.rbegin(), childAreas.rend(), sort_pred());
	for (int i = 0; i < childAreas.size(); i++)
	{
		childContourClosedIndexes.at(i) = childAreas.at(i).first;
	}

	return childContourClosedIndexes;
}


bool isConcentricContour(vector<vector<Point>> contours, int parentCntrIdx, int childCntrIndx, float perimeterThresh)
{
	double parentPerimeter, childPerimeter;
	parentPerimeter = arcLength(contours[parentCntrIdx], true);
	childPerimeter = arcLength(contours[childCntrIndx], true);

	vector<Moments> mu(2);
	//get moments
	mu[0] = moments(contours[parentCntrIdx], false);
	mu[1] = moments(contours[childCntrIndx], false);
	//parentChildContoursAreaThresh(contours, parentCntrIdx, childCntrIndx, areaThresh);
	//get mass centers
	vector<Point2f> mc(2);

	mc[1] = Point2f(mu[1].m10 / mu[1].m00, mu[1].m01 / mu[1].m00); //// child center of mass Point 
	mc[0] = Point2f(mu[0].m10 / mu[0].m00, mu[0].m01 / mu[0].m00); // parent center of mass Point 

	double res = cv::norm(mc[1] - mc[0]); //distance between center of mass

	//Area of contours
	//double parentArea, childArea;
	//parentArea = contourArea(Mat(contours[parentCntrIdx]));
	//childArea = contourArea(Mat(contours[childCntrIndx]));

	//Compactness of child
	//float compactness;
	//compactness = ((4 * pi*childArea) / (childPerimeter*childPerimeter));
	//double matchScore ;
	//matchScore=matchShapes(contours[parentCntrIdx], contours[childCntrIndx], CV_CONTOURS_MATCH_I1, 0);

	//HardCoding perimter threshold to 85 % and distance between the center of mass of the two concentric ones is now 5 units
	if (((childPerimeter / parentPerimeter) * 100 > 40) && (res < 5) )
	{
		cout << "\n";
		cout << "child contours " << childCntrIndx << " is concentric to parent " << parentCntrIdx;
		cout << "\n Perimeter Ratio child/parent is % " << (childPerimeter / parentPerimeter) * 100 << "  distance between center of mass is \n" << res;
		return true;
	}
	else
	{
		cout << "\n child contours " << childCntrIndx << " is NOT concentric to parent " << parentCntrIdx;
		cout << "\n Perimeter Ratio child/parent is % " << (childPerimeter / parentPerimeter) * 100 << "  distance between center of mass is " << res;
		return false;
	}

}

int getNextNonConcentricContour(vector<vector<Point>> contours, vector<Vec4i> hierarchy, int concentricContrIdx, int parentCntrIdx, float perimeterThresh)
{
	if ((hierarchy[concentricContrIdx][2] >= 0))
	{
		for (int i = hierarchy[concentricContrIdx][2]; i < contours.size(); i = hierarchy[i][0])
		{ //looping through all child of the concentric contour and trying to weed out noise ...
			if ((hierarchy[i][2] >= 0) && parentChildContoursPerimeterThresh(contours, parentCntrIdx, i, 5)) // Create a global variable for this permThresh --this is suppose to be the parentChild one !
			{
				if (isConcentricContour(contours, concentricContrIdx, i, perimeterThresh))
				{
					return getNextNonConcentricContour(contours, hierarchy, i,  parentCntrIdx, perimeterThresh);
				}
				else
					return i;
			}
		} //end for
	}
	else return concentricContrIdx;
}

bool parentChildContoursPerimeterThresh(vector<vector<Point>> contours, int parentCntrIdx, int childCntrIndx, float perimeterThresh)
{

	double parentPerimeter, childPerimeter;
	parentPerimeter = arcLength(contours[parentCntrIdx], true);
	childPerimeter = arcLength(contours[childCntrIndx], true);

	//Area of contours
	double parentArea, childArea;
	parentArea = contourArea(Mat(contours[parentCntrIdx]));
	childArea = contourArea(Mat(contours[childCntrIndx]));

	//Compactness of child
	//float compactness;
	//compactness = ((4 * pi*childArea) / (childPerimeter*childPerimeter));

	Rect bRectParent, bRectChild;
	//get bounding rect ... parent and child
	//also experimenting with the contour areas ... 

	bRectParent = boundingRect(contours[parentCntrIdx]);
	bRectChild = boundingRect(contours[childCntrIndx]);

	double pthresh = (childPerimeter / parentPerimeter) * 100;
	//if (((childPerimeter / parentPerimeter) * 100  > perimeterThresh) && !doContoursIntersect(contours,parentCntrIdx, childCntrIndx))
	//if ((pthresh > perimeterThresh) && (pthresh < (100 - perimeterThresh)))
	if (pthresh > perimeterThresh)
		return true;
	else
		return false;

}



int findNoOfChildren(vector<vector<Point>> contours, vector<Vec4i> hierarchy, int parentCntrIdx)
{
	int cntr = 0;
	if (hierarchy[parentCntrIdx][2] > 0)
	{
		for (int i = hierarchy[parentCntrIdx][2]; i < contours.size(); i = hierarchy[i][0])
			cntr++;
		return cntr;
	}
	else
		return 0;
}

bool doContoursIntersect(vector<vector<Point>> contours, int parentCntrIdx, int childCntrIndx)
{
	//It should be docontoursRectanglesintersect as we dont want that ...
	//vector<Point> opPoints;
	RotatedRect rec1, rec2;
	cv::Rect r1 = cv::boundingRect(contours[parentCntrIdx]);
	cv::Rect r2 = cv::boundingRect(contours[childCntrIndx]);



	Mat opRegion;
	rec1 = minAreaRect(contours[parentCntrIdx]);
	rec2 = minAreaRect(contours[childCntrIndx]);

	//bool intersecStatus = true;

	if (rotatedRectangleIntersection(rec1, rec2, opRegion) == INTERSECT_FULL)
	{
		return false;
	}
	else
	{
		cout << "\n child contour " << childCntrIndx << " lies outside or on parent  " << parentCntrIdx;
		return true;
	}


	/*
	for (int i = 0; i < contours[childCntrIndx].size() ; i+=10)
	{
	if (pointPolygonTest(contours[parentCntrIdx], contours[childCntrIndx][i],false) <= 0) //checking here if any point lies outside or on the parent contour
	{
	cout << "\n child contour lies outside or on parent  " << parentCntrIdx << " and " << childCntrIndx;
	intersecStatus = true;
	break;
	}
	}
	*/



}


bool doContoursIntersect1(vector<vector<Point>> contours, int parentCntrIdx, int childCntrIndx)
{

	bool intersecStatus = false;

	for (int i = 0; i < contours[childCntrIndx].size(); i++)
	{
		if (pointPolygonTest(contours[parentCntrIdx], contours[childCntrIndx][i], false) <= 0) //checking here if any point lies outside or on the parent contour
		{
			cout << "\n child contour " << childCntrIndx << " lies outside or on parent  " << parentCntrIdx;
			return intersecStatus = true;
			break;
		}
	}

	return intersecStatus;
}

bool isContourClosed(vector<vector<Point>> contours, int contourIdx)
{
	// 1.5 is approximately sqrt(2)
	if (cv::norm(contours[contourIdx].front() - contours[contourIdx].back()) < 1.5)
		return true;
	else
		return false;
}


// approach to extract child contours !
//Another probable approach might be -- from the top level parent -- dive in ... check for all elements that satisfy the parentchildperimeterTest , -- maybe sort all first based on Area ... to discard 
/// unsuitable ones .... see if its enclosed then only calculate area .... / then sort top 5/10 ... then do the perimter sum check ... then do intersection check (we also dont want any intersecting contours with parent
//child that are closed , child that are completely
// within the parent - rotatedRectangleIntersection(rec1, rec2, opRegion); ...... and we ignore child that are concentric to parent .....


