#include <opencv2\opencv.hpp>

using namespace cv;
using namespace std;

class ParallelCompute : public cv::ParallelLoopBody
{

private:
	//cv::Mat img;
	//cv::Mat& retVal;
	vector<vector<Point>>  contoursPin;
	vector<Vec4i> hierarchyPin;
	float perimeterThreshin;
	vector<pair <int, double>> areasPin;
	int maxTopLevelContoursIn;
	vector<pair<int, vector<int>>>& allTopLevelContoursPout;

public:
	//Parallel_process(cv::Mat inputImgage, cv::Mat& outImage)
	//	: img(inputImgage), retVal(outImage){}

	ParallelCompute(vector<vector<Point>>  contours, vector<Vec4i> hierarchy, float perimeterThresh, vector<pair <int, double>> areas1, int maxTopLevelContours, vector<pair<int, vector<int>>> allTopLevelContours)
		: contoursPin(contours), hierarchyPin(hierarchy), perimeterThreshin(perimeterThresh), areasPin(areas1), maxTopLevelContoursIn(maxTopLevelContours), allTopLevelContoursPout(allTopLevelContours){}

	virtual void operator()(const cv::Range& range) const
	{
		for (int i = range.start; i < range.end; i++)
		{
			// Your code here
			allTopLevelContoursPout.at(i).first = areasPin.at(i).first;
			allTopLevelContoursPout.at(i).second = getChildContoursClosed(contoursPin, hierarchyPin, areasPin.at(i).first, perimeterThreshin);
		}
	}



	vector<int> getChildContoursClosed(vector<vector<Point>> contours, vector<Vec4i> hierarchy, int parentCntrIdx, float perimeterThresh) const
	{
		//	Mat childContours_img;
		vector<int> childContourClosedIndexes;
		int nextNonConcentricCntrIdx = 0;
		for (int i = hierarchy[parentCntrIdx][2]; i < contours.size(); i = hierarchy[i][0]) //Starting at child of parent and looking from there onwards
		{
			if ((hierarchy[i][2] >= 0) && parentChildContoursPerimeterThresh(contours, parentCntrIdx, i, perimeterThresh))
			{
				if (isConcentricContour(contours, parentCntrIdx, i, 95))
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
						if ((hierarchy[k][2] >= 0) && (parentChildContoursPerimeterThresh(contours, parentCntrIdx, k, perimeterThresh)))
						{
							childContourClosedIndexes.push_back(k);
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

		/*
		for (int i = 0; i < contours.size(); i++)
		{
		if ((hierarchy[i][3] == parentCntrIdx) && (hierarchy[i][2]>0)) //Contour is a child of parentCntrIdx & the contour is closed i.e. it has a child enclosed ...
		{
		for (int j = hierarchy[i][2]; j <contours.size(); j = hierarchy[j][0]) //Check all next ones at that level i.e. at level hierarchy[i][2]
		{
		if (parentChildContoursAreaSimilarityThresh(contours, i, j, 95))
		{
		for (int k = hierarchy[j][2]; k < contours.size(); k = hierarchy[k][0])
		{
		if (parentChildContoursAreaThresh(contours, parentCntrIdx, k, areaThresh))
		{
		childContourClosedIndexes.push_back(k);
		}
		}
		}
		else if (parentChildContoursAreaThresh(contours, parentCntrIdx, j, areaThresh))
		{
		childContourClosedIndexes.push_back(j); //Add to/Insert to vector
		}//end if

		} //end for j

		} //end if

		} //end for i
		*/
		return childContourClosedIndexes;
	}


	bool isConcentricContour(vector<vector<Point>> contours, int parentCntrIdx, int childCntrIndx, float perimeterThresh) const
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



		//HardCoding perimter threshold to 85 % and distance between the center of mass of the two concentric ones is now 5 units
		if (((childPerimeter / parentPerimeter) * 100 > 90) && (res < 5))
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

	int getNextNonConcentricContour(vector<vector<Point>> contours, vector<Vec4i> hierarchy, int concentricContrIdx, int parentCntrIdx, float perimeterThresh) const
	{
		if ((hierarchy[concentricContrIdx][2] >= 0))
		{
			for (int i = hierarchy[concentricContrIdx][2]; i < contours.size(); i = hierarchy[i][0])
			{ //looping through all child of the concentric contour and trying to weed out noise ...
				if ((hierarchy[i][2] >= 0) && parentChildContoursPerimeterThresh(contours, parentCntrIdx, i, 5)) // Create a global variable for this permThresh --this is suppose to be the parentChild one !
				{
					if (isConcentricContour(contours, concentricContrIdx, i, perimeterThresh))
					{
						return getNextNonConcentricContour(contours, hierarchy, i, parentCntrIdx, perimeterThresh);
					}
					else
						return i;
				}
			} //end for
		}
		else return concentricContrIdx;
	}

	bool parentChildContoursPerimeterThresh(vector<vector<Point>> contours, int parentCntrIdx, int childCntrIndx, float perimeterThresh) const
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


		//if (((childPerimeter / parentPerimeter) * 100  > perimeterThresh) && !doContoursIntersect(contours,parentCntrIdx, childCntrIndx))
		if (((childPerimeter / parentPerimeter) * 100  > perimeterThresh))
			return true;
		else
			return false;

	}



	int findNoOfChildren(vector<vector<Point>> contours, vector<Vec4i> hierarchy, int parentCntrIdx) const
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

	bool doContoursIntersect(vector<vector<Point>> contours, int parentCntrIdx, int childCntrIndx) const
	{
		//vector<Point> opPoints;
		RotatedRect rec1, rec2;


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


	bool doContoursIntersect1(vector<vector<Point>> contours, int parentCntrIdx, int childCntrIndx) const
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

	bool isContourClosed(vector<vector<Point>> contours, int contourIdx) const
	{
		// 1.5 is approximately sqrt(2)
		if (cv::norm(contours[contourIdx].front() - contours[contourIdx].back()) < 1.5)
			return true;
		else
			return false;
	}




};
/*
int main(int argc, char* argv[])
{
cv::Mat img, out;
img = cv::imread(argv[1]);
out = cv::Mat::zeros(img.size(), CV_8UC3);

// create 8 threads and use TBB
cv::parallel_for_(cv::Range(0, 8), Parallel_process(img, out));
return(1);
}
*/