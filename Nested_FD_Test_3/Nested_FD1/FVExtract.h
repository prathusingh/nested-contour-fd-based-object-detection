#include <opencv2\opencv.hpp>
#include <opencv2\imgproc\imgproc.hpp>
//#include <vector>
//#include <numeric>


#include "FDPersoonFu.h"
#include "EllipticalFD.h"


#include<thread>
#include <future>  

#include <opencv2/ml/ml.hpp>

#define pi  3.141592653
using namespace cv;
using namespace std;


vector<vector<Point>> preProcessContour(String imgName);
void appendFVToFileEFD(String fileName, String imagePath, String fvName, int numberOfCoeff);
//Mat queryFvEFD(String imagePath, int numberOfCoeff);
//vector<Point2d> queryFvPFu(String imagePath, int numberOfCoeff);
void appendFVToFilePFu(String fileName, String imagePath, String fvName, int numberOfCoeff);

void writeVectorOfVector(FileStorage &fs, string name, vector<vector<Point2d>> &vov);
void readVectorOfVector(FileStorage &fs, string name, vector<vector<Point2d>> &vov);

//struct to sort the vector of pairs <int,double> based on the second double value
struct sort_pred {
	bool operator()(const std::pair<int, double> &left, const std::pair<int, double> &right) {
		return left.second < right.second;
	}
};

//Utility function to parse a path name in order to get the filename
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

Mat imageTest, emptImage;
Mat globalFV;
vector<vector<Point2d>>globalFvPFu;


class FVExtract {
public:
	/*
	bool extractFV = false;
	bool useSVM = false;
	bool useKNN = false;
	bool useEcldDist = false;
	bool extractFVForEFD = false;
	bool extractFVForPFuFD = false;

	bool extractFVForEFDChild = false;
	bool extractFVForPFuFDChild = false;
	*/

	void  extractFVForEFDChild()
	{
		vector<string> folderNames;
		//folderNames.push_back("WALKING_MAN");
		folderNames.push_back("S");
		//folderNames.push_back("T");
		//folderNames.push_back("O");
		//folderNames.push_back("P");
		String ROOT_DIR = "C:\\Users\\UTRGVCS\\Downloads\\DataAugmentation-master\\Out";

		//Access individual folders ...
		for (int k = 0; k < folderNames.size(); k++)
		{
			vector<String> filenames;
			String path1 = ROOT_DIR + "\\" + folderNames[k];
			glob(path1, filenames);
			vector<int> coeffs;
			coeffs.push_back(360); coeffs.push_back(512); coeffs.push_back(1024);

			for (int x = 0; x < coeffs.size(); x++)
			{
				for (size_t i = 0; i < filenames.size(); ++i)
				{
					try{
						cout << "\n " << filenames[i];
						appendFVToFileEFD("./FV/" + folderNames[k] + to_string(coeffs[x]) + ".yml", filenames[i], folderNames[k], coeffs[x]);
					}
					catch (string std)
					{
						cout << "  " + folderNames[k] + "  " + std;
					}

				}

			}
			filenames.clear();
		}


	}



	void extractFVForPFuFDChild()
	{
		vector<string> folderNames;
		folderNames.push_back("WALKING_MAN");
		folderNames.push_back("S");
		folderNames.push_back("T");
		folderNames.push_back("O");
		folderNames.push_back("P");
		String ROOT_DIR = "C:\\Users\\UTRGVCS\\Downloads\\DataAugmentation-master\\Out";

		//Access individual folders ...
		for (int k = 0; k < folderNames.size(); k++)
		{
			vector<String> filenames;
			String path1 = ROOT_DIR + "\\" + folderNames[k];
			glob(path1, filenames);
			vector<int> coeffs;
			coeffs.push_back(256); coeffs.push_back(512); coeffs.push_back(1024);

			for (int x = 0; x < coeffs.size(); x++)
			{
				for (size_t i = 0; i < filenames.size(); ++i)
				{
					try{
						cout << "\n " << filenames[i];
						appendFVToFilePFu("./FV/" + folderNames[k] + "P_Fu" + to_string(coeffs[x]) + ".yml", filenames[i], folderNames[k], coeffs[x]);
					}
					catch (string std)
					{
						cout << "  " + folderNames[k] + "  " + std;
					}
				}
				FileStorage PFu("./FV/" + folderNames[k] + "P_Fu" + to_string(coeffs[x]) + ".yml", FileStorage::WRITE);
				writeVectorOfVector(PFu, folderNames[k], globalFvPFu);
				globalFvPFu.clear();
			}
			filenames.clear();
		}


	}




	void extractFVForEFD()
	{
		vector<String> filenames, filenames2; // notice here that we are using the Opencv's embedded "String" class
		String tr1 = "C:\\Users\\UTRGVCS\\Downloads\\DataAugmentation-master\\Out\\TRIANGLE"; // again we are using the Opencv's embedded "String" class
		String oct1 = "C:\\Users\\UTRGVCS\\Downloads\\DataAugmentation-master\\Out\\OCTAGON";

		glob(tr1, filenames); // new function that does the job ;-)

		for (size_t i = 0; i < filenames.size(); ++i)
		{
			cout << "\n " << filenames[i];
			appendFVToFileEFD("./FV/TRIANGLE122.yml", filenames[i], "TRIANGLE", 1024);

		}
		filenames.clear();
		globalFV.release();


		glob(oct1, filenames2);
		for (size_t i = 0; i < filenames2.size(); ++i)
		{
			cout << "\n " << filenames2[i];
			appendFVToFileEFD("./FV/OCTAGON122.yml", filenames2[i], "OCTAGON", 1024);

		}
		filenames2.clear();
		globalFV.release();

	}

	void extractFVForPFuFD()
	{
		vector<String> filenames; // notice here that we are using the Opencv's embedded "String" class
		String tr1 = "C:\\Users\\UTRGVCS\\Downloads\\DataAugmentation-master\\Out\\TRIANGLE"; // again we are using the Opencv's embedded "String" class
		String oct1 = "C:\\Users\\UTRGVCS\\Downloads\\DataAugmentation-master\\Out\\OCTAGON";
		String dia1 = "C:\\Users\\UTRGVCS\\Downloads\\DataAugmentation-master\\Out\\DIAMOND";
		String circ1 = "C:\\Users\\UTRGVCS\\Downloads\\DataAugmentation-master\\Out\\circles";
		String rect1 = "C:\\Users\\UTRGVCS\\Downloads\\DataAugmentation-master\\Out\\RECTANGLE";


		glob(tr1, filenames);
		for (size_t i = 0; i < filenames.size(); ++i)
		{
			appendFVToFilePFu("./FV/TRIANGLE_PFu.yml", filenames[i], "TRIANGLE", 1024);
		}
		filenames.clear();
		FileStorage triPFu("./FV/TRIANGLE_PFu.yml", FileStorage::WRITE);
		writeVectorOfVector(triPFu, "TRIANGLE", globalFvPFu);
		globalFvPFu.clear();


		glob(oct1, filenames);
		for (size_t i = 0; i < filenames.size(); ++i)
		{
			appendFVToFilePFu("./FV/OCTAGON_PFu.yml", filenames[i], "OCTAGON", 1024);
		}
		filenames.clear();
		FileStorage octPFu("./FV/OCTAGON_PFu.yml", FileStorage::WRITE);
		writeVectorOfVector(octPFu, "OCTAGON", globalFvPFu);
		globalFvPFu.clear();


		glob(dia1, filenames);
		for (size_t i = 0; i < filenames.size(); ++i)
		{
			appendFVToFilePFu("./FV/DIAMOND_PFu.yml", filenames[i], "DIAMOND", 1024);
		}
		filenames.clear();
		FileStorage diaPFu("./FV/DIAMOND_PFu.yml", FileStorage::WRITE);
		writeVectorOfVector(diaPFu, "DIAMOND", globalFvPFu);
		globalFvPFu.clear();


		glob(circ1, filenames);
		for (size_t i = 0; i < filenames.size(); ++i)
		{
			appendFVToFilePFu("./FV/CIRCLES_PFu.yml", filenames[i], "CIRCLES", 1024);
		}
		filenames.clear();
		FileStorage circPFu("./FV/CIRCLES_PFu.yml", FileStorage::WRITE);
		writeVectorOfVector(circPFu, "CIRCLES", globalFvPFu);
		globalFvPFu.clear();


		glob(rect1, filenames);
		for (size_t i = 0; i < filenames.size(); ++i)
		{
			appendFVToFilePFu("./FV/RECTANGLE_PFu.yml", filenames[i], "RECTANGLE", 1024);
		}
		filenames.clear();
		FileStorage rectPFu("./FV/RECTANGLE_PFu.yml", FileStorage::WRITE);
		writeVectorOfVector(rectPFu, "RECTANGLE", globalFvPFu);
		globalFvPFu.clear();


	}



	void extractFV()
	{
		//Octagon shapes
		appendFVToFileEFD("./FV/OCTAGON.yml", "./STOP/Parent/s1.jpg", "OCTAGON", 1024);
		appendFVToFileEFD("./FV/OCTAGON.yml", "./STOP/Parent/s2.jpg", "OCTAGON", 1024);
		appendFVToFileEFD("./FV/OCTAGON.yml", "./STOP/Parent/sign2.jpg", "OCTAGON", 1024);
		appendFVToFileEFD("./FV/OCTAGON.yml", "./STOP/Parent/s4.jpg", "OCTAGON", 1024);
		appendFVToFileEFD("./FV/OCTAGON.yml", "./STOP/Parent/s5.jpg", "OCTAGON", 1024);
		appendFVToFileEFD("./FV/OCTAGON.yml", "./STOP/Parent/s6.jpg", "OCTAGON", 1024);
		appendFVToFileEFD("./FV/OCTAGON.yml", "./STOP/Parent/stop11.jpg", "OCTAGON", 1024);
		globalFV.release();

		//diamond shapes
		appendFVToFileEFD("./FV/DIAMOND.yml", "./TrafficSigns/d1.jpg", "DIAMOND", 1024);
		appendFVToFileEFD("./FV/DIAMOND.yml", "./TrafficSigns/d2.jpg", "DIAMOND", 1024);
		appendFVToFileEFD("./FV/DIAMOND.yml", "./TrafficSigns/d3.jpg", "DIAMOND", 1024);
		appendFVToFileEFD("./FV/DIAMOND.yml", "./TrafficSigns/d4.jpg", "DIAMOND", 1024);
		globalFV.release();

		//circle / oval shapes
		appendFVToFileEFD("./FV/CIRCLE.yml", "./TrafficSigns/c1.jpg", "CIRCLE", 1024);
		appendFVToFileEFD("./FV/CIRCLE.yml", "./TrafficSigns/c2.jpg", "CIRCLE", 1024);
		appendFVToFileEFD("./FV/CIRCLE.yml", "./TrafficSigns/oval1.jpg", "CIRCLE", 1024);
		appendFVToFileEFD("./FV/CIRCLE.yml", "./TrafficSigns/test.jpg", "CIRCLE", 1024);
		appendFVToFileEFD("./FV/CIRCLE.yml", "./TrafficSigns/test1.jpg", "CIRCLE", 1024);
		appendFVToFileEFD("./FV/CIRCLE.yml", "./TrafficSigns/test2.jpg", "CIRCLE", 1024);
		appendFVToFileEFD("./FV/CIRCLE.yml", "./TrafficSigns/test3.jpg", "CIRCLE", 1024);
		appendFVToFileEFD("./FV/CIRCLE.yml", "./TrafficSigns/conctest.jpg", "CIRCLE", 1024);
		appendFVToFileEFD("./FV/CIRCLE.yml", "./TrafficSigns/conctest1.jpg", "CIRCLE", 1024);
		appendFVToFileEFD("./FV/CIRCLE.yml", "./TrafficSigns/conc2.jpg", "CIRCLE", 1024);
		globalFV.release();


		//Rectangle / Squares
		appendFVToFileEFD("./FV/RECTANGLE.yml", "./TrafficSigns/rec1.jpg", "RECTANGLE", 1024);
		appendFVToFileEFD("./FV/RECTANGLE.yml", "./TrafficSigns/rec2.jpg", "RECTANGLE", 1024);
		appendFVToFileEFD("./FV/RECTANGLE.yml", "./TrafficSigns/rec3.jpg", "RECTANGLE", 1024);
		appendFVToFileEFD("./FV/RECTANGLE.yml", "./TrafficSigns/rec4.jpg", "RECTANGLE", 1024);
		globalFV.release();

		//Cross
		appendFVToFileEFD("./FV/CROSS.yml", "./TrafficSigns/rr1.jpg", "CROSS", 1024);
		globalFV.release();
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


			//imshow(imgName, drwCntr1);

			//String folder = "./Child_Test/" + getFileName(imgName);
			//imwrite(folder, drwCntr1);

			//cout << "\n Contour size : " + imgName << contoursTopLevel.size();

			return contoursTopLevel;
		}

	}


};