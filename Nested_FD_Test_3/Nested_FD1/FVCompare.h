#include <opencv2\opencv.hpp>
#include <opencv2\imgproc\imgproc.hpp>
//#include <vector>
//#include <numeric>
#include "FDPersoonFu.h"
#include<thread>
#include <future>  

#include <opencv2/ml/ml.hpp>

#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

#define pi  3.141592653
using namespace cv;
using namespace std;

class FVCompare {
public:

	
	static vector <vector < pair <string, int>>> namesLabelPairParent, namesLabelPairChild;
	static vector <Mat> allFvMatParent, allFvMatChild;
	static vector<Mat> allFvLabelsParent, allFvLabelsChild;
	static vector<vector< pair <vector<vector<Point2d>>, string >>>mergedFVParent, mergedFVChild;



	//De-serialize from the stored FV's
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


	void deserializeFvEFDParent()
	{
		vector<int> coeffs;
		coeffs.push_back(1024); 

		for (int y = 0; y < coeffs.size();y++)
		{
			vector<String> filenames;
			String dir = "./FV/Parents/EFD/" + to_string(coeffs[y]) + "/";
		glob(dir, filenames);
	
		for (int i = 0; i < filenames.size(); i++)
		{
			filenames[i] = getFileName(filenames[i]);
		}

		vector<Mat> parentShapes;
		//trying labelling
		
		vector<Mat> indivLabels;
		vector<pair < string, int >>namesLabelPairParentL;

		for (int i = 0; i < filenames.size(); i++)
		{
			cout << "\n  " << dir + filenames[i];
			FileStorage pr(dir+filenames[i], FileStorage::READ);
			
			cout << "\n  " << splitStringDelim(filenames[i], '_');
			
			Mat tempPS;
			pr[splitStringDelim(filenames[i], '_')] >> tempPS;
			
			pair < string, int > tempNamesLabelPair;
			tempNamesLabelPair.first = splitStringDelim(filenames[i], '_');
			tempNamesLabelPair.second = i;
			namesLabelPairParentL.push_back(tempNamesLabelPair);

			Mat labels1(tempPS.rows, 1, CV_32SC1);
			labels1.rowRange(0, labels1.rows).setTo((Scalar(i)));
			indivLabels.push_back(labels1);
			parentShapes.push_back(tempPS);

			labels1.release();
			pr.release();
			tempPS.release();
		}
		
		namesLabelPairParent.push_back(namesLabelPairParentL);
		namesLabelPairParentL.clear();

		filenames.clear();
		Mat tempPr, tempL;
		vconcat(parentShapes, tempPr);
		vconcat(indivLabels, tempL);
		allFvMatParent.push_back(tempPr);
		allFvLabelsParent.push_back(tempL);

		parentShapes.clear();
		indivLabels.clear();
		tempPr.release();
		tempL.release();

		}

		cout << " \n  allFvMatParent " << allFvMatParent.size();
		cout << " \n  allFvLabelParent " << allFvLabelsParent.size();
		cout << endl;
		//cout << allFvLabelsParent << endl;


	}



	void deserializeFvEFDChild()
	{
		vector<int> coeffs;
		coeffs.push_back(1024); coeffs.push_back(512); coeffs.push_back(360);

		for (int y = 0; y < coeffs.size(); y++)
		{

			vector<String> filenames;
			String dir = "./FV/Child/EFD/" + to_string(coeffs[y]) + "/";
			glob(dir, filenames);

			for (int i = 0; i < filenames.size(); i++)
			{
				filenames[i] = getFileName(filenames[i]);
			}

			vector<Mat> childShapes;
			//trying labelling

			vector<Mat> indivLabels;
			vector<pair < string, int >>namesLabelPairChildL;


			for (int i = 0; i < filenames.size(); i++)
			{
				cout << "\n  " << dir + filenames[i];
				FileStorage pr(dir + filenames[i], FileStorage::READ);

				cout << "\n  " << splitStringDelim(filenames[i], '_');

				Mat tempPS;
				pr[splitStringDelim(filenames[i], '_')] >> tempPS;

				pair < string, int > tempNamesLabelPair;
				tempNamesLabelPair.first = splitStringDelim(filenames[i], '_');
				tempNamesLabelPair.second = i;
				namesLabelPairChildL.push_back(tempNamesLabelPair);

				Mat labels1(tempPS.rows, 1, CV_32SC1);
				labels1.rowRange(0, labels1.rows).setTo((Scalar(i)));
				indivLabels.push_back(labels1);
				childShapes.push_back(tempPS);

				labels1.release();
				pr.release();
				tempPS.release();
			}

			namesLabelPairChild.push_back(namesLabelPairChildL);
			namesLabelPairChildL.clear();

			filenames.clear();
			Mat tempPr, tempL;

			vconcat(childShapes, tempPr);
			vconcat(indivLabels, tempL);
			allFvMatChild.push_back(tempPr);
			allFvLabelsChild.push_back(tempL);

			childShapes.clear();
			indivLabels.clear();
			tempPr.release();
			tempL.release();
		

		}

		cout << " \n  allFvMatChild " << allFvMatChild.size();
		cout << " \n  allFvLabelChild " << allFvLabelsChild.size();
		cout << endl;


	}


	void deserializeFvPFuParent(int numberOfFV)
	{

		vector<int> coeffs;
		coeffs.push_back(1024);

		for (int y = 0; y < coeffs.size(); y++)
		{


		vector<String> filenames;
		String dir = "./FV/Parents/PFu/" + to_string(coeffs[y]) + "/";
		glob(dir, filenames);

		for (int i = 0; i < filenames.size(); i++)
		{
			filenames[i] = getFileName(filenames[i]);
		}

		
		
		vector<pair <vector<vector<Point2d>>, string >> mergedFVParentT;
		for (int i = 0; i < filenames.size(); i++)
		{
			cout << "\n  " << dir + filenames[i];
			FileStorage pr(dir + filenames[i], FileStorage::READ);

			cout << "\n  " << splitStringDelim(filenames[i], '_');

			vector<vector<Point2d>> tempPS;
			readVectorOfVector(pr, splitStringDelim(filenames[i], '_'), tempPS);

			pair <vector<vector<Point2d>>, string > fvPair;
			
			//Shuffling the FV for the shape
			std::random_shuffle(tempPS.begin(), tempPS.end());
			//Now selecting first n values 
			tempPS.resize(numberOfFV);

			fvPair.first = tempPS;
			fvPair.second = splitStringDelim(filenames[i], '_');
			mergedFVParentT.push_back(fvPair);

			pr.release();
			tempPS.clear();
		}

		mergedFVParent.push_back(mergedFVParentT);
		mergedFVParentT.clear();
		filenames.clear();
		}


		//cout << mergedFVParent.size();
		
		cout << "\n  mergedFVParent PFu " << mergedFVParent.size();
		cout << endl;


	}


	void deserializeFvPFuChild(int numberOfFV)
	{

		vector<int> coeffs;
		coeffs.push_back(1024); coeffs.push_back(512); coeffs.push_back(256);

		for (int y = 0; y < coeffs.size(); y++)
		{



			vector<String> filenames;
			String dir = "./FV/Child/PFu/" + to_string(coeffs[y]) + "/";
			glob(dir, filenames);

			for (int i = 0; i < filenames.size(); i++)
			{
				filenames[i] = getFileName(filenames[i]);
			}



			vector<pair <vector<vector<Point2d>>, string >> mergedFVChildT;
			for (int i = 0; i < filenames.size(); i++)
			{
				cout << "\n  " << dir + filenames[i];
				FileStorage pr(dir + filenames[i], FileStorage::READ);

				cout << "\n  " << splitStringDelim(filenames[i], '_');

				vector<vector<Point2d>> tempPS;
				readVectorOfVector(pr, splitStringDelim(filenames[i], '_'), tempPS);

				pair <vector<vector<Point2d>>, string > fvPair;

				//Shuffling the FV for the shape
				std::random_shuffle(tempPS.begin(), tempPS.end());
				//Now selecting first n values 
				tempPS.resize(numberOfFV);

				fvPair.first = tempPS;
				fvPair.second = splitStringDelim(filenames[i], '_');
				mergedFVChildT.push_back(fvPair);

				pr.release();
				tempPS.clear();
			}

			mergedFVChild.push_back(mergedFVChildT);
			mergedFVChildT.clear();
			filenames.clear();



		}

		//cout << mergedFVChild.size();
		
		cout << "\n  mergedFVChild PFu " << mergedFVChild.size();
		cout << endl;


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

	void useEcldDist_Parents( int coeff, Mat query,int compareCoeff)
	{
		int c;
		switch (coeff)
		{
		case 1024: c = 0;
			break;
		default: cout << " \n Coeff entered do not exist : 1024"; break;
		}


		//Min Euclidean distance over the entire data2 matrix
		
		float minScore = 10000;
		Mat q = query.colRange(0, compareCoeff);
		vector<float> vQ;
		vQ.assign((float*)q.datastart, (float*)q.dataend);


		int rowIndex;


		for (int i = 0; i < allFvMatParent[c].rows; i++)
		{
			Mat row = allFvMatParent[c].row(i);
			Mat dbvec = row.colRange(0, compareCoeff);
			//Convert to vector
			//vector<float> vD;
			//vD.assign((float*)dbvec.datastart, (float*)dbvec.dataend);

			float score = 0;
			score = norm(q, dbvec, cv::NormTypes::NORM_L2SQR);

			if (score < minScore)
			{
				minScore = score;
				rowIndex = i;
			}

		}


		cout << "\n \n EUCLIDEAN DIST -- Distance between : " << minScore;

		cout << "\n Label : " << allFvLabelsParent[c].at<int>(rowIndex);

		//integer label
		int label;
		label = allFvLabelsParent[c].at<int>(rowIndex);

		vector< pair <string, int> >::iterator it = find_if(namesLabelPairParent[c].begin(), namesLabelPairParent[c].end(), [&label](const pair<string, int>& element){ return element.second == label; });
		//pair found
		cout << "\n the shape is  " << it->first;
		cout << endl;


	}


	void useEcldDist_Child(int coeff, Mat query, int compareCoeff)
	{
		int c;
		switch (coeff)
		{
		case 1024: c = 0; break;
		case 512: c = 1; break;
		case 360: c = 2; break;
		default: cout << " \n Coeff entered do not exist : 1024,512 or 360 !"; break;
		}


		//Min Euclidean distance over the entire data2 matrix

		float minScore = 10000;
		Mat q = query.colRange(0, compareCoeff);
		vector<float> vQ;
		vQ.assign((float*)q.datastart, (float*)q.dataend);


		int rowIndex;


		for (int i = 0; i < allFvMatChild[c].rows; i++)
		{
			Mat row = allFvMatChild[c].row(i);
			Mat dbvec = row.colRange(0, compareCoeff);
			//Convert to vector
			//vector<float> vD;
			//vD.assign((float*)dbvec.datastart, (float*)dbvec.dataend);

			float score = 0;
			score = norm(q, dbvec, cv::NormTypes::NORM_L2SQR);

			if (score < minScore)
			{
				minScore = score;
				rowIndex = i;
			}

		}


		cout << "\n \n EUCLIDEAN DIST -- Distance between : " << minScore;

		cout << "\n Label : " << allFvLabelsChild[c].at<int>(rowIndex);

		//integer label
		int label;
		label = allFvLabelsChild[c].at<int>(rowIndex);

		vector< pair <string, int> >::iterator it = find_if(namesLabelPairParent[c].begin(), namesLabelPairParent[c].end(), [&label](const pair<string, int>& element){ return element.second == label; });
		//pair found
		cout << "\n the shape is  " << it->first;
		cout << endl;


	}


	void useKNN_Parents(int K,int coeff,Mat query)
	{
		int c;
		switch (coeff)
		{
		case 1024: c = 0;
			break;
		default: cout << " \n Coeff entered do not exist : 1024"; break;
		}

		//KNN
		
		Ptr<ml::KNearest> knn = ml::KNearest::create();
		knn->setDefaultK(K);
		knn->setIsClassifier(true);


		knn->setAlgorithmType(ml::KNearest::Types::BRUTE_FORCE);
		knn->train(allFvMatParent[c], ml::ROW_SAMPLE, allFvLabelsParent[c]);

		cout << "\n Data Rows " << allFvMatParent[c].rows << " Data cols " << allFvMatParent[c].cols;

		Mat res2;
		knn->predict(query, res2);
		//knn->findNearest(query, knn->getDefaultK(), res2);
		cout << "\n  RESULT OF KNN  " << res2.at<float>(0,0) << endl << "\n";
		//integer label
		int label;
		label = int(res2.at<float>(0, 0));

		vector< pair <string, int> >::iterator it = find_if(namesLabelPairParent[c].begin(), namesLabelPairParent[c].end(), [&label](const pair<string, int>& element){ return element.second == label; });
		//pair found
		cout << "\n the shape is  " << it->first;
		cout << endl;


	}



	void useKNN_Child(int K, int coeff, Mat query)
	{
		int c;
		switch (coeff)
		{
		case 1024: c = 0; break;
		case 512: c = 1; break;
		case 360: c = 2; break;
		default: cout << " \n Coeff entered do not exist : 1024,512 or 360 !"; break;
		}

		//KNN

		Ptr<ml::KNearest> knn = ml::KNearest::create();
		knn->setDefaultK(K);
		knn->setIsClassifier(true);

		cout << "\n Data Rows " << allFvMatChild[c].rows << " Data cols " << allFvMatChild[c].cols;

		//cout << allFvMatChild[c] << endl;
		knn->setAlgorithmType(ml::KNearest::Types::BRUTE_FORCE);
		knn->train(allFvMatChild[c], ml::ROW_SAMPLE, allFvLabelsChild[c]);

		Mat res2;
		knn->predict(query, res2);
		//knn->findNearest(query, knn->getDefaultK(), res2);
		cout << "\n  RESULT OF KNN  " << res2.at<float>(0, 0) << endl << "\n";
		//integer label
		int label;
		label = int(res2.at<float>(0, 0));

		vector< pair <string, int> >::iterator it = find_if(namesLabelPairChild[c].begin(), namesLabelPairChild[c].end(), [&label](const pair<string, int>& element){ return element.second == label; });
		//pair found
		cout << "\n the shape is  " << it->first;
		cout << endl;


	}


	String  usePFu_Parent(vector<Point2d> query, int coeff, int compareCoeff)
	{
		int c;
		switch (coeff)
		{
		case 1024: c = 0; break;
		case 512: c = 1; break;
		case 256: c = 2; break;
		default: cout << " \n Coeff entered do not exist : 1024,512 or 256 !"; break;
		}

		FDPersoonFu fdComp;
		//Set the reference contour (one to be matched)
		fdComp.sContour = query;
		fdComp.nbDesFit = compareCoeff;
		float alpha, phi, s, minScore1 = 100000;
		string label;

		for (int i = 0; i < mergedFVParent[c].size(); i++)
		{
			int samples = mergedFVParent[c].at(i).first.size();

			for (int k = 0; k < samples; k++)
			{
				vector<Point2d> tmp;
				tmp = mergedFVParent[c].at(i).first[k];
				fdComp.AjustementRtSafe(tmp, alpha, phi, s);
				complex<float> expitheta = s * complex<float>(cos(phi), sin(phi));
				float score1 = 0;
				score1 = fdComp.Distance(expitheta, alpha);
				if (score1 < minScore1)
				{
					minScore1 = score1;
					label = mergedFVParent[c].at(i).second;
				}
				tmp.clear();

			}

		}

		cout << "\n \n P-Fu Distance between : " << minScore1;

		cout << "\n Label Parent: " << label;

		return label;


	}


	String  usePFu_Child(vector<Point2d> query,int coeff,int compareCoeff)
	{
		int c;
		switch (coeff)
		{
		case 1024: c = 0; break;
		case 512: c = 1; break;
		case 256: c = 2; break;
		default: cout << " \n Coeff entered do not exist : 1024,512 or 256 !"; break;
		}
		
		FDPersoonFu fdComp;
		//Set the reference contour (one to be matched)
		fdComp.sContour = query;
		fdComp.nbDesFit = compareCoeff;
		float alpha, phi, s, minScore1 = 100000;
		string label;

		for (int i = 0; i < mergedFVChild[c].size(); i++)
		{
			int samples = mergedFVChild[c].at(i).first.size();

			for (int k = 0; k < samples; k++)
			{
				vector<Point2d> tmp;
				tmp = mergedFVChild[c].at(i).first[k];
				fdComp.AjustementRtSafe(tmp, alpha, phi, s);
				complex<float> expitheta = s * complex<float>(cos(phi), sin(phi));
				float score1 = 0;
				score1 = fdComp.Distance(expitheta, alpha);
				if (score1 < minScore1)
				{
					minScore1 = score1;
					label = mergedFVChild[c].at(i).second;
				}
				tmp.clear();

			}

		}

		

		if (minScore1 >= 6)
		{
			cout << "\n \n P-Fu Distance between : " << minScore1;
			cout << "\n Child  Label :  UNKNOWN  ";
			return "UN";
		}
		else
		{ 
			cout << "\n \n P-Fu Distance between : " << minScore1;
			cout << "\n Child  Label : " << label;
		return label;
		}

	}



	String usePFu_ChildFLANN(vector<Point2d> query, int coeff, int compareCoeff)
	{
		int c;
		switch (coeff)
		{
		case 1024: c = 0; break;
		case 512: c = 1; break;
		case 256: c = 2; break;
		default: cout << " \n Coeff entered do not exist : 1024,512 or 256 !"; break;
		}

		Mat localFVChildCombined;

		map<int, String> labels;
		int countLabel = 1;
		int sc = 30;
		vector<Mat> labels1;
		Mat labelsF;

		//Build vector for inserting in FLANN - DB
		for (int i = 0; i < mergedFVChild[c].size(); i++)
		{
			
			for (int j = 0; j < mergedFVChild[c][i].first.size(); j++)
			{
				//Mat temp = Mat(sc, 2, CV_32FC1);
				vector<Point2d> tempVec = mergedFVChild[c][i].first[j];
				Mat temp(tempVec,true); // = Mat(tempVec).reshape(1);
				transpose(temp, temp);
				temp.convertTo(temp, CV_32FC1);
				
				localFVChildCombined.push_back(temp);	
			}

			Mat labelsTemp(mergedFVChild[c][i].first.size(), 1, CV_32SC1);
			labelsTemp.rowRange(0, mergedFVChild[c][i].first.size()).setTo((Scalar(countLabel)));
			labels1.push_back(labelsTemp);
			labels[countLabel] = mergedFVChild[c][i].second;
			countLabel++;
			labelsTemp.release();

		}

		vconcat(labels1, labelsF);



		
		Mat tempQ(query, true);// = Mat(query).reshape(1);
		transpose(tempQ, tempQ);
		tempQ.convertTo(tempQ, CV_32FC1);

		cout << tempQ << endl;

		cout << "\n Mat format " << localFVChildCombined.rows << "    " << localFVChildCombined.cols << "   ";
	

		//KNN
		int K = 1;
		Ptr<ml::KNearest> knn = ml::KNearest::create();
		knn->setDefaultK(K);
		knn->setIsClassifier(true);

		knn->setAlgorithmType(ml::KNearest::Types::BRUTE_FORCE);
		knn->train(localFVChildCombined, ml::ROW_SAMPLE, labelsF);

		Mat res2;
		knn->predict(tempQ, res2);
		//knn->findNearest(query, knn->getDefaultK(), res2);
		cout << "\n  RESULT OF KNN  " << res2 << endl << "\n";







		//Mat dest_32f; localFVChildCombined.convertTo(dest_32f, CV_32FC2);
		//Mat obj_32f; tempQ.convertTo(obj_32f, CV_32FC2);


		//assert(dest_32f.type() == CV_32F);

		//vector<int> m_indices;
		//vector<float> m_dists;

		//cv::flann::Index flann_index(localFVChildCombined, cv::flann::KDTreeIndexParams(1));  // using 2 randomized kdtrees
		//flann_index.knnSearch(tempQ, m_indices, m_dists, 1, cv::flann::SearchParams(64));

		return "S";


	}


	void convertPFutoMat()
	{

		//converting the vector<Point2d> type fv of Pfu to abs/magnitude Mat CV_32FC1 types .... 

		vector<Mat> PFu_Mat;

		for (int i = 0; i < mergedFVParent[0].size(); i++)
		{
			
			Mat shape;
			
			for (int j = 0; j < mergedFVParent.at(0).at(i).first.size(); j++)
			{
				FileStorage fs("./FVNewPFu/" + mergedFVParent.at(0).at(i).second + "_1024.yml", FileStorage::WRITE);
				vector<Point2d> temp;
				vector<float> tempFloat;
				temp = mergedFVParent.at(0).at(i).first.at(j);
				//convert temp to magnitude vector
				for (int k = 0; k < temp.size(); k++)
				{
					tempFloat.push_back(sqrtf(powf(temp[k].x,2)+powf(temp[k].y, 2))); /// Getting magnitude x+iy of complex number
				}

				Mat converted(1, temp.size(), CV_32FC1);
				memcpy(converted.data, tempFloat.data(), tempFloat.size()*sizeof(float));
				shape.push_back(converted);
				fs << mergedFVParent.at(0).at(i).second << shape;
				fs.release();
				converted.release();
				tempFloat.clear(); temp.clear();
			}
			shape.release();

		}

		
	}


	/*
	void useSVM()
	{
		//SVM - Train
		int matchCf = 256;
		Mat octCol = octagon.colRange(0, matchCf);
		Mat diaCol = diamond.colRange(0, matchCf);
		Mat rectCol = rectangle.colRange(0, matchCf);
		Mat circCol = circle.colRange(0, matchCf);
		Mat crossCol = cross.colRange(0, matchCf);
		// Combine your features from different classes into one big matrix
		int numPostives = rectCol.rows, numNegatives = octCol.rows + diaCol.rows + circCol.rows + crossCol.rows;

		int numSamples = numPostives + numNegatives;
		int featureSize = octCol.cols;

		cout << "\n numPostives " << numPostives;
		cout << "\n numNegatives " << numNegatives;

		cv::Mat data(numSamples, featureSize, CV_32FC1);

	

		vector<Mat> mergeMat;
		mergeMat.push_back(rectCol); // Positive sample
		mergeMat.push_back(octCol);
		mergeMat.push_back(diaCol);
		mergeMat.push_back(circCol);
		mergeMat.push_back(crossCol);

		vconcat(mergeMat, data);

		cout << "\n data rows " << data.rows << "  data cols " << data.cols;


		//Create label matrix according to the big feature matrix
		Mat labels(numSamples, 1, CV_32SC1);
		labels.rowRange(0, numPostives).setTo((Scalar(1)));
		labels.rowRange(numPostives, numSamples).setTo(Scalar(-1));


		// Train the SVM
		Ptr<ml::SVM> svm = ml::SVM::create();
		svm->setType(ml::SVM::C_SVC);
		svm->setKernel(ml::SVM::LINEAR);
		svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
		svm->setGamma(3);
		svm->train(data, ml::ROW_SAMPLE, labels);

		Mat q = query.colRange(0, matchCf);
		svm->predict(q, res);

		cout << "\n  RESULT OF SVM  " << res << endl << "\n";
	}

	void useKNN()
	{
		Mat data1;
		//octagon.copyTo(data);   //1
		//diamond.copyTo(data);   //2
		//rectangle.copyTo(data); //3
		//circle.copyTo(data);    //4
		//cross.copyTo(data);     //5

		vector<Mat> mergeMat1;
		mergeMat1.push_back(rectangle); //positve sample
		mergeMat1.push_back(octagon);
		mergeMat1.push_back(diamond);
		mergeMat1.push_back(circle);
		mergeMat1.push_back(cross);
		//mergeMat1.push_back(triangle);

		vconcat(mergeMat1, data1);

		cout << "\n data rows " << data1.rows << "  data cols " << data1.cols;
		int numPostives = rectangle.rows, numNegatives = octagon.rows + diamond.rows + circle.rows + cross.rows; //+ triangle.rows; 
		int numSamples = numPostives + numNegatives;
		//Create label matrix according to the big feature matrix
		Mat labels1(numSamples, 1, CV_32SC1);
		//labels1.rowRange(0, numPostives).setTo((Scalar(1)));
		//labels1.rowRange(numPostives, numSamples).setTo(Scalar(-1));


		labels1.rowRange(0, rectangle.rows).setTo((Scalar(1)));
		labels1.rowRange(rectangle.rows, rectangle.rows + octagon.rows).setTo((Scalar(2)));
		labels1.rowRange(rectangle.rows + octagon.rows, rectangle.rows + octagon.rows + diamond.rows).setTo((Scalar(3)));
		labels1.rowRange(rectangle.rows + octagon.rows + diamond.rows, rectangle.rows + octagon.rows + diamond.rows + circle.rows).setTo((Scalar(4)));
		labels1.rowRange(rectangle.rows + octagon.rows + diamond.rows + circle.rows, rectangle.rows + octagon.rows + diamond.rows + circle.rows + cross.rows).setTo((Scalar(5)));
		//labels1.rowRange(rectangle.rows + octagon.rows + diamond.rows + circle.rows + triangle.rows, rectangle.rows + octagon.rows + diamond.rows + circle.rows + triangle.rows + cross.rows).setTo((Scalar(6)));


		//KNN
		int K = 1;
		Ptr<ml::KNearest> knn = ml::KNearest::create();
		knn->setDefaultK(K);
		knn->setIsClassifier(true);

		knn->setAlgorithmType(ml::KNearest::Types::BRUTE_FORCE);
		knn->train(data1, ml::ROW_SAMPLE, labels1);

		Mat res2;
		knn->predict(query, res2);
		//knn->findNearest(query, knn->getDefaultK(), res2);
		cout << "\n  RESULT OF KNN  " << res2 << endl << "\n";

		cout << "\n  KEY :: ";
		cout << "\n  1 - RECTANGLE ";
		cout << "\n  2 - OCTAGON ";
		cout << "\n  3 - DIAMOND ";
		cout << "\n  4 - CIRCLE ";
		//cout << "\n  5 - TRIANGLE ";
		cout << "\n  5 - CROSS ";


	}


	void useEcldDist()
	{
		Mat data2;
		vector<Mat> mergeMat2;
		int numPostives = rectangle.rows, numNegatives = octagon.rows + diamond.rows + circle.rows + cross.rows + triangle.rows;
		int numSamples = numPostives + numNegatives;
		//Create label matrix according to the big feature matrix
		Mat labels1(numSamples, 1, CV_32SC1);

		mergeMat2.push_back(rectangle);
		mergeMat2.push_back(octagon);
		mergeMat2.push_back(diamond);
		mergeMat2.push_back(circle);
		mergeMat2.push_back(cross);
		mergeMat2.push_back(triangle);

		vconcat(mergeMat2, data2);

		cout << "\n data rows " << data2.rows << "  data cols " << data2.cols;

		//labels1.rowRange(0, numPostives).setTo((Scalar(1)));
		//labels1.rowRange(numPostives, numSamples).setTo(Scalar(-1));


		labels1.rowRange(0, rectangle.rows).setTo((Scalar(1)));
		labels1.rowRange(rectangle.rows, rectangle.rows + octagon.rows).setTo((Scalar(2)));
		labels1.rowRange(rectangle.rows + octagon.rows, rectangle.rows + octagon.rows + diamond.rows).setTo((Scalar(3)));
		labels1.rowRange(rectangle.rows + octagon.rows + diamond.rows, rectangle.rows + octagon.rows + diamond.rows + circle.rows).setTo((Scalar(4)));
		labels1.rowRange(rectangle.rows + octagon.rows + diamond.rows + circle.rows, rectangle.rows + octagon.rows + diamond.rows + circle.rows + cross.rows).setTo((Scalar(5)));
		labels1.rowRange(rectangle.rows + octagon.rows + diamond.rows + circle.rows + cross.rows, rectangle.rows + octagon.rows + diamond.rows + circle.rows + cross.rows + triangle.rows).setTo((Scalar(6)));


		//Min Euclidean distance over the entire data2 matrix
		int noOfCoeff = 70;
		float minScore = 10000;
		Mat q = query.colRange(0, noOfCoeff);
		vector<float> vQ;
		vQ.assign((float*)q.datastart, (float*)q.dataend);


		int rowIndex;


		for (int i = 0; i < data2.rows; i++)
		{
			Mat row = data2.row(i);
			Mat dbvec = row.colRange(0, noOfCoeff);
			//Convert to vector
			//vector<float> vD;
			//vD.assign((float*)dbvec.datastart, (float*)dbvec.dataend);

			float score = 0;
			score = norm(q, dbvec, cv::NormTypes::NORM_L2SQR);
		
			if (score < minScore)
			{
				minScore = score;
				rowIndex = i;
			}

		}


		cout << "\n \n EUCLIDEAN DIST -- Distance between : " << minScore;

		cout << "\n Label : " << labels1.at<int>(rowIndex);

		cout << "\n  KEY :: ";
		cout << "\n  1 - RECTANGLE ";
		cout << "\n  2 - OCTAGON ";
		cout << "\n  3 - DIAMOND ";
		cout << "\n  4 - CIRCLE ";
		cout << "\n  5 - CROSS ";
		cout << "\n  6 - TRIANGLE ";


	}


	void  usePFu()
	{
		vector<vector<Point2d>> octagonV, rectangleV, diamondV, circleV, triangleV;
		// Load the data
		FileStorage octV("./FV/OCTAGON_PFu.yml", FileStorage::READ);
		readVectorOfVector(octV, "OCTAGON", octagonV);


		FileStorage diaV("./FV/DIAMOND_PFu.yml", FileStorage::READ);
		readVectorOfVector(diaV, "DIAMOND", diamondV);

		FileStorage rectV("./FV/RECTANGLE_PFu.yml", FileStorage::READ);
		readVectorOfVector(rectV, "RECTANGLE", rectangleV);

		FileStorage circV("./FV/CIRCLES_PFu.yml", FileStorage::READ);
		readVectorOfVector(circV, "CIRCLES", circleV);

		FileStorage triV("./FV/TRIANGLE_PFu.yml", FileStorage::READ);
		readVectorOfVector(triV, "TRIANGLE", triangleV);



		vector< pair <vector<vector<Point2d>>, int >> mergedFV;

		pair <vector<vector<Point2d>>, int > fvPair1, fvPair2, fvPair3, fvPair4, fvPair5;
		fvPair1.first = rectangleV;
		fvPair1.second = 1;
		mergedFV.push_back(fvPair1);
		fvPair2.first = octagonV;
		fvPair2.second = 2;
		mergedFV.push_back(fvPair2);
		fvPair3.first = diamondV;
		fvPair3.second = 3;
		mergedFV.push_back(fvPair3);
		fvPair4.first = circleV;
		fvPair4.second = 4;
		mergedFV.push_back(fvPair4);
		fvPair5.first = triangleV;
		fvPair5.second = 5;
		mergedFV.push_back(fvPair5);


		queryPFu = queryFvPFu("stop5.jpg", 1024);
		FDPersoonFu fdComp;
		//Set the reference contour (one to be matched)
		fdComp.sContour = queryPFu;
		fdComp.nbDesFit = 20;
		float alpha, phi, s, minScore1 = 100000;
		int label;

		for (int i = 0; i < mergedFV.size(); i++)
		{
			int samples = mergedFV[i].first.size();

			for (int k = 0; k < samples; k++)
			{
				vector<Point2d> tmp;
				tmp = mergedFV[i].first[k];
				fdComp.AjustementRtSafe(tmp, alpha, phi, s);
				complex<float> expitheta = s * complex<float>(cos(phi), sin(phi));
				float score1 = 0;
				score1 = fdComp.Distance(expitheta, alpha);
				if (score1 < minScore1)
				{
					minScore1 = score1;
					label = mergedFV[i].second;
				}
				tmp.clear();

			}

		}

		cout << "\n \n P-Fu Distance between : " << minScore1;

		cout << "\n Label : " << label;


		cout << "\n \n \n KEY :: ";
		cout << "\n  1 - RECTANGLE ";
		cout << "\n  2 - OCTAGON ";
		cout << "\n  3 - DIAMOND ";
		cout << "\n  4 - CIRCLE ";
		cout << "\n  5 - TRIANGLE ";

	}


	Mat queryFvEFD(String imagePath, int numberOfCoeff)
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



	vector<Point2d> queryFvPFu(String imagePath, int numberOfCoeff)
	{
		FDPersoonFu fd1;
		vector<Point2d>  dft1;

		vector<vector<Point>> contoursTopLevel1;
		contoursTopLevel1 = preProcessContour(imagePath);

		dft1 = fd1.generateFDHarmonicsPFu(contoursTopLevel1, 0, numberOfCoeff);

		return dft1;
	}



	
	*/


};

vector<vector< pair <vector<vector<Point2d>>, string >>>  FVCompare::mergedFVParent, FVCompare::mergedFVChild;
vector <vector < pair <string, int>>> FVCompare::namesLabelPairParent, FVCompare::namesLabelPairChild;
vector<Mat> FVCompare::allFvMatParent, FVCompare::allFvMatChild;
vector<Mat> FVCompare::allFvLabelsParent, FVCompare::allFvLabelsChild;