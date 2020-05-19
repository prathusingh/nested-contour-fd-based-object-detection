#include <opencv2\opencv.hpp>
#include <opencv2\imgproc\imgproc.hpp>
//#include <vector>
//#include <numeric>

#include<thread>
#include <future>  
#include <algorithm>
#include <iterator>

#define pi  3.141592653
using namespace cv;
using namespace std;

class RoadSignModels {
public:
	vector<tuple< String, String, tuple<vector<String>, vector<double>, vector<double>> > > shapesRead;
	pair<String, tuple<vector<String>, vector<double>, vector<double>>> modelPair, modelPairRead;


	void createModel(vector<pair<String, tuple<vector<String>, vector<double>, vector<double>, vector<int>>>> shapePred, String modelName)
	{
		//Creates an instance of ofstream, and opens example.txt
		ofstream a_file("./Models/" + modelName + ".txt");

		for (int i = 0; i < shapePred.size(); i++)
		{
			//cout << "\n Parent is a : " << shapePred[i].first;
			//cout << "\n Children are : ";
			a_file << shapePred[i].first << "\n";
			tuple<vector<String>, vector<double>, vector<double>, vector<int>> localTuple;
			localTuple = shapePred[i].second;
			vector<String> localStringVec;
			vector<double> localCenterDistVec;
			vector<double> localOrientVec;
			localStringVec = std::get<0>(localTuple);
			localCenterDistVec = std::get<1>(localTuple);
			localOrientVec = std::get<2>(localTuple);
			for (int j = 0; j < localStringVec.size(); j++)
			{
				a_file << localStringVec[j] << " " << localCenterDistVec[j] << " " << localOrientVec[j] << "\n";
			}

		}
		a_file.close();

	}


	void readModels()
	{
		vector<String> filenames;
		String dir = "./Models/";
		glob(dir, filenames);

		vector<vector<String>> alldataString;

		tuple< String, String, tuple<vector<String>, vector<double>, vector<double>> > lTuple;

		for (int i = 0; i < filenames.size(); i++)
		{
			std::ifstream file(filenames[i]);
			std::string str;
			vector<String> localStr;
			//Put the file name STOP from STOP.txt and full path at the first place in the vector ...
			string fil = getFileName(filenames[i]);
			size_t lastindex = fil.find_last_of(".");
			string rawname = fil.substr(0, lastindex);
			localStr.push_back(rawname);
			while (std::getline(file, str))
			{
				if (!str.empty())
					localStr.push_back(str);
			}
			alldataString.push_back(localStr);
			localStr.clear();
		}


		for (int i = 0; i < alldataString.size(); i++)
		{
			//	cout << "\n New Model";
			vector<String> lPairStr;
			vector<double> lPairDist;
			vector<double> lPairOrient;
			for (int j = 2; j < alldataString[i].size(); j++)
			{
				istringstream iss(alldataString[i][j]);
				vector<string> tokens;
				copy(istream_iterator<string>(iss), istream_iterator<string>(), back_inserter(tokens));
				lPairStr.push_back(tokens[0]);
				lPairDist.push_back(std::stod(tokens[1]));
				lPairOrient.push_back(std::stod(tokens[2]));
			}
			lTuple = std::make_tuple(alldataString[i][0], alldataString[i][1], std::make_tuple(lPairStr, lPairDist, lPairOrient));
			shapesRead.push_back(lTuple);
			lPairStr.clear();
			lPairDist.clear();
			lPairOrient.clear();
		}


		/*
		cout << "\n \n ------------------  This is a TEST \n";
		for (int i = 0; i < shapesRead.size(); i++)
		{
		tuple< String, String, tuple<vector<String>, vector<double>, vector<double>> >localTuple;
		localTuple = shapesRead[i];

		cout << "\n Model is : " << std::get<0>(localTuple);
		cout << "\n Parent is a : " << std::get<1>(localTuple);
		cout << "\n Children are : ";
		tuple<vector<String>, vector<double>, vector<double>> lt2;
		lt2 = std::get<2>(localTuple);
		vector<String> localStringVec;
		vector<double> localCenterDistVec;
		vector<double> localOrientVec;

		localStringVec = std::get<0>(lt2);
		localCenterDistVec = std::get<1>(lt2);
		localOrientVec = std::get<2>(lt2);

		for (int j = 0; j < localStringVec.size(); j++)
		{
		cout << "\n" << localStringVec[j] << "\t" << localCenterDistVec[j] << "\t" << localOrientVec[j] ;
		}

		}
		*/



	}





	vector<tuple< String, tuple<vector<String>, vector<double>, vector<double>, vector<int>>>>  compareModels(vector< pair<String, tuple< vector<String>, vector<double>, vector<double>, vector<int>> > > queryModel)
	{
		//vector<tuple< String, String, tuple<vector<String>, vector<double>, vector<double>> > > shapesRead;
		//use set_intersection --> where the size is max ....

		/*
		//store all parents from the query in a vector
		vector<String> parentsQuery;
		for (int i = 0; i < queryModel.size(); i++)
		{
		parentsQuery.push_back(queryModel[i].first);
		}

		//store all parents stored model in a vector
		vector<String> parentsModel;
		for (int i = 0; i < shapesRead.size(); i++)
		{
		tuple< String, String, tuple<vector<String>, vector<double>, vector<double>> >localTuple;
		localTuple = shapesRead[i];
		parentsModel.push_back(std::get<1>(localTuple));
		}
		*/

		vector<tuple< String, tuple<vector<String>, vector<double>, vector<double>, vector<int>>>> matchedChild;

		for (int i = 0; i < queryModel.size(); i++)
		{

			for (int j = 0; j < shapesRead.size(); j++)
			{
				tuple< String, String, tuple<vector<String>, vector<double>, vector<double>> >localTuple;
				localTuple = shapesRead[j];
				if (queryModel[i].first.compare(std::get<1>(localTuple)) == 0)
				{

					vector<String> childQuery;
					vector<String> childModel;

					vector<double> childQueryDist;
					vector<double> childModelDist;

					vector<double> childQueryOrient;
					vector<double> childModelOrient;

					vector<int>queryCntrIndex;


					tuple<vector<String>, vector<double>, vector<double>, vector<int>> qMt;
					qMt = queryModel[i].second;
					childQuery = std::get<0>(qMt);
					childQueryDist = std::get<1>(qMt);
					childQueryOrient = std::get<2>(qMt);
					queryCntrIndex = std::get<3>(qMt);

					tuple<vector<String>, vector<double>, vector<double>> sRt;
					sRt = std::get<2>(localTuple);
					childModel = std::get<0>(sRt);
					childModelDist = std::get<1>(sRt);
					childModelOrient = std::get<2>(sRt);

					int itemsMatched = 0, itemsNotMatched=0;

					tuple<String, tuple<vector<String>, vector<double>, vector<double>, vector<int>>> mcLocal;
					tuple<vector<String>, vector<double>, vector<double>, vector<int>> mcl1;

					vector<String> matched_wp_child;
					vector<double> matched_wp_centroiddist;
					vector<double> matched_wp_orientation;
					vector<int> matched_wp_cntrIdx;

					for (int x = 0; x < childQuery.size(); x++)
					{
						for (int y = 0; y < childModel.size(); y++)
						{
							if (childQuery[x].compare(childModel[y]) == 0)
							{
								//compare the location and orientation
								double percentDiffDist, percentDiffAspectRatio;
								percentDiffDist = fabs(((childQueryDist[x] - childModelDist[y]) / childModelDist[y]) * 100);
								percentDiffAspectRatio = fabs(((childQueryOrient[x] - childModelOrient[y]) / childModelOrient[y]) * 100);

								if ((percentDiffDist <= 70))
								{
									//Matched Childs
									matched_wp_child.push_back(childQuery[x]);
									matched_wp_centroiddist.push_back(percentDiffDist);
									matched_wp_orientation.push_back(childQueryOrient[x]);
									matched_wp_cntrIdx.push_back(queryCntrIndex[x + 1]);
									itemsMatched++;
								}
								else
								{
									//Matched the string ... but did not match centroid dist/orientation
									matched_wp_child.push_back(childQuery[x] + "_ERROR");
									matched_wp_centroiddist.push_back(percentDiffDist);
									matched_wp_orientation.push_back(childQueryOrient[x]);
									matched_wp_cntrIdx.push_back(queryCntrIndex[x + 1]);
									itemsNotMatched++;
								}

							}

						}
					}
					//Add the parent cntr index to the end of matched_wp_cntrIdx
					matched_wp_cntrIdx.push_back(queryCntrIndex[0]);

					mcl1 = std::make_tuple(matched_wp_child, matched_wp_centroiddist, matched_wp_orientation, matched_wp_cntrIdx);
					if (itemsMatched >= (childModel.size() / 2))
					{
						mcLocal = std::make_tuple(std::get<0>(localTuple), mcl1);
						matchedChild.push_back(mcLocal);
					}
					else 
					{
						if ( (itemsMatched + itemsNotMatched) >= (childModel.size() / 2) ) 
						{
							mcLocal = std::make_tuple("Nearest Model : "+std::get<0>(localTuple), mcl1);
							matchedChild.push_back(mcLocal);
						}
						else
						{ 
						mcLocal = std::make_tuple("Parent contour matches " + std::get<1>(localTuple) +" Model unknown", mcl1);
						matchedChild.push_back(mcLocal);
						}
					}


				}
			}
		}

		return matchedChild;
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

};


