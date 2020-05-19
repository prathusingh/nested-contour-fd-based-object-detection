
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

class IPToolBox{

public:
void color_reduce(cv::Mat &input, cv::Mat &output, size_t div)
{
	if (input.data != output.data) {
		output.create(input.size(), input.type());
	}

	uchar buffer[256];
	for (size_t i = 0; i != 256; ++i) {
		buffer[i] = i / div * div + div / 2;
	}
	cv::Mat table(1, 256, CV_8U, buffer, sizeof(buffer));
	cv::LUT(input, table, output);
}
void gray_level_quantize(cv::Mat &input, cv::Mat &output,int bits)
{
	uchar **myArray, **myArray2;
	myArray = new uchar*[input.rows];

	int rows = input.rows;
	int cols = input.cols;

	for (int i = 0; i < input.rows; ++i)
		myArray[i] = new uchar[input.cols];

	for (int i = 0; i < input.rows; ++i)
		myArray[i] = input.ptr<uchar>(i);

	myArray2 = new unsigned char*[rows];
	for (int i = 0; i< rows; i++)
		myArray2[i] = new unsigned char[cols];

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			myArray2[i][j] = (uchar)(myArray[i][j]>>bits);
		}
	}

	output.create(input.size(), input.type());

	for (int i = 0; i<output.rows; ++i)
	{
		uchar* outputImage = output.ptr<uchar>(i);
		for (int j = 0; j < output.cols; ++j)
		{
			*outputImage++ = myArray2[i][j];
		}
	}




}


};