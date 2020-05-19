#include <opencv2/opencv.hpp> 
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>

// CUDA structures and methods
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/plot.hpp>




//Reference Paper
//http://uosis.mif.vu.lt/~valdas/PATTERN_RECOGNITION/Boundaries/Persoon_Fu86.pdf

using namespace cv;
using namespace std;


class FDPersoonFu {
public:
	vector<Point2d> sContour;
	vector<complex<float>> b;
	vector<complex<float>> a;
	vector<float> frequence;
	vector<float> rho, psi;
	double piI;
	int nbDesFit;

public:
	FDPersoonFu(){ nbDesFit = 7; piI = acos(-1.0); };
	//float AjustementRtSafe(vector<Point2d> &c, float &alphaMin, float &phiMin, float &sMin);
	//float Ajustement(vector<Point2d> &c, float &alphaMin, float &phiMin, float &sMin);
	//void falpha(float x, float *fn, float *df);
	//void InitFrequence();
	//float rtsafe(float x1, float x2, float xacc);
	float Distance(complex<float> r, float alpha)
	{
		long            i;
		complex<float>      j(0, 1);
		float       d = 0;

		for (i = 1; i <= nbDesFit; i++)
		{
			d += abs(a[i] - b[i] * r*exp(j*float(alpha*frequence[i]))) + abs(a[a.size() - i] - b[a.size() - i] * r*exp(j*float(alpha*frequence[a.size() - i])));
			//d += powf( abs(a[i] - b[i]) ,2);
		}
		return sqrtf(d);

	}



void FDPersoonFu::InitFrequence()
{
	long i;
	int nbElt = sContour.size();
	frequence.resize(sContour.size());

	for (i = 0; i <= nbElt / 2; i++)
		frequence[i] = 2 * piI*(float)i / nbElt;
	for (i = nbElt / 2 + 1; i<nbElt; i++)
		frequence[i] = 2 * piI*(float)(i - nbElt) / nbElt;
}


void FDPersoonFu::falpha(float x, float *fn, float *df)
{
	long    n, nbElt = sContour.size();
	float   s1 = 0, s2 = 0, s3 = 0, s4 = 0;
	float   ds1 = 0, ds2 = 0, ds3 = 0, ds4 = 0;

	for (n = 1; n <= nbDesFit; n++)
	{
		s1 += rho[n] * sin(psi[n] + frequence[n] * x) +
			rho[nbElt - n] * sin(psi[nbElt - n] + frequence[nbElt - n] * x);
		s2 += frequence[n] * rho[n] * cos(psi[n] + frequence[n] * x) +
			frequence[nbElt - n] * rho[nbElt - n] * cos(psi[nbElt - n] + frequence[nbElt - n] * x);
		s3 += rho[n] * cos(psi[n] + frequence[n] * x) +
			rho[nbElt - n] * cos(psi[nbElt - n] + frequence[nbElt - n] * x);
		s4 += frequence[n] * rho[n] * sin(psi[n] + frequence[n] * x) +
			frequence[nbElt - n] * rho[nbElt - n] * sin(psi[nbElt - n] + frequence[nbElt - n] * x);
		ds1 += frequence[n] * rho[n] * cos(psi[n] + frequence[n] * x) +
			frequence[nbElt - n] * rho[nbElt - n] * cos(psi[nbElt - n] + frequence[nbElt - n] * x);
		ds2 += -frequence[n] * frequence[n] * rho[n] * sin(psi[n] + frequence[n] * x) -
			frequence[nbElt - n] * frequence[nbElt - n] * rho[nbElt - n] * sin(psi[nbElt - n] + frequence[nbElt - n] * x);
		ds3 += -frequence[n] * rho[n] * sin(psi[n] + frequence[n] * x) -
			frequence[nbElt - n] * rho[nbElt - n] * sin(psi[nbElt - n] + frequence[nbElt - n] * x);
		ds4 += frequence[n] * frequence[n] * rho[n] * cos(psi[n] + frequence[n] * x) +
			frequence[nbElt - n] * frequence[nbElt - n] * rho[nbElt - n] * cos(psi[nbElt - n] + frequence[nbElt - n] * x);
	}
	*fn = s1 * s2 - s3 *s4;
	*df = ds1 * s2 + s1 * ds2 - ds3 * s4 - s3 * ds4;
}


float FDPersoonFu::AjustementRtSafe(vector<Point2d> &c, float &alphaMin, float &phiMin, float &sMin)
{
	long            n, nbElt = sContour.size();
	float           s1, s2, sign1, sign2, df, x1 = nbElt, x2 = nbElt, dx;
	float           dist, distMin = 10000, alpha, s, phi;
	complex<float>  j(0, 1), zz;

	InitFrequence();
	rho.resize(nbElt);
	psi.resize(nbElt);

	b.resize(nbElt);
	a.resize(nbElt);
	if (nbElt != c.size())
		return -1;
	for (n = 0; n<nbElt; n++)
	{
		b[n] = complex<float>(sContour[n].x, sContour[n].y);
		a[n] = complex<float>(c[n].x, c[n].y);
		zz = conj(a[n])*b[n];
		rho[n] = abs(zz);
		psi[n] = arg(zz);
	}

	float xp = -nbElt, fnp, dfp;
	falpha(xp, &fnp, &dfp);

	x1 = nbElt, x2 = nbElt;
	sMin = 1;
	alphaMin = 0;
	phiMin = arg(a[1] / b[1]);
	do
	{
		x2 = x1;
		falpha(x2, &sign2, &df);
		dx = 1;
		x1 = x2;
		do
		{
			x2 = x1;
			x1 -= dx;
			falpha(x1, &sign1, &df);
		} while ((sign1*sign2>0) && (x1>-nbElt));
		if (sign1*sign2<0)
		{
			alpha = rtsafe(x1, x2, 1e-8);
			falpha(alpha, &sign1, &df);
			//alpha = alpha;
			s1 = 0;
			s2 = 0;
			for (n = 1; n<nbElt; n++)
			{
				s1 += rho[n] * sin(psi[n] + frequence[n] * alpha);
				s2 += rho[n] * cos(psi[n] + frequence[n] * alpha);
			}
			phi = -atan(s1 / s2);
			phi = -atan2(s1, s2);
			s1 = 0;
			s2 = 0;
			for (n = 1; n < nbElt; n++)
			{
				s1 += rho[n] * cos(psi[n] + frequence[n] * alpha + phi);
				s2 += abs(b[n] * conj(b[n]));
			}
			s = s1 / s2;
			zz = s*exp(j*phi);
			if (s>0)
				dist = Distance(zz, alpha);
			else
				dist = 10000;
			if (dist<distMin)
			{
				distMin = dist;
				alphaMin = alpha;
				phiMin = phi;
				sMin = s;
			}
		}
	} while ((x1>-nbElt));
	return distMin;
}


#define MAXIT 100

float FDPersoonFu::rtsafe(float x1, float x2, float xacc)
{
	long j;
	float df, dx, dxold, f, fh, fl;
	float temp, xh, xl, rts;

	falpha(x1, &fl, &df);
	falpha(x2, &fh, &df);
	if (fl < 0.0) {
		xl = x2;
		xh = x1;
	}
	else {
		xh = x2;
		xl = x1;
	}
	rts = 0.5*(x1 + x2);
	dxold = fabs(x2 - x1);
	dx = dxold;
	falpha(rts, &f, &df);
	for (j = 1; j <= MAXIT; j++)
	{
		if ((((rts - xh)*df - f)*((rts - xl)*df - f) >= 0.0)
			|| (fabs(2.0*f) > fabs(dxold*df)))
		{
			dxold = dx;
			dx = 0.5*(xh - xl);
			rts = xl + dx;
			if (xl == rts) return rts;
		}
		else
		{
			dxold = dx;
			dx = f / df;
			temp = rts;
			rts -= dx;
			if (temp == rts)
				return rts;
		}
		if (fabs(dx) < xacc)
			return rts;
		falpha(rts, &f, &df);
		if (f < 0.0)
			xl = rts;
		else
			xh = rts;
	}
	return 0.0;
}






Point2d Echantillon(vector<Point> &c, long i, float l1, float l2, float s)
{

	Point2d d = c[(i + 1) % c.size()] - c[i % c.size()];
	Point2d p = Point2d(c[i % c.size()]) + d * (s - l1) / (l2 - l1);
	return p;
}


vector<Point2d> ReSampleContour(vector<Point> &c, int nbElt)
{
	long        nb = c.size();
	float       l1 = 0, l2, p, d, s;
	vector<Point2d> r;
	int j = 0;
	p = arcLength(c, true);

	l2 = norm(c[j] - c[j + 1]) / p;
	for (int i = 0; i<nbElt; i++)
	{
		s = (float)i / (float)nbElt;
		while (s >= l2)
		{
			j++;
			l1 = l2;
			d = norm(c[j % nb] - c[(j + 1) % nb]);
			l2 = l1 + d / p;
		}
		if ((s >= l1) && (s<l2))
			r.push_back(Echantillon(c, j, l1, l2, s));
	}
	return r;
}


void generateFDHarmonicsNew(vector<vector<Point>> contours1, int contourIndex, int noOfPoints)
{
	vector<Point2d> c = ReSampleContour(contours1[contourIndex], noOfPoints);
	vector<Point2d>  z;
	vector<Point2d>  Z;

	for (int j = 0; j<c.size(); j++)
		z.push_back(c[(j) % c.size()]);


	vector<Point> origContr = contours1[contourIndex];
	vector<complex<float> > complexC,complexCDFT;

	for (int i = 0; i < z.size(); i++)
	{
		complexC.push_back(complex<float>(z[i].x, z[i].y));
	}

	dft(z, Z);


	cout << "\n \n DFT";
	for (int i = 0; i < Z.size(); i++)
	{
		cout << "\n"<<Z[i];
	}


}



//Reconstructing a contour based on the number of specified Fourier coefficeints in the DFT 
//Input -  single Contour vector<vector<Point>> , No. of coeff to recreate the contour
//Output - vector<Point> 

vector<float> generateFDHarmonics(vector<vector<Point>> contours1, int contourIndex, int noOfPoints)
{

	
	cout << "\n Original Size of  contour \n" << contours1[contourIndex].size();

	vector<Point2d> c = ReSampleContour(contours1[contourIndex], noOfPoints);

	cout << "\n Size of Resampled contour \n" << c.size();

	//cout << c;
	vector<Point2d>  z;
	vector<Point2d>  Z;
	
	for (int j = 0; j<c.size(); j++)
		z.push_back(c[(j) % c.size()]);

	cout << "\n Size of z \n" << z.size();
	
	//cuda
	Size dftSize;
	// calculate the size of DFT transform
	dftSize.width = getOptimalDFTSize(1);
	dftSize.height = getOptimalDFTSize(z.size());

	//cuda::GpuMat zMat(1,z.size(),CV_32FC2);
	//cuda::GpuMat ZMat(1,z.size(), CV_32FC2);

	//cuda::dft(zMat, ZMat, dftSize, DFT_SCALE | DFT_REAL_OUTPUT);


	//cuda idft

	//cuda::dft(ZMat, zMat, dftSize, DFT_INVERSE);

	
	dft(z, Z, DFT_SCALE | DFT_REAL_OUTPUT);

	cout << "\n Size of Z \n" << Z.size();

	//cout << Z;





	//Trying to make these coeff translation,scale & roatation etc invariant
	//Set dc coeff to 0

	//Magnitude of the DFT
	vector<float> magDFT;

	for (int i = 0; i < Z.size(); i++)
	{
		magDFT.push_back( sqrtf(pow(Z[i].x, 2) + pow(Z[i].y, 2)) );
	}


	/*
	Mat M2 = Mat(magDFT.size(), 1, CV_64FC1);
	memcpy(M2.data, magDFT.data(), magDFT.size()*sizeof(float));
	
	Mat plot_result;
	Ptr<plot::Plot2d> plot;
	plot = plot::createPlot2d(M2);
	plot->render(plot_result);
	//-------------------------------//
	//         show the plot
	//-------------------------------//
	imshow("plot", plot_result);
	*/



	for (int i = 0; i < magDFT.size(); i++)
	{
	//	cout <<" \n"<< magDFT[i];
	}
	
	
	
	vector<float> norm;
	float dcCoeff = Z[0].x;



	for (int i = 1; i < magDFT.size(); i++)
	{
		norm.push_back(magDFT[i]);
	}


	//Divide by dc coeff
	for (int i = 0; i < norm.size(); i++)
	{
		norm[i] = norm[i] / dcCoeff;
		//Print norm
	//	cout << norm[i];
	}
	





	/*
	FDPersoonFu fd;
	fd.sContour = Z;
	//fd.nbDesFit = noOfCoeff;
	float alpha, phi, s;

	vector<vector<Point> > ctrRotated;

	fd.AjustementRtSafe(Z, alpha, phi, s);
	complex<float> expitheta = s * complex<float>(cos(phi), sin(phi));

	for (int j = 1; j<Z.size(); j++)
	{
		complex<float> zr(Z[j].x, Z[j].y);
		zr = zr*expitheta*exp(alpha * fd.frequence[j] * complex<float>(0, 1));
		Z[j].x = zr.real();
		Z[j].y = zr.imag();
	}
	*/

	dft(Z, z, DFT_INVERSE);

	vector<Point> c1;
	for (int k = 0; k < z.size(); k++)
	{
		c1.push_back(Point(z[k].x, z[k].y));
	}

	cout << "\n BINGO !! \n";
	return magDFT;

	//ctrRotated.push_back(c1);
}



vector<Point2d> generateFDHarmonicsPFu(vector<vector<Point>> contours1, int contourIndex, int noOfPoints)
{

	cout << "\n Original Size of  contour \n" << contours1[contourIndex].size();

	vector<Point2d> c = ReSampleContour(contours1[contourIndex], noOfPoints);

	cout << "\n Size of Resampled contour \n" << c.size();

	//cout << c;
	vector<Point2d>  z;
	vector<Point2d>  Z;

	for (int j = 0; j<c.size(); j++)
		z.push_back(c[(j) % c.size()]);

	cout << "\n Size of z \n" << z.size();

	//cuda
	Size dftSize;
	// calculate the size of DFT transform
	dftSize.width = getOptimalDFTSize(1);
	dftSize.height = getOptimalDFTSize(z.size());

	//cuda::GpuMat zMat(1,z.size(),CV_32FC2);
	//cuda::GpuMat ZMat(1,z.size(), CV_32FC2);

	//cuda::dft(zMat, ZMat, dftSize, DFT_SCALE | DFT_REAL_OUTPUT);


	//cuda idft

	//cuda::dft(ZMat, zMat, dftSize, DFT_INVERSE);


	dft(z, Z, DFT_SCALE | DFT_REAL_OUTPUT);

	
	
	return Z;

	//ctrRotated.push_back(c1);
}




};

/*
int main(int argc, char **argv)
{
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat mTest, mThresh, mConnected;

	Mat m1, m2, m3, m4, m5;
	Mat  m = imread("C:/Users/Laurent.PC-LAURENT-VISI/Downloads/scene2.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	bitwise_not(m, m1);
	threshold(m1, mThresh, 5, 255, THRESH_BINARY);
	findContours(mThresh, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
	int sizeMax = 0, idx = 0;
	vector<int> ctrSelec;
	for (int i = 0; i < contours.size(); i++)
	{
		if (contours[i].size() >= 500)
		{
			ctrSelec.push_back(i);
		}
	}

	Mat mc = Mat::zeros(m.size(), CV_8UC3);
	vector<vector<Point2d> > z;
	vector<vector<Point2d> > Z;

	z.resize(ctrSelec.size());
	Z.resize(ctrSelec.size());
	for (int i = 0; i<ctrSelec.size(); i++)
	{
		vector<Point2d> c = ReSampleContour(contours[ctrSelec[i]], 1024);
		for (int j = 0; j<c.size(); j++)
			z[i].push_back(c[(j + i * 10) % c.size()]);
		dft(z[i], Z[i], DFT_SCALE | DFT_REAL_OUTPUT);
	}
	int indRef = 1;
	MatchDescriptor md;
	md.sContour = Z[indRef];
	md.nbDesFit = 20;
	vector<float> alpha, phi, s;
	vector<vector<Point> > ctrRotated;
	alpha.resize(ctrSelec.size());
	phi.resize(ctrSelec.size());
	s.resize(ctrSelec.size());


	for (int i = 0; i<ctrSelec.size(); i++)
	{
	    //I guess this is for comparisions next 3 lines ....
		md.AjustementRtSafe(Z[i], alpha[i], phi[i], s[i]);
		complex<float> expitheta = s[i] * complex<float>(cos(phi[i]), sin(phi[i]));
		cout << "Contour " << indRef << " with " << i << " origin " << alpha[i] << " and rotated of " << phi[i] * 180 / md.pi << " and scale " << s[i] << " Distance between contour is " << md.Distance(expitheta, alpha[i]) << " " << endl;
		

		for (int j = 1; j<Z[i].size(); j++)
		{
			complex<float> zr(Z[indRef][j].x, Z[indRef][j].y);
			zr = zr*expitheta*exp(alpha[i] * md.frequence[j] * complex<float>(0, 1));
			Z[i][j].x = zr.real();
			Z[i][j].y = zr.imag();
		}
		dft(Z[i], z[i], DFT_INVERSE);
		vector<Point> c;
		for (int j = 0; j<z[i].size(); j++)
			c.push_back(Point(z[i][j].x, z[i][j].y));
		ctrRotated.push_back(c);
	}


	for (int i = 0; i < ctrSelec.size(); i++)
	{
		if (i != indRef)
			drawContours(mc, contours, ctrSelec[i], Scalar(255, 0, 0));
		else
			drawContours(mc, contours, ctrSelec[i], Scalar(255, 255, 255));
		putText(mc, format("%d", i), Point(Z[i][0].x, Z[i][0].y), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0));
	}

	for (int i = 0; i < ctrSelec.size(); i++)
	{
		drawContours(mc, ctrRotated, i, Scalar(0, 0, 255));
	}
	imshow("mc ", mc);
	waitKey();

	return 0;

};
*/