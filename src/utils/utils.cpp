/*------------------------------------------------------------------------------
 This file contains the definitions of functions/modules used in MFG
 ------------------------------------------------------------------------------*/

#include "utils.h"
#include "settings.h"
#include "random.h"

#include <math.h>
#include <iostream>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>

extern "C"
{
#include "lsd/lsd.h"
}

#ifdef _MSC_VER
#include <unordered_map>
#else
// TODO: FIXME
#include <unordered_map>
#endif

extern int IDEAL_IMAGE_WIDTH;
extern double THRESH_POINT_MATCH_RATIO;
extern MfgSettings* mfgSettings;

using namespace std;
using namespace cv;

//////////////////////////// Utility functions /////////////////////////

// output GRAY scale images
Mat* grayImage(Mat* src)
{
	if(src->channels() == 3) {
		unsigned int width  = src->size().width;
		unsigned int height = src->size().height;
		Mat *dst = new Mat(width, height, CV_8UC1);
		// CV_RGB2GRAY: convert RGB image to grayscale
		cvtColor( *src, *dst, CV_RGB2GRAY );
   	return dst;
	} else {
		return src;
   }
}

// use LSD to extract line segments from an image
// input : an image color/grayscale
// output: a list of line segments - endpoint x-y coords + ...
ntuple_list callLsd (Mat* src)
{
	Mat* src_gray = grayImage(src);
	image_double image; //image_double is a struct defined in 'lsd.h'
	ntuple_list lsd_out;
	unsigned int w = src->size().width;
	unsigned int h = src->size().height;
	image = new_image_double(w,h);
	unsigned char s = 0;//to get image values
   for (int x = 0; x < w; ++x) {
		for(int y = 0; y < h; ++y) {
			s=src_gray->at<uchar>(y, x);
			image->data[x + y*image->xsize] = s;
		}
	}
	lsd_out = lsd(image);
	free_image_double(image);
	return lsd_out;
}

cv::Mat findNearestPointOnLine (cv::Mat l, cv::Mat p)
// find the nearest point of p on line ax+by+c=0;
// l=(a, b, c), p = (x,y) or (x,y,z)
{
	double a = l.at<double>(0),
           b = l.at<double>(1),
           c = l.at<double>(2),
           x = p.at<double>(0),
           y = p.at<double>(1);
	if (p.cols*p.rows==3) {
		x = x/p.at<double>(2);
		y = y/p.at<double>(2);
	}
	double la[2][2] = {{a,b},{b,-a}},
           ra[2] = {-c,b*x-a*y};
	cv::Mat lhs(2,2,CV_64F,la), rhs(2,1,CV_64F,ra);
	return lhs.inv()*rhs;
}

double point2LineDist(double l[3], cv::Point2d p)
// distance from point(x,y) to line ax+by+c=0;
// l=(a, b, c), p = (x,y)
{
	double a = l[0],
           b = l[1],
           c = l[2],
           x = p.x,
           y = p.y;
	return abs((a*x+b*y+c))/sqrt(a*a+b*b);
}

double point2LineDist(cv::Mat l, cv::Point2d p)
// distance from point(x,y) to line ax+by+c=0;
// l=(a, b, c), p = (x,y)
{
	if(l.cols*l.rows !=3)
		cout<<"point2LineDist: error,l should be 3d\n";
	double a = l.at<double>(0),
           b = l.at<double>(1),
           c = l.at<double>(2),
           x = p.x,
           y = p.y;
	return abs((a*x+b*y+c))/sqrt(a*a+b*b);
}

double point2LineDist(cv::Mat l, cv::Mat p)
// distance from point(x,y) to line ax+by+c=0;
// l=(a, b, c), p = (x,y)
{
	double a = l.at<double>(0),
           b = l.at<double>(1),
           c = l.at<double>(2), x,y;
	if (p.cols*p.rows==2)	{
		x = p.at<double>(0),
              y = p.at<double>(1);
	} else {
		if (p.at<double>(2)!=0){
			x = p.at<double>(0)/p.at<double>(2);
			y = p.at<double>(1)/p.at<double>(2);
		}
	}
	return abs((a*x+b*y+c))/sqrt(a*a+b*b);
}

// This is a robust measurement of the difference of two image line segments.
// Compute the sum of distances from l1's two endpoints to l2 ==> d12
//                                   nomalize by l1's length: d12/len(l1)
// Compute the sum of distances from l2's two endpoints to l1 ==> d21
//									nomalize by l2's length: d21/len(l2)
// Choose the smaller one as the difference.
double normalizedLs2LstDist (IdealLine2d l1, IdealLine2d l2)
{
	//d12 = d(p1A,l2)+d(p1B,l2);
	cv::Mat eq1 = l1.lineEq(), eq2 = l2.lineEq();
	double a1 = eq1.at<double>(0),
           b1 = eq1.at<double>(1), c1 = eq1.at<double>(2),
           a2 = eq2.at<double>(0),
           b2 = eq2.at<double>(1), c2 = eq2.at<double>(2);
	double d12 = abs(a2*l1.extremity1.x+b2*l1.extremity1.y+c2)/sqrt(a2*a2 +b2*b2)
   +abs(a2*l1.extremity2.x+b2*l1.extremity2.y+c2)/sqrt(a2*a2 +b2*b2);
	d12 = d12/l1.length();
	double d21 = abs(a1*l2.extremity1.x+b1*l2.extremity1.y+c1)/sqrt(a1*a1 +b1*b1)
   +abs(a1*l2.extremity2.x+b1*l2.extremity2.y+c1)/sqrt(a1*a1 +b1*b1);
	d21 = d21/l2.length();
	return min(d12,d21);
}


std::string num2str(double i)
{
	std::stringstream ss;
	ss<<i;
	return ss.str();
}

cv::Mat vec2SkewMat (cv::Mat vec)
{
	cv::Mat m = (cv::Mat_<double>(3,3) <<
           0, -vec.at<double>(2), vec.at<double>(1),
           vec.at<double>(2), 0, -vec.at<double>(0),
           -vec.at<double>(1), vec.at<double>(0), 0);
	return m;
}

cv::Mat normalizeLines (cv::Mat& src, cv::Mat& dst)
//Zeng's algorithm (Pattern Recognition Letters 2008)
{
	cv::Mat vectOne = cv::Mat::ones(1,src.cols,CV_64F);
	cv::Mat Ta = cv::Mat::eye(3,3,CV_64F);
	Ta.at<double>(0,2) = -vectOne.dot(src.row(0))/vectOne.dot(src.row(2));
	Ta.at<double>(1,2) = -vectOne.dot(src.row(1))/vectOne.dot(src.row(2));
	cv::Mat lp = Ta*src;
	double s = sqrt((pow(cv::norm(lp.row(0)),2)+pow(cv::norm(lp.row(1)),2))
   /(2*pow(cv::norm(lp.row(2)),2)));
	cv::Mat Tb = cv::Mat::eye(3,3,CV_64F);
	Tb.at<double>(2,2) = s;
	dst.create(src.rows, src.cols, src.type());
	dst = Tb*lp;
	return Tb*Ta;
}

void showImage(std::string name, cv::Mat *img, int width)
{
	double ratio = double(width)/img->cols;
	cv::namedWindow(name,0);
	cv::imshow(name,*img);
	cvResizeWindow(name.c_str(),width,int(img->rows*ratio));
}

int sgn(double x)
{
	if (x>0)
		return 1;
	if (x<0)
		return -1;
	else
		return 0;
}

std::string nextImgName (std::string name, int n, int step)
{
	std::string head = name.substr(0,name.size()-n-4);
	std::string tail = name.substr(name.size()-4, 4);
	int idx = atoi (name.substr(name.size()-n-4, n).c_str());

	std::string nextName = num2str(idx+step);
	for (int i = 0; i < n- int(log10(double(idx+step))) -1; ++i ) {
		nextName = '0'+nextName;
	}
	nextName = head+nextName+tail;
	//	cout<<nextName<<endl;
	return nextName;
}

std::string prevImgName (std::string name, int n, int step)
{
	std::string head = name.substr(0,name.size()-n-4);
	std::string tail = name.substr(name.size()-4, 4);
	int idx = atoi (name.substr(name.size()-n-4, n).c_str());

	std::string prevName = num2str(idx-step);
	for (int i = 0; i < n- int(log10(double(idx-step))) -1; ++i ) {
		prevName = '0'+prevName;
	}
	prevName = head+prevName+tail;
	//	cout<<nextName<<endl;
	return prevName;
}


double getLineEndPtInterval (IdealLine2d a, IdealLine2d b)
// return the smallest distance between endpoints of two line segments
{
	return min(min(cv::norm(a.extremity1-b.extremity1),
           cv::norm(a.extremity1-b.extremity2)),
           min(cv::norm(a.extremity2-b.extremity1),
           cv::norm(a.extremity2-b.extremity2)));
}

cv::Point2d mat2cvpt (cv::Mat m)
// 3x1 mat => point
{
	if (m.cols * m.rows == 2) {
		return cv::Point2d(m.at<double>(0), m.at<double>(1));
   }
	if (m.cols * m.rows == 3) {
		return cv::Point2d(m.at<double>(0)/m.at<double>(2),
              m.at<double>(1)/m.at<double>(2));
	}

   std::cerr<<"input matrix dimmension wrong!";
   return cv::Point2d();
}

cv::Point3d mat2cvpt3d (cv::Mat m)
// 3x1 mat => point
{
	if (m.cols * m.rows ==3) {
		return cv::Point3d(m.at<double>(0),
              m.at<double>(1),
              m.at<double>(2));
   } else {
		std::cerr<<"input matrix dimmension wrong!";
   }
   return cv::Point3d();
}

cv::Mat cvpt2mat(cv::Point2d p, bool homo)
{
	if (homo)
		return (cv::Mat_<double>(3,1)<<p.x, p.y, 1);
	else
		return (cv::Mat_<double>(2,1)<<p.x, p.y);
}

cv::Mat cvpt2mat(cv::Point3d p, bool homo)
{
	if (homo)
		return (cv::Mat_<double>(4,1)<<p.x, p.y, p.z, 1);
	else
		return (cv::Mat_<double>(3,1)<<p.x, p.y, p.z);
}

bool isPtOnLineSegment (cv::Point2d p, IdealLine2d l)
// NOTE: point p is collinear with l
// if point is between two endpoints, return 1
{
	return (p-l.extremity1).dot(p-l.extremity2) < 0;
}

double compMsldDiff (IdealLine2d a, IdealLine2d b)
// minimum of all possible msld comparison
{
	vector<double> tmp;
	for (int ii=0; ii<a.msldDescs.size(); ++ii) {
		for (int jj=0; jj<b.msldDescs.size(); ++jj) {
			tmp.push_back(cv::norm(a.msldDescs[ii] - b.msldDescs[jj]));
		}
	}
	return *std::min_element(tmp.begin(), tmp.end());
}

double aveLine2LineDist (IdealLine2d a, IdealLine2d b)
// compute line to line distance by averaging 4 endpoint to line distances
// a and b must be almost parallel
{
	return 0.25*point2LineDist(a.lineEq(), b.extremity1)
           + 0.25*point2LineDist(a.lineEq(), b.extremity2)
           + 0.25*point2LineDist(b.lineEq(), a.extremity1)
           + 0.25*point2LineDist(b.lineEq(), a.extremity2);
}

double ln2LnDist_H(IdealLine2d& l1,IdealLine2d& l2,cv::Mat& H)
// supposedly, x2 = H*x1, for point correspondence x1<->x2
// distance/error
// effectively average of 4 point-to-line distances.
{
	cv::Mat l1A_in2 = H* cvpt2mat(l1.extremity1),
           l1B_in2 = H* cvpt2mat(l1.extremity2),
           l2A_in1 = H.inv()* cvpt2mat(l2.extremity1),
           l2B_in1 = H.inv()* cvpt2mat(l2.extremity2);

	return 0.25*point2LineDist(l1.lineEq(), l2A_in1)
           + 0.25*point2LineDist(l1.lineEq(), l2B_in1)
           + 0.25*point2LineDist(l2.lineEq(), l1A_in2)
           + 0.25*point2LineDist(l2.lineEq(), l1B_in2);

}

void projectImgPt2Plane (cv::Mat imgPt, PrimPlane3d pi, cv::Mat K, cv::Mat& result)
// compute the intersection point between the shooting line passing origin and
// the image point and the 3d plane pi
// Input: imgPt is a point on image,
//		  pi is a 3d plane,
//		  K is camera matrix
// Output: result is a 3d point in inhomogeneous coordinate
{
	cv::Mat pt;
	if (imgPt.cols*imgPt.rows==2)
		pt = cvpt2mat(mat2cvpt(imgPt));
	else
		pt = imgPt;
	pt = K.inv() *pt; // normalize by camera matrix inverse
	double t = -pi.d*cv::norm(pi.n)/pt.dot(pi.n);
	result = t*pt;
}


void computeEpipolar (vector<vector<cv::Point2d>>& pointMatches, cv::Mat K,
        cv::Mat& F, cv::Mat& R, cv::Mat& E, cv::Mat& t)
// compute F, E, R, t, and prune pointmatches
{
	int nPts = pointMatches.size();
	cv::Mat pts1(nPts,1,CV_32FC2), pts2(nPts,1,CV_32FC2),
           p1(3,nPts,CV_64F), p2(3,nPts,CV_64F);

	for (int i=0; i<pointMatches.size(); ++i) {
		pts1.at<cv::Vec2f>(i,0)[0] = pointMatches[i][0].x;
		pts1.at<cv::Vec2f>(i,0)[1] = pointMatches[i][0].y;
		pts2.at<cv::Vec2f>(i,0)[0] = pointMatches[i][1].x;
		pts2.at<cv::Vec2f>(i,0)[1] = pointMatches[i][1].y;
	}

	// normalize pts before computing fundmental matrix
	cv::Mat zPts1(3,nPts,CV_64F), zPts2(3,nPts,CV_64F);
	for (int i=0; i<pointMatches.size(); ++i) {
		cv::Mat tmp1 = (cv::Mat_<double>(3,1)<<pointMatches[i][0].x,
              pointMatches[i][0].y, 1),
              tmp2 = (cv::Mat_<double>(3,1)<<pointMatches[i][1].x,
              pointMatches[i][1].y, 1);
		cv::Mat pt1 = K.inv() * tmp1,
              pt2 = K.inv() * tmp2;
		pt1.copyTo(zPts1.col(i));
		pt2.copyTo(zPts2.col(i));
	}

	cv::Mat inSetF;
	// --- recover extrinsic parameters---
	cv::Mat inSetF2 = inSetF.clone();
	while (!essn_ransac (&zPts1, &zPts2, &E, K,	&inSetF, IDEAL_IMAGE_WIDTH))
		;

	// ---  delete outliers from pointMatches
	for (int i=pointMatches.size()-1; i>=0; --i) {
		if (inSetF.at<uchar>(i) >= 1) {
		} 	else	{ // eliminate false matches
			pointMatches.erase(pointMatches.begin()+i);
		}
	}

	cv::Mat R1, R2, trans;
	decEssential (&E, &R1, &R2, &trans);
	//--- check Cheirality ---
	// use all point matches instead of just one for robustness (very necessary!)
	int posTrans = 0, negTrans = 0;
	cv::Mat Rt, pRt, nRt;
	for (int i=0; i < pointMatches.size(); ++i) {
		Rt = findTrueRt(R1, R2, trans, mat2cvpt(K.inv()*cvpt2mat(pointMatches[i][0])),
              mat2cvpt(K.inv()*cvpt2mat(pointMatches[i][1])));
		if(trans.dot(Rt.col(3)) > 0) {
			++posTrans;
			pRt = Rt;
		} else	{
			++negTrans;
			nRt = Rt;
		}
	}
	if (posTrans > negTrans) {
		t = trans;
		R = pRt.colRange(0,3);
	} else {
		t = -trans;
		R = nRt.colRange(0,3);
	}
	F = K.t().inv()*E*K.inv();
}

void computeEpipolar (vector<vector<cv::Point2d>>& pointMatches, vector<vector<int>>& pairIdx,
        cv::Mat K,	cv::Mat& F, cv::Mat& R, cv::Mat& E, cv::Mat& t, bool useMultiE)
// compute F, E, R, t, and prune pointmatches
{
	int nPts = pointMatches.size();
	cv::Mat pts1(nPts,1,CV_32FC2), pts2(nPts,1,CV_32FC2),
           p1(3,nPts,CV_64F), p2(3,nPts,CV_64F);

	for (int i=0; i<pointMatches.size(); ++i) {
		pts1.at<cv::Vec2f>(i,0)[0] = pointMatches[i][0].x;
		pts1.at<cv::Vec2f>(i,0)[1] = pointMatches[i][0].y;
		pts2.at<cv::Vec2f>(i,0)[0] = pointMatches[i][1].x;
		pts2.at<cv::Vec2f>(i,0)[1] = pointMatches[i][1].y;
	}

	// normalize pts before computing fundmental matrix
	cv::Mat zPts1(3,nPts,CV_64F), zPts2(3,nPts,CV_64F);
	for (int i=0; i<pointMatches.size(); ++i) {
		cv::Mat tmp1 = (cv::Mat_<double>(3,1)<<pointMatches[i][0].x,
              pointMatches[i][0].y, 1),
              tmp2 = (cv::Mat_<double>(3,1)<<pointMatches[i][1].x,
              pointMatches[i][1].y, 1);
		cv::Mat pt1 = K.inv() * tmp1,
              pt2 = K.inv() * tmp2;
		pt1.copyTo(zPts1.col(i));
		pt2.copyTo(zPts2.col(i));
	}

	cv::Mat inSetF;
	// --- recover extrinsic parameters---
	cv::Mat inSetF2 = inSetF.clone();

	if ( useMultiE ) {
		vector<cv::Mat> Es, inlierMasks;
		essn_ransac (&zPts1, &zPts2, Es, K, inlierMasks, IDEAL_IMAGE_WIDTH);
		double maxNum = 0;
		for(int i=0; i<Es.size(); ++i){
			if(cv::norm(inlierMasks[i]) > maxNum)	{
				maxNum = cv::norm(inlierMasks[i]);
				E = Es[i];
				inSetF = inlierMasks[i];
			}
		}
	} else {
		while (!essn_ransac (&zPts1, &zPts2, &E, K,	&inSetF, IDEAL_IMAGE_WIDTH))
			;
	}


	// ---  delete outliers from pointMatches
	for (int i=pointMatches.size()-1; i>=0; --i) {
		if (inSetF.at<uchar>(i) >= 1) {

		} 	else	{ // eliminate false matches
			pointMatches.erase(pointMatches.begin()+i);
			pairIdx.erase(pairIdx.begin()+i);
		}
	}

	cv::Mat R1, R2, trans;
	decEssential (&E, &R1, &R2, &trans);
	//--- check Cheirality ---
	// use all point matches instead of just one for robustness (very necessary!)
	int posTrans = 0, negTrans = 0;
	cv::Mat Rt, pRt, nRt;
	for (int i=0; i < pointMatches.size(); ++i) {
		Rt = findTrueRt(R1, R2, trans, mat2cvpt(K.inv()*cvpt2mat(pointMatches[i][0])),
              mat2cvpt(K.inv()*cvpt2mat(pointMatches[i][1])));
		if (Rt.cols != 4) continue;
		if(trans.dot(Rt.col(3)) > 0) {
			++posTrans;
			pRt = Rt;
		} else	{
			++negTrans;
			nRt = Rt;
		}
	}
	if (posTrans > negTrans) {
		t = trans;
		R = pRt.colRange(0,3);
	} else {
		t = -trans;
		R = nRt.colRange(0,3);
	}
	//	cout<<posTrans<<":"<<negTrans<<endl;
	F = K.t().inv()*E*K.inv();
}

void computePotenEpipolar (vector<vector<cv::Point2d>>& pointMatches, vector<vector<int>>& pairIdx,
        cv::Mat K, vector<cv::Mat>& Fs, vector<cv::Mat>& Es, vector<cv::Mat>& Rs, vector<cv::Mat>& ts
, bool usePrior, cv::Mat t_prior)
// compute F, E, R, t, and prune pointmatches
{
	int nPts = pointMatches.size();
	cv::Mat pts1(nPts,1,CV_32FC2), pts2(nPts,1,CV_32FC2),
           p1(3,nPts,CV_64F), p2(3,nPts,CV_64F);

	for (int i=0; i<pointMatches.size(); ++i) {
		pts1.at<cv::Vec2f>(i,0)[0] = pointMatches[i][0].x;
		pts1.at<cv::Vec2f>(i,0)[1] = pointMatches[i][0].y;
		pts2.at<cv::Vec2f>(i,0)[0] = pointMatches[i][1].x;
		pts2.at<cv::Vec2f>(i,0)[1] = pointMatches[i][1].y;
	}

	// normalize pts before computing fundmental matrix
	cv::Mat zPts1(3,nPts,CV_64F), zPts2(3,nPts,CV_64F);
	for (int i=0; i<pointMatches.size(); ++i) {
		cv::Mat tmp1 = (cv::Mat_<double>(3,1)<<pointMatches[i][0].x,
              pointMatches[i][0].y, 1),
              tmp2 = (cv::Mat_<double>(3,1)<<pointMatches[i][1].x,
              pointMatches[i][1].y, 1);
		cv::Mat pt1 = K.inv() * tmp1,
              pt2 = K.inv() * tmp2;
		pt1.copyTo(zPts1.col(i));
		pt2.copyTo(zPts2.col(i));
	}

	cv::Mat inSetF;
	// --- recover extrinsic parameters---
	cv::Mat inSetF2 = inSetF.clone();
	vector<cv::Mat> inlierMasks;
	essn_ransac (&zPts1,&zPts2, Es, K, inlierMasks, IDEAL_IMAGE_WIDTH, usePrior, t_prior);

	// ---  delete outliers from pointMatches
	cv::Mat maxInlierMask = cv::Mat::zeros(nPts,1, CV_8U);
	for(int i=0; i<inlierMasks.size(); ++i) {
		maxInlierMask = maxInlierMask + inlierMasks[i];
	}

	for (int i=pointMatches.size()-1; i>=0; --i) {
		if (maxInlierMask.at<uchar>(i) >= 1) {
		} 	else	{ // eliminate false matches
			pointMatches.erase(pointMatches.begin()+i);
			pairIdx.erase(pairIdx.begin()+i);
		}
	}
	for (int i=0; i<Es.size(); ++i) {
		cv::Mat R1, R2, trans, F, R, t;
		decEssential (&Es[i], &R1, &R2, &trans);

		//--- check Cheirality ---
		// use all point matches instead of just one for robustness (very necessary!)
		int posTrans = 0, negTrans = 0;
		cv::Mat Rt, pRt, nRt;
		for (int j = 0; j < pointMatches.size(); ++j) {
			Rt = findTrueRt(R1, R2, trans, mat2cvpt(K.inv()*cvpt2mat(pointMatches[j][0])),
                 mat2cvpt(K.inv()*cvpt2mat(pointMatches[j][1])));
			if (Rt.cols != 4) continue;
			if(trans.dot(Rt.col(3)) > 0) {
				++posTrans;
				pRt = Rt;
			} else	{
				++negTrans;
				nRt = Rt;
			}
		}
		if (posTrans > negTrans) {
			t = trans;
			R = pRt.colRange(0,3);
		} else {
			t = -trans;
			R = nRt.colRange(0,3);
		}
		F = K.t().inv()*Es[i]*K.inv();
		Rs.push_back(R);
		ts.push_back(t);
		Fs.push_back(F);
	}
}

void drawLineMatches(cv::Mat im1,cv::Mat im2, vector<IdealLine2d>lines1,
        vector<IdealLine2d>lines2, vector<vector<int>> pairs)
{
	cv::Mat canv1 = im1.clone();
	cv::Mat canv2 = im2.clone();
	for (int i=0; i < pairs.size(); ++i)
	{
		cv::Scalar color(rand()/200+55,rand()/200+55,rand()/200+55,0);
		cv::line(canv1, lines1[pairs[i][0]].extremity1,
              lines1[pairs[i][0]].extremity2,color,2);

		cv::putText(canv1, num2str(lines1[pairs[i][0]].gid),
              (lines1[pairs[i][0]].extremity1+lines1[pairs[i][0]].extremity2)*0.5, 1,1, color);
		cv::line(canv2, lines2[pairs[i][1]].extremity1,
              lines2[pairs[i][1]].extremity2,color,2);
		cv::putText(canv2, num2str(lines2[pairs[i][1]].gid),
              (lines2[pairs[i][1]].extremity1+lines2[pairs[i][1]].extremity2)*0.5, 1, 1, color);
		showImage("I1-line match"+num2str(pairs.size()),&canv1);
		showImage("I2-line match"+num2str(pairs.size()),&canv2);

	}
	cv::waitKey();
}

vector<vector<int>>matchKeyPoints (const vector<FeatPoint2d>& kps1,
        const vector<FeatPoint2d>& kps2, vector<vector<cv::Point2d>>& pointMatches)
{
	cv::Mat sift1(kps1[0].siftDesc.rows, kps1.size(),
           kps1[0].siftDesc.type()),
           sift2(kps2[0].siftDesc.rows, kps2.size(),
           kps2[0].siftDesc.type());
	for (int i=0; i<kps1.size(); ++i)
		kps1[i].siftDesc.copyTo(sift1.col(i));
	for (int i=0; i<kps2.size(); ++i)
		kps2[i].siftDesc.copyTo(sift2.col(i));

	vector<vector<cv::DMatch>> knnMatches;
	if (kps1.size() * kps2.size() > 1e8) {
		cv::FlannBasedMatcher matcher;	// this gives fast inconsistent output
		matcher.knnMatch(sift1.t(), sift2.t(), knnMatches,2);
	}
	else { // BF is slower but result is consistent
		//cv::BruteForceMatcher<cv::L2<float>> matcher; // for opencv2.3.0
      cv::BFMatcher matcher( cv::NORM_L2, false ); // for opencv2.4.2
		matcher.knnMatch(sift1.t(), sift2.t(),	knnMatches,2);
	}
	pointMatches.clear();
	vector<vector<int>> pairIdx;
	for (int i=0; i<knnMatches.size(); ++i) {
		double ratio = knnMatches[i][0].distance/knnMatches[i][1].distance;
		if ( ratio < THRESH_POINT_MATCH_RATIO) 		{
			//			vector<double> ptPair;
			//			ptPair.push_back(view1.featurePoints[knnMatches[i][0].queryIdx].x);
			//			ptPair.push_back(view1.featurePoints[knnMatches[i][0].queryIdx].y);
			//			ptPair.push_back(view2.featurePoints[knnMatches[i][0].trainIdx].x);
			//			ptPair.push_back(view2.featurePoints[knnMatches[i][0].trainIdx].y);
			vector<cv::Point2d> ptPair;
			ptPair.push_back(cv::Point2d(
         kps1[knnMatches[i][0].queryIdx].x,
                 kps1[knnMatches[i][0].queryIdx].y));
			ptPair.push_back(cv::Point2d(
         kps2[knnMatches[i][0].trainIdx].x,
                 kps2[knnMatches[i][0].trainIdx].y));
			pointMatches.push_back(ptPair);

			vector<int> idpair;
			idpair.push_back(knnMatches[i][0].queryIdx);
			idpair.push_back(knnMatches[i][0].trainIdx);
			pairIdx.push_back(idpair);
		}
	}
	//	cout<<"Keypoint Matches: "<<pointMatches.size()<<endl;
	return pairIdx;
}

double compParallax (cv::Point2d x1, cv::Point2d x2, cv::Mat K, cv::Mat R1, cv::Mat R2)
// compute parallax of a feature point between two frames
{
	cv::Mat R = R1.t()*R2; // relative rotation matrix between frame 1 and 2
	cv::Mat Hp = K * R * K.inv(); //homography maps point from view 1 to view 2

	return cv::norm( mat2cvpt(Hp * cvpt2mat(x1)) - x2 );
}

double compParallaxDeg (cv::Point2d x1, cv::Point2d x2, cv::Mat K, cv::Mat R1, cv::Mat R2)
// compute parallax of a feature point between two frames
{
	cv::Mat R = R1.t()*R2; // relative rotation matrix between frame 1 and 2
	cv::Mat Hp = K * R * K.inv(); //homography maps point from view 1 to view 2

	cv::Mat nz_x2 =  K.inv() * cvpt2mat(x2);
	cv::Mat nz_x1 =  K.inv() * ( Hp * cvpt2mat(x1));
	return acos(abs(nz_x1.dot(nz_x2)/cv::norm(nz_x1)/cv::norm(nz_x2)))*180/PI;
}

double compParallax (IdealLine2d l1, IdealLine2d l2, cv::Mat K, cv::Mat R1, cv::Mat R2)
// // compute parallax of an ideal line between two frames
{
	cv::Mat R = R1.t()*R2; // relative rotation matrix between frame 1 and 2
	cv::Mat Hp = K * R * K.inv();   // homography maps point from view 1 to view 2

	IdealLine2d  l1in2 = IdealLine2d(
           LineSegmt2d(mat2cvpt(Hp * cvpt2mat(l1.extremity1)),
           mat2cvpt(Hp * cvpt2mat(l1.extremity2))) );
	cv::Point2d l2dir = (l2.extremity1 - l2.extremity2);
	cv::Point2d l1in2dir = (l1in2.extremity1 - l1in2.extremity2);
	double cos = abs( l2dir.dot(l1in2dir)/cv::norm(l1in2dir)/cv::norm(l2dir) );
	//	cout<<aveLine2LineDist(l1in2, l2) <<"\t"<<aveLine2LineDist(l1in2, l2) * cos<<endl;
	return aveLine2LineDist(l1in2, l2) * cos; // suppress by cos(a)
}

// TODO: remove this function
bool isFileExist(std::string imgName)
{
	ifstream file(imgName);
	if (file.is_open())
	{
		file.close();
		return true;
	} else
		return false;
}

cv::Mat triangulatePoint (const cv::Mat& P1, const cv::Mat& P2, const cv::Mat& K,
        cv::Point2d pt1, cv::Point2d pt2)
{
	CvMat cvP1 = P1, cvP2 = P2; // projection matrix
	cv::Mat X(4,1,CV_64F);
	cv::Mat x1 = cvpt2mat(mat2cvpt(K.inv()*cvpt2mat(pt1)),0),
           x2 = cvpt2mat(mat2cvpt(K.inv()*cvpt2mat(pt2)),0);
	CvMat cx1 = x1, cx2 = x2, cX = X;
	cvTriangulatePoints(&cvP1, &cvP2, &cx1, &cx2, &cX);
	return (cv::Mat_<double>(3,1)<<cX.data.db[0]/cX.data.db[3],
           cX.data.db[1]/cX.data.db[3], cX.data.db[2]/cX.data.db[3]);
}

cv::Mat triangulatePoint (const cv::Mat& R1, const cv::Mat& t1, const cv::Mat& R2,
        const cv::Mat& t2,const cv::Mat& K,	cv::Point2d pt1, cv::Point2d pt2)
// output 3d vector
{
	cv::Mat P1(3,4,CV_64F) , P2(3,4,CV_64F);
	R1.copyTo(P1.colRange(0,3)); 	 t1.copyTo(P1.col(3));
	R2.copyTo(P2.colRange(0,3)); 	 t2.copyTo(P2.col(3));
	CvMat cvP1 = P1, cvP2 = P2; // projection matrix
	cv::Mat X(4,1,CV_64F);
	cv::Mat x1 = cvpt2mat(mat2cvpt(K.inv()*cvpt2mat(pt1)),0),
           x2 = cvpt2mat(mat2cvpt(K.inv()*cvpt2mat(pt2)),0);
	CvMat cx1 = x1, cx2 = x2, cX = X;
	cvTriangulatePoints(&cvP1, &cvP2, &cx1, &cx2, &cX);
	return (cv::Mat_<double>(3,1)<<cX.data.db[0]/cX.data.db[3],
           cX.data.db[1]/cX.data.db[3], cX.data.db[2]/cX.data.db[3]);
}

Eigen::Quaterniond r2q(cv::Mat R)
{
	Eigen::Matrix3d Rx;
	Rx << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
           R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
           R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2);
	Eigen::Quaterniond q(Rx);
	return q;
}
cv::Mat q2r (double* q)
// input: unit quaternion representing rotation
// output: 3x3 rotation matrix
// note: q=(a,b,c,d)=a + b i + c j + d k, where (b,c,d) is the rotation axis
{
	double  a = q[0],	b = q[1],
           c = q[2],	d = q[3];
	double nm = sqrt(a*a+b*b+c*c+d*d);
   a = a/nm;
   b = b/nm;
   c = c/nm;
   d = d/nm;
	cv::Mat R = (cv::Mat_<double>(3,3)<<
           a*a+b*b-c*c-d*d,	2*b*c-2*a*d,		2*b*d+2*a*c,
           2*b*c+2*a*d,		a*a-b*b+c*c-d*d,	2*c*d-2*a*b,
           2*b*d-2*a*c,		2*c*d+2*a*b,		a*a-b*b-c*c+d*d);
	return R.clone();
}
void termReason(int info)
{
	switch(info) {
      case 1:
		{cout<<"Termination reason 1: stopped by small gradient J^T e."<<endl;break;}
      case 2:
		{cout<<"Termination reason 2: stopped by small Dp."<<endl;break;}
      case 3:
		{cout<<"Termination reason 3: stopped by itmax."<<endl;break;}
      case 4:
		{cout<<"Termination reason 4: singular matrix. Restart from current p with increased mu."<<endl;break;}
      case 5:
		{cout<<"Termination reason 5: no further error reduction is possible. Restart with increased mu."<<endl;break;}
      case 6:
		{cout<<"Termination reason 6: stopped by small ||e||_2."<<endl;break;}
      case 7:
		{cout<<"Termination reason 7: stopped by invalid (i.e. NaN or Inf) 'func' values; a user error."<<endl;break;}
      default:
		{cout<<"Termination reason: Unknown..."<<endl;}
	}
}

void unitVec2angle(cv::Mat v, double* a, double* b)
// v = (x, y, z) = (cbca, cbsa, sb) , ||v|| = 1
// a in (-pi,pi), b in (-pi/2, pi/2)
{
	if (v.cols*v.rows == 3) {
		*b = asin(v.at<double>(2));
		if (v.at<double>(1) >= 0)
			*a = acos(v.at<double>(0)/cos(*b));
		else
			*a = -acos(v.at<double>(0)/cos(*b));
	} else {
		std::cerr<<"error in 'unitVec2angle': vector should be of 3 dim.\n";
	}
}

cv::Mat angle2unitVec (double a, double b)
{
	return (cv::Mat_<double>(3,1)<<cos(b)*cos(a), cos(b)*sin(a), sin(b));
}

IdealLine3d triangluateLine (cv::Mat R1, cv::Mat t1, cv::Mat R2, cv::Mat t2, cv::Mat K,
        IdealLine2d a, IdealLine2d b )
{
	cv::Mat F = K.t().inv()*(vec2SkewMat(t2-R2*R1.t()*t1)*R2*R1.t())*K.inv();
	cv::Point2d a1_r = mat2cvpt(b.lineEq().cross(F*cvpt2mat(a.extremity1,1))),
           a2_r = mat2cvpt(b.lineEq().cross(F*cvpt2mat(a.extremity2,1))),
           a1_l = a.extremity1,
           a2_l = a.extremity2;
	cv::Mat A1 = triangulatePoint(R1, t1, R2, t2, K, a1_l, a1_r);
	cv::Mat A2 = triangulatePoint(R1, t1, R2, t2, K, a2_l, a2_r);

	cv::Point3d md((A1.at<double>(0)+A2.at<double>(0))/2,
           (A1.at<double>(1)+A2.at<double>(1))/2,
           (A1.at<double>(2)+A2.at<double>(2))/2);
	double len = cv::norm(A1-A2);
	cv::Mat drct = (A1-A2)/len;
	IdealLine3d line(md, drct);
	//	IdealLine3d line(cv::Point3d(A1.at<double>(0),A1.at<double>(1),A1.at<double>(2)),
	//		cv::Point3d(A2.at<double>(0),A2.at<double>(1),A2.at<double>(2)));
	line.is3D = true;
	line.length = len;
	return line;
}

double pesudoHuber(double e, double band)
// pesudo Huber cost function
// e : error (distance)
// ouput: bounded e*e
{
	return 2*band*band*(sqrt(1+(e/band)*(e/band))-1);
}


double ls2lnArea(cv::Mat ln, cv::Point2d a, cv::Point2d b)
// compute the area between line segment (a,b) and straight line ln
{
	// project points to line
	cv::Mat a0 = findNearestPointOnLine(ln, cvpt2mat(a));
	cv::Mat b0 = findNearestPointOnLine(ln, cvpt2mat(b));
	double a_a0 = cv::norm(a-mat2cvpt(a0));
	double b_b0 = cv::norm(b-mat2cvpt(b0));
	double a0_b0 = cv::norm(mat2cvpt(a0)-mat2cvpt(b0));
	double area;
	if (cvpt2mat(a).dot(ln) * cvpt2mat(b).dot(ln) > 0) {// a and b on the same side of line
		area = (a_a0 + b_b0) * a0_b0 /2;
	} else { // on opposite sides of line
		double ha = a0_b0 * a_a0 / (a_a0 + b_b0);
		double hb = a0_b0 * b_b0 / (a_a0 + b_b0);
		area = a_a0*ha/2 + b_b0*hb/2;
	}
	return area;
}

cv::Mat projectLine(IdealLine3d l, cv::Mat R, cv::Mat t, cv::Mat K)
// project a 3d line to camera K[R t]
{
	cv::Mat a = K * (R * cvpt2mat(l.midpt,0) + t);
	cv::Mat b = K * R * l.direct;
	return a.cross(b);
}

cv::Mat projectLine(IdealLine3d l, cv::Mat P)
// project a 3d line to camera P
{
	cv::Mat a = P * cvpt2mat(l.midpt) ;
	cv::Mat b = P * (cv::Mat_<double>(4,1)<<l.direct.at<double>(0),
           l.direct.at<double>(1),
           l.direct.at<double>(2),
           0);
	return a.cross(b);
}

bool checkCheirality (cv::Mat R, cv::Mat t, IdealLine3d line) {
	return	checkCheirality(R, t, cvpt2mat(line.extremity1())) &&
           checkCheirality(R, t, cvpt2mat(line.extremity2()));
}

cv::Point3d projectPt3d2Ln3d (IdealLine3d ln, cv::Point3d P)
{
	cv::Point3d A = ln.extremity1();
	cv::Point3d B = ln.extremity2();
	cv::Point3d AB = B-A;
	cv::Point3d AP = P-A;
	return A + (AB.dot(AP)/(AB.dot(AB)))*AB;
}

cv::Point3d projectPt3d2Ln3d (cv::Point3d endptA, cv::Point3d endptB, cv::Point3d P)
// A,B are line endpoints
{
	cv::Point3d A = endptA;
	cv::Point3d B = endptB;
	cv::Point3d AB = B-A;
	cv::Point3d AP = P-A;
	return A + (AB.dot(AP)/(AB.dot(AB)))*AB;
}

bool comparator_valIdxPair ( const valIdxPair& l, const valIdxPair& r)
{ return l.first < r.first; }


cv::Mat pnp_withR (vector<cv::Point3d> X, vector<cv::Point2d> x, cv::Mat K, cv::Mat R)
{
	cv::Mat LHS = cv::Mat::zeros(3*X.size(), 3, CV_64F);
	cv::Mat RHS = cv::Mat::zeros(3*X.size(), 1, CV_64F);
	for(int i=0; i < X.size(); ++i) {
		vec2SkewMat(K.inv()*cvpt2mat(x[i])).copyTo(LHS.rowRange(i*3, i*3+3));
		cv::Mat tmp =  -(K.inv()*cvpt2mat(x[i])).cross(R*cvpt2mat(X[i],0));
		tmp.copyTo(RHS.rowRange(i*3, i*3+3));
	}

	return (LHS.t()*LHS).inv()*LHS.t()*RHS;
}


double vecSum (vector<double> v)
{
	double sum = 0;
	for(int i=0; i<v.size(); ++i)	{
		sum += v[i];
	}
	return sum;
}

double vecMedian(vector<double> v)
{
	size_t n = v.size() / 2;		  // median scale
	nth_element(v.begin(), v.begin()+n, v.end());
	return v[n];
}

cv::Mat inhomogeneous(cv::Mat x)
{
	cv::Mat y(x.rows-1, 1, x.type());
	for(int i=0; i<y.rows; ++i) {
		y.at<double>(i) = x.at<double>(i)/x.at<double>(y.rows);
	}
	return y;
}

cv::Mat pantilt(cv::Mat vp, cv::Mat K) { //compute pan tilt angles, refer to yiliang's vp paper eq 6
	// vp should be in image cooridnate (before applying K-1)
	if (vp.rows !=2)
		std::cerr<<"pantilt: vp should be 2x1 \n";
	double u = vp.at<double>(0) - K.at<double>(0,2);
	double v = vp.at<double>(1) - K.at<double>(1,2);
	double f = (K.at<double>(1,1) + K.at<double>(0,0))/2;
	cv::Mat pt = (cv::Mat_<double>(2,1)<< atan(u/f), -atan(v/sqrt(u*u+f*f))); // in radian
	return pt;
}

cv::Mat reversePanTilt(cv::Mat pt) // covert a pan-tilt angle to its opposite direction
{
	if (pt.rows !=2 )
		std::cerr<<"error in reversePanTilt: input should be 2-D \n";
	double pr, tr;
	if( pt.at<double>(0) >= 0 )
		pr = pt.at<double>(0) - PI;
	else
		pr = pt.at<double>(0) + PI;

	if( pt.at<double>(1) >= 0 )
		tr = pt.at<double>(1) - PI;
	else
		tr = pt.at<double>(1) + PI;

	cv::Mat out = (cv::Mat_<double>(2,1)<<pr, tr);
	return out;
}

void find3dPlanes_pts (vector<KeyPoint3d> pts, vector<vector<int>>& groups,
        vector<cv::Mat>& planeVecs)
// find 3d planes from a set of 3d points, using sequential ransac
// input: pts,
{
	int maxIterNo = 500;
	int minSolSetSize = 3;
	double pt2planeDistThresh = 0.2;
	int planeSetSizeThresh = 50;

	// ----- sequential ransac -----
	for (int seq_i = 0; seq_i < 5; ++seq_i) {
		// ---- ransac ----
		vector<int> indexes(pts.size());
		for (int i=0; i<indexes.size(); ++i) indexes[i]=i;
		vector<int> maxInlierSet;
		for(int iter=0; iter<maxIterNo;iter++) {
			vector<int> inlierSet;
			random_unique(indexes.begin(), indexes.end(),minSolSetSize);// shuffle
         //		cout<<"indexes: "<<indexes[0]<<","<<indexes[1]<<","<<indexes[2]<<endl;
			cv::Point3d n = (pts[indexes[0]].cvpt()-pts[indexes[1]].cvpt()).cross(
                 pts[indexes[0]].cvpt()-pts[indexes[2]].cvpt());
			double d = -n.dot(pts[indexes[0]].cvpt()); // plane=[n' d];
			for(int i=0; i<pts.size(); ++i) {
				double dist = abs(n.dot(pts[i].cvpt())+d)/cv::norm(n);
				if (dist < pt2planeDistThresh) {
					inlierSet.push_back(i);
				}
			}
			if(inlierSet.size()>maxInlierSet.size())
				maxInlierSet = inlierSet;
		}
		if (maxInlierSet.size() > planeSetSizeThresh) {// found a new plane
			vector<int> plane;// contains gid of coplanar pts
			cv::Mat cpPts(4,maxInlierSet.size(),CV_64F);
			for(int i=maxInlierSet.size()-1; i>=0; --i) {
				plane.push_back(pts[maxInlierSet[i]].gid);
				pts[maxInlierSet[i]].mat().clone().copyTo(cpPts.col(i));
				pts.erase(pts.begin()+maxInlierSet[i]);
			}
			groups.push_back(plane);
			cv::SVD svd(cpPts.t());
         //		cout<<"plane eq="<<svd.vt.t().col(svd.vt.rows-1)<<endl;
			planeVecs.push_back(svd.vt.t().col(svd.vt.rows-1));
		}
	}
}

void find3dPlanes_pts_lns_VPs (vector<KeyPoint3d> pts, vector<IdealLine3d> lns, vector<VanishPnt3d> vps,
        vector<vector<int>>& planePtIdx, vector<vector<int>>& planeLnIdx,
        vector<cv::Mat>& planeVecs)
// find 3d planes from a set of 3d points, using sequential ransac
// input: pts, lines
{
	int maxIterNo = 500;
	double pt2planeDistThresh = mfgSettings->getMfgPointToPlaneDistance();
	int planeSetSizeThresh = mfgSettings->getMfgPointsPerPlane();
	double normal_tolerance_deg = 2;
	// ===== find possible plane normal vectors compatible with VPs =====
	vector<cv::Point3d> normals;
	if(vps.size() > 1) {
		for(int i=0; i<1; ++i) {
			cv::Point3d vpi = mat2cvpt3d(vps[i].mat(0));
			vpi = vpi*(1/cv::norm(vpi));
			for(int j=i+1; j<vps.size(); ++j) {
				cv::Point3d vpj = mat2cvpt3d(vps[j].mat(0));
				vpj = vpj*(1/cv::norm(vpj));
				if(abs(vpi.dot(vpj)) < cos(45*PI/180)) {// ensure vps are not ill-posed
					normals.push_back(vpi.cross(vpj));
				}
			}
		}
	}

	// ====== sequential ransac ======
	for (int seq_i = 0; seq_i < 5; ++seq_i) {
		// ---- ransac ----
		vector<int> ptIdxes(pts.size()), lnIdxes(lns.size());
		for (int i=0; i<ptIdxes.size(); ++i) ptIdxes[i] = i;
		for (int i=0; i<lnIdxes.size(); ++i) lnIdxes[i] = i;
		vector<int> maxInSet_pt, maxInSet_ln;
		for(int iter=0; iter<maxIterNo; iter++) {
			vector<int> insetpt, insetln;
			cv::Point3d pt0, pt1, pt2;
			int solMode = xrand()%10;
         // 0-4: 3 pts, 5-7: 1 pt + 1 line, 8-9: 2 lines
			if (solMode < 5) {// use 3 points
				if (pts.size() < 3) continue;
				random_unique(ptIdxes.begin(), ptIdxes.end(), 3);// shuffle
				pt0 = pts[ptIdxes[0]].cvpt();
				pt1 = pts[ptIdxes[1]].cvpt();
				pt2 = pts[ptIdxes[2]].cvpt();
			} else if(solMode<8) {// 1 pt + 1 line
				if(pts.size()<1 || lns.size()<1) continue;
				int a = xrand()%pts.size(), b = xrand()%lns.size();
				pt0 = pts[a].cvpt();
				pt1 = lns[b].extremity1();
				pt2 = lns[b].extremity2();
			} else { // 2 line
				if(lns.size()<2) continue;
				random_unique(lnIdxes.begin(), lnIdxes.end(), 2);// shuffle
				if(lns[lnIdxes[0]].vpGid != lns[lnIdxes[1]].vpGid) continue;
				pt0 = lns[lnIdxes[0]].extremity1();
				pt1 = lns[lnIdxes[0]].extremity2();
				pt2 = lns[lnIdxes[1]].midpt;
			}
			// compute minimal solution
			cv::Point3d n = (pt0 - pt1).cross(pt0 - pt2);
			double d = -n.dot(pt0); // plane=[n' d];
			// --- check compatibility with vps ---
			if(normals.size() > 0) {
				bool validNormal = false;
				for (int i = 0; i < normals.size();++i) {
					if(abs(normals[i].dot(n)/cv::norm(n)) > cos(normal_tolerance_deg*PI/180)) {
						validNormal = true;
						break;
					}
				}
				if(!validNormal) continue; // discard this solution
			}

			// find inliers
			for(int i=0; i<pts.size(); ++i) {
				double dist = abs(n.dot(pts[i].cvpt())+d)/cv::norm(n);
				if (dist < pt2planeDistThresh) {
					insetpt.push_back(i);
				}
			}
			for(int i=0; i<lns.size(); ++i) {
				double dist = (abs(n.dot(lns[i].extremity1())+d)+abs(n.dot(lns[i].extremity2())+d))
            /(2*cv::norm(n));
				if(dist < pt2planeDistThresh) {
					insetln.push_back(i);
				}
			}
			if(insetpt.size()+insetln.size()*2 > maxInSet_pt.size()+maxInSet_ln.size()*2) {
				maxInSet_pt = insetpt;
				maxInSet_ln = insetln;
			}
		}
		if (maxInSet_pt.size()+maxInSet_ln.size()*2 > planeSetSizeThresh
              && maxInSet_ln.size() > 1) {// found a new plane
			vector<int> planePts, planeLns;// contains gid of coplanar pts,lines
			cv::Mat cpPts(4,maxInSet_pt.size()+maxInSet_ln.size()*2,CV_64F);
			for(int i=maxInSet_pt.size()-1; i>=0; --i) {
				planePts.push_back(pts[maxInSet_pt[i]].gid);
				pts[maxInSet_pt[i]].mat().clone().copyTo(cpPts.col(i));
				pts.erase(pts.begin()+maxInSet_pt[i]);
			}
			planePtIdx.push_back(planePts);
			for (int i=maxInSet_ln.size()-1; i>=0; --i) {
				planeLns.push_back(lns[maxInSet_ln[i]].gid);
				cvpt2mat(lns[maxInSet_ln[i]].extremity1()).copyTo(cpPts.col(maxInSet_pt.size()+i*2+1));
				cvpt2mat(lns[maxInSet_ln[i]].extremity2()).copyTo(cpPts.col(maxInSet_pt.size()+i*2));
				lns.erase(lns.begin()+maxInSet_ln[i]);
			}
			planeLnIdx.push_back(planeLns);
			cv::SVD svd(cpPts.t());
         //		cout<<"plane eq="<<svd.vt.t().col(svd.vt.rows-1)<<endl;
			planeVecs.push_back(svd.vt.t().col(svd.vt.rows-1));
		}
	}
}


void detect_featpoints_buckets (cv::Mat grayImg, int n, vector<cv::Point2f>& pts, int maxNumPts,
        double qualityLevel, double minDistance)
{// devide image into nxn regions(buckets)
	pts.clear();
	pts.reserve(maxNumPts);
	int w = grayImg.cols, h = grayImg.rows;
	for (int i=0; i<n; ++i) {
		for(int j=0; j<n; ++j) {
         cv::Mat mask = cv::Mat::zeros(grayImg.size(), CV_8UC1);
         cv::Mat roi = cv::Mat(mask, cv::Rect(i*w/n,j*h/n,w/n,h/n));
         roi = 1;
         vector<cv::Point2f> partpts;
         cv::goodFeaturesToTrack(grayImg, // the image
                 partpts,   // the output detected features
                 maxNumPts/n/n,  // the maximum number of features
                 qualityLevel,     // quality level
                 minDistance,     // min distance between two features
                 mask,
                 3,
                 false
         );
         pts.insert(pts.end(), partpts.begin(), partpts.end());
		}
	}
	cv::TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03);
	cv::cornerSubPix(grayImg, pts, cv::Size(10,10), cv::Size(-1,-1),termcrit);
}


double rotateAngle(cv::Mat R) // return radian
{
	return acos(abs((R.at<double>(0,0) + R.at<double>(1,1) + R.at<double>(2,2) - 1)/2));
}

double rotateAngleDeg(cv::Mat R) // return degree
{
	return acos(abs((R.at<double>(0,0) + R.at<double>(1,1) + R.at<double>(2,2) - 1)/2))
           * 180/PI;
}
