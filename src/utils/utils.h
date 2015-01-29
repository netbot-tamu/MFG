#ifndef UTILS_H_
#define UTILS_H_

//#include <Windows.h>
// C++ 11
//#include <chrono>
// Boost Chrono
//#include <boost/chrono/system_clocks.hpp>
// Qt QTime
#include <QTime>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <eigen3/Eigen/Geometry>

#include "lsd/lsd.h"
#include "features2d.h"
#include "features3d.h"

#include "random.h"
using namespace std;

cv::Mat* grayImage(cv::Mat*); // output GRAY scale images

ntuple_list callLsd(cv::Mat*); // detect line segments by LSD

double point2LineDist(double l[3], cv::Point2d p);
double point2LineDist(cv::Mat l, cv::Point2d p);
double point2LineDist(cv::Mat l, cv::Mat p);

double symmErr_PtPair(cv::Mat p1, cv::Mat p2,cv::Mat& H);

cv::Mat vec2SkewMat (cv::Mat vec);

cv::Mat normalizeLines (cv::Mat& src, cv::Mat& dst);

class MyTimer
{
public:
	MyTimer() {}

   QTime Tstart, Tend;
	double time_ms;
	double time_s;
	void start()  {
      Tstart = QTime::currentTime();
   }
	void end() 	{
      Tend = QTime::currentTime();
      int elapsed = Tstart.msecsTo(Tend);
      time_ms = (double) elapsed;
      time_s = (double) elapsed/1000;
	}
};

std::string num2str(double i);

void showImage(std::string, cv::Mat *img, int width=800);

int sgn(double x);

std::string nextImgName (std::string name, int n, int step=1);
std::string prevImgName (std::string name, int n, int step=1);

void fivepoint_stewnister (const cv::Mat& P1, const cv::Mat& P2, std::vector<cv::Mat> &Es);
int essn_ransac (cv::Mat* pts1, cv::Mat* pts2, cv::Mat* E, cv::Mat K,
	cv::Mat* inlierMask, int imsize=640);

int essn_ransac_slow (cv::Mat* pts1, cv::Mat* pts2, std::vector<cv::Mat>& bestEs, cv::Mat K,
	std::vector<cv::Mat>& inlierMasks, int imsize);

void essn_ransac (cv::Mat* pts1, cv::Mat* pts2, std::vector<cv::Mat>& bestEs, cv::Mat K,
	std::vector<cv::Mat>& inlierMasks, int imsize,  bool usePrior = false, cv::Mat t_prior = cv::Mat(0,0,CV_8U));

void decEssential (cv::Mat *E, cv::Mat *R1, cv::Mat *R2, cv::Mat *t) ;
cv::Mat
	findTrueRt(cv::Mat R1,cv::Mat R2, cv::Mat t,cv::Point2d q1,cv::Point2d q2) ;
cv::Mat linearTriangulate (cv::Mat P1,cv::Mat P2,cv::Point2d q1,cv::Point2d q2);
bool checkCheirality (cv::Mat P, cv::Mat X);
void opt_essn_pts (cv::Mat p1, cv::Mat p2, cv::Mat *E);
void optimizeEmat (cv::Mat p1, cv::Mat p2, cv::Mat K, cv::Mat *E);


//======================= newly added since 4/2/2013 ================
int computeMSLD ( LineSegmt2d& l, cv::Mat* xGradient, cv::Mat* yGradient) ;

void refineVanishPt (const std::vector<LineSegmt2d>& allLs, std::vector<int>& lsIdx,
					cv::Mat& vp );
void refineVanishPt (const std::vector<LineSegmt2d>& allLs, std::vector<int>& lsIdx,
						cv::Mat& vp, cv::Mat& cov, cv::Mat& covhomo);

double mleVp2LineDist (cv::Mat vp, LineSegmt2d l);

void optimizeVainisingPoint (std::vector<LineSegmt2d>& lines, cv::Mat& vp);
void optimizeVainisingPoint (std::vector<LineSegmt2d>& lines, cv::Mat& vp, cv::Mat& covMat, cv::Mat& covHomo);

double normalizedLs2LstDist (IdealLine2d l1, IdealLine2d l2);
double getLineEndPtInterval (IdealLine2d a, IdealLine2d b);

cv::Point2d mat2cvpt (cv::Mat m);
cv::Point3d mat2cvpt3d (cv::Mat m);
cv::Mat cvpt2mat(cv::Point2d p, bool homo=true);
cv::Mat cvpt2mat(cv::Point3d p, bool homo=true);

bool isPtOnLineSegment (cv::Point2d p, IdealLine2d l);
double compMsldDiff (IdealLine2d a, IdealLine2d b);
double aveLine2LineDist (IdealLine2d a, IdealLine2d b);

void matchLinesByPointPairs (double imWidth,
	std::vector<IdealLine2d>& lines1,std::vector<IdealLine2d>& lines2,
	std::vector< std::vector<cv::Point2d> >& pointPairs,
	std::vector< std::vector<int> >& linePairIdx);

std::vector<cv::Point2d> sampleFromLine (IdealLine2d l, double stepSize);
std::vector<cv::Point2d> sampleFromLine (IdealLine2d l, int ptNum);

double ln2LnDist_H(IdealLine2d& l1,IdealLine2d& l2,cv::Mat& H);

void projectImgPt2Plane (cv::Mat imgPt, PrimPlane3d pi, cv::Mat K, cv::Mat& result);

void computeEpipolar (std::vector< std::vector<cv::Point2d> >& pointMatches, cv::Mat K,
						cv::Mat& F, cv::Mat& R,cv::Mat& E,cv::Mat& t);
void computeEpipolar (std::vector< std::vector<cv::Point2d> >& pointMatches,
	std::vector< std::vector<int> >& pairIdx, cv::Mat K, cv::Mat& F, cv::Mat& R,cv::Mat& E,cv::Mat& t, bool useMultiE = false);
void computeEpipolar (vector<vector<cv::Point2d>>& pointMatches, vector<vector<int>>& pairIdx,
        cv::Mat K,	vector<cv::Mat>& Fs, vector<cv::Mat>& Es, vector<cv::Mat>& Rs, vector<cv::Mat>& ts) ;

void computePotenEpipolar (std::vector< std::vector<cv::Point2d> >& pointMatches, std::vector< std::vector<int> >& pairIdx,
 	cv::Mat K, std::vector<cv::Mat>& Fs, std::vector<cv::Mat>& Es, std::vector<cv::Mat>& Rs, std::vector<cv::Mat>& ts,
	bool usePrior=false, cv::Mat t_prior = cv::Mat(0,0,CV_8U));


void drawLineMatches(cv::Mat im1,cv::Mat im2, std::vector<IdealLine2d>lines1,
	std::vector<IdealLine2d>lines2, std::vector< std::vector<int> > pairs);

std::vector< std::vector<int> > matchKeyPoints (const std::vector<FeatPoint2d>& kps1,
	const std::vector<FeatPoint2d>& kps2, std::vector< std::vector<cv::Point2d> >& ptmatch);


double compParallax (cv::Point2d x1, cv::Point2d x2, cv::Mat K, cv::Mat R1, cv::Mat R2);
double compParallax (IdealLine2d l1, IdealLine2d l2, cv::Mat K, cv::Mat R1, cv::Mat R2);


int computePnP(std::vector<cv::Point3d> X, std::vector<cv::Point2d> x, cv::Mat K,cv::Mat& R, cv::Mat& t);

bool isFileExist(std::string imgName);


cv::Mat triangulatePoint (const cv::Mat& P1, const cv::Mat& P2, const cv::Mat& K,
					cv::Point2d pt1, cv::Point2d pt2);

cv::Mat triangulatePoint (const cv::Mat& R1, const cv::Mat& t1, const cv::Mat& R2,
			const cv::Mat& t2,const cv::Mat& K,	cv::Point2d pt1, cv::Point2d pt2);

cv::Mat triangulatePoint_nonlin (const cv::Mat& R1, const cv::Mat& t1, const cv::Mat& R2,
			const cv::Mat& t2,const cv::Mat& K,	cv::Point2d pt1, cv::Point2d pt2);


double fund_samperr (cv::Mat x1, cv::Mat x2, cv::Mat F) ;

void optimizeRt_withVP (cv::Mat K, std::vector< std::vector<cv::Mat> > vppairs,  double weightVP,
						std::vector< std::vector<cv::Point2d> >& featPtMatches,
						cv::Mat R, cv::Mat t);

void optimize_t_givenR (cv::Mat K, cv::Mat R, std::vector< std::vector<cv::Point2d> >& featPtMatches,
						cv::Mat t);

bool isEssnMatSimilar(cv::Mat E1, cv::Mat E2);

bool checkCheirality (cv::Mat R, cv::Mat t, cv::Mat X);

Eigen::Quaterniond r2q(cv::Mat R);
cv::Mat q2r (double* q);


void termReason(int info);

void unitVec2angle(cv::Mat v, double* a, double* b);
cv::Mat angle2unitVec (double a, double b);

IdealLine3d triangluateLine (cv::Mat R1, cv::Mat t1, cv::Mat R2, cv::Mat t2, cv::Mat K,
					  IdealLine2d a, IdealLine2d b );

cv::Mat findNearestPointOnLine (cv::Mat l, cv::Mat p);

double pesudoHuber(double e, double band);

std::vector<cv::Mat> detectVP_Jlink(std::vector<LineSegmt2d>& lines, std::vector<unsigned int>& LableCount, double lenThresh);
std::vector<cv::Mat> detectVP_Jlink(std::vector<LineSegmt2d>& lines, std::vector<unsigned int>& LableCount, double lenThresh,
	std::vector<cv::Mat>& vpCov);

double ls2lnArea(cv::Mat ln, cv::Point2d a, cv::Point2d b);

cv::Mat projectLine(IdealLine3d l, cv::Mat R, cv::Mat t, cv::Mat K) ;
cv::Mat projectLine(IdealLine3d l, cv::Mat P);

bool checkCheirality (cv::Mat R, cv::Mat t, IdealLine3d line);

cv::Point3d projectPt3d2Ln3d (IdealLine3d ln, cv::Point3d P);
cv::Point3d projectPt3d2Ln3d (cv::Point3d endptA, cv::Point3d endptB, cv::Point3d P);


typedef std::pair<double,int> valIdxPair;
bool comparator_valIdxPair ( const valIdxPair& l, const valIdxPair& r);

cv::Mat pnp_withR (std::vector<cv::Point3d> X, std::vector<cv::Point2d> x, cv::Mat K, cv::Mat R);

double vecSum (std::vector<double> v);
double vecMedian(std::vector<double> v);

void est3dpt (std::vector<cv::Mat> Rs, std::vector<cv::Mat> ts, cv::Mat K,
				std::vector<cv::Point2d> pt, cv::Mat& X, int maxIter = 200);
void est3dpt_g2o (std::vector<cv::Mat> Rs, std::vector<cv::Mat> ts, cv::Mat K, std::vector<cv::Point2d> pts2d, cv::Mat& X);

double optimizeScale (std::vector<cv::Point3d> X, std::vector<cv::Point2d> x, cv::Mat K,
					cv::Mat Rn, cv::Mat Rtn_1, cv::Mat t, double s);

cv::Mat inhomogeneous(cv::Mat x);

void twoview_ba (cv::Mat K, cv::Mat& R, cv::Mat& t, std::vector<KeyPoint3d>& pt3d, std::vector< std::vector<cv::Point2d> > pt2d);

cv::Mat pantilt(cv::Mat vp, cv::Mat K) ;

cv::Mat reversePanTilt(cv::Mat pt) ;

template<class bidiiter> //Fisher-Yates shuffle
bidiiter random_unique(bidiiter begin, bidiiter end, size_t num_random) {
	size_t left = std::distance(begin, end);
	while (num_random--) {
		bidiiter r = begin;
		std::advance(r, xrand()%left);
		std::swap(*begin, *r);
		++begin;
		--left;
	}
	return begin;
}


void find3dPlanes_pts (std::vector<KeyPoint3d> pts, std::vector< std::vector<int> >& groups,
							  std::vector<cv::Mat>& planes);
void find3dPlanes_pts_lns_VPs (std::vector<KeyPoint3d> pts, std::vector<IdealLine3d> lns, std::vector<VanishPnt3d> vps,
								std::vector< std::vector<int> >& planePtIdx, std::vector< std::vector<int> >& planeLnIdx,
								std::vector<cv::Mat>& planeVecs) ;

cv::Mat vanishpoint_cov_xy2ab(cv::Mat vp, cv::Mat K, cv::Mat cov_xy);
cv::Mat vanishpoint_cov_xyw2ab(cv::Mat vp, cv::Mat K, cv::Mat cov_xyw);

void detect_featpoints_buckets (cv::Mat grayImg, int n, std::vector<cv::Point2f>& pts,
	int maxNumPts = 1000, double qualityLevel = 0.01, double minDistance = 5);
void detect_featpoints_buckets (cv::Mat grayImg, int m, int n, std::vector<cv::Point2f>& pts,
	int maxNumPts = 1000, double qualityLevel = 0.01, double minDistance = 5);

int computePnP_ransac (std::vector<cv::Point3d> X, std::vector<cv::Point2d> x, cv::Mat K,
			cv::Mat& R, cv::Mat& t, int maxIter = 50);
double compParallaxDeg (cv::Point2d x1, cv::Point2d x2, cv::Mat K, cv::Mat R1, cv::Mat R2);
double rotateAngleDeg(cv::Mat R) ;
double rotateAngle(cv::Mat R) ;

bool fund_ransac (cv::Mat pts1, cv::Mat pts2, cv::Mat F, std::vector<uchar>& mask, double distThresh, double confidence);

vector<int> findGroundPlaneFromPoints (const vector<cv::Point3f>& pts, cv::Point3f& norm_vec, double& depth);

bool detectGroundPlane (const cv::Mat& im1, const cv::Mat& im2, const cv::Mat& R, const cv::Mat& t, const cv::Mat& K,
						int& n_pts, double& depth, cv::Point3f& normal, double& quality, cv::Mat&, double);

#endif
