#ifndef MFG_UTILS_HEADER
#define MFG_UTILS_HEADER

//#include <Windows.h>
// C++ 11
//#include <chrono>
// Boost Chrono
//#include <boost/chrono/system_clocks.hpp>
// Qt QTime
#include <QTime>
#include "lsd/lsd.h"
#include "mfg.h"


IplImage* grayImage(IplImage*); // output GRAY scale images

ntuple_list callLsd(IplImage*,bool); // detect line segments by LSD

double point2LineDist(double l[3], cv::Point2d p);
double point2LineDist(cv::Mat l, cv::Point2d p);
double point2LineDist(cv::Mat l, cv::Mat p);

double symmErr_PtPair(cv::Mat p1, cv::Mat p2,cv::Mat& H);

cv::Mat vec2SkewMat (cv::Mat vec);

cv::Mat normalizeLines (cv::Mat& src, cv::Mat& dst);

class MyTimer
{
public:
	MyTimer() {
      //QueryPerformanceFrequency(&TickPerSec);	}
   }

	//long long int TickPerSec;     // ticks per second
	//boost::chrono::system_clock::time_point Tstart, Tend;   // ticks
   QTime Tstart, Tend;
	double time_ms;
	double time_s;
	void start()  {
      //QueryPerformanceCounter(&Tstart);
      //Tstart = boost::chrono::system_clock::now();
      Tstart = QTime::currentTime();
   }
	void end() 	{
		//QueryPerformanceCounter(&Tend);
		//time_ms = (Tend-Tstart)*1000.0/TickPerSec;
		//time_s = time_ms/1000.0;
      //Tend = boost::chrono::system_clock::now();
      Tend = QTime::currentTime();
      int elapsed = Tstart.msecsTo(Tend);
      time_ms = (double) elapsed;
      time_s = (double) elapsed/1000;
	}
};

string num2str(double i);

void showImage(string, cv::Mat *img, int width=800);

int sgn(double x); 

void getConfigration (string* img1, cv::Mat& K, cv::Mat& distCoeffs,
						int& imgwidth);
string nextImgName (string name, int n, int step=1);
string prevImgName (string name, int n, int step=1);

void fivepoint_stewnister (const cv::Mat& P1, const cv::Mat& P2, vector<cv::Mat> &Es);
int essn_ransac (cv::Mat* pts1, cv::Mat* pts2, cv::Mat* E, cv::Mat K, 
	cv::Mat* inlierMask, int imsize=640);

int essn_ransac_slow (cv::Mat* pts1, cv::Mat* pts2, vector<cv::Mat>& bestEs, cv::Mat K, 
	vector<cv::Mat>& inlierMasks, int imsize);

void essn_ransac (cv::Mat* pts1, cv::Mat* pts2, vector<cv::Mat>& bestEs, cv::Mat K, 
	vector<cv::Mat>& inlierMasks, int imsize,  bool usePrior = false, cv::Mat t_prior = cv::Mat(0,0,CV_8U));

void decEssential (cv::Mat *E, cv::Mat *R1, cv::Mat *R2, cv::Mat *t) ;
cv::Mat 
	findTrueRt(cv::Mat R1,cv::Mat R2, cv::Mat t,cv::Point2d q1,cv::Point2d q2) ;
cv::Mat linearTriangulate (cv::Mat P1,cv::Mat P2,cv::Point2d q1,cv::Point2d q2);
bool checkCheirality (cv::Mat P, cv::Mat X);
void opt_essn_pts (cv::Mat p1, cv::Mat p2, cv::Mat *E);
void optimizeEmat (cv::Mat p1, cv::Mat p2, cv::Mat K, cv::Mat *E);


//======================= newly added since 4/2/2013 ================
int computeMSLD ( LineSegmt2d& l, cv::Mat* xGradient, cv::Mat* yGradient) ;

void refineVanishPt (const vector<LineSegmt2d>& allLs, vector<int>& lsIdx, 
					cv::Mat& vp );
void refineVanishPt (const vector<LineSegmt2d>& allLs, vector<int>& lsIdx, 
						cv::Mat& vp, cv::Mat& cov, cv::Mat& covhomo);

double mleVp2LineDist (cv::Mat vp, LineSegmt2d l);

void optimizeVainisingPoint (vector<LineSegmt2d>& lines, cv::Mat& vp);
void optimizeVainisingPoint (vector<LineSegmt2d>& lines, cv::Mat& vp, cv::Mat& covMat, cv::Mat& covHomo);

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
	vector<IdealLine2d>& lines1,vector<IdealLine2d>& lines2,
	vector< vector<cv::Point2d> >& pointPairs,
	vector< vector<int> >& linePairIdx);

vector<cv::Point2d> sampleFromLine (IdealLine2d l, double stepSize);
vector<cv::Point2d> sampleFromLine (IdealLine2d l, int ptNum);

double ln2LnDist_H(IdealLine2d& l1,IdealLine2d& l2,cv::Mat& H);

void projectImgPt2Plane (cv::Mat imgPt, PrimPlane3d pi, cv::Mat K, cv::Mat& result);

void computeEpipolar (vector< vector<cv::Point2d> >& pointMatches, cv::Mat K, 
						cv::Mat& F, cv::Mat& R,cv::Mat& E,cv::Mat& t);
void computeEpipolar (vector< vector<cv::Point2d> >& pointMatches, 
	vector< vector<int> >& pairIdx, cv::Mat K, cv::Mat& F, cv::Mat& R,cv::Mat& E,cv::Mat& t, bool useMultiE = false);

void computePotenEpipolar (vector< vector<cv::Point2d> >& pointMatches, vector< vector<int> >& pairIdx,
 	cv::Mat K, vector<cv::Mat>& Fs, vector<cv::Mat>& Es, vector<cv::Mat>& Rs, vector<cv::Mat>& ts,
	bool usePrior=false, cv::Mat t_prior = cv::Mat(0,0,CV_8U));


void F_guidedLinematch (cv::Mat F, View view1, View view2);
vector< vector<int> > F_guidedLinematch (cv::Mat F, vector<IdealLine2d> lines1, 
					vector<IdealLine2d> lines2, cv::Mat img1, cv::Mat img2);

void drawLineMatches(cv::Mat im1,cv::Mat im2, vector<IdealLine2d>lines1,
	vector<IdealLine2d>lines2, vector< vector<int> > pairs);

vector< vector<int> > matchKeyPoints (const vector<FeatPoint2d>& kps1, 
	const vector<FeatPoint2d>& kps2, vector< vector<cv::Point2d> >& ptmatch);

bool isKeyframe (const View& v0, const View& v1, int, int);
bool isKeyframe ( Mfg& map, const View& v1, int th_pair, int th_overlap);

double compParallax (cv::Point2d x1, cv::Point2d x2, cv::Mat K, cv::Mat R1, cv::Mat R2);
double compParallax (IdealLine2d l1, IdealLine2d l2, cv::Mat K, cv::Mat R1, cv::Mat R2);

void drawFeatPointMatches(View&, View& , vector< vector<cv::Point2d> >);

int computePnP(vector<cv::Point3d> X, vector<cv::Point2d> x, cv::Mat K,cv::Mat& R, cv::Mat& t);

bool isFileExist(string imgName);

void  matchIdealLines(View& view1, View& view2, vector< vector<int> > vpPairIdx,
	vector< vector<cv::Point2d> > featPtMatches, cv::Mat F, vector< vector<int> >& ilinePairIdx,
	bool usePtMatch);

void detectPlanes_2Views (View& view1, View& view2, cv::Mat R, cv::Mat t, vector <vector<int> > vpPairIdx,
	vector< vector<int> > ilinePairIdx, vector <PrimPlane3d>&	primaryPlanes);

cv::Mat triangulatePoint (const cv::Mat& P1, const cv::Mat& P2, const cv::Mat& K,
					cv::Point2d pt1, cv::Point2d pt2);

cv::Mat triangulatePoint (const cv::Mat& R1, const cv::Mat& t1, const cv::Mat& R2, 
			const cv::Mat& t2,const cv::Mat& K,	cv::Point2d pt1, cv::Point2d pt2);

cv::Mat triangulatePoint_nonlin (const cv::Mat& R1, const cv::Mat& t1, const cv::Mat& R2, 
			const cv::Mat& t2,const cv::Mat& K,	cv::Point2d pt1, cv::Point2d pt2);

vector< vector<int> > matchVanishPts_withR(View& view1, View& view2, cv::Mat R, bool& goodR);

double fund_samperr (cv::Mat x1, cv::Mat x2, cv::Mat F) ;

void optimizeRt_withVP (cv::Mat K, vector< vector<cv::Mat> > vppairs,  double weightVP,
						vector< vector<cv::Point2d> >& featPtMatches,
						cv::Mat R, cv::Mat t);

void optimize_t_givenR (cv::Mat K, cv::Mat R, vector< vector<cv::Point2d> >& featPtMatches,
						cv::Mat t);

bool isEssnMatSimilar(cv::Mat E1, cv::Mat E2);

bool checkCheirality (cv::Mat R, cv::Mat t, cv::Mat X);

Quaterniond r2q(cv::Mat R);
cv::Mat q2r (double* q);


void termReason(int info);

void unitVec2angle(cv::Mat v, double* a, double* b);
cv::Mat angle2unitVec (double a, double b);

IdealLine3d triangluateLine (cv::Mat R1, cv::Mat t1, cv::Mat R2, cv::Mat t2, cv::Mat K,
					  IdealLine2d a, IdealLine2d b );

cv::Mat findNearestPointOnLine (cv::Mat l, cv::Mat p);

double pesudoHuber(double e, double band);

vector<cv::Mat> detectVP_Jlink(vector<LineSegmt2d>& lines, std::vector<unsigned int>& LableCount, double lenThresh);
vector<cv::Mat> detectVP_Jlink(vector<LineSegmt2d>& lines, std::vector<unsigned int>& LableCount, double lenThresh,
	vector<cv::Mat>& vpCov);

double ls2lnArea(cv::Mat ln, cv::Point2d a, cv::Point2d b);

cv::Mat projectLine(IdealLine3d l, cv::Mat R, cv::Mat t, cv::Mat K) ;
cv::Mat projectLine(IdealLine3d l, cv::Mat P);

bool checkCheirality (cv::Mat R, cv::Mat t, IdealLine3d line);

cv::Point3d projectPt3d2Ln3d (IdealLine3d ln, cv::Point3d P);
cv::Point3d projectPt3d2Ln3d (cv::Point3d endptA, cv::Point3d endptB, cv::Point3d P);

void exportCamPose (Mfg& m, string fname) ;

typedef std::pair<double,int> valIdxPair;
bool comparator_valIdxPair ( const valIdxPair& l, const valIdxPair& r);

cv::Mat pnp_withR (vector<cv::Point3d> X, vector<cv::Point2d> x, cv::Mat K, cv::Mat R);

double vecSum (vector<double> v);
double vecMedian(vector<double> v);

void est3dpt (vector<cv::Mat> Rs, vector<cv::Mat> ts, cv::Mat K, 
				vector<cv::Point2d> pt, cv::Mat& X, int maxIter = 200);
void est3dpt_g2o (vector<cv::Mat> Rs, vector<cv::Mat> ts, cv::Mat K, vector<cv::Point2d> pts2d, cv::Mat& X);

double optimizeScale (vector<cv::Point3d> X, vector<cv::Point2d> x, cv::Mat K, 
					cv::Mat Rn, cv::Mat Rtn_1, cv::Mat t, double s);

cv::Mat inhomogeneous(cv::Mat x);

void twoview_ba (cv::Mat K, cv::Mat& R, cv::Mat& t, vector<KeyPoint3d>& pt3d, vector< vector<cv::Point2d> > pt2d);

void exportMfgNode(Mfg& m, string fname);

cv::Mat pantilt(cv::Mat vp, cv::Mat K) ;

cv::Mat reversePanTilt(cv::Mat pt) ;

template<class bidiiter> //Fisher-Yates shuffle
bidiiter random_unique(bidiiter begin, bidiiter end, size_t num_random) {
	size_t left = std::distance(begin, end);
	while (num_random--) {
		bidiiter r = begin;
		std::advance(r, rand()%left);
		std::swap(*begin, *r);
		++begin;
		--left;
	}    
	return begin;
} 


void find3dPlanes_pts (vector<KeyPoint3d> pts, vector< vector<int> >& groups,
							  vector<cv::Mat>& planes);
void find3dPlanes_pts_lns_VPs (vector<KeyPoint3d> pts, vector<IdealLine3d> lns, vector<VanishPnt3d> vps,
								vector< vector<int> >& planePtIdx, vector< vector<int> >& planeLnIdx,
								vector<cv::Mat>& planeVecs) ;

cv::Mat vanishpoint_cov_xy2ab(cv::Mat vp, cv::Mat K, cv::Mat cov_xy);
cv::Mat vanishpoint_cov_xyw2ab(cv::Mat vp, cv::Mat K, cv::Mat cov_xyw);

void detect_featpoints_buckets (cv::Mat grayImg, int n, vector<cv::Point2f>& pts, 
	int maxNumPts = 1000, double qualityLevel = 0.01, double minDistance = 5);

int computePnP_ransac (vector<cv::Point3d> X, vector<cv::Point2d> x, cv::Mat K, 
			cv::Mat& R, cv::Mat& t, int maxIter = 50);
double compParallaxDeg (cv::Point2d x1, cv::Point2d x2, cv::Mat K, cv::Mat R1, cv::Mat R2);
double rotateAngleDeg(cv::Mat R) ;
double rotateAngle(cv::Mat R) ;

void optimize_E_g2o (cv::Mat p1, cv::Mat p2, cv::Mat K, cv::Mat *E);
bool fund_ransac (cv::Mat pts1, cv::Mat pts2, cv::Mat F, vector<uchar>& mask, double distThresh, double confidence);

#endif
