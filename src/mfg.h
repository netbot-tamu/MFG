/*------------------------------------------------------------------------------
This file contains the decalrations of functions/modules used in MFG; 
definations are in 'mfg.cpp' file.
------------------------------------------------------------------------------*/
#ifndef MFG_HEADER
#define MFG_HEADER

#include <iostream>
#include <fstream>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <opencv2/nonfree/nonfree.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include <eigen3/Eigen/Geometry>

#include "lsd/lsd.h"
#include "levmar-2.6/levmar.h"
// Eigen includes replace above with linux variant
// TODO: make this windows-compatible
//#include "C:\Study\Research\Toolbox\C_C++\Eigen311\Eigen\Dense"
//#include "C:\Study\Research\Toolbox\C_C++\Eigen311\Eigen\Eigenvalues"
//#include "C:\Study\Research\Toolbox\C_C++\Eigen311\Eigen\Geometry"

#include <QThread>


using namespace Eigen;
using namespace std;
#define PI (3.14159265)
//#define HIGH_SPEED_NO_GRAPHICS
//#define DEBUG

class FeatPoint2d
// This class stores a feature point information in 2d image
{
public:
	double		x, y;	// position (x,y)
	cv::Mat		siftDesc;	// 
	int			lid;	// local id in view
	int			gid;    // global id = the corresponding 3d keypoint's gid

	FeatPoint2d(double x_, double y_) {x = x_; y = y_; lid = -1; gid = -1;}
	FeatPoint2d(double x_, double y_, int _lid) {x = x_; y = y_; lid = -1; lid = _lid; gid = -1;}
	FeatPoint2d(double x_, double y_, cv::Mat des) 
	{x = x_; y = y_; siftDesc = des; lid = -1; gid = -1;}

	FeatPoint2d(double x_, double y_, cv::Mat des, int l, int g) 
	{x = x_; y = y_; siftDesc = des; lid = l; gid = g;}

	cv::Mat mat()
	{
		return (cv::Mat_<double>(3,1)<<x,y,1);
	}
	cv::Point2d cvpt()
	{
		return cv::Point2d(x,y);
	}

};


class LineSegmt2d
{
public:
	cv::Point2d		endpt1, endpt2;	
	int				lid;
	int				vpLid;  // the local id of parent vanishing point
	int				idlnLid;  // the local id of parent ideal line
	cv::Mat			msldDesc;
	// gid of il, vp?
	cv::Point2d		gradient;

	LineSegmt2d(){}
	LineSegmt2d(cv::Point2d pt1, cv::Point2d pt2, int l =-1)
	{
		endpt1 = pt1;
		endpt2 = pt2;
		lid	   = l;
		vpLid  = -1;
		idlnLid= -1;
	}

	cv::Point2d getGradient(cv::Mat* xGradient, cv::Mat* yGradient);
	double length()
		{return sqrt(pow(endpt1.x-endpt2.x,2)+pow(endpt1.y-endpt2.y,2));}
	cv::Mat lineEq ();
};

class VanishPnt2d
{
public:
	double			x, y, w;
	int				lid;
	int				gid;
	vector <int>	idlnLids; // local id of child ideal line
	cv::Mat			cov; //2x2 inhomo image
	cv::Mat			cov_ab; //2x2 of ab representation
	cv::Mat			cov_homo;//3x3
	VanishPnt2d(double x_, double y_, double w_, int l, int g)
	{x = x_; y = y_; w = w_; lid = l; gid = g; }

	cv::Mat mat(bool homo=true)
	{
		if (homo)
			return (cv::Mat_<double>(3,1)<<x,y,w);
		else
			return (cv::Mat_<double>(2,1)<<x/w,y/w);
	}
	cv::Mat pantilt(cv::Mat K) { //compute pan tilt angles, refer to yiliang's vp paper eq 6
		double u = x/w - K.at<double>(0,2);
		double v = y/w - K.at<double>(1,2);
	//	double f = (K.at<double>(1,1) + K.at<double>(0,0))/2;	
	//	cv::Mat pt = (cv::Mat_<double>(2,1)<< atan(u/f), -atan(v/sqrt(u*u+f*f))); // in radian
		cv::Mat pt = (cv::Mat_<double>(2,1)<< atan(u/K.at<double>(0,0)), 
						-atan(v/sqrt(u*u+K.at<double>(1,1)*K.at<double>(1,1)))); // in radian
		return pt;
	}
	cv::Mat cov_pt(cv::Mat K) { // cov of pan, tilt angle
		double u = x/w - K.at<double>(0,2);
		double v = y/w - K.at<double>(1,2);
		double f = (K.at<double>(1,1) + K.at<double>(0,0))/2;
		double rho = sqrt(u*u+f*f);
		cv::Mat H = (cv::Mat_<double>(2,2)<<f/(rho*rho), 0, u*v/((rho*rho+v*v)*rho), -rho/(rho*rho+v*v));
		return H * cov * H.t();
	}
};

class IdealLine2d
{
public:
	int					vpLid;
	int					lid;
	int					gid;
	int					pGid;  // plane global id
	vector<int>			lsLids; // member line segments' local id
	vector<cv::Point2d> lsEndpoints;
	vector<cv::Mat>		msldDescs;
	cv::Point2d			extremity1, extremity2;
	
	cv::Point2d			gradient;

	IdealLine2d(LineSegmt2d s) // cast a linesegment to ideal line
	{
		vpLid = s.vpLid;
		lid	= -1;
		gid = -1;
		pGid = -1;
		extremity1 = s.endpt1;
		extremity2 = s.endpt2;
		gradient = s.gradient;
		msldDescs.push_back(s.msldDesc);
		lsLids.push_back(s.lid);
	}

	cv::Mat lineEq()
	{
		cv::Mat pt1 = (cv::Mat_<double>(3,1)<<extremity1.x, extremity1.y, 1);
		cv::Mat pt2 = (cv::Mat_<double>(3,1)<<extremity2.x, extremity2.y, 1);
		cv::Mat lnEq = pt1.cross(pt2); // lnEq = pt1 x pt2		
		lnEq = lnEq/sqrt(lnEq.at<double>(0)*lnEq.at<double>(0)
			+lnEq.at<double>(1)*lnEq.at<double>(1)); // normalize, optional
		return lnEq;
	}
	double length()
	{
		return sqrt(pow(extremity1.x-extremity2.x,2)
			+pow(extremity1.y-extremity2.y,2));
	}
};


class View 
// View class collects information from a single view, including points,
// lines etc
{
public:
	// ***** data members *****
	int							id; //		keyframe id
	int							frameId;//  rawframe id
	string						filename;
	vector <FeatPoint2d>		featurePoints;
	vector <LineSegmt2d>		lineSegments;
	vector <VanishPnt2d>		vanishPoints;
	vector <IdealLine2d>		idealLines;
	cv::Mat						K;
	cv::Mat						R;	// rotation matrix w.r.t. {W}
	cv::Mat						t;	// translation vector in {W}
	cv::Mat						t_loc;// local relative translation with previous view
	vector< vector<int> >	vpGrpIdLnIdx;
	cv::Point2d					epipoleA, epipoleB; // A is with respect to next frame
	double						angVel; // angular velocity, in degree

	
	// ****** 
	cv::Mat						img, grayImg; // resized image
	double						lsLenThresh;

	// ****** for debugging ******
	double errPt, errVp, errLn, errPl, errAll, 
		errPtMean, errLnMean,  errVpMean, errPlMean;

	// ***** methods *****
	View () {}	
	View (string imgName, cv::Mat _K, cv::Mat dc);
	View (string imgName, cv::Mat _K, cv::Mat dc, int _id);
	void detectFeatPoints ();			// detect feature points from image
	void detectLineSegments (IplImage);			// detect line segments from image
	void compMsld4AllSegments (cv::Mat grayImg);
	void detectVanishPoints ();
	void extractIdealLines();
	void drawLineSegmentGroup(vector<int> idx);
	void drawAllLineSegments(bool write2file = false);
	void drawIdealLineGroup(vector<IdealLine2d>);
	void drawIdealLines();
	void drawPointandLine();
};

class KeyPoint3d
// This class stores the 3d keypoint's information
{
public:
	double					x, y, z;
	int						gid;
	int						pGid;		  // plane gid 
	vector< vector<int> > viewId_ptLid; // (viewId, lid of featpt)
	bool					is3D;
	int						estViewId;

	KeyPoint3d() {is3D = false;  pGid = -1;}
	KeyPoint3d(double x_, double y_, double z_) 
	{
		x = x_; y = y_; z = z_;
		gid = -1;
		pGid = -1;
		is3D = false;
		estViewId = -1;
	}
	KeyPoint3d(double x_, double y_, double z_, int gid_, bool is3d_) 
	{
		x = x_; y = y_; z = z_;
		gid = gid_;
		pGid = -1;
		is3D = is3d_;
		estViewId = -1;
	}

	cv::Mat mat(bool homo=true) const
	{
		if (homo)
			return (cv::Mat_<double>(4,1)<<x,y,z,1);
		else
			return (cv::Mat_<double>(3,1)<<x,y,z);
	}
	cv::Point3d cvpt() const
	{
		return cv::Point3d(x,y,z);
	}
};

class IdealLine3d
{
public:
	cv::Point3d				midpt;
	cv::Mat					direct;  // line direction vector
	double					length;	 // length between endpoints
	int						gid;	// global id of line
	int						pGid;	// global id of the associated plane	
	int						vpGid;  // global id of the associated vanishing point
	vector< vector<int> > viewId_lnLid;
	bool					is3D;
	int						estViewId;

	IdealLine3d() {}

	IdealLine3d(cv::Point3d mpt, cv::Mat d) 
	{
		midpt = mpt;
		direct = d.clone();
		vpGid = -1;
		pGid = -1;
		gid = -1;
		estViewId = -1;
	}
		
	cv::Point3d extremity1() const
	{
		cv::Mat d = direct/cv::norm(direct);
		return midpt + 0.5*length*cv::Point3d(d.at<double>(0),d.at<double>(1),d.at<double>(2));
	}

	cv::Point3d extremity2() const
	{
		cv::Mat d = direct/cv::norm(direct);
		return midpt - 0.5*length*cv::Point3d(d.at<double>(0),d.at<double>(1),d.at<double>(2));
	}

/*	cv::Mat dirVec() {
		cv::Mat d = (cv::Mat_<double>(3,1)<<(extremity1 - extremity2).x,
			(extremity1 - extremity2).y, (extremity1 - extremity2).z);
		return d/cv::norm(d);
	}
*/	
};

class PrimPlane3d
{
public:
	cv::Mat			n;
	double			d;
	int				gid;
	vector <int>	ilnGids;  // global ids of coplanar ideal lines
	vector <int>	kptGids; // global ids of coplanar key points
	vector< pair<int,int> >	viewID_vpLid;
	int				estViewId;
	int				recentViewId;

	PrimPlane3d()
	{
		estViewId = -1;
		gid = -1;
	}
	PrimPlane3d(cv::Mat nd, int _gid) 
	{
		gid = _gid;
		if (nd.cols*nd.rows==3) {// nd = n/d, is a 3-vector
			n = nd/cv::norm(nd);
			d = 1/cv::norm(nd);
		} else { // nd=[n d] is 4-vector
			n = (cv::Mat_<double>(3,1)<<nd.at<double>(0),nd.at<double>(1),nd.at<double>(2));
			d = nd.at<double>(3);
		}
		estViewId = -1;
	}
	void setPlane(cv::Mat nd) 
	{
		if (nd.cols*nd.rows==3) {// nd = n/d, is a 3-vector
			n = nd/cv::norm(nd);
			d = 1/cv::norm(nd);
		} else { // nd=[n d] is 4-vector
			n = (cv::Mat_<double>(3,1)<<nd.at<double>(0),nd.at<double>(1),nd.at<double>(2));
			d = nd.at<double>(3);
		}
	}
};

class VanishPnt3d
{
public:
	double					x, y, z, w;
	int						gid;		// global id
	vector <int>			idlnGids;
	vector< vector<int> > viewId_vpLid;
	int						estViewId;

	VanishPnt3d(double x_, double y_, double z_) 
	{
		double len = sqrt(x_*x_+y_*y_+z_*z_);
		x = x_/len; y = y_/len; z = z_/len; w = 0;
		gid = -1;
		estViewId = -1;
	}

	cv::Mat mat(bool homo = true)
	{
		cv::Mat v;
		if (homo) {
			v=(cv::Mat_<double>(4,1)<<x, y, z, 0);
		} else {
			v=(cv::Mat_<double>(3,1)<<x, y, z);
		}
		return v/cv::norm(v);
	}

};


class TwoView
{
public:
	View view1, view2;
	vector< vector<cv::Point2d> > featPtMatches;
	vector< vector<int> >			vpPairIdx;
	vector< vector<int> >			ilinePairIdx;
	cv::Mat						E, F, R, t;

	vector <KeyPoint3d>			keyPoints;
	vector <IdealLine3d>		idealLines;
	vector <PrimPlane3d>		primaryPlanes;
	vector <VanishPnt3d>		vanishingPoints;

	TwoView(){}
	TwoView(View&, View&);

	vector< vector<int> > matchVanishPts();
	void matchIdealLines(bool usePtMatch = false);
	void triangulateIdealLines();
	void triangulateFeatPoints();
	void detectPlanes();
	void optimize();

	void drawFeatPointMatches();
	void draw3D ();
};

class Frame // raw frame, not key image, for feature tacking use 
{
public:
//	int frameId;  //raw frame id 
	string filename;
	cv::Mat image; // gray, not necessary key frame
	vector<cv::Point2f> featpts;
	vector<int> pt_lid_in_last_view;
};

class Mfg
// This class stores 3d information of MFG nodes, and the connection with views
{
public:
	vector <View>				views;		
	vector <KeyPoint3d>			keyPoints;
	vector <IdealLine3d>		idealLines;
	vector <PrimPlane3d>		primaryPlanes;
	vector <VanishPnt3d>		vanishingPoints;
	cv::Mat						K;
	vector <KeyPoint3d>			pointTrack;  // point track, not triangulated yet
	vector <IdealLine3d>		lineTrack;

	double angVel; // angle velocity deg/sec
	double linVel; // linear velocity
	double fps;

	//== for pt traking use ==
	vector<Frame> trackFrms;
	double angleSinceLastKfrm;

	Mfg(){}
	Mfg(View v0, View v1) {
		views.push_back(v0);
		views.push_back(v1);
		initialize();
	//	adjustBundle();
	}
	Mfg(View v0, View v1, double fps_) {
		views.push_back(v0);
		views.push_back(v1);
		fps = fps_;
		initialize();
	//	adjustBundle();
	}
	void initialize(); // initialize with first two views
	void expand(View&, int frameId);
	void expand_keyPoints (View& prev, View& nview);
	void expand_idealLines (View& prev, View& nview);
	void detectLnOutliers(double threshPt2LnDist);
	void detectPtOutliers(double threshPt2PtDist);
	void adjustBundle ();
	void adjustBundle_Pt_G2O(int numPos, int numFrm);
	void adjustBundle_G2O(int numPos, int numFrm);
	void est3dIdealLine(int lnGid);
	void update3dIdealLine(vector< vector<int> > ilinePairIdx, View& nview);
	void updatePrimPlane();
	void draw3D() const;
	bool rotateMode ();
};


class MfgThread : public QThread
{
     Q_OBJECT
protected:
     void run();

public:
	Mfg* pMap;
	string imgName;			// first image name
	cv::Mat K, distCoeffs;	// distortion coeff: k1, k2, p1, p2
	int imIdLen;			// n is the image number length,
	int ini_incrt;
	int increment;
	int totalImg;

	MfgThread(){}
};


struct SysPara // system parameters
{
	int		use_img_width; // user designated image width
	int		kpt_detect_alg; // feature point detection algorithm// 1: SIFT, 2: SURF, 3: GoodFeat
	

	// === parameters for gftt ===
	int		gftt_max_ptnum; // image 
	double  gftt_qual_levl; // quality level, default 0.01
	double	gftt_min_ptdist;//min dist between two features
	
	// === parameters for opticflow-lk ===
	double	oflk_min_eigval; // default 0.0001, minEignVal threshold for the 2x2 spatial motion matrix, to eleminate bad points
	int		oflk_win_size;
	
	// === keypoint region radius for descriptor ===
	double	kpt_desc_radius;

	int		frm_increment;
	int		ini_increment; // first step size
	int		nFrm4VptBA; // number of frames used for vp in BA

	// === mfg discover 3d planes ===
	double  mfg_pt2plane_dist; //
	int		mfg_num_recent_pts; // use recent points to discover planes
	int		mfg_num_recent_lns;
	int		mfg_min_npt_plane; // min num of points to claim a new plane

	// === ba ===
	double ba_weight_vp;
	double ba_weight_ln;
	double ba_weight_pl;
	bool   ba_use_kernel;
	double ba_kernel_delta_pt;
	double ba_kernel_delta_vp;
	double ba_kernel_delta_ln;
	double ba_kernel_delta_pl;

	double angle_thresh_vp2d3d; // degree, match local 2d vp to existing 3d vp

	SysPara() {}
	void init();

};

#endif //MFG_HEADER
