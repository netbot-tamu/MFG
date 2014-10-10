
#ifndef FEATURES2D_H_
#define FEATURES2D_H_

#include <vector>
#include <math.h>

#include <opencv2/core/core.hpp>

// This class stores a feature point information in 2d image
class FeatPoint2d
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
	std::vector <int>	idlnLids; // local id of child ideal line
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
	std::vector<int>			lsLids; // member line segments' local id
	std::vector<cv::Point2d> lsEndpoints;
	std::vector<cv::Mat>		msldDescs;
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


#endif

