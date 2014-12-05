
#ifndef VIEW_H_
#define VIEW_H_

#include <vector>
#include <string>

#include <opencv2/core/core.hpp>

#include "features2d.h"

class MfgSettings;

// View class collects information from a single view, including points,
// lines etc
class View
{
public:
   // ***** data members *****
   int							id; //		keyframe id
   int							frameId;//  rawframe id
   std::string						filename;
   std::vector <FeatPoint2d>		featurePoints;
   std::vector <LineSegmt2d>		lineSegments;
   std::vector <VanishPnt2d>		vanishPoints;
   std::vector <IdealLine2d>		idealLines;
   cv::Mat						K;
   cv::Mat						R;	// rotation matrix w.r.t. {W}
   cv::Mat						t;	// translation std::vector in {W}
   cv::Mat						t_loc;// local relative translation with previous view
   std::vector< std::vector<int> >	vpGrpIdLnIdx;
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
   View (std::string imgName, cv::Mat _K, cv::Mat dc, MfgSettings* _settings);
   View (std::string imgName, cv::Mat _K, cv::Mat dc, int _id, MfgSettings* _settings);
   void detectFeatPoints ();			// detect feature points from image
   void detectLineSegments (cv::Mat);			// detect line segments from image
   void compMsld4AllSegments (cv::Mat grayImg);
   void detectVanishPoints ();
   void extractIdealLines();
   void drawLineSegmentGroup(std::vector<int> idx);
   void drawAllLineSegments(bool write2file = false);
   void drawIdealLineGroup(std::vector<IdealLine2d>);
   void drawIdealLines();
   void drawPointandLine();

private:
   MfgSettings* mfgSettings;
};

#endif
