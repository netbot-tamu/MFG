/*------------------------------------------------------------------------------
  This file contains the decalrations of functions/modules used in MFG; 
  definations are in 'mfg.cpp' file.
  ------------------------------------------------------------------------------*/
#ifndef MFG_HEADER
#define MFG_HEADER

#include <iostream>
#include <fstream>
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <opencv2/nonfree/nonfree.hpp>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include "lsd/lsd.h"
#include "levmar-2.6/levmar.h"
// Eigen includes replace above with linux variant
// TODO: make this windows-compatible
//#include "C:\Study\Research\Toolbox\C_C++\Eigen311\Eigen\Dense"
//#include "C:\Study\Research\Toolbox\C_C++\Eigen311\Eigen\Eigenvalues"
//#include "C:\Study\Research\Toolbox\C_C++\Eigen311\Eigen\Geometry"

#include <QThread>

#include "view.h"
#include "features2d.h"
#include "features3d.h"


using namespace Eigen;
using namespace std;
#define PI (3.14159265)
//#define HIGH_SPEED_NO_GRAPHICS
//#define DEBUG

class MfgSettings;

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

   MfgThread(MfgSettings* _settings) : mfgSettings(_settings) {}

private:
   MfgSettings* mfgSettings;
};


#endif //MFG_HEADER
