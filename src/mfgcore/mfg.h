/////////////////////////////////////////////////////////////////////////////////
//
//  Multilayer Feature Graph (MFG), version 1.0
//  Copyright (C) 2011-2015 Yan Lu, Dezhen Song
//  Netbot Laboratory, Texas A&M University, USA
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
//
/////////////////////////////////////////////////////////////////////////////////

#ifndef MFG_HEADER
#define MFG_HEADER

// Standard library
#include <iostream>
#include <fstream>

// OpenCV
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <opencv2/nonfree/nonfree.hpp>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

// Qt
#include <QThread>

// MFG
#include "view.h"
#include "features2d.h"
#include "features3d.h"

using namespace Eigen; 	// this should be removed
using namespace std;	// this should be removed

//#define PLOT_MID_RESULTS

class MfgSettings;

// Frame type: This class is used for feature tacking in raw frames
class Frame {
public:
    //int frameId;  						// raw frame id
    std::string filename;					
    cv::Mat image; 							// gray image
    std::vector<cv::Point2f> featpts;
    std::vector<int> pt_lid_in_last_view;
};

// Mfg type: This class contains the 3D information of MFG views
class Mfg
{
public:
    cv::Mat						K;
	std::vector<View>			views;
    std::vector<KeyPoint3d>		keyPoints;
    std::vector<KeyPoint3d>		pointTrack; // points that are being tracked but not triangulated yet
    std::vector<IdealLine3d>	idealLines;
    std::vector<IdealLine3d>	lineTrack;
    std::vector<VanishPnt3d>	vanishingPoints;
    std::vector<PrimPlane3d>	primaryPlanes;

    double angVel; // angle velocity (deg/sec)
    double linVel; // linear velocity (m/s)
    double fps;

	// Used when ground plane detection is enabled
    bool need_scale_to_real;
    std::vector<int>  scale_since;
    std::vector<double> scale_vals;
    double camera_height;
    std::vector<std::vector<double> > camdist_constraints; // cam1_id, cam2_id, dist, confidence

    // For feature point tracking
    std::vector<Frame> trackFrms;
    double angleSinceLastKfrm;

    Mfg() {}
    Mfg(View v0, int ini_incrt, cv::Mat dc, double fps_);
    Mfg(View v0, View v1, double fps_) 
	{
        views.push_back(v0);
        views.push_back(v1);
        fps = fps_;
        initialize();
    }

    void initialize(); // initializes MFG with first two views
    void expand(View &, int frameId);
    void expand_keyPoints(View &prev, View &nview);
    void expand_idealLines(View &prev, View &nview);
    void detectLnOutliers(double threshPt2LnDist);
    void detectPtOutliers(double threshPt2PtDist);
    void adjustBundle();
    void adjustBundle_G2O(int numPos, int numFrm);
    void est3dIdealLine(int lnGid);
    void update3dIdealLine(std::vector< std::vector<int> > ilinePairIdx, View &nview);
    void updatePrimPlane();
    void draw3D() const;
    bool rotateMode();
    void exportAll(std::string root_dir);
    void scaleLocalMap(int from_view_id, int to_view_id, double, bool);
    void cleanup(int n_keep);
    void bundle_adjust_between(int from_view, int to_view, int);
};

// MfgThread type
class MfgThread : public QThread
{
    Q_OBJECT

signals:
    void closeAll();

protected:
    void run();

public:
    Mfg *pMap;
    std::string imgName;	// first image name
    cv::Mat K;				// camera intrinsic matrix
	cv::Mat distCoeffs;		// distortion coefficients (k1, k2, p1, p2)
    int imIdLen;			// num of digits in frame file id
    int ini_incrt;			// first frame step
    int increment;			// frame step
    int totalImg;			// num of frames to process

    MfgThread(MfgSettings *_settings) : mfgSettings(_settings) {}

private:
    MfgSettings *mfgSettings;
};

#endif
