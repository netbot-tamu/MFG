/////////////////////////////////////////////////////////////////////////////////
//
//  Multilayer Feature Graph (MFG), version 1.0
//  Copyright (C) 2011-2015 Yan Lu, Madison Treat, Dezhen Song
//  Netbot Laboratory, Texas A&M University, USA
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU Lesser General Public License as published by
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

#include "mfg.h"

// OpenGL
#ifdef __APPLE__
#include <GLUT/glut.h>
#include <OpenGL/gl.h>
#elif __linux__
#include <GL/glut.h>
#include <GL/gl.h>
#else
#include <gl/glut.h>
#include <gl/gl.h>
#endif

// Standard library
#include <math.h>
#include <fstream>
#ifdef _MSC_VER
#include <unordered_map>
#else
#include <unordered_map>
#endif

// MFG
#include "mfgutils.h"
#include "export.h"
#include "utils.h"
#include "settings.h"

#define THRESH_PARALLAX (7) 
#define THRESH_PARALLAX_DEGREE (0.9)

extern double THRESH_POINT_MATCH_RATIO, SIFT_THRESH, SIFT_THRESH_HIGH, SIFT_THRESH_LOW;
extern int IDEAL_IMAGE_WIDTH;
extern MfgSettings *mfgSettings;
extern bool mfg_writing;

using namespace std;
using namespace cv;

struct cvpt2dCompare
{
    bool operator()(const cv::Point2d &lhs, const cv::Point2d &rhs) const
    {
        if (cv::norm(lhs - rhs) > 1e-7 && lhs.x < rhs.x)
            return true;
        else
            return false;
    }
};

/**
 * Given the fist view, this constructor reads in raw frames until the first frame step 
 * is reached. Feature points are tracked in each raw frame to establish feature point
 * correspondence between key frames (i.e. views). The initial MFG is constructed using 
 * the first two views.
 *
 * v0: first view
 * ini_incrt: first frame step
 * dc: distortion coefficients
 *
 */
Mfg::Mfg(View v0, int ini_incrt, Mat dc, double fps_)
{
    angVel = 0;
    angleSinceLastKfrm = 0;
    fps = fps_;

	// Store 1st view
    views.push_back(v0);

	//----------------------------------------------------------------------
	// Do feature point tracking to establish feature point correspondence 
	// between first view and second view
	//----------------------------------------------------------------------
	
	// Set 1st view feature points as "previous" feature points
    vector<Point2f> prev_pts;
    vector<int> v0_idx; // point ids in first view that is being tracked using optical flow
    for (int i = 0; i < v0.featurePoints.size(); ++i) {
        prev_pts.push_back(Point2f(v0.featurePoints[i].x, v0.featurePoints[i].y));
        v0_idx.push_back(v0.featurePoints[i].lid);
    }
   
   	// Set 1st view image as "previous" image
	K = v0.K;
    Mat prev_img = v0.grayImg;
    string imgName = v0.filename;
	
	// Track feature points in subsequent views until first frame step is reached
    for (int idx = 0; idx < ini_incrt; ++idx)
    {
		// Set next image
        imgName = nextImgName(imgName, 5, 1);

		// Read next image
        Mat oriImg = cv::imread(imgName, 1);

		// Remove lens distortion
        if (cv::norm(dc) > 1e-6)
            cv::undistort(oriImg, oriImg, K, dc);

        // Resize image and ajust camera matrix
        if (mfgSettings->getImageWidth() > 1) {
            double scl = mfgSettings->getImageWidth() / double(oriImg.cols);
            cv::resize(oriImg, oriImg, cv::Size(), scl, scl, cv::INTER_AREA);
        }

		// Set gray-scale image (for optical flow)
        Mat grayImg;
        if (oriImg.channels() == 3)
            cv::cvtColor(oriImg, grayImg, CV_RGB2GRAY);
        else
            grayImg = oriImg;

		// Compute optical flow
        vector<Point2f> curr_pts;
        vector<uchar> status;
        vector<float> err;
        calcOpticalFlowPyrLK(prev_img, grayImg, prev_pts, curr_pts, status, err,
            Size(mfgSettings->getOflkWindowSize(), mfgSettings->getOflkWindowSize()), 3,
            TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01), 0, mfgSettings->getOflkMinEigenval());

		// Extract points with matches
		vector<Point2f> tracked_pts;
        vector<int> tracked_idx;
        for (int i = 0; i < status.size(); ++i) {
            if (status[i]) { // match found
                tracked_idx.push_back(v0_idx[i]);
                tracked_pts.push_back(curr_pts[i]);
            }
        }

		// Set "previous" feature points and image for next tracking
        prev_pts = tracked_pts;
        v0_idx = tracked_idx;
        prev_img = grayImg.clone();
    }

	//----------------------------------------------------------------------
	// Feature point tracking done. Create second view.
	//----------------------------------------------------------------------
    
    double distThresh_PtHomography = 1;
    double vpLnAngleThrseh = 50; // degree, tolerance angle between 3d line direction and vp direction
    double maxNoNew3dPt = 400;
    Mat F, E, R, t;

	// Create 2nd view
    View v1(imgName, K, dc, 1, mfgSettings);
    v1.featurePoints.clear();

	// Set feature point correspondence between 1st and 2nd views
    vector<vector<Point2d>> featPtMatches;
    vector<vector<int>> pairIdx;
    for (int i = 0; i < prev_pts.size(); ++i)
    {
        v1.featurePoints.push_back(FeatPoint2d(prev_pts[i].x, prev_pts[i].y, i));
		// Store corresponding feature point location
        vector<Point2d> match;
        match.push_back(v0.featurePoints[v0_idx[i]].cvpt());
        match.push_back(v1.featurePoints.back().cvpt());
        featPtMatches.push_back(match);
       	// Store corresponding index 
		vector<int> pair;
        pair.push_back(v0_idx[i]);
        pair.push_back(i);
        pairIdx.push_back(pair);
    }

	// Store 2nd view
    views.push_back(v1);

	//----------------------------------------------------------------------
	// Compute R and t using 1st and 2nd view epipolar geometry
	//----------------------------------------------------------------------
    
    View &view0 = views[0];
	View &view1 = views[1];
    view1.frameId = atoi(imgName.substr(imgName.size() - 5 - 4, 5).c_str());

	// Compute R and t between first two views
    computeEpipolar(featPtMatches, pairIdx, K, F, R, E, t, true);
	cout << "R=" << R << endl << "t=" << t << endl;
	v1.t_loc = t;
    
	bool isRgood = true;
    vector<vector<int>> vpPairIdx;
    vector<vector<int>> ilinePairIdx;
    vpPairIdx = matchVanishPts_withR(v0, v1, R, isRgood);

    vector<KeyPoint3d> tmpkp;
    for (int i = 0; i < featPtMatches.size(); ++i) {
        Mat X = triangulatePoint_nonlin(Mat::eye(3, 3, CV_64F), Mat::zeros(3, 1, CV_64F), R, t, K, featPtMatches[i][0], featPtMatches[i][1]);
        tmpkp.push_back(KeyPoint3d(X.at<double>(0), X.at<double>(1), X.at<double>(2)));
    }

	// Set epipoles
    Point2d ep1 = mat2cvpt(K * R.t() * t);
    Point2d ep2 = mat2cvpt(K * t);
    view0.epipoleA = ep1;
    view1.epipoleB = ep2;

	//----------------------------------------------------------------------
	// Set 3D feature points
	//----------------------------------------------------------------------
	
    int numNew3dPt = 0;
    Mat X(4, 1, CV_64F);

	// Loop through each feature point match
    for (int i = 0; i < featPtMatches.size(); ++i)
    {
		// Compute parallax
        double parallax = compParallax(featPtMatches[i][0], featPtMatches[i][1], K, Mat::eye(3, 3, CV_64F), R);

        // If parallax is not enough...
        if (parallax < THRESH_PARALLAX)
        {
            // Set 3D-2D correspondence
            KeyPoint3d kp;
            kp.gid = keyPoints.size();
            view0.featurePoints[pairIdx[i][0]].gid = kp.gid;
            view1.featurePoints[pairIdx[i][1]].gid = kp.gid;
			// Store index correspondence
            vector<int> view_point;
            view_point.push_back(view0.id);
            view_point.push_back(pairIdx[i][0]);
            kp.viewId_ptLid.push_back(view_point);
            view_point.clear();
            view_point.push_back(view1.id);
            view_point.push_back(pairIdx[i][1]);
			kp.viewId_ptLid.push_back(view_point);
			// Do not triangulate yet
            kp.is3D = false;
		   	// Add point
			keyPoints.push_back(kp);
        }
        else
        {
			// if enough points constructed
            if (numNew3dPt > maxNoNew3dPt) 
				continue;

			// Triangulatate point
            Mat X = triangulatePoint_nonlin(
				Mat::eye(3, 3, CV_64F), 
				Mat::zeros(3, 1, CV_64F), R, t, K, featPtMatches[i][0],  featPtMatches[i][1]);

			// Is point in front of both cameras?
            if (!checkCheirality(R, t, X) || !checkCheirality(Mat::eye(3, 3, CV_64F), Mat::zeros(3, 1, CV_64F), X))
                continue;

			// Is point too far?
            if (cv::norm(X) > 20) 
				continue;

            // Set 3D-2D correspondence
            KeyPoint3d kp(X.at<double>(0), X.at<double>(1), X.at<double>(2));
            kp.gid = keyPoints.size();
            view0.featurePoints[pairIdx[i][0]].gid = kp.gid;
            view1.featurePoints[pairIdx[i][1]].gid = kp.gid;
           	// Store index correspondence 
			vector<int> view_point;
            view_point.push_back(view0.id);
            view_point.push_back(pairIdx[i][0]);
            kp.viewId_ptLid.push_back(view_point);
            view_point.clear();
            view_point.push_back(view1.id);
            view_point.push_back(pairIdx[i][1]);
            kp.viewId_ptLid.push_back(view_point);
           	// Exists in 3D 
			kp.is3D = true;
            kp.estViewId = 1;
			// Add point
            keyPoints.push_back(kp);
			numNew3dPt++;
        }
    }

    view0.R = Mat::eye(3, 3, CV_64F);
    view0.t = Mat::zeros(3, 1, CV_64F);
    view1.R = R.clone();
    view1.t = t.clone();

	//----------------------------------------------------------------------
	// Set 3D vanishing points
	//----------------------------------------------------------------------
   
   	// Loop through each vanishing point matches 
	for (int i = 0; i < vpPairIdx.size(); ++i)
    {
        Mat tmpvp = K.inv() * view0.vanishPoints[vpPairIdx[i][0]].mat();
        tmpvp = tmpvp / cv::norm(tmpvp);

        // Set 3D-2D correspondence
        VanishPnt3d vp(tmpvp.at<double>(0), tmpvp.at<double>(1), tmpvp.at<double>(2));
        vp.gid = vanishingPoints.size();
        vanishingPoints.push_back(vp);
        view0.vanishPoints[vpPairIdx[i][0]].gid = vp.gid;
        view1.vanishPoints[vpPairIdx[i][1]].gid = vp.gid;
        // Store index correspondence 
		vector<int> vid_vpid;
        vid_vpid.push_back(view0.id);
        vid_vpid.push_back(vpPairIdx[i][0]);
        vanishingPoints.back().viewId_vpLid.push_back(vid_vpid);
        vid_vpid.clear();
        vid_vpid.push_back(view1.id);
        vid_vpid.push_back(vpPairIdx[i][1]);
		vanishingPoints.back().viewId_vpLid.push_back(vid_vpid);
    }

	// Match ideal lines
    matchIdealLines(view0, view1, vpPairIdx, featPtMatches, F, ilinePairIdx, 1);

	//----------------------------------------------------------------------
	// Set 3D coplanar points
	//----------------------------------------------------------------------
   
   	// Loop through each key points
	for (int i = 0; i < keyPoints.size(); ++i)
    {
		// If point is not triangulated yet...
        if (!keyPoints[i].is3D || keyPoints[i].gid < 0) 
			continue;

        FeatPoint2d p0 = view0.featurePoints[keyPoints[i].viewId_ptLid[0][1]];
        FeatPoint2d p1 = view1.featurePoints[keyPoints[i].viewId_ptLid[1][1]];

		// Loop through each plane
        for (int j = 0; j < primaryPlanes.size(); ++j)
        {
			// Compute homography
            Mat H = K * (R - t * primaryPlanes[j].n.t() / primaryPlanes[j].d) * K.inv();

            double dist = 
				0.5 * cv::norm(mat2cvpt(H * p0.mat()) - mat2cvpt(p1.mat())) + 
				0.5 * cv::norm(mat2cvpt(H.inv() * p1.mat()) - mat2cvpt(p0.mat()));
			
			// Check if point is on plane
            if (dist < distThresh_PtHomography) {
                keyPoints[i].pGid = primaryPlanes[j].gid;
                break;
            }
        }
    }

    Mat canv1, canv2;

    //----------------------------------------------------------------------
	// Set 3D lines (only coplanar ones)
	//----------------------------------------------------------------------
    
	for (int i = 0; i < ilinePairIdx.size(); ++i)
    {
		// Compute parallax
        double prlx = compParallax(
			view0.idealLines[ilinePairIdx[i][0]],
            view1.idealLines[ilinePairIdx[i][1]], K, view0.R, view1.R);

		// If parallax not enough...
        if (prlx < THRESH_PARALLAX || view0.idealLines[ilinePairIdx[i][0]].pGid < 0)
        {
            IdealLine3d line;
            line.is3D = false;
        	// Set 3D-2D correspondence
            line.gid = idealLines.size();
            line.vpGid = view0.vanishPoints[view0.idealLines[ilinePairIdx[i][0]].vpLid].gid; // assign vpGid to line
            view0.idealLines[ilinePairIdx[i][0]].gid = line.gid;
            view1.idealLines[ilinePairIdx[i][1]].gid = line.gid;
            line.pGid = view0.idealLines[ilinePairIdx[i][0]].pGid;
            view1.idealLines[ilinePairIdx[i][1]].pGid = line.pGid;
			// Store corresponding index            
			vector<int> pair;
            pair.push_back(view0.id);
            pair.push_back(ilinePairIdx[i][0]);
            line.viewId_lnLid.push_back(pair);
            pair.clear();
            pair.push_back(view1.id);
            pair.push_back(ilinePairIdx[i][1]);
            line.viewId_lnLid.push_back(pair);
			// Add line
            idealLines.push_back(line);
        }
        else
        {
			// Triangulate line
            IdealLine2d a = view0.idealLines[ilinePairIdx[i][0]], b = view1.idealLines[ilinePairIdx[i][1]];
            IdealLine3d line = triangluateLine(Mat::eye(3, 3, CV_64F), Mat::zeros(3, 1, CV_64F), R, t, K, a, b);

            // 3D line and vp angle too large
            int vpGid = view0.vanishPoints[view0.idealLines[ilinePairIdx[i][0]].vpLid].gid;
            if (abs(vanishingPoints[vpGid].mat().dot(line.direct) / cv::norm(line.direct)) < cos(vpLnAngleThrseh * PI / 180))
                continue; // invalid line

            line.is3D = true;
        	
			// Set 3D-2D correspondence
            line.gid = idealLines.size();
            line.vpGid = view0.vanishPoints[view0.idealLines[ilinePairIdx[i][0]].vpLid].gid;
            view0.idealLines[ilinePairIdx[i][0]].gid = line.gid;
            view1.idealLines[ilinePairIdx[i][1]].gid = line.gid;
            line.pGid = view0.idealLines[ilinePairIdx[i][0]].pGid;
            view1.idealLines[ilinePairIdx[i][1]].pGid = line.pGid;
			// Store corresponding index            
			vector<int> pair;
            pair.push_back(view0.id);
            pair.push_back(ilinePairIdx[i][0]);
            line.viewId_lnLid.push_back(pair);
            pair.clear();
            pair.push_back(view1.id);
            pair.push_back(ilinePairIdx[i][1]);
            line.viewId_lnLid.push_back(pair);
			// Add line
            idealLines.push_back(line);
#ifdef PLOT_MID_RESULTS
            cv::Scalar color(rand() % 255, rand() % 255, rand() % 255, 0);
            cv::line(canv1, a.extremity1, a.extremity2, color, 2);
            cv::line(canv2, b.extremity1, b.extremity2, color, 2);
#endif
        }
    }

   	// Compute angular velocity
    Matrix3d Rx;
    Rx <<
		R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
    	R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
    	R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
	Quaterniond q(Rx);
    angVel = 2 * acos(q.w()) * 180 / PI / ((views[views.size() - 1].frameId - views[views.size() - 2].frameId) / fps);
    view0.angVel = 0;
    view1.angVel = angVel;

	// Update planes
    updatePrimPlane();

    cout << endl << " >>>>>>>>>> MFG initialized using two views <<<<<<<<<<" << endl;

	// If fixed height from ground is assumed...
    if (mfgSettings->getDetectGround()) {
        need_scale_to_real = true;
        scale_since.push_back(1);
        camera_height = 1.65; // meter
    }

    return;
}

// Construct MFG with first two views
void Mfg::initialize()
{
    angVel = 0;
    angleSinceLastKfrm = 0;

    double distThresh_PtHomography = 1;
    double vpLnAngleThrseh = 50; // degree, tolerance angle between 3d line direction and vp direction
    double maxNoNew3dPt = 400;

	//----------------------------------------------------------------------
	// Compute R and t using 1st and 2nd view epipolar geometry
	//----------------------------------------------------------------------
    
	View &view0 = views[0];
    View &view1 = views[1];
    K = view0.K;

    vector<vector<Point2d>> featPtMatches, allFeatPtMatches;
    vector<vector<int>>	pairIdx, allPairIdx;
    vector<vector<int>>	vpPairIdx;
    vector<vector<int>>	ilinePairIdx;
    Mat F, R, E, t;

    pairIdx = matchKeyPoints(view0.featurePoints, view1.featurePoints, featPtMatches);

	// Compute R and t
    allFeatPtMatches = featPtMatches;
    allPairIdx = pairIdx;
    computeEpipolar(featPtMatches, pairIdx, K, F, R, E, t, true);
    cout << "R=" << R << endl << "t=" << t << endl;
    view1.t_loc = t;
   
   	// Match vanishing points with R
	bool isRgood = true;
    vpPairIdx = matchVanishPts_withR(view0, view1, R, isRgood);

	// If R is not consistent with VPs...
    if (!isRgood)     
	{
		// Estimate R and t with VPs
        vector<vector<Mat>> vppairs;
        for (int i = 0; i < vpPairIdx.size(); ++i) {
            vector<Mat> pair;
            pair.push_back(view0.vanishPoints[vpPairIdx[i][0]].mat());
            pair.push_back(view1.vanishPoints[vpPairIdx[i][1]].mat());
            vppairs.push_back(pair);
        }
        optimizeRt_withVP(K, vppairs, 1000, featPtMatches, R, t);
        
		// Compute all potential relative poses from 5-point ransac algorithm
        vector<Mat> Fs, Es, Rs, ts;
        computePotenEpipolar(allFeatPtMatches, allPairIdx, K, Fs, Es, Rs, ts);
       
	   	// Find best R and t
		double difR, idx = 0, minDif = 100;
        for (int i = 0; i < Rs.size(); ++i) {
            difR =  cv::norm(R - Rs[i]);
            if (minDif > difR) {
                minDif = difR;
                idx = i;
            }
        }
        R = Rs[idx];
        t = ts[idx];

		// Match vanishing points with R
        matchVanishPts_withR(view0, view1, R, isRgood);
    }

	// triangulate points using adjacent keyframes, to estimate scale
    vector<KeyPoint3d> tmpkp;
    for (int i = 0; i < featPtMatches.size(); ++i) {
        Mat X = triangulatePoint_nonlin(Mat::eye(3, 3, CV_64F), Mat::zeros(3, 1, CV_64F), R, t, K, featPtMatches[i][0],	featPtMatches[i][1]);
        tmpkp.push_back(KeyPoint3d(X.at<double>(0), X.at<double>(1), X.at<double>(2)));
    }

	// Set epipoles
    Point2d ep1 = mat2cvpt(K * R.t() * t);
    Point2d ep2 = mat2cvpt(K * t);
    view0.epipoleA = ep1;
    view1.epipoleB = ep2;

	//----------------------------------------------------------------------
    // Sort point matches by parallax
	//----------------------------------------------------------------------
    
	vector<valIdxPair> prlxVec;
    for (int i = 0; i < featPtMatches.size(); ++i) {
        valIdxPair pair;
        pair.first = compParallax(featPtMatches[i][0], featPtMatches[i][1], K, Mat::eye(3, 3, CV_64F), R);
        pair.second = i;
        prlxVec.push_back(pair);
    }

    sort(prlxVec.begin(), prlxVec.end(), comparator_valIdxPair);
    
	vector<vector<int>> copyPairIdx = pairIdx;
    vector<vector<Point2d>> copyFeatPtMatches = featPtMatches;
	pairIdx.clear();
    featPtMatches.clear();

    for (int i = prlxVec.size() - 1; i >= 0; --i) {
        pairIdx.push_back(copyPairIdx[prlxVec[i].second]);
        featPtMatches.push_back(copyFeatPtMatches[prlxVec[i].second]);
    }

#ifdef PLOT_MID_RESULTS
    Mat canv1 = view0.img.clone();
    Mat canv2 = view1.img.clone();
#endif

	//----------------------------------------------------------------------
	// Set 3D feature points
	//----------------------------------------------------------------------
    
	int numNew3dPt = 0;
    Mat X(4, 1, CV_64F);

	// Loop through each feature point match
    for (int i = 0; i < featPtMatches.size(); ++i)
    {
		// Compute parallax
        double parallax = compParallax(featPtMatches[i][0], featPtMatches[i][1], K, Mat::eye(3, 3, CV_64F), R);

        // If parallax is not enough...
        if (parallax < THRESH_PARALLAX)
        {
            // Set 3D-2D correspondence
            KeyPoint3d kp;
            kp.gid = keyPoints.size();
            view0.featurePoints[pairIdx[i][0]].gid = kp.gid;
            view1.featurePoints[pairIdx[i][1]].gid = kp.gid;
			// Store index correspondence
			vector<int> view_point;
            view_point.push_back(view0.id);
            view_point.push_back(pairIdx[i][0]);
            kp.viewId_ptLid.push_back(view_point);
            view_point.clear();
            view_point.push_back(view1.id);
            view_point.push_back(pairIdx[i][1]);
            kp.viewId_ptLid.push_back(view_point);
			// Do not triangulate yet
            kp.is3D = false;
			// Add point
            keyPoints.push_back(kp);
        }
        else
        {
			
            if (numNew3dPt > maxNoNew3dPt) 
				continue;

			// Triangulate point
            Mat X = triangulatePoint_nonlin(
				Mat::eye(3, 3, CV_64F), 
				Mat::zeros(3, 1, CV_64F), R, t, K, featPtMatches[i][0], featPtMatches[i][1]);

			// Is point in front of both cameras?
            if (!checkCheirality(R, t, X) || !checkCheirality(Mat::eye(3, 3, CV_64F), Mat::zeros(3, 1, CV_64F), X))
                continue;

			// Is point to far?
            if (cv::norm(X) > 20) 
				continue;

            // Set 3D-2D correspondence
            KeyPoint3d kp(X.at<double>(0), X.at<double>(1), X.at<double>(2));
            kp.gid = keyPoints.size();
            view0.featurePoints[pairIdx[i][0]].gid = kp.gid;
            view1.featurePoints[pairIdx[i][1]].gid = kp.gid;
           	// Store index correspondence 
			vector<int> view_point;
            view_point.push_back(view0.id);
            view_point.push_back(pairIdx[i][0]);
            kp.viewId_ptLid.push_back(view_point);
            view_point.clear();
            view_point.push_back(view1.id);
            view_point.push_back(pairIdx[i][1]);
            kp.viewId_ptLid.push_back(view_point);
           	// Exists in 3D 
            kp.is3D = true;
            kp.estViewId = 1;
			// Add point
            keyPoints.push_back(kp);
            numNew3dPt++;
        }
    }

    view0.R = Mat::eye(3, 3, CV_64F);
    view0.t = Mat::zeros(3, 1, CV_64F);
    view1.R = R.clone();
    view1.t = t.clone();

	//----------------------------------------------------------------------
	// Set 3D vanishing points
	//----------------------------------------------------------------------
    
   	// Loop through each vanishing point matches 
	for (int i = 0; i < vpPairIdx.size(); ++i)
    {
        Mat tmpvp = K.inv() * view0.vanishPoints[vpPairIdx[i][0]].mat();
        tmpvp = tmpvp / cv::norm(tmpvp);

        // Set 3D-2D correspondence
        VanishPnt3d vp(tmpvp.at<double>(0),	tmpvp.at<double>(1), tmpvp.at<double>(2));
        vp.gid = vanishingPoints.size();
        vanishingPoints.push_back(vp);
        view0.vanishPoints[vpPairIdx[i][0]].gid = vp.gid;
        view1.vanishPoints[vpPairIdx[i][1]].gid = vp.gid;
        // Store index correspondence 
        vector<int> vid_vpid;
        vid_vpid.push_back(view0.id);
        vid_vpid.push_back(vpPairIdx[i][0]);
        vanishingPoints.back().viewId_vpLid.push_back(vid_vpid);
        vid_vpid.clear();
        vid_vpid.push_back(view1.id);
        vid_vpid.push_back(vpPairIdx[i][1]);
        vanishingPoints.back().viewId_vpLid.push_back(vid_vpid);
    }

	// Match ideal lines
    matchIdealLines(view0, view1, vpPairIdx, featPtMatches, F, ilinePairIdx, 1);

	//----------------------------------------------------------------------
	// Set 3D coplanar points
	//----------------------------------------------------------------------
    
   	// Loop through each key points
	for (int i = 0; i < keyPoints.size(); ++i)
    {
		// If point is not triangulated yet...
        if (! keyPoints[i].is3D || keyPoints[i].gid < 0) 
			continue;

        FeatPoint2d p0 = view0.featurePoints[keyPoints[i].viewId_ptLid[0][1]];
        FeatPoint2d p1 = view1.featurePoints[keyPoints[i].viewId_ptLid[1][1]];

		// Loop through each plane
        for (int j = 0; j < primaryPlanes.size(); ++j)
        {
			// Compute homography
            Mat H = K * (R - t * primaryPlanes[j].n.t() / primaryPlanes[j].d) * K.inv();

            double dist = 
				0.5 * cv::norm(mat2cvpt(H * p0.mat()) - mat2cvpt(p1.mat())) + 
				0.5 * cv::norm(mat2cvpt(H.inv() * p1.mat()) - mat2cvpt(p0.mat()));

			// Check if point is on plane
            if (dist < distThresh_PtHomography) {
                keyPoints[i].pGid = primaryPlanes[j].gid;
                break;
            }
        }
    }

    //----------------------------------------------------------------------
	// Set 3D lines (only coplanar ones)
	//----------------------------------------------------------------------
    
	for (int i = 0; i < ilinePairIdx.size(); ++i)
    {
		// Compute parallax
        double prlx = compParallax(
			view0.idealLines[ilinePairIdx[i][0]],
            view1.idealLines[ilinePairIdx[i][1]], K, view0.R, view1.R);

		// If parallax not enough...
        if (prlx < THRESH_PARALLAX ||  view0.idealLines[ilinePairIdx[i][0]].pGid < 0)  // set up 2d line track
        {
            IdealLine3d line;
            line.is3D = false;
        	// Set 3D-2D correspondence
            line.gid = idealLines.size();
            line.vpGid = view0.vanishPoints[view0.idealLines[ilinePairIdx[i][0]].vpLid].gid; // assign vpGid to line
            view0.idealLines[ilinePairIdx[i][0]].gid = line.gid;
            view1.idealLines[ilinePairIdx[i][1]].gid = line.gid;
            line.pGid = view0.idealLines[ilinePairIdx[i][0]].pGid;
            view1.idealLines[ilinePairIdx[i][1]].pGid = line.pGid;
			// Store corresponding index            
			vector<int> pair;
            pair.push_back(view0.id);
            pair.push_back(ilinePairIdx[i][0]);
            line.viewId_lnLid.push_back(pair);
            pair.clear();
            pair.push_back(view1.id);
            pair.push_back(ilinePairIdx[i][1]);
            line.viewId_lnLid.push_back(pair);
			// Add line
            idealLines.push_back(line);
        }
        else
        {
			// Triangulate line
            IdealLine2d a = view0.idealLines[ilinePairIdx[i][0]], b = view1.idealLines[ilinePairIdx[i][1]];
            IdealLine3d line = triangluateLine(Mat::eye(3, 3, CV_64F), Mat::zeros(3, 1, CV_64F), R, t, K, a, b);

            // 3D line and vp angle too large
            int vpGid = view0.vanishPoints[view0.idealLines[ilinePairIdx[i][0]].vpLid].gid;
            if (abs(vanishingPoints[vpGid].mat().dot(line.direct) / cv::norm(line.direct)) < cos(vpLnAngleThrseh * PI / 180))
                continue; // invalid line

            line.is3D = true;

			// Set 3D-2D correspondence
            line.gid = idealLines.size();
            line.vpGid = view0.vanishPoints[view0.idealLines[ilinePairIdx[i][0]].vpLid].gid;
            view0.idealLines[ilinePairIdx[i][0]].gid = line.gid;
            view1.idealLines[ilinePairIdx[i][1]].gid = line.gid;
            line.pGid = view0.idealLines[ilinePairIdx[i][0]].pGid;
            view1.idealLines[ilinePairIdx[i][1]].pGid = line.pGid;
			// Store corresponding index            
			vector<int> pair;
            pair.push_back(view0.id);
            pair.push_back(ilinePairIdx[i][0]);
            line.viewId_lnLid.push_back(pair);
            pair.clear();
            pair.push_back(view1.id);
            pair.push_back(ilinePairIdx[i][1]);
            line.viewId_lnLid.push_back(pair);
			// Add line
            idealLines.push_back(line);
#ifdef PLOT_MID_RESULTS
            cv::Scalar color(rand() % 255, rand() % 255, rand() % 255, 0);
            cv::line(canv1, a.extremity1, a.extremity2, color, 2);
            cv::line(canv2, b.extremity1, b.extremity2, color, 2);
#endif
        }
    }

   	// Compute angular velocity
    Matrix3d Rx;
    Rx << 
		R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
    	R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
    	R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
    Quaterniond q(Rx);
    angVel = 2 * acos(q.w()) * 180 / PI / ((views[views.size() - 1].frameId - views[views.size() - 2].frameId) / fps);
    view0.angVel = 0;
    view1.angVel = angVel;

	// Update planes
    updatePrimPlane();
    
	cout << endl << " >>>>>>>>>> MFG initialized using two views <<<<<<<<<<" << endl;

	// If fixed height from ground is assumed...
    if (mfgSettings->getDetectGround()) {
        need_scale_to_real = true;
        scale_since.push_back(1);
        camera_height = 1.65; // meter
    }

    return;
}

void Mfg::expand(View &nview, int fid)
{
	// Store current view
    nview.id = views.size();
    nview.frameId = fid;
    views.push_back(nview);

	// Expand MFG using two views
    expand_keyPoints(views[views.size() - 2], views[views.size() - 1]);

    // Remove point and line outliers
    detectPtOutliers(5 * IDEAL_IMAGE_WIDTH / 640.0);
    detectLnOutliers(10 * IDEAL_IMAGE_WIDTH / 640.0);

	// Do bundle adjustment
    adjustBundle();

    // Further remove point and line outliers
    detectPtOutliers(2 * IDEAL_IMAGE_WIDTH / 640.0);
    detectLnOutliers(3 * IDEAL_IMAGE_WIDTH / 640.0);

}

void Mfg::expand_keyPoints(View &prev, View &nview)
{
    double numPtpairLB = 30; 							// lower bound of pair number to trust epi than vp
    double reprjThresh = 5 * IDEAL_IMAGE_WIDTH / 640.0; // projected point distance threshold
    double scale = -1;  								// relative scale of baseline
    double vpLnAngleThrseh = 50; 						// degree, tolerance angle between 3d line direction and vp direction
    double ilineLenLimit = 10;  						// length limit of 3d line, ignore too long lines
    double maxNumNew3dPt = 200;
    double maxNumNew2d_3dPt = 500;
    
	bool   sortbyPrlx = false;
    double parallaxThresh = THRESH_PARALLAX;
    double parallaxDegThresh = THRESH_PARALLAX_DEGREE;
    double accel_max = 10; // m/s/s

	//----------------------------------------------------------------------
	// Find feature point correspondence between two views
	//----------------------------------------------------------------------

    vector<vector<Point2d>> featPtMatches;
    vector<vector<int>> pairIdx;

	// Use SIFT or SURF for feature matching? 
    if (mfgSettings->getKeypointAlgorithm() < 3)
	{
        pairIdx = matchKeyPoints(prev.featurePoints, nview.featurePoints, featPtMatches);
    } 
	else // use optical flow
    {
        bool found_in_track = false;

        Frame frm_same_nview; // frame corresponding to the current keyframe
        for (int i = 0; i < trackFrms.size(); ++i) {
            if (trackFrms[i].filename == nview.filename) {
                found_in_track = true;
                frm_same_nview = trackFrms[i];
                break;
            }
        }

        if (!found_in_track)
        {
			// Track feature points with optical flow
            vector<Point2f> curr_pts;
            vector<uchar> status;
            vector<float> err;
            calcOpticalFlowPyrLK(trackFrms.back().image, nview.grayImg, trackFrms.back().featpts, curr_pts, status, err,
                Size(mfgSettings->getOflkWindowSize(), mfgSettings->getOflkWindowSize()), 3,
                TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01), 0,
                mfgSettings->getOflkMinEigenval());
			
			// Extract feature points with match
            for (int i = 0; i < status.size(); ++i) {
                if (status[i]) { // found match
                    frm_same_nview.featpts.push_back(curr_pts[i]);
                    frm_same_nview.pt_lid_in_last_view.push_back(trackFrms.back().pt_lid_in_last_view[i]);
                }
            }
        }

        // record tracked points to avoid future use in matching
        std::map<Point2d, int, cvpt2dCompare> pt_idx;
        vector<cv::KeyPoint> kpts;
        for (int j = 0; j < frm_same_nview.featpts.size(); ++j) {
            pt_idx[frm_same_nview.featpts[j]] = frm_same_nview.pt_lid_in_last_view[j];
            kpts.push_back(cv::KeyPoint(frm_same_nview.featpts[j], mfgSettings->getFeatureDescriptorRadius()));
        }

	
        for (int j = 0; j < nview.featurePoints.size(); ++j)
        {
            bool pt_exist = false;
            for (int k = 0; k < frm_same_nview.featpts.size(); ++k) {
                float dx = frm_same_nview.featpts[k].x - nview.featurePoints[j].x;
                float dy = frm_same_nview.featpts[k].y - nview.featurePoints[j].y;
				if (sqrt(dx * dx + dy * dy) < mfgSettings->getGfttMinimumPointDistance()) {
                    pt_exist = true;
                    break;
                }
            }
            if (!pt_exist)
                kpts.push_back(cv::KeyPoint(nview.featurePoints[j].cvpt(), mfgSettings->getFeatureDescriptorRadius()));
        }

        // Compute descriptors for feature points using SURF
        cv::SurfDescriptorExtractor surfext;
        Mat descs;
        surfext.compute(nview.grayImg, kpts, descs); // note that some ktps can be removed
        nview.featurePoints.clear();
        nview.featurePoints.reserve(kpts.size());

        std::map<int, int> idx_pair_exist; // avoid one-to-many matches

        for (int i = 0; i < kpts.size(); ++i)
        {
            int lid = nview.featurePoints.size();
            
			nview.featurePoints.push_back(FeatPoint2d(kpts[i].pt.x, kpts[i].pt.y, descs.row(i).t(), lid, -1));

            if (pt_idx.find(kpts[i].pt) != pt_idx.end()) // matched with last keyframe
            {
                if (idx_pair_exist.find(pt_idx[kpts[i].pt]) != idx_pair_exist.end())
                    continue;

                vector<int> pair(2);
                vector<Point2d> match(2);
                pair[0] = pt_idx[kpts[i].pt];
                pair[1] = lid;
                idx_pair_exist[pair[0]] = pair[1];
                match[0] = prev.featurePoints[pair[0]].cvpt();
                match[1] = kpts[i].pt;
				pairIdx.push_back(pair);
                featPtMatches.push_back(match);
            }
        }
    }

#ifdef PLOT_MID_RESULTS
    Mat canv1 = prev.img.clone(), canv2 = nview.img.clone();
    for (int i = 0; i < featPtMatches.size(); ++i) {
        cv::circle(canv1, featPtMatches[i][0], 2, cv::Scalar(100, 200, 200), 2);
        cv::circle(canv2, featPtMatches[i][1], 2, cv::Scalar(100, 200, 200), 2);
    }
#endif

    trackFrms.clear();

	//----------------------------------------------------------------------
	// Compute R and t candidates (using epipolar geometry)
	//----------------------------------------------------------------------
    
	//Mat t_prev = prev.t - (prev.R * views[views.size()-3].R.t()) * views[views.size()-3].t;
    Mat t_prev = views[views.size() - 2].t_loc;

    Mat F, R, E, t; // relative pose between last two views
    vector<Mat> Fs, Es, Rs, ts;
    
	if (mfgSettings->getDetectGround())
        computeEpipolar(featPtMatches, pairIdx, K, Fs, Es, Rs, ts);
    else
        computePotenEpipolar(featPtMatches, pairIdx, K, Fs, Es, Rs, ts, false, t_prev);

    R = Rs[0];
    t = ts[0];

	//----------------------------------------------------------------------
	// Compute R and t (using PnP)
	//----------------------------------------------------------------------
    
	// Get visible 3D points
	
	vector<Point3d> pt3d, pt3d_old;
    vector<Point2d> pt2d, pt2d_old;
	
	// Loop through feature point matches
    for (int i = 0; i < featPtMatches.size(); ++i)
    {
        int gid = prev.featurePoints[pairIdx[i][0]].gid;

#ifdef PLOT_MID_RESULTS
        cv::Scalar color(rand() % 255, rand() % 255, rand() % 255, 0);
		if (gid >= 0 && keyPoints[gid].is3D) { // observed 3d pts
            cv::circle(canv1, featPtMatches[i][0], 4, color, 2);
            cv::circle(canv2, featPtMatches[i][1], 4, color, 2);
        }
        else {
            cv::circle(canv1, featPtMatches[i][0], 1, color, 1);
            cv::circle(canv2, featPtMatches[i][1], 1, color, 1);
        }
#endif
		
		// Is this key point triangulated?
        if (gid >= 0 && keyPoints[gid].is3D) {
            pt3d.push_back(Point3d(keyPoints[gid].x, keyPoints[gid].y, keyPoints[gid].z));
            pt2d.push_back(featPtMatches[i][1]);
			// Is key point observed at least 3 times?
            if (keyPoints[gid].viewId_ptLid.size() >= 3) { 
                pt3d_old.push_back(keyPoints[gid].cvpt());
                pt2d_old.push_back(featPtMatches[i][1]);
            }
        }
    }

    // Compute R and t
    
	Mat Rn_pnp, tn_pnp; // relative R and t
	Mat R_pnp, t_pnp;	

    if (pt3d.size() > 3) // more than minimum num of points to solve PnP
    {
		// Convert Point3d to Point3f (for solvePnPRansac function)
        vector<Point3f> pt3f;
        vector<Point2f> pt2f;
        for (int i = 0; i < pt3d.size(); ++i) {
            pt3f.push_back(Point3f(pt3d[i].x, pt3d[i].y, pt3d[i].z));
            pt2f.push_back(Point2f(pt2d[i].x, pt2d[i].y));
        }

		// Solve PnP with RANSAC
        Mat rvec;
        solvePnPRansac(pt3f, pt2f, K, Mat(), rvec, tn_pnp, false, 100, 8.0, 200);
        cv::Rodrigues(rvec, Rn_pnp);

        R_pnp = Rn_pnp * prev.R.t();
        t_pnp = tn_pnp - R_pnp * prev.t; // relative t, with scale
    }

	//----------------------------------------------------------------------
	// Find best R and t from candidate R and ts. 
	//----------------------------------------------------------------------
    
	bool use_const_vel_model = false; // When not enough features are available or no good estimation
    
	vector<int> maxInliers_Rt;
    vector<KeyPoint3d> localKeyPts;
    double bestScale = -100;
    vector<double> sizes;

	// Do multiple trials...
    for (int trial = 0; trial < 10; ++trial)
    {
        maxInliers_Rt.clear();
        localKeyPts.clear();
        sizes.clear();

		// Loop through candidate Rs
        for (int iter = 0; iter < Rs.size(); ++iter)
        {
            Mat Ri = Rs[iter], ti = ts[iter];

            bool isRgood = true;
            vector<vector<int>> vpPairIdx = matchVanishPts_withR(prev, nview, Ri, isRgood);

			//----------------------------------------------------------------------
            // Triangulate points 
			//----------------------------------------------------------------------
            
			vector<KeyPoint3d> tmpkp; // temporary keypoints
            int num_usable_pts = 0;

			// Loop through each feature point matches	
            for (int i = 0; i < featPtMatches.size(); ++i)
            {
				// Compute parallax degree	
                double parallaxDeg = compParallaxDeg(featPtMatches[i][0], featPtMatches[i][1], K, Mat::eye(3, 3, CV_64F), Ri);
                
				int iGid = prev.featurePoints[pairIdx[i][0]].gid;

				// If parallax is enough...
                if ((parallaxDeg > parallaxDegThresh * 0.8) && iGid >= 0 && keyPoints[iGid].is3D)
                {
					// Triangulate point
                    Mat X = triangulatePoint_nonlin(Mat::eye(3, 3, CV_64F), Mat::zeros(3, 1, CV_64F), 
						Ri, ti, K, featPtMatches[i][0], featPtMatches[i][1]);
                    tmpkp.push_back(KeyPoint3d(X.at<double>(0), X.at<double>(1), X.at<double>(2)));
                    tmpkp.back().is3D = true;
                    ++num_usable_pts;
                }
                else // point not useful for estimating scale
                {
                    tmpkp.push_back(KeyPoint3d(0, 0, 0));
                    tmpkp.back().is3D = false;
                }
            }

			// If there is not enough useful points...
            if (num_usable_pts < 1 || (num_usable_pts < 10 && pt3d.size() > 50))
                continue; // abandon this R and t

			// Fix rotation
            Mat Rn = Ri * prev.R;

			//----------------------------------------------------------------------
            // Find best t based on 3D points
			//----------------------------------------------------------------------
            
			int maxIter = 100, maxmaxIter = 1000;
            vector<int> maxInliers;
            double best_s = -1;

			// Generate and test different ts
            for (int it = 0, itt = 0; it < maxIter && itt < maxmaxIter; ++it, ++itt)
            {
				// Randomly pick feature point
                int i = xrand() % featPtMatches.size();

                int iGid = prev.featurePoints[pairIdx[i][0]].gid;

				// If this feature point not observed before...
                if (iGid < 0) {
                    --it;                        
					continue;
                }

				// If this feature point is not triangulated yet...
                if (!keyPoints[iGid].is3D || keyPoints[iGid].gid < 0) {
                    --it;    
                    continue;
                }

                if (!tmpkp[i].is3D) {
                    --it;
                    continue;
                }

                // Compute scale (minimum solution)
                double s = cv::norm(keyPoints[iGid].mat(0) + (prev.R.t() * prev.t)) / cv::norm(tmpkp[i].mat());

				// Compute t
                Mat tn = Ri * prev.t + ti * s;
               
				// Find inliers that conform to this R and t pair
				vector<int> inliers;
                for (int j = 0; j < featPtMatches.size(); ++j)
                {
                    int jGid = prev.featurePoints[pairIdx[j][0]].gid;

                    if (jGid < 0) 
						continue;

                    if (!keyPoints[jGid].is3D || keyPoints[jGid].gid < 0) 
						continue;

                    if (!tmpkp[j].is3D) 
						continue;

                    // Project 3D to n-th view
                    Mat pt = K * Rn * (keyPoints[jGid].mat(0) + Rn.t() * tn);

					// Is reprojection error less than threshold
                    if (cv::norm(featPtMatches[j][1] - mat2cvpt(pt)) < reprjThresh)
                        inliers.push_back(j); // add to inlier set
                }

				// Does this t have the largest inlier set?
                if (maxInliers.size() < inliers.size()) {
                    maxInliers = inliers;
                    best_s = s;
                }
            }

			cout << ti.t() << " " << maxInliers.size() << endl;
			
            if (cv::norm(t_prev) > 0.1 && abs(ti.dot(t_prev)) < cos(50 * PI / 180.0))
                continue;

            sizes.push_back(maxInliers.size());

			// Does this R and t pair have the largest inlier set?
            if (maxInliers.size() > maxInliers_Rt.size())
            {
                maxInliers_Rt = maxInliers;
                bestScale = best_s;
                R = Ri;
                t = ti;
                F = Fs[iter];
                localKeyPts = tmpkp;
            }
        }

        sort(sizes.begin(), sizes.end());

        if (sizes.size() > 1 && (
			sizes[sizes.size() - 2] >= sizes[sizes.size() - 1] * 0.9 || 
			sizes[sizes.size() - 2] >= sizes[sizes.size() - 1] - 1))
        {
            
            // Filter out some R and t
            double minVal = 1, minIdx = 0;
            for (int i = 0; i < Rs.size(); ++i) {
                if (cv::norm(t_prev) > 0.1 && ts[i].dot(t_prev) < minVal) {
                    minVal = ts[i].dot(t_prev);
                    minIdx = i;
                }
            }

            // Remove the most different from t_prev
            Rs.erase(Rs.begin() + minIdx);
            ts.erase(ts.begin() + minIdx);
            
        }
        else
            break;
    }

	

    if (maxInliers_Rt.size() > 0)
    {
        // Re-compute scale from max inlier set

        nview.R = R * prev.R;
        cv::Scalar color(rand() % 255, rand() % 255, rand() % 255, 0);
        
		double wsum = 0, tw = 0, sum3 = 0;
        vector<double> scales, scales3;

		// Loop through inliers
        for (int i = 0; i < maxInliers_Rt.size(); ++i)
        {
            int j = maxInliers_Rt[i];

            int gid = prev.featurePoints[pairIdx[j][0]].gid;

            if (!localKeyPts[j].is3D) 
				continue;

			// Compute scale
            double s = cv::norm(keyPoints[gid].mat(0) + (prev.R.t() * prev.t)) / cv::norm(localKeyPts[j].mat());
			
            if (maxInliers_Rt.size() > 5 && (abs(s - bestScale) > 0.3 || (bestScale > 0.2 && max(s / bestScale, bestScale / s) > 1.5)))
                continue ; // could be outlier

            int obsTimesSinceEst = 0; // observed num of times since established
            for (int k = 0; k < keyPoints[gid].viewId_ptLid.size(); ++k) {
                if (keyPoints[gid].viewId_ptLid[k][0] >= keyPoints[gid].estViewId)
                    obsTimesSinceEst++;
            }

#ifdef PLOT_MID_RESULTS
            cv::putText(canv2, "  " + num2str(s), featPtMatches[j][1], 1, 1, color);
#endif

            wsum = wsum + s * pow((double)obsTimesSinceEst + 1, 3);
            tw = tw + pow((double)obsTimesSinceEst + 1, 3);
            scales.push_back(s);

            if (keyPoints[gid].viewId_ptLid.size() >= 3 && obsTimesSinceEst >= 2)  // point observed at least 3 times
            {
                sum3 = sum3 + s;
                scales3.push_back(s);
            }
        }

        if (scales.size() > 0) // fuse scales computed in different ways
        {
            scale = wsum / tw;
            if (scales3.size() >= 15)
                scale = vecMedian(scales3);

            scale = (scale + bestScale + vecMedian(scales)) / 3;
            cout << "keyframe " << nview.id << ", " << nview.frameId - prev.frameId << " frames: mixed scale = " << scale;
        }
        else
            scale = bestScale;

        if (pt3d.size() > 3)
            cout << ", " << cv::norm(t_pnp) << "(" << pt3d.size() << ")\n";

        double dt = (nview.frameId - prev.frameId) / fps;

		// If we are able to compute acceleration...
        if (prev.speed > 0 && !need_scale_to_real)
        {
			// Compute acceleration
            double a0 = abs((scale / dt - prev.speed) / dt);

			// If acceleration not out of bound...
            if (a0 < accel_max)
            {
                nview.t = R * prev.t + t * scale;
                nview.R_loc = R;
                nview.t_loc = t;
            }
            else
			{
				// Just use constant velocity assumption
                use_const_vel_model = true;
			}

			// If R and t inconsistent with PnP output
            if (pt3d.size() > 10 && (
				abs(scale - cv::norm(t_pnp)) > 0.7 * scale || 
				abs(scale - cv::norm(t_pnp)) > 0.7 * cv::norm(t_pnp)))
            {
                cout << "Inconsistent pnp and 5-pt: " << cv::norm(t_pnp) << ", " << scale << endl;
               
			   	// Compute acceleration
				double a1 = abs((cv::norm(t_pnp) / dt - prev.speed) / dt);

                if (a0 >= accel_max && a1 >= accel_max)
                {
					// Just use constant velocity assumption
                    use_const_vel_model = true;
                }
                else if (a1 < a0) // prefer small acc    
                {
					// Just use PnP output
                    cout << "checking accelaraion " << a0 << " > " << a1 << ", use pnp instead" << endl;
                    nview.R = Rn_pnp;
                    nview.t = tn_pnp;
                    R = R_pnp;
                    t = t_pnp / cv::norm(t_pnp);
                    nview.R_loc = R;
                    nview.t_loc = t;
                    use_const_vel_model = false;
                }
            }
        }
        else
        {   
			nview.t = R * prev.t + t * scale;
            nview.R_loc = R;
            nview.t_loc = t;

			// If R and t inconsistent with PnP output
            if (pt3d.size() > 10 && (
				abs(scale - cv::norm(t_pnp)) > 0.7 * scale || 
				abs(scale - cv::norm(t_pnp)) > 0.7 * cv::norm(t_pnp)))
            {
                cout << "Inconsistent pnp and 5-pt: " << cv::norm(t_pnp) << ", " << scale << endl;

				// prefer small acc
                if (abs(cv::norm(t_pnp) / dt - prev.speed) < abs(scale / dt - prev.speed) && prev.speed > 0)
				{
					// Just use PnP output
                    cout << "checking accelaraion, use pnp instead" << endl;
                    nview.R = Rn_pnp;
                    nview.t = tn_pnp;
                    R = R_pnp;
                    t = t_pnp / cv::norm(t_pnp);
                    nview.R_loc = R;
                    nview.t_loc = t;
                    use_const_vel_model = false;
                }
            }
        }
    }
    else // small movement, use R-PnP
    {
        vector <int> maxInliers_R;
        Mat best_tn;
        vector<double> sizes;

		//----------------------------------------------------------------------
		// Find best R and t from candidate R and ts. 
		//----------------------------------------------------------------------
		
		// Do multiple trials...
        for (int trial = 0; trial < 10; ++trial)
        {
            maxInliers_R.clear();
            sizes.clear();
			
			// For each R
            for (int iter = 0; iter < Rs.size(); ++iter)
            {
                Mat Ri = Rs[iter];
				Mat ti = ts[iter];
                
				bool isRgood = true;
                vector<vector<int>> vpPairIdx = matchVanishPts_withR(prev, nview, Ri, isRgood);
           
		   		// Fix rotation
				Mat Rn = Ri * prev.R;

				//----------------------------------------------------------------------
                // Find best t based on 3D points
				//----------------------------------------------------------------------
                
				int maxIter = 100, maxmaxIter = 1000;
                vector<int> maxInliers;
                Mat good_tn;
                double best_s = -1;

				// Generate and test different ts
                for (int it = 0, itt = 0; it < maxIter && itt < maxmaxIter; ++it, ++itt)
                {
                    if (pt3d.size() < 2) 
						break;

					// Randomly pick two points
                    int i = xrand() % pt3d.size();
                    int j = xrand() % pt3d.size();
                    if (i == j) {
                        --it;
                        continue;
                    }

					// Set 3D-2D point correspondence
                    vector<Point3d> samplePt3d;
                    samplePt3d.push_back(pt3d[i]);
                    samplePt3d.push_back(pt3d[j]);
                    vector<Point2d> samplePt2d;
                    samplePt2d.push_back(pt2d[i]);
                    samplePt2d.push_back(pt2d[j]);

					// Given R, compute t with PnP
                    Mat tn = pnp_withR(samplePt3d, samplePt2d, K, Rn);
                    Mat t_rel = tn - Ri * prev.t;
                    t_rel = t_rel / cv::norm(t_rel);

					//  motion smooth
                    if (cv::norm(t_prev) > 0.1 && abs(t_rel.dot(t_prev) < cos(40 * PI / 180)))
                        continue;

					// Find inliers that conform to R and t
                    vector<int> inliers;
                    for (int j = 0; j < pt3d.size(); ++j) 
					{
                        // Project 3D to n-th view
                        Mat pt = K * (Rn * cvpt2mat(pt3d[j], 0) + tn);

						// If reprojection error is small enough
                        if (cv::norm(pt2d[j] - mat2cvpt(pt)) < reprjThresh)
                            inliers.push_back(j); // add as inlier
                    }

					// Is inlier set larger than max inlier set?
                    if (maxInliers.size() < inliers.size()) {
                        maxInliers = inliers;
                        good_tn = tn;
                    }
                }

                cout << ti.t() << ",, " << maxInliers.size() << endl;
                
				sizes.push_back(maxInliers.size());

				// update maxInliers_R
                if (maxInliers.size() > maxInliers_R.size())
                {
                    maxInliers_R = maxInliers;
                    best_tn = good_tn;
                    R = Ri;
                    t = ti;
                    F = Fs[iter];
                }
            }

            sort(sizes.begin(), sizes.end());

			// If inlier sets are similar in size, lower the reprojection threshold and repeat 
            if (sizes.size() > 1 && sizes[sizes.size() - 2] >= sizes[sizes.size() - 1] * 0.95)
                reprjThresh = reprjThresh * 0.9;
            else
                break;
        }

#ifdef PLOT_MID_RESULTS
        cv::Scalar color(rand() % 255, rand() % 255, rand() % 255, 0);
#endif

	   	// Is max inlier set large enough?
		if (maxInliers_R.size() >= 5)
        {
        	// Re-compute scale on max inlier set
           
		  	vector<Point3d> inlierPt3d;
            vector<Point2d> inlierPt2d;
            for (int i = 0; i < maxInliers_R.size(); ++i) {
                inlierPt3d.push_back(pt3d[maxInliers_R[i]]);
                inlierPt2d.push_back(pt2d[maxInliers_R[i]]);
            }

            Mat nvR = R * prev.R; // only trust the relative motion R 
            Mat nvt = pnp_withR(inlierPt3d, inlierPt2d, K, nvR); // compute position given orientation

            cout << nview.frameId - prev.frameId << " frames," << "Scale (2pt_alg)="
                 << cv::norm(-nvR.t()*nvt + prev.R.t()*prev.t) << ", "
                 << maxInliers_R.size() << "/" << pt3d.size() << endl;

            double base = cv::norm(nvR.t() * nvt - prev.R.t() * prev.t);
            
			double dt = (nview.frameId - prev.frameId) / fps;

			// If we are able to compute acceleration...
            if (prev.speed > 0 && !need_scale_to_real)
            {
				// Compute acceleration
                double a0 = abs((base / dt - prev.speed) / dt);

				// If acceleration not out of bound...
                if (a0 < accel_max)
                {					
                    nview.R = nvR.clone();
                    nview.t = nvt.clone();
                    nview.R_loc = R.clone();
                    nview.t_loc = nvt - R * prev.t; // relative t with scale
                    nview.t_loc = nview.t_loc / cv::norm(nview.t_loc);
                }
                else
				{
					// Just use constant velocity assumption
                    use_const_vel_model = true;
				}

				// If R and t inconsistent with PnP output
                if (pt3d.size() > 10 && (
					abs(base - cv::norm(t_pnp)) > 0.7 * base || 
					abs(base - cv::norm(t_pnp)) > 0.7 * cv::norm(t_pnp)))
                {
                    cout << "Inconsistent pnp and 2-pt: " << cv::norm(t_pnp) << ", " << base << endl;

					// Compute acceleration
                    double a1 = abs((cv::norm(t_pnp) / dt - prev.speed) / dt);

                    if (a0 >= accel_max && a1 >= accel_max)
                    {
						// Just use constant velocity assumption
                        use_const_vel_model = true;
                    }
                    else if (a1 < a0)
                    {
						// Just use PnP output
                        cout << "checking accelaraion " << a0 << " > " << a1 << ", use pnp instead" << endl;
						nview.R = Rn_pnp;
                        nview.t = tn_pnp;
                        R = R_pnp;
                        t = t_pnp / cv::norm(t_pnp);
                        nview.R_loc = R;
                        nview.t_loc = t;
                        
						use_const_vel_model = false;
                    }
                }
            }
            else
            {
                nview.R = nvR.clone();
                nview.t = nvt.clone();
                nview.R_loc = R.clone();
                nview.t_loc = nvt - R * prev.t; // relative t with scale
                nview.t_loc = nview.t_loc / cv::norm(nview.t_loc);

				// If R and t inconsistent with PnP output
                if (pt3d.size() > 10 && (
					abs(base - cv::norm(t_pnp)) > 0.7 * base || 
					abs(base - cv::norm(t_pnp)) > 0.7 * cv::norm(t_pnp)))
                {
                    cout << "Inconsistent pnp and 2-pt: " << cv::norm(t_pnp) << ", " << base << endl;

					// prefer small acc
                    if (abs(cv::norm(t_pnp) / dt - prev.speed) < abs(base / dt - prev.speed) && prev.speed > 0)
                    {
						// Just use PnP output
                        cout << "checking accelaraion, use pnp instead" << endl;
                        nview.R = Rn_pnp;
                        nview.t = tn_pnp;
                        R = R_pnp;
                        t = t_pnp / cv::norm(t_pnp);
                        nview.R_loc = R;
                        nview.t_loc = t;
                    }
                }
            }
        }
        else // max inlier set not large enough
        {
            if (pt3d.size() > 7) // enough 3D points 
            {
				// Just use PnP output
                cout << "fallback to pnp : " << cv::norm(t_pnp) << endl;
                nview.R = Rn_pnp;
                nview.t = tn_pnp;
                R = R_pnp;
                t = t_pnp / cv::norm(t_pnp);
                nview.R_loc = R;
                nview.t_loc = t;
            }
            else
			{
            	// Just use constant velocity assumption
                use_const_vel_model = true;
			}
        }
    }

	// Notify if we will use constant velocity assumption
    if (use_const_vel_model)
        cout << "fallback to const-vel model ...\n";

	//----------------------------------------------------------------------
	// Ground plane detection for scale estimation	
	//----------------------------------------------------------------------
	
	// If we are assuming fixed camera height from ground...
    if (mfgSettings->getDetectGround())
    {
        int n_gp_pts;
        Point3f gp_norm_vec;
        double gp_depth;
        double gp_quality;
        double scale_to_real = -1;
        Mat gp_roi;
        double gp_qual_thres = 0.55;
        double baseline = -1;

		// Compute baseline distance
        if (!need_scale_to_real && !use_const_vel_model)
            baseline = cv::norm(nview.t - R * prev.t);

		// Detect ground plane
        bool gp_detect_valid = false;
        gp_detect_valid = detectGroundPlane(prev.grayImg, nview.grayImg, R, t, K, 
			n_gp_pts, gp_depth, gp_norm_vec, gp_quality, gp_roi, baseline);

		// If we have good ground detection...
        if (gp_detect_valid && gp_quality >= gp_qual_thres - 0.02)
        {
            if (!use_const_vel_model) // translation scale obtained from epipolar geometry or PnP
            {
                scale_to_real = camera_height / abs((cv::norm(nview.t - R * prev.t) * gp_depth));

                if (need_scale_to_real)
                {
                    if (gp_quality > gp_qual_thres)
                        scale_vals.push_back(scale_to_real);
					
					// if having enough evidence (prompted to scale at least 3 times) or haven't done scaling for long time
                    if (scale_vals.size() >= 3 || nview.id - scale_since.back() > 50)
                    {
                        if (scale_vals.size() == 0)
                            scale_vals.push_back(scale_to_real);

                        cout << "Scale local map by " << vecSum(scale_vals) / scale_vals.size() << endl;
                        
						vector<double> scale_constraint(4);
                        scale_constraint[0] = prev.id;
                        scale_constraint[1] = nview.id;
                        scale_constraint[2] = cv::norm(nview.t - R * prev.t) * vecSum(scale_vals) / scale_vals.size();
                        scale_constraint[3] = 1e6 * gp_quality * gp_quality * gp_quality;
                        camdist_constraints.push_back(scale_constraint);

                        scaleLocalMap(scale_since.back(), nview.id, vecSum(scale_vals) / scale_vals.size(), false);
                        bundle_adjust_between(scale_since.back(), nview.id, scale_since.back() + 1); // when window large, not convergent
                        
						need_scale_to_real = false;
                        
						scale_since.push_back(nview.id);
                        scale_vals.clear();
                    }
                }
                else // already in real scale, check consistency
                {
					// Compute camera height
                    double est_ch = cv::norm(nview.t - R * prev.t) * abs(gp_depth);
                    
					cout << "estimated camera height " << est_ch << " m, qual = " << gp_quality << endl;

                    if ((abs(est_ch - camera_height) > 0.02 && gp_quality > gp_qual_thres) ||
                        (abs(est_ch - camera_height) > 0.1 && gp_quality > gp_qual_thres) ||
                      //(abs(est_ch - camera_height) > 0.2 && gp_quality > gp_qual_thres - 0.05) ||
                        (abs(scale_since.back() - nview.id) > 20 && 
						 abs(est_ch - camera_height) > 0.1 &&
                         abs(est_ch - camera_height) < 0.3 && 
						 gp_quality > gp_qual_thres - 0.02)) // inconsistence
                    {
                        cout << "Scale inconsistence, do scaling " << scale_to_real << " from " << scale_since.back() << endl;

						
                        if (nview.id - scale_since.back() == 1 && scale_since.size() > 2)
                            scaleLocalMap(scale_since[scale_since.size() - 2], nview.id, scale_to_real, false);

                        if (nview.id - scale_since.back() > 1 && nview.frameId - views[scale_since.back()].frameId < 100)
                            scaleLocalMap(scale_since.back(), nview.id, scale_to_real, false);
                        else if (nview.frameId - views[scale_since.back()].frameId > 100)
                            scaleLocalMap(max(0, nview.id - 50), nview.id, scale_to_real, false);

                        vector<double> scale_constraint(4);
                        scale_constraint[0] = prev.id;
                        scale_constraint[1] = nview.id;
                        scale_constraint[2] = cv::norm(nview.t - R * prev.t);
                        scale_constraint[3] = 1e6 * gp_quality * gp_quality * gp_quality;
                        camdist_constraints.push_back(scale_constraint);

                        if (nview.id - scale_since.back() == 1 && scale_since.size() > 2)
                            bundle_adjust_between(scale_since[scale_since.size() - 2], nview.id, scale_since[scale_since.size() - 2] + 1);

                        if (nview.id - scale_since.back() > 1 && nview.frameId - views[scale_since.back()].frameId < 100)
                            bundle_adjust_between(scale_since.back() - 1, nview.id, scale_since.back() + 1);
                        else if (nview.frameId - views[scale_since.back()].frameId > 60)
                            bundle_adjust_between(max(0, nview.id - 20), nview.id, max(0, nview.id - 20) + 1);

                        cv::putText(gp_roi, "scale since " + num2str(scale_since.back()) + " by " + num2str(scale_to_real), 
							Point2f(10, 200), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(200, 0, 0));
                        
						scale_since.push_back(nview.id);
                    }
                    else if (gp_quality >= gp_qual_thres) // good so far
                    {
						// add one more constraint to BA
                        vector<double> scale_constraint(4);
                        scale_constraint[0] = prev.id;
                        scale_constraint[1] = nview.id;
                        scale_constraint[2] = cv::norm(nview.t - R * prev.t);
                        scale_constraint[3] = 1e6 * gp_quality * gp_quality * gp_quality;
                        camdist_constraints.push_back(scale_constraint);
                        scale_since.push_back(nview.id);
                    }
                }

                cv::imwrite("./tmpimg/gp/gp_" + num2str(nview.id) + ".jpg", gp_roi);
            }
            else // use_const_vel_model, t-scale not available from epipolar/pnp
            {
                scale_to_real = camera_height / abs(gp_depth);

                if (!need_scale_to_real)  // already in real scale, and R nad t computed
                {
                    R = Rs[0];
                    t = ts[0];
                    nview.R = R * prev.R;
                    nview.t = R * prev.t + scale_to_real * t;
                    nview.R_loc = R;
                    nview.t_loc = t;
                    use_const_vel_model = false;     // avoid using constent velocity  via gp info
                    cout << "t-scale unavailable ... GP recovers real scale by scaling " << scale_to_real << endl;
                }
            }
        }
        else // no ground plane detected
        {
			// start a new local scale
            if (use_const_vel_model && !need_scale_to_real) 
            {
                need_scale_to_real = true;
                scale_since.push_back(nview.id);
                R = Rs[0];
                t = ts[0];
                nview.R = R * prev.R;
                nview.t = R * prev.t + t;
                nview.R_loc = R;
                nview.t_loc = t;
                use_const_vel_model = false;

                cout << "started a new local system since view " << nview.id << endl;
            }
        }
    }
 
	//----------------------------------------------------------------------
    // Use constant velocity to predict R t
	//----------------------------------------------------------------------
	
	Mat R_const, t_const;

    if (nview.id > 2)
    {
        Eigen::Quaterniond q_prev = r2q(prev.R_loc);
        double theta_prev = acos(q_prev.w()) * 2;
        double theta_curt = theta_prev * (nview.frameId - prev.frameId) / (prev.frameId - views[(prev.id - 1)].frameId);
        
		Eigen::Vector3d xyz(q_prev.x(), q_prev.y(), q_prev.z());
        xyz = xyz / xyz.norm();
        double q_curt[4] = {
			cos(theta_curt / 2),
            xyz(0) * sin(theta_curt / 2),
            xyz(1) * sin(theta_curt / 2),
            xyz(2) * sin(theta_curt / 2)
        };

        R_const = q2r(q_curt);
        t_const = (prev.t - (prev.R * views[views.size() - 3].R.t()) * views[views.size() - 3].t)
                  * (nview.frameId - prev.frameId) / (prev.frameId - views[(prev.id - 1)].frameId);
    }

    if (use_const_vel_model)
    {
		// R_const is not available, no way to proceed
        if (R_const.cols < 3)
        {
            cout << "R_const not available. Terminated.\n";
            exit(0);
        }

        if (R.cols == 3) // R computed
        {
            nview.R = R * prev.R;

            if (t_const.dot(t) > 0)
                nview.t = R * prev.t + cv::norm(t_const) * t;
            else
                nview.t = R * prev.t - cv::norm(t_const) * t;
        }
        else
        {
            nview.R = R_const * prev.R;
            nview.t = R_const * prev.t + t_const;
            R = R_const;
            t = t_const / cv::norm(t_const);
        }

        nview.R_loc = R;
        nview.t_loc = Mat::zeros(3, 1, CV_64F);
    }

	//----------------------------------------------------------------------
    
	// Compute epipoles
    Point2d ep1 = mat2cvpt(K * R.t() * t);
    Point2d ep2 = mat2cvpt(K * t);
    prev.epipoleA = ep1;
    nview.epipoleB = ep2;
	double epNeibRds = 0.0 * IDEAL_IMAGE_WIDTH / 640;
#ifdef PLOT_MID_RESULTS
    cv::circle(canv1, ep1, epNeibRds, cv::Scalar(0, 0, 100), 1);
    cv::circle(canv2, ep2, epNeibRds, cv::Scalar(0, 0, 100), 1);
#endif

   	// Compute angular velocity
    Matrix3d Rx;
    Rx << 
		R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
    	R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
    	R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
    Quaterniond q(Rx);
    double delta_t = (nview.frameId - prev.frameId) / fps;
    angVel = 2 * acos(q.w()) * 180 / PI / delta_t;
    nview.angVel = angVel;

    nview.speed = -1;
    nview.accel = -1;

	//----------------------------------------------------------------------
   	

	if (!need_scale_to_real)
    {
        nview.speed = cv::norm(nview.R.t() * nview.t - prev.R.t() * prev.t) / delta_t;

		// Update previous speed (after potential scaling)
        if (prev.id > 0) {             
			int ppid = prev.id - 1;
            prev.speed = cv::norm(prev.R.t() * prev.t - views[ppid].R.t() * views[ppid].t) / (prev.frameId - views[ppid].frameId) * fps;
        }

        if (prev.speed > 0.1)
            nview.accel = abs(nview.speed - prev.speed) / delta_t;
    }

	//----------------------------------------------------------------------
    // Sort point matches by parallax
	//----------------------------------------------------------------------
    if (sortbyPrlx)
    {
        vector<valIdxPair> prlxVec;
        for (int i = 0; i < featPtMatches.size(); ++i) {
            valIdxPair pair;
            pair.first = compParallax(featPtMatches[i][0], featPtMatches[i][1], K, Mat::eye(3, 3, CV_64F), R);
            pair.second = i;
            prlxVec.push_back(pair);
        }

        sort(prlxVec.begin(), prlxVec.end(), comparator_valIdxPair);

        vector<vector<int>> copyPairIdx = pairIdx;
        vector<vector<Point2d>> copyFeatPtMatches = featPtMatches;
        
		pairIdx.clear();
        featPtMatches.clear();

        for (int i = prlxVec.size() - 1; i >= 0; --i) {
            pairIdx.push_back(copyPairIdx[prlxVec[i].second]);
            featPtMatches.push_back(copyFeatPtMatches[prlxVec[i].second]);
        }
    }

	//----------------------------------------------------------------------
    // Set 3D keypoints and point tracks
	//----------------------------------------------------------------------
	
	vector<int> view_point;

	// Loop through each feature point match
    for (int i = 0; i < featPtMatches.size(); ++i)
    {
        view_point.clear();
        int ptGid = prev.featurePoints[pairIdx[i][0]].gid;
        nview.featurePoints[pairIdx[i][1]].gid = ptGid;

		// If point is triangulated...
        if (ptGid >= 0 && keyPoints[ptGid].is3D) {
            view_point.push_back(nview.id);
            view_point.push_back(pairIdx[i][1]);
            keyPoints[ptGid].viewId_ptLid.push_back(view_point);
        }
    }

    int new3dPtNum = 0, new2_3dPtNum = 0;

	// Loop through each feature point match
    for (int i = 0; i < featPtMatches.size(); ++i)
    {
        view_point.clear();

        int ptGid = prev.featurePoints[pairIdx[i][0]].gid;
        nview.featurePoints[pairIdx[i][1]].gid = ptGid;

		// If point is triangulated...
        if (ptGid >= 0 && keyPoints[ptGid].is3D)
        {
        }
        else if (ptGid >= 0 && !keyPoints[ptGid].is3D && keyPoints[ptGid].gid >= 0) // point exist as 2D track
        {
        }
        else // newly found points
        {
			// Compute parallax degree
            double parallaxDeg = compParallaxDeg(featPtMatches[i][0], featPtMatches[i][1], K, Mat::eye(3, 3, CV_64F), R);

			// Triangulate point
            Mat Xw = triangulatePoint_nonlin(prev.R, prev.t, nview.R, nview.t, K, featPtMatches[i][0], featPtMatches[i][1]);
            Mat Xc = triangulatePoint(Mat::eye(3, 3, CV_64F), Mat::zeros(3, 1, CV_64F), R, t, K, featPtMatches[i][0], featPtMatches[i][1]);

#ifdef PLOT_MID_RESULTS
            cv::Scalar color(rand() % 255, rand() % 255, rand() % 255, 0);
            if (parallaxDeg > parallaxDegThresh) {
                cv::circle(canv1, featPtMatches[i][0], 2, color, 2);
                cv::circle(canv2, featPtMatches[i][1], 2, color, 2);
            }
#endif

        	// If parallax is not enough...
            if (parallaxDeg < parallaxDegThresh || (
		      //cv::norm(Xw+nview.R.t()*nview.t) > mfgSettings->getDepthLimit() &&
                cv::norm(Xc) > mfgSettings->getDepthLimit() * 1) || // depth too large
                new3dPtNum > maxNumNew3dPt || 
				cv::norm(featPtMatches[i][1] - ep2) < epNeibRds)
            {
            	// Set 3D-2D correspondence
                KeyPoint3d kp;
                kp.gid = keyPoints.size();
                prev.featurePoints[pairIdx[i][0]].gid = kp.gid;
                nview.featurePoints[pairIdx[i][1]].gid = kp.gid;
				// Store index correspondence
                view_point.clear();
                view_point.push_back(prev.id);
                view_point.push_back(pairIdx[i][0]);
                kp.viewId_ptLid.push_back(view_point);
                view_point.clear();
                view_point.push_back(nview.id);
                view_point.push_back(pairIdx[i][1]);
                kp.viewId_ptLid.push_back(view_point);
				// Do not triangulate yet
                kp.is3D = false;
				// Add point
                keyPoints.push_back(kp);
            }
            else // Add point to MFG as 3D point
            {
				// Is point in front of both cameras?
                if (!checkCheirality(prev.R, prev.t, Xw) || !checkCheirality(nview.R, nview.t, Xw))
                    continue;

            	// Set 3D-2D correspondence
                KeyPoint3d kp(Xw.at<double>(0), Xw.at<double>(1), Xw.at<double>(2));
                kp.gid = keyPoints.size();
                kp.is3D = true;
                kp.estViewId = nview.id;
                prev.featurePoints[pairIdx[i][0]].gid = kp.gid;
                nview.featurePoints[pairIdx[i][1]].gid = kp.gid;
           		// Store index correspondence 
                view_point.clear();
                view_point.push_back(prev.id);
                view_point.push_back(pairIdx[i][0]);
                kp.viewId_ptLid.push_back(view_point);
                view_point.clear();
                view_point.push_back(nview.id);
                view_point.push_back(pairIdx[i][1]);
                kp.viewId_ptLid.push_back(view_point);
				// Add point
                keyPoints.push_back(kp);

#ifdef PLOT_MID_RESULTS
                cv::circle(canv1, featPtMatches[i][0], 7, color, 1);
                cv::circle(canv2, featPtMatches[i][1], 7, color, 1);
                cv::putText(canv1, " " + num2str(cv::norm(Xw + nview.R.t()*nview.t)), featPtMatches[i][0], 1, 1, color);
#endif
                new3dPtNum++;
            }
        }
    }

	//----------------------------------------------------------------------
    // Conver image point track to 3D points
	//----------------------------------------------------------------------
    
	#pragma omp parallel for

	// Loop through feature matches
    for (int i = 0; i < featPtMatches.size(); ++i)
    {
        vector<int> view_point;
        int ptGid = prev.featurePoints[pairIdx[i][0]].gid;
        nview.featurePoints[pairIdx[i][1]].gid = ptGid;

		// If point is triangulated...
        if (ptGid >= 0 && keyPoints[ptGid].is3D)
        {
        }
        else if (ptGid >= 0 && !keyPoints[ptGid].is3D && keyPoints[ptGid].gid >= 0) // point exist as 2D track
        {
            cv::Scalar color(rand() % 255, rand() % 255, rand() % 255, 0);
           
		   	// Compute parallax
			double parallaxDeg = compParallaxDeg(featPtMatches[i][0], featPtMatches[i][1], K, Mat::eye(3, 3, CV_64F), R);

#ifdef PLOT_MID_RESULTS
            if (parallaxDeg > parallaxDegThresh) {
                cv::circle(canv1, featPtMatches[i][0], 2, color, 2);
                cv::circle(canv2, featPtMatches[i][1], 2, color, 2);
            }
#endif
            // 1) Add current observation into map
            view_point.push_back(nview.id);
            view_point.push_back(pairIdx[i][1]);
            keyPoints[ptGid].viewId_ptLid.push_back(view_point);

            // if(new2_3dPtNum >= maxNumNew2d_3dPt) continue;  //not thread safe

            // 2) Check if a 2D track is ready to establish 3D point
            
			// start a voting/ransac --
            Mat bestKeyPt;
            vector<int> maxInlier;
            int obs_size = keyPoints[ptGid].viewId_ptLid.size();

            for (int p = 0; p < obs_size - 1 && p < 2; ++p)
            {
                int pVid = keyPoints[ptGid].viewId_ptLid[p][0],
                    pLid = keyPoints[ptGid].viewId_ptLid[p][1];

                if (!views[pVid].matchable) 
					break;

                if (mfgSettings->getDetectGround() && need_scale_to_real && scale_since.back() != 1 && pVid < scale_since.back())
                    continue;

                for (int q = obs_size - 1; q >= p + 1 && q > obs_size - 3; --q)
                {
                    // try to triangulate every pair of 2d pt
                    int qVid = keyPoints[ptGid].viewId_ptLid[q][0],
                        qLid = keyPoints[ptGid].viewId_ptLid[q][1];

                    if (!views[qVid].matchable) 
						break;

                    Point2d ppt = views[pVid].featurePoints[pLid].cvpt(),
                            qpt = views[qVid].featurePoints[qLid].cvpt();
                    
					double pqPrlxDeg = compParallaxDeg(ppt, qpt, K,	views[pVid].R, views[qVid].R);

                    if (pqPrlxDeg < parallaxDegThresh * 1) 	
						break; // not usable

                    //   Mat ptpq = triangulatePoint_nonlin (views[pVid].R, views[pVid].t,views[qVid].R, views[qVid].t, K, ppt, qpt);
                    Mat ptpq = triangulatePoint(views[pVid].R, views[pVid].t, views[qVid].R, views[qVid].t, K, ppt, qpt);

                    // Check Cheirality
                    if (!checkCheirality(views[pVid].R, views[pVid].t, ptpq))
                        continue;

                    // Check if every 2D observation agrees with this 3d point
                    vector<int> inlier;
                    for (int j = 0; j < keyPoints[ptGid].viewId_ptLid.size(); ++j)
                    {
                        int vid = keyPoints[ptGid].viewId_ptLid[j][0];
                        int lid = keyPoints[ptGid].viewId_ptLid[j][1];
                        Point2d pt_j = mat2cvpt(K * (views[vid].R * ptpq + views[vid].t));
                        
						double dist = cv::norm(views[vid].featurePoints[lid].cvpt() - pt_j);
                        if (dist < 2) // inlier;
                            inlier.push_back(j);
                    }

                    if (maxInlier.size() < inlier.size())
                    {
                        maxInlier = inlier;
                        bestKeyPt = ptpq;
                    }
                }
            }

            // Due to outlier(s), not ready to establish 3d , append to 2d track
            if (maxInlier.size() < 3)
                continue;

            // Non-linear estimation of 3D points
            vector<Mat> Rs, ts;
            vector<Point2d> pt;
            for (int a = 0; a < maxInlier.size(); ++a) {
                int vid = keyPoints[ptGid].viewId_ptLid[maxInlier[a]][0];
                int lid = keyPoints[ptGid].viewId_ptLid[maxInlier[a]][1];
                if (!views[vid].matchable) 
					continue;
                Rs.push_back(views[vid].R);
                ts.push_back(views[vid].t);
                pt.push_back(views[vid].featurePoints[lid].cvpt());
            }

            est3dpt_g2o(Rs, ts, K, pt, bestKeyPt);  // nonlinear estimation

            // 3) Establish 3D point
            keyPoints[ptGid].is3D = true;
            keyPoints[ptGid].estViewId = nview.id;
            keyPoints[ptGid].x = bestKeyPt.at<double>(0);
            keyPoints[ptGid].y = bestKeyPt.at<double>(1);
            keyPoints[ptGid].z = bestKeyPt.at<double>(2);
#ifdef PLOT_MID_RESULTS
            cv::circle(canv1, featPtMatches[i][0], 7, color, 2);
            cv::circle(canv2, featPtMatches[i][1], 7, color, 2);
            cv::putText(canv1, " " + num2str(cv::norm(bestKeyPt + nview.R.t()*nview.t)), featPtMatches[i][0], 1, 1, color);
#endif
            new2_3dPtNum++;
        }
    }

    // Remove outliers
    if (!use_const_vel_model)
    {
        for (int i = 0; i < maxInliers_Rt.size(); ++i)
        {
            int j = maxInliers_Rt[i];
            
			int gid = prev.featurePoints[pairIdx[j][0]].gid;
            if (gid < 0) 
				continue;

			// Compute scale
            double s = cv::norm(keyPoints[gid].mat(0) + (prev.R.t() * prev.t)) / cv::norm(localKeyPts[j].mat());
		
            if (abs(s - cv::norm(-nview.R.t()*nview.t + prev.R.t()*prev.t)) > 0.15)
            {
                // delete outliers
                keyPoints[gid].gid = -1;
                keyPoints[gid].is3D = false;

                for (int k = 0; k < keyPoints[gid].viewId_ptLid.size(); ++k) {
                    int vid = keyPoints[gid].viewId_ptLid[k][0];
                    int lid = keyPoints[gid].viewId_ptLid[k][1];
                    if (views[vid].matchable)
                        views[vid].featurePoints[lid].gid = -1;
                }

                keyPoints[gid].viewId_ptLid.clear();
            }
        }
    }

	//----------------------------------------------------------------------
	// Set 3D vanishing points
	//----------------------------------------------------------------------
    
    bool isRgood;
    vector<vector<int>> vpPairIdx = matchVanishPts_withR(prev, nview, nview.R * prev.R.t(), isRgood);
    vector<int> vid_vpid;

   	// Loop through each vanishing point matches 
    for (int i = 0; i < vpPairIdx.size(); ++i)
    {
        vid_vpid.clear();

        if (prev.vanishPoints[vpPairIdx[i][0]].gid >= 0) // pass correspondence on
        {
            nview.vanishPoints[vpPairIdx[i][1]].gid = prev.vanishPoints[vpPairIdx[i][0]].gid;
            vid_vpid.push_back(nview.id);
            vid_vpid.push_back(vpPairIdx[i][1]);
            vanishingPoints[prev.vanishPoints[vpPairIdx[i][0]].gid].viewId_vpLid.push_back(vid_vpid);
        }
        else // establish new vanishing point in 3d
        {
            Mat vp = prev.R.t() * K.inv() * prev.vanishPoints[vpPairIdx[i][0]].mat();
            vp = vp / cv::norm(vp);
    
        	// Set 3D-2D correspondence
			vanishingPoints.push_back(VanishPnt3d(vp.at<double>(0), vp.at<double>(1), vp.at<double>(2)));
            vanishingPoints.back().gid = vanishingPoints.size() - 1;
            prev.vanishPoints[vpPairIdx[i][0]].gid = vanishingPoints.back().gid;
            nview.vanishPoints[vpPairIdx[i][1]].gid = vanishingPoints.back().gid;
        	// Store index correspondence 
			vid_vpid.clear();
            vid_vpid.push_back(prev.id);
            vid_vpid.push_back(vpPairIdx[i][0]);
            vanishingPoints.back().viewId_vpLid.push_back(vid_vpid);
            vid_vpid.clear();
            vid_vpid.push_back(nview.id);
            vid_vpid.push_back(vpPairIdx[i][1]);
            vanishingPoints.back().viewId_vpLid.push_back(vid_vpid);
        }
    }

    // Check if VP observed before previous frame
    for (int i = 0; i < nview.vanishPoints.size(); ++i)
    {
        if (nview.vanishPoints[i].gid >= 0) 
			continue;

        Mat vpi = nview.R.t() * K.inv() * nview.vanishPoints[i].mat(); // in world coord
        vpi = vpi * (1 / cv::norm(vpi));

        // try matching with existing 3d vp
        for (int j = 0; j < vanishingPoints.size(); ++j) // existing vp 3D
        {
            Mat vpj = vanishingPoints[j].mat(0) / cv::norm(vanishingPoints[j].mat(0));

            if (abs(vpi.dot(vpj)) > cos(mfgSettings->getVPointAngleThresh() *PI / 180)) // degree
            {
                // matched
                nview.vanishPoints[i].gid = vanishingPoints[j].gid;
                vector<int> vid_vpid;
                vid_vpid.push_back(nview.id);
                vid_vpid.push_back(nview.vanishPoints[i].lid);
                vanishingPoints[nview.vanishPoints[i].gid].viewId_vpLid.push_back(vid_vpid);
                break;
            }
        }
    }

    if (F.empty())
        F = K.t().inv() * vec2SkewMat(t) * R * K.inv();

	// Update ideal lines
    vector<vector<int>>	ilinePairIdx;
    matchIdealLines(prev, nview, vpPairIdx, featPtMatches, F, ilinePairIdx, 1);
	update3dIdealLine(ilinePairIdx, nview);

	// Update primary planes (when ground detection is off)
    if (mfgSettings->getDetectGround() == 0 || (mfgSettings->getDetectGround() && !need_scale_to_real))
        updatePrimPlane();

#ifdef PLOT_MID_RESULTS
    // Write to images for debugging
    cv::imwrite("./tmpimg/" + num2str(views.back().id) + "_pt1.jpg", canv1);
    cv::imwrite("./tmpimg/" + num2str(views.back().id) + "_pt2.jpg", canv2);
    views.back().drawAllLineSegments(true);
#endif
}

// Establish a 3D line only when it had been seen in 3 or more views
void Mfg::update3dIdealLine(vector<vector<int>> ilinePairIdx, View &nview)
{
    View &prev = views[views.size() - 2];

    double ilineLenLimit	= 10;
    double ilineDepthLimit	= 10;
    double threshPt2LnDist  = 2;  // endpt to reprojected line dist
    double epNeibRds = 60.0 * IDEAL_IMAGE_WIDTH / 640;
    double parallaxThresh = THRESH_PARALLAX;
    double parallaxAngleThresh = THRESH_PARALLAX_DEGREE;

    if (rotateMode())
        parallaxThresh = parallaxThresh * 1.5;

#ifdef PLOT_MID_RESULTS
    Mat canv1 = prev.img.clone(), canv2 = nview.img.clone();
#endif

    vector<int> vid_lid;

    for (int i = 0; i < ilinePairIdx.size(); ++i)
    {
        int lnGid = prev.idealLines[ilinePairIdx[i][0]].gid;  // line gid from prev view
        nview.idealLines[ilinePairIdx[i][1]].gid = lnGid;     // pass to current view
        nview.idealLines[ilinePairIdx[i][1]].pGid = prev.idealLines[ilinePairIdx[i][0]].pGid;

        cv::Scalar color(rand() % 255, rand() % 255, rand() % 255, 0);
        int linewidth = 0;

        if (lnGid < 0)	  // not existent in map, setup a new 2d track, not a 3d line yet
        {
            IdealLine3d line;
            line.is3D = false;
            line.gid = idealLines.size();
            prev.idealLines[ilinePairIdx[i][0]].gid = line.gid;		// assign gid for a new line
            nview.idealLines[ilinePairIdx[i][1]].gid = line.gid;
            line.vpGid = prev.vanishPoints[prev.idealLines[ilinePairIdx[i][0]].vpLid].gid;
            //	line.pGid = prev.idealLines[ilinePairIdx[i][0]].pGid;
            //	nview.idealLines[ilinePairIdx[i][1]].pGid = line.pGid;
            vid_lid.clear();
            vid_lid.push_back(prev.id);
            vid_lid.push_back(ilinePairIdx[i][0]);
            line.viewId_lnLid.push_back(vid_lid);
            vid_lid.clear();
            vid_lid.push_back(nview.id);
            vid_lid.push_back(ilinePairIdx[i][1]);
            line.viewId_lnLid.push_back(vid_lid);
            idealLines.push_back(line);
        }
        else // existent 3D line or 2D line track
        {
            // 1) add current observation into map
            vid_lid.clear();
            vid_lid.push_back(nview.id);
            vid_lid.push_back(ilinePairIdx[i][1]);
            idealLines[lnGid].viewId_lnLid.push_back(vid_lid);

            if (idealLines[lnGid].is3D)
                linewidth = 1; // for debugging plotting purpose

            // Avoid those close to epipole
            if (point2LineDist(prev.idealLines[ilinePairIdx[i][0]].lineEq(), prev.epipoleA) < epNeibRds || 
				point2LineDist(nview.idealLines[ilinePairIdx[i][1]].lineEq(), nview.epipoleB) < epNeibRds)
                continue;

            // 2) check if a 2D track is ready to establish 3D line
            if (!idealLines[lnGid].is3D)
            {
                // Start voting/ransac
                IdealLine3d bestLine;
                int maxNumInlier = 0;

                for (int p = 0; p < idealLines[lnGid].viewId_lnLid.size() - 1; ++p)
                {
                    int pVid = idealLines[lnGid].viewId_lnLid[p][0],
                        pLid = idealLines[lnGid].viewId_lnLid[p][1];

                    if (!views[pVid].matchable) 
						break;

                    if (mfgSettings->getDetectGround() && 
						need_scale_to_real && 
						scale_since.back() != 1 && 
						pVid < scale_since.back())
                        continue;

                    for (int q = p + 1; q < idealLines[lnGid].viewId_lnLid.size(); ++q)
                    {
                        // try to triangulate every pair of 2d line
                        int   qVid = idealLines[lnGid].viewId_lnLid[q][0],
                              qLid = idealLines[lnGid].viewId_lnLid[q][1];

                        if (!views[qVid].matchable) 
							break;

                        IdealLine2d pln = views[pVid].idealLines[pLid],
                                    qln = views[qVid].idealLines[qLid];
                        double pqPrlx = compParallax(pln, qln, K, views[pVid].R, views[qVid].R);

                        if (pqPrlx < parallaxThresh * 1.5) 	
							continue; // not usable

                        // Triangulate 3D line
                        IdealLine3d lnpq = triangluateLine(
							views[pVid].R, views[pVid].t,
                            views[qVid].R, views[qVid].t, K, pln, qln);

                        // check Cheirality
                        if (!checkCheirality(views[pVid].R, views[pVid].t, lnpq) ||
                            !checkCheirality(views[qVid].R, views[qVid].t, lnpq))
                            continue;

                        // Ignore lines that are too long (possible false matching)
                        if (lnpq.length > ilineLenLimit)
                            continue;

                        int numInlier = 0;

                        // Check if every 2D observation agrees with this 3D line
                        for (int j = 0; j < idealLines[lnGid].viewId_lnLid.size(); ++j)
                        {
                            int vid = idealLines[lnGid].viewId_lnLid[j][0];
                            int lid = idealLines[lnGid].viewId_lnLid[j][1];
                            Mat ln2d = projectLine(lnpq, views[vid].R, views[vid].t, K);
                            double sumDist = 0;

                            for (int k = 0; k < views[vid].idealLines[lid].lsEndpoints.size(); ++k)
                                sumDist += point2LineDist(ln2d, views[vid].idealLines[lid].lsEndpoints[k]);

                            if (sumDist / views[vid].idealLines[lid].lsEndpoints.size() < threshPt2LnDist) // inlier;
                                numInlier++;
                        }

                        if (maxNumInlier < numInlier)
                        {
                            maxNumInlier = numInlier;
                            bestLine = lnpq;
                        }
                    }
                }

                if (maxNumInlier < 3) // due to outlier(s), not ready to establish 3D line, append to 2D track
                    continue;

                Point3d curCamPos = mat2cvpt3d(-views.back().R.t() * views.back().t);

                if (cv::norm(projectPt3d2Ln3d(bestLine, curCamPos) - curCamPos) > ilineDepthLimit)
                    continue;

                // 3) Establish a new 3D line in MFG
                idealLines[lnGid].midpt = bestLine.midpt;
                idealLines[lnGid].direct = bestLine.direct.clone();
                idealLines[lnGid].length = bestLine.length;
                idealLines[lnGid].is3D = true;
                idealLines[lnGid].pGid = prev.idealLines[ilinePairIdx[i][0]].pGid;
                idealLines[lnGid].estViewId = nview.id;
                linewidth = 2; // 2D to 3D conversion
            }
        }

#ifdef PLOT_MID_RESULTS
        if (linewidth > 0) {
            cv::line(canv1, prev.idealLines[ilinePairIdx[i][0]].extremity1, prev.idealLines[ilinePairIdx[i][0]].extremity2, color, linewidth);
            cv::line(canv2, nview.idealLines[ilinePairIdx[i][1]].extremity1, nview.idealLines[ilinePairIdx[i][1]].extremity2, color, linewidth);
        }
#endif
    }

#ifdef PLOT_MID_RESULTS
    cv::imwrite("./tmpimg/ln_" + num2str(views.back().id) + "_1.jpg", canv1);
    cv::imwrite("./tmpimg/ln_" + num2str(views.back().id) + "_2.jpg", canv2);
#endif
}

// Primary plane update
//
// 1. assosciate new points or lines to exiting planes
// 2. use 3d key points and ideal lines to discover new planes
void Mfg::updatePrimPlane()
{
    // 0. set prameters/thresholds
    double pt2PlaneDistThresh = mfgSettings->getMfgPointToPlaneDistance();

    // 1. check if any newly added points/lines belong to existing planes

    // 1.1 check points
    for (int i = 0; i < keyPoints.size(); ++i)
    {
		// If point is not triangulated yet...
        if (!keyPoints[i].is3D || keyPoints[i].gid < 0) 
			continue;

        if (keyPoints[i].estViewId < views.back().id) 
			continue; // not newly added feature

        double minDist = 1e6; //initialized to be very large
        int minIdx = -1;

        for (int j = 0; j < primaryPlanes.size(); ++j)
        {
            if (views.back().id - primaryPlanes[j].recentViewId > 5) 
				continue; // obsolete, not use

            double dist = abs(keyPoints[i].mat(0).dot(primaryPlanes[j].n) + primaryPlanes[j].d) / cv::norm(primaryPlanes[j].n);
            if (dist < minDist) {
                minDist = dist;
                minIdx = j;
            }
        }

        if (minIdx >= 0 && minDist <= pt2PlaneDistThresh)  // pt-plane distance under threshold
        {
            keyPoints[i].pGid = primaryPlanes[minIdx].gid;
            primaryPlanes[minIdx].kptGids.push_back(keyPoints[i].gid); // connect pt and plane
            primaryPlanes[minIdx].recentViewId = views.back().id;
        }
    }

    // 1.2 check lines
    for (int i = 0; i < idealLines.size(); ++i)
    {
        if (!idealLines[i].is3D || idealLines[i].gid < 0) continue;

        if (idealLines[i].estViewId < views.back().id) continue;

        double	minDist = 1e6; //initialized to be very large
        int		minIdx = -1;

        for (int j = 0; j < primaryPlanes.size(); ++j)
        {
            if (views.back().id - primaryPlanes[j].recentViewId > 5) continue; // obsolete

            double dist = (abs(cvpt2mat(idealLines[i].extremity1(), 0).dot(primaryPlanes[j].n) + primaryPlanes[j].d)
                           + abs(cvpt2mat(idealLines[i].extremity2(), 0).dot(primaryPlanes[j].n) + primaryPlanes[j].d))
                          / (2 * cv::norm(primaryPlanes[j].n));

            if (dist < minDist)
            {
                minDist = dist;
                minIdx = j;
            }
        }

        if (minIdx >= 0 && minDist <= pt2PlaneDistThresh)  // line-plane distance under threshold
        {
            idealLines[i].pGid = primaryPlanes[minIdx].gid;
            primaryPlanes[minIdx].ilnGids.push_back(idealLines[i].gid); // connect line and plane
            primaryPlanes[minIdx].recentViewId = views.back().id;
        }
    }

    // ========== 2. discover new planes (using seq-ransac) =========
    vector<KeyPoint3d> lonePts; // used for finding new planes
    int ptNumLimit = mfgSettings->getMfgNumRecentPoints();

    for (int i = keyPoints.size() - 1; i >= 0; --i)
    {
        if (!keyPoints[i].is3D || keyPoints[i].gid < 0 || keyPoints[i].pGid >= 0) continue;

        lonePts.push_back(keyPoints[i]);

        if (lonePts.size() > ptNumLimit) break;
    }

    vector<IdealLine3d> loneLns;
    int lnNumLimit = mfgSettings->getMfgNumRecentLines();

    for (int i = idealLines.size() - 1; i >= 0; --i)
    {
        if (!idealLines[i].is3D || idealLines[i].gid < 0 || idealLines[i].pGid >= 0) continue;

        loneLns.push_back(idealLines[i]);

        if (loneLns.size() > lnNumLimit) break;
    }

    vector<vector<int>> ptIdxGroups, lnIdxGroups;
    vector<Mat> planeVecs;
    find3dPlanes_pts_lns_VPs(lonePts, loneLns, vanishingPoints, ptIdxGroups, lnIdxGroups, planeVecs);

    for (int i = 0; i < ptIdxGroups.size(); ++i)
    {
        int newPlaneGid = primaryPlanes.size();

        for (int j = 0; j < ptIdxGroups[i].size(); ++j)
            keyPoints[ptIdxGroups[i][j]].pGid = newPlaneGid;

        if (lnIdxGroups.size() > i)
        {
            for (int j = 0; j < lnIdxGroups[i].size(); ++j)
                idealLines[lnIdxGroups[i][j]].pGid = newPlaneGid;

            cout << "Add plane " << newPlaneGid << " with " << ptIdxGroups[i].size() << '\t' << lnIdxGroups[i].size() << endl;
        }

        primaryPlanes.push_back(PrimPlane3d(planeVecs[i], newPlaneGid)); // need compute plane equation
        primaryPlanes.back().kptGids = ptIdxGroups[i];
        primaryPlanes.back().ilnGids = lnIdxGroups[i];
        primaryPlanes.back().estViewId = views.back().id;
        primaryPlanes.back().recentViewId = views.back().id;
    }

}

void Mfg::draw3D() const
{
    if (mfg_writing)
        return;

    // plot first camera, small
    glLineWidth(1);
    glBegin(GL_LINES);
    glColor3f(1, 0, 0); // x-axis
    glVertex3f(0, 0, 0);
    glVertex3f(1, 0, 0);
    glColor3f(0, 1, 0);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 1, 0);
    glColor3f(0, 0, 1); // z axis
    glVertex3f(0, 0, 0);
    glVertex3f(0, 0, 1);
    glEnd();

    Mat xw = (Mat_<double>(3, 1) << 0.5, 0, 0),
            yw = (Mat_<double>(3, 1) << 0, 0.5, 0),
            zw = (Mat_<double>(3, 1) << 0, 0, 0.5);

    for (int i = 1; i < views.size(); ++i)
    {
        if (!(views[i].R.dims == 2)) continue; // handle the in-process view

        Mat c = -views[i].R.t() * views[i].t;
        Mat x_ = views[i].R.t() * (xw - views[i].t),
                y_ = views[i].R.t() * (yw - views[i].t),
                z_ = views[i].R.t() * (zw - views[i].t);
        glBegin(GL_LINES);

        glColor3f(1, 0, 0);
        glVertex3f(c.at<double>(0), c.at<double>(1), c.at<double>(2));
        glVertex3f(x_.at<double>(0), x_.at<double>(1), x_.at<double>(2));
        glColor3f(0, 1, 0);
        glVertex3f(c.at<double>(0), c.at<double>(1), c.at<double>(2));
        glVertex3f(y_.at<double>(0), y_.at<double>(1), y_.at<double>(2));
        glColor3f(0, 0, 1);
        glVertex3f(c.at<double>(0), c.at<double>(1), c.at<double>(2));
        glVertex3f(z_.at<double>(0), z_.at<double>(1), z_.at<double>(2));
        glEnd();
    }

    glPointSize(3.0);
    glBegin(GL_POINTS);

    for (int i = 0; i < keyPoints.size(); ++i)
    {
        if (!keyPoints[i].is3D || keyPoints[i].gid < 0) 
			continue;

        if (keyPoints[i].pGid < 0) // red
            glColor3f(0.6, 0.6, 0.6);
        else // coplanar green
            glColor3f(0.0, 1.0, 0.0);

        glVertex3f(keyPoints[i].x, keyPoints[i].y, keyPoints[i].z);
    }

    glEnd();

    glColor3f(0, 1, 1);
    glLineWidth(2);
    glBegin(GL_LINES);

    for (int i = 0; i < idealLines.size(); ++i)
    {
        if (!idealLines[i].is3D || idealLines[i].gid < 0) 
			continue;

        if (idealLines[i].pGid < 0)
            glColor3f(0, 0, 0);
        else
            glColor3f(0.0, 1.0, 0.0);

        glVertex3f(idealLines[i].extremity1().x, idealLines[i].extremity1().y, idealLines[i].extremity1().z);
        glVertex3f(idealLines[i].extremity2().x, idealLines[i].extremity2().y, idealLines[i].extremity2().z);
    }

    glEnd();
}

// Local bundle adjustment
void Mfg::adjustBundle()
{
    int numPos = 8, numFrm = 10;

    if (views.size() <= numFrm)
        numPos = numFrm;

    adjustBundle_G2O(numPos, numFrm);

	// Write to file
    exportCamPose(*this, "camPose.txt");
    exportMfgNode(*this, "mfgNode.txt");
}

// Detect lines that are too far or too long.
void Mfg::detectLnOutliers(double threshPt2LnDist)
{
    double ilineLenLimit = 10;

	// Loop through each ideal line
    for (int i = 0; i < idealLines.size(); ++i)
    {
		// Skip lines that are not triangulated yet
        if (!idealLines[i].is3D || idealLines[i].gid < 0) 
			continue;

		// If line too long...
        if (idealLines[i].length > ilineLenLimit)
        {
			// Remove from ideal line
            idealLines[i].gid = -1;
            idealLines[i].is3D = false;

            // Remove 2D feature's info
            for (int j = 0; j < idealLines[i].viewId_lnLid.size(); ++j)
            {
                int vid = idealLines[i].viewId_lnLid[j][0];
                int lid = idealLines[i].viewId_lnLid[j][1];

                if (views[vid].matchable)
                    views[vid].idealLines[lid].gid = -1;
            }

            idealLines[i].viewId_lnLid.clear();
        }
    }

    // Line reprojection

    for (int i = 0; i < idealLines.size(); ++i)
    {
        if (!idealLines[i].is3D || idealLines[i].gid < 0) 
			continue;

        if (idealLines[i].viewId_lnLid.size() == 3 && views.back().id == idealLines[i].viewId_lnLid.back()[0]) 
			continue;

		// Loop through each 2D line
        for (int j = 0; j < idealLines[i].viewId_lnLid.size(); ++j)
        {
            int vid = idealLines[i].viewId_lnLid[j][0];
            int lid = idealLines[i].viewId_lnLid[j][1];
            if (!views[vid].matchable) 
				continue;

            Mat lneq = projectLine(idealLines[i], views[vid].R, views[vid].t, K);
           
		    double sumDist = 0;
            for (int k = 0; k < views[vid].idealLines[lid].lsEndpoints.size(); ++k)
                sumDist += point2LineDist(lneq, views[vid].idealLines[lid].lsEndpoints[k]);

            if (sumDist / views[vid].idealLines[lid].lsEndpoints.size() > threshPt2LnDist) // outlier;
            {
                views[vid].idealLines[lid].gid = -1;
                idealLines[i].viewId_lnLid.erase(idealLines[i].viewId_lnLid.begin() + j); // delete
                --j;
            }
        }

		// if remaining observations of a line are less than 3 times, or current keyframe doesn't observe it 
        if (idealLines[i].viewId_lnLid.size() < 3 || (
			idealLines[i].viewId_lnLid.size() == 3 && abs(views.back().id - idealLines[i].viewId_lnLid.back()[0]) >= 1))
        {
			// Remove from 3D MFG
            idealLines[i].gid = -1;
            idealLines[i].is3D = false;
			// Remove from 2D views
            for (int j = 0; j < idealLines[i].viewId_lnLid.size(); ++j) {
                int vid = idealLines[i].viewId_lnLid[j][0];
                int lid = idealLines[i].viewId_lnLid[j][1];
                if (views[vid].matchable)
                    views[vid].idealLines[lid].gid = -1;
            }
            idealLines[i].viewId_lnLid.clear();
        }
    }
}

// Detect lines that are too far or too long.
void Mfg::detectPtOutliers(double threshPt2PtDist)
{
    //double threshPt2PtDist = 2;
    double rangeLimit = 30;

	// Loop through each key point
    for (int i = 0; i < keyPoints.size(); ++i)
    {
		// Skip key point if it is not triangulated yet
        if (!keyPoints[i].is3D || keyPoints[i].gid < 0) 
			continue;

        // delete too-far points
        double minD = 100;
        for (int j = 0; j < keyPoints[i].viewId_ptLid.size(); ++j) {
            int vid = keyPoints[i].viewId_ptLid[j][0];
            double dist = cv::norm(keyPoints[i].mat(0) + views[vid].R.t() * views[vid].t);
            if (dist < minD)
                minD = dist;
        }

        if (minD > rangeLimit) // delete
        {
			// Remove from 3D MFG
            keyPoints[i].is3D = false;
            keyPoints[i].gid = -1;
			// Remove from 2D views
            for (int j = 0; j < keyPoints[i].viewId_ptLid.size(); ++j) {
                int vid = keyPoints[i].viewId_ptLid[j][0];
                int lid = keyPoints[i].viewId_ptLid[j][1];
                if (views[vid].matchable)
                    views[vid].featurePoints[lid].gid = -1;
            }
            keyPoints[i].viewId_ptLid.clear();
        }
    }

    int n_del = 0;

	// Loop through each key point
    for (int i = 0; i < keyPoints.size(); ++i)
    {
		// Skip key point if it is not triangulated yet
        if (!keyPoints[i].is3D || keyPoints[i].gid < 0) 
			continue;

        //if (keyPoints[i].viewId_ptLid.size()<5) continue;  // allow some time to ajust position via lba
        Mat pt3d = keyPoints[i].mat(0);

        for (int j = 0; j < keyPoints[i].viewId_ptLid.size(); ++j) // for each
        {
            int vid = keyPoints[i].viewId_ptLid[j][0];
            int lid = keyPoints[i].viewId_ptLid[j][1];
            if (!views[vid].matchable) 
				continue;

            double dist = cv::norm(mat2cvpt(K * (views[vid].R * pt3d + views[vid].t)) - views[vid].featurePoints[lid].cvpt());

            if (dist > threshPt2PtDist)  // outlier;
            {
                views[vid].featurePoints[lid].gid = -1;
                keyPoints[i].viewId_ptLid.erase(keyPoints[i].viewId_ptLid.begin() + j); // delete
                --j;
            }
        }

		// if too few observations survive
        if (keyPoints[i].viewId_ptLid.size() < 2 || (
			keyPoints[i].viewId_ptLid.size() == 22 && abs(views.back().id - keyPoints[i].viewId_ptLid.back()[0]) >= 1))
        {
			// Remove from 3D MFG
            keyPoints[i].gid = -1;
            keyPoints[i].is3D = false;
			// Remove from 2D views
            for (int j = 0; j < keyPoints[i].viewId_ptLid.size(); ++j) {
                int vid = keyPoints[i].viewId_ptLid[j][0];
                int lid = keyPoints[i].viewId_ptLid[j][1];
                if (views[vid].matchable)
                    views[vid].featurePoints[lid].gid = -1;
            }
            keyPoints[i].viewId_ptLid.clear();
            n_del++;
        }
    }

    // detect newly established points when baseline too short
    for (int i = 0; i < keyPoints.size(); ++i)
    {
        if (!keyPoints[i].is3D || keyPoints[i].gid < 0) continue;

        if (views.size() < 2) continue;

        for (int j = 1; j < 3; ++j)
        {
            if (views.size() - j - 1 < 0) break;

            if (cv::norm(-views[views.size() - j].R.t()*views[views.size() - j].t
                         + views[views.size() - j - 1].R.t()*views[views.size() - j - 1].t) < 0.02
                    && keyPoints[i].estViewId == views[views.size() - j].id) // cancel these new 3d pts
            {
                keyPoints[i].is3D = false;
                n_del++;
            }
        }
    }
}

bool Mfg::rotateMode()
{
    double avThresh = 15; // angular velo degree/sec
    int	   winLen = 5;
    bool mode = false;

    for (int i = 0; i < winLen; ++i)
    {
        int id = views.size() - 1 - i;

        if (id < 1)
            break;
        else
        {
            Mat R = views[id - 1].R.inv() * views[id].R;
            double angle = acos(abs((R.at<double>(0, 0) + R.at<double>(1, 1) + R.at<double>(2, 2) - 1) / 2));

            if (views[id].angVel > avThresh ||
                    angle > 10 * PI / 180)
            {
                mode = true;
                break;
            }
        }
    }

    return mode;
}

// transform all campose and landmarks to the local one, scale, then transform back to global
void Mfg::scaleLocalMap(int from_view_id, int to_view_id, double scale, bool interpolate)
{
    // do incremental scaling, not all the same

    Mat R0 = views[from_view_id - 1].R.clone(), t0 = views[from_view_id - 1].t.clone();

    for (int i = from_view_id; i <= to_view_id; ++i)
    {
        // views[i].t = views[i].t * scale;
        double interp_scale = scale;

        if (interpolate)
            interp_scale = 1 + (i - from_view_id) * (scale - 1) / (to_view_id - from_view_id);

        views[i].t = views[i].R * R0.t() * t0 + interp_scale * (views[i].t - views[i].R * R0.t() * t0);
    }

    for (int i = 0; i < keyPoints.size(); ++i)
    {
        if (keyPoints[i].is3D && keyPoints[i].gid >= 0
                && keyPoints[i].estViewId >= from_view_id)
        {
            double interp_scale = scale;

            if (interpolate)
                interp_scale  = 1 + (keyPoints[i].estViewId - from_view_id) * (scale - 1) / (to_view_id - from_view_id) ;

            Mat X = R0.t() * ((R0 * keyPoints[i].mat(false) + t0) * interp_scale - t0);
            keyPoints[i].x = X.at<double>(0);
            keyPoints[i].y = X.at<double>(1);
            keyPoints[i].z = X.at<double>(2);
        }
    }

    for (int i = 0; i < idealLines.size(); ++i)
    {
        if (idealLines[i].is3D && idealLines[i].gid >= 0 && idealLines[i].estViewId >= from_view_id)
        {
            double interp_scale = scale;

            if (interpolate)
                interp_scale  = 1 + (idealLines[i].estViewId - from_view_id) * (scale - 1) / (to_view_id - from_view_id) ;

            idealLines[i].length = idealLines[i].length * interp_scale;
            Mat mp = R0.t() * ((R0 * cvpt2mat(idealLines[i].midpt, false) + t0) * interp_scale - t0);
            idealLines[i].midpt.x = mp.at<double>(0);
            idealLines[i].midpt.y = mp.at<double>(1);
            idealLines[i].midpt.z = mp.at<double>(2);
        }
    }
}

void Mfg::cleanup(int n_kf_keep)
{
    for (int i = 0; i < (int)views.size() - n_kf_keep; ++i)
    {
        vector<FeatPoint2d>().swap(views[i].featurePoints);
        vector<LineSegmt2d>().swap(views[i].lineSegments);
        vector<VanishPnt2d>().swap(views[i].vanishPoints);
        vector<IdealLine2d>().swap(views[i].idealLines);
        views[i].img.release();
        views[i].grayImg.release();
        views[i].matchable = false;
    }
}
