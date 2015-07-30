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

/********************************************************************************
 * View class organizing features of a key frame
 ********************************************************************************/


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
    cv::Mat                 R_loc;
    std::vector< std::vector<int> >	vpGrpIdLnIdx;
    cv::Point2d					epipoleA, epipoleB; // A is with respect to next frame
    double						angVel; // angular velocity, in degree
    double                  speed; // m/s,
    double                  accel;

    bool                    matchable;

    // ******
    cv::Mat						img, grayImg; // resized image
    double						lsLenThresh;



    // ****** for debugging ******
    double errPt, errVp, errLn, errPl, errAll,
           errPtMean, errLnMean,  errVpMean, errPlMean;

    // ***** methods *****
    View() {}
    View(std::string imgName, cv::Mat _K, cv::Mat dc, MfgSettings *_settings);
    View(std::string imgName, cv::Mat _K, cv::Mat dc, int _id, MfgSettings *_settings);
    void detectFeatPoints();			// detect feature points from image
    void detectLineSegments(cv::Mat);			// detect line segments from image
    void compMsld4AllSegments(cv::Mat grayImg);
    void detectVanishPoints();
    void extractIdealLines();
    void drawLineSegmentGroup(std::vector<int> idx);
    void drawAllLineSegments(bool write2file = false);
    void drawIdealLineGroup(std::vector<IdealLine2d>);
    void drawIdealLines();
    void drawPointandLine();

private:
    MfgSettings *mfgSettings;
};

#endif
