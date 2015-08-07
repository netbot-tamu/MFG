/////////////////////////////////////////////////////////////////////////////////
//
//  Multilayer Feature Graph (MFG), version 1.0
//  Copyright (C) 2011-2015 Yan Lu, Madison Treat, Dezhen Song
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
 * MFG utility functions
 ********************************************************************************/

#ifndef MFGUTILS_H
#define MFGUTILS_H
#include <vector>
#include <opencv2/core/core.hpp>

class View;
class IdealLine2d;
class PrimPlane3d;
class Mfg;

std::vector< std::vector<int> > F_guidedLinematch(cv::Mat F, std::vector<IdealLine2d> lines1,
        std::vector<IdealLine2d> lines2, cv::Mat img1, cv::Mat img2);                // see ../features/linematch.cpp

bool isKeyframe(Mfg &map, const View &v1, int th_pair, int th_overlap);              // TODO: FIXME: circular dependency  
                                                                                     // see mfgutils.cpp 
void drawFeatPointMatches(View &, View &, std::vector< std::vector<cv::Point2d> >);  // see mfgutils.cpp 

void  matchIdealLines(View &view1, View &view2, std::vector< std::vector<int> > vpPairIdx,
                      std::vector< std::vector<cv::Point2d> > featPtMatches, cv::Mat F, std::vector< std::vector<int> > &ilinePairIdx,
                      bool usePtMatch);                                              // see twoview.cpp 

std::vector< std::vector<int> > matchVanishPts_withR(View &view1, View &view2, cv::Mat R, bool &goodR);// see mfgutils.cpp 

#endif
