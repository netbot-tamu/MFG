/*
 * Utility functions specific to MFG applications
 */

#ifndef MFGUTILS_H
#define MFGUTILS_H

#include <vector>
#include <opencv2/core/core.hpp>

#include "consts.h"

class View;
class IdealLine2d;
class PrimPlane3d;
class Mfg;

// TODO: find where this is defined
std::vector< std::vector<int> > F_guidedLinematch (cv::Mat F, std::vector<IdealLine2d> lines1,
					std::vector<IdealLine2d> lines2, cv::Mat img1, cv::Mat img2);

bool isKeyframe ( Mfg& map, const View& v1, int th_pair, int th_overlap); // TODO: FIXME: circular dependency

void drawFeatPointMatches(View&, View& , FeaturePointPairs);

void  matchIdealLines(View& view1, View& view2, std::vector< std::vector<int> > vpPairIdx,
	FeaturePointPairs featPtMatches, cv::Mat F, std::vector< std::vector<int> >& ilinePairIdx,
	bool usePtMatch);

std::vector< std::vector<int> > matchVanishPts_withR(View& view1, View& view2, cv::Mat R, bool& goodR);

#endif
