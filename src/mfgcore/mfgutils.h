/*
 * Utility functions specific to MFG applications
 */

#ifndef MFGUTILS_H
#define MFGUTILS_H
#include <vector>
#include <opencv2/core/core.hpp>

class View;
class IdealLine2d;
class PrimPlane3d;
class Mfg;

void F_guidedLinematch (cv::Mat F, View view1, View view2);
std::vector< std::vector<int> > F_guidedLinematch (cv::Mat F, std::vector<IdealLine2d> lines1,
					std::vector<IdealLine2d> lines2, cv::Mat img1, cv::Mat img2);

bool isKeyframe (const View& v0, const View& v1, int, int);
bool isKeyframe ( Mfg& map, const View& v1, int th_pair, int th_overlap); // TODO: FIXME: circular dependency

void drawFeatPointMatches(View&, View& , std::vector< std::vector<cv::Point2d> >);

void  matchIdealLines(View& view1, View& view2, std::vector< std::vector<int> > vpPairIdx,
	std::vector< std::vector<cv::Point2d> > featPtMatches, cv::Mat F, std::vector< std::vector<int> >& ilinePairIdx,
	bool usePtMatch);

void detectPlanes_2Views(View& view1, View& view2, cv::Mat R, cv::Mat t, std::vector <std::vector<int> > vpPairIdx,
	std::vector< std::vector<int> > ilinePairIdx, std::vector <PrimPlane3d>&	primaryPlanes);

std::vector< std::vector<int> > matchVanishPts_withR(View& view1, View& view2, cv::Mat R, bool& goodR);

#endif
