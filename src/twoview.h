
#ifndef TWOVIEW_H
#define TWOVIEW_H

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>

#include "view.h"
#include "features2d.h"
#include "features3d.h"

class TwoView
{
public:
   View view1, view2;
   std::vector< std::vector<cv::Point2d> > featPtMatches;
   std::vector< std::vector<int> >			vpPairIdx;
   std::vector< std::vector<int> >			ilinePairIdx;
   cv::Mat						E, F, R, t;

   std::vector <KeyPoint3d>			keyPoints;
   std::vector <IdealLine3d>		idealLines;
   std::vector <PrimPlane3d>		primaryPlanes;
   std::vector <VanishPnt3d>		vanishingPoints;

   TwoView(){}
   TwoView(View&, View&);

   std::vector< std::vector<int> > matchVanishPts();
   void matchIdealLines(bool usePtMatch = false);
   void triangulateIdealLines();
   void triangulateFeatPoints();
   void detectPlanes();
   void optimize();

   void drawFeatPointMatches();
   void draw3D ();
};

#endif

