/*------------------------------------------------------------------------------
  This file contains the definations of functions/modules used in MFG;
  delarations are in 'mfg.h' file.
  ------------------------------------------------------------------------------*/

#include "mfg.h"

//#include <QtGui>
//#include <QtOpenGL/QtOpenGL>

//#include <gl/GLU.h>
// replaced with:
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

#include <math.h>
#include <fstream>
#ifdef _MSC_VER
#include <unordered_map>
#else
// TODO: FIXME
#include <unordered_map>
#endif
//#include "glwidget.h"

#include "mfgutils.h"
#include "export.h"

#include "utils.h"
#include "settings.h"

//#define USE_PT_ONLY_FOR_LBA
#define PREDICT_POSE_USE_R_AND_T
//#define PREDICT_POSE_USE_R_NOT_T
#define THRESH_PARALLAX 7 //(12.0*IDEAL_IMAGE_WIDTH/1000.0)
#define THRESH_PARALLAX_DEGREE (0.9)

extern double THRESH_POINT_MATCH_RATIO, SIFT_THRESH, SIFT_THRESH_HIGH, SIFT_THRESH_LOW;
extern int IDEAL_IMAGE_WIDTH;
extern MfgSettings* mfgSettings;
extern vector<vector<double>> planeColors;
struct cvpt2dCompare
{
   bool operator() (const cv::Point2d& lhs, const cv::Point2d& rhs) const
   {
      if(cv::norm(lhs-rhs) > 1e-7 && lhs.x < rhs.x )
         return true;
      else
         return false;
   }
};

// build up a mfg with first two views
void Mfg::initialize()
{
   // ----- setting parameters -----
   double distThresh_PtHomography = 1;
   double vpLnAngleThrseh = 50; // degree, tolerance angle between 3d line direction and vp direction
   double maxNoNew3dPt = 100;

   View& view0 = views[0];
   View& view1 = views[1];
   K = view0.K;

   vector<vector<cv::Point2d>> featPtMatches, allFeatPtMatches;
   vector<vector<int>>			pairIdx, allPairIdx;
   vector<vector<int>>			vpPairIdx;
   vector<vector<int>>			ilinePairIdx;
   cv::Mat F, R, E, t;

   pairIdx = matchKeyPoints (view0.featurePoints, view1.featurePoints, featPtMatches);
   allFeatPtMatches = featPtMatches;
   allPairIdx = pairIdx;
   //	drawFeatPointMatches(view0, view1, featPtMatches);
   computeEpipolar (featPtMatches, pairIdx, K, F, R, E, t, true);
   cout<<"R="<<R<<endl<<"t="<<t<<endl;
   view1.t_loc = t;
   //	drawFeatPointMatches(view0, view1, featPtMatches);
   bool isRgood = true;
   vpPairIdx = matchVanishPts_withR(view0, view1, R, isRgood);

   if (!isRgood) { // if R is not consistent with VPs, reestimate with VPs)
      vector<vector<cv::Mat>> vppairs;
      for(int i = 0; i < vpPairIdx.size(); ++i) {
         vector<cv::Mat> pair;
         pair.push_back(view0.vanishPoints[vpPairIdx[i][0]].mat());
         pair.push_back(view1.vanishPoints[vpPairIdx[i][1]].mat());
         vppairs.push_back(pair);
      }
      optimizeRt_withVP (K, vppairs, 1000, featPtMatches,R, t);
      cout<<"optR="<<R<<endl<<t<<endl;
      // compute all potential relative poses from 5-point ransac algorithm
      vector<cv::Mat> Fs, Es, Rs, ts;
      computePotenEpipolar (allFeatPtMatches, allPairIdx,K, Fs, Es, Rs, ts);
      double difR, idx=0, minDif=100;
      for (int i=0; i < Rs.size(); ++i) {
         difR =  cv::norm(R - Rs[i]);
         if (minDif > difR) {
            minDif = difR;
            idx = i;
         }
      }
      R = Rs[idx];
      t = ts[idx];
      matchVanishPts_withR(view0, view1, R, isRgood);
      cout<<"after vp recitify,\n R="<<R<<endl<<t<<endl;
   }


   vector<KeyPoint3d> tmpkp; // temporary keypoints
   for(int i=0; i < featPtMatches.size(); ++i) {
      cv::Mat X = triangulatePoint_nonlin (cv::Mat::eye(3,3,CV_64F), cv::Mat::zeros(3,1,CV_64F),
            R, t, K, featPtMatches[i][0],	featPtMatches[i][1]);
      tmpkp.push_back(KeyPoint3d(X.at<double>(0),X.at<double>(1),X.at<double>(2)));
   }
   //	twoview_ba(K, R, t, tmpkp, featPtMatches);
   cv::Point2d ep1 = mat2cvpt(K*R.t()*t);
   cv::Point2d ep2 = mat2cvpt(K*t);

   view0.epipoleA = ep1;
   view1.epipoleB = ep2;

   // ----- sort point matches by parallax -----
   vector<valIdxPair> prlxVec;
   for(int i=0; i<featPtMatches.size(); ++i) {
      valIdxPair pair;
      pair.first = compParallax (featPtMatches[i][0], featPtMatches[i][1], K,
            cv::Mat::eye(3,3,CV_64F), R);
      pair.second = i;
      prlxVec.push_back(pair);
   }
   sort(prlxVec.begin(), prlxVec.end(), comparator_valIdxPair);
   vector<vector<int>>  copyPairIdx = pairIdx;
   vector<vector<cv::Point2d>> copyFeatPtMatches = featPtMatches;
   pairIdx.clear();
   featPtMatches.clear();
   for(int i=prlxVec.size()-1; i >= 0; --i) {
      pairIdx.push_back(copyPairIdx[prlxVec[i].second]);
      featPtMatches.push_back(copyFeatPtMatches[prlxVec[i].second]);
   }


   cv::Mat canv1 = view0.img.clone();
   cv::Mat canv2 = view1.img.clone();
   // --- set up 3d key points ---
   int numNew3dPt = 0;
   cv::Mat X(4,1,CV_64F);
   for(int i=0; i<featPtMatches.size(); ++i) {
      double parallax = compParallax (featPtMatches[i][0], featPtMatches[i][1], K,
            cv::Mat::eye(3,3,CV_64F), R);
      // don't triangulate small-parallax point
      if (parallax < THRESH_PARALLAX) { // add as a 2d track
         // set up and 3d-2d correspondence
         KeyPoint3d kp;
         kp.gid = keyPoints.size();
         view0.featurePoints[pairIdx[i][0]].gid = kp.gid;
         view1.featurePoints[pairIdx[i][1]].gid = kp.gid;
         vector<int> view_point;
         view_point.push_back(view0.id);
         view_point.push_back(pairIdx[i][0]);
         kp.viewId_ptLid.push_back(view_point);
         view_point.clear();
         view_point.push_back(view1.id);
         view_point.push_back(pairIdx[i][1]);
         kp.viewId_ptLid.push_back(view_point);
         kp.is3D = false;
         keyPoints.push_back(kp);
#ifndef HIGH_SPEED_NO_GRAPHICS
         cv::circle(canv1, featPtMatches[i][0], 2, cv::Scalar(0,0,0,1), 1);
         cv::putText(canv1, " "+num2str(i), featPtMatches[i][0], 1, 1, cv::Scalar(0,0,0,1));
#endif
      } else {
         if(numNew3dPt > maxNoNew3dPt) continue;
         cv::Mat X = triangulatePoint_nonlin ( cv::Mat::eye(3,3,CV_64F), cv::Mat::zeros(3,1,CV_64F),
               R, t, K, featPtMatches[i][0],  featPtMatches[i][1]);
         if(!checkCheirality(R,t,X) || !checkCheirality( cv::Mat::eye(3,3,CV_64F), cv::Mat::zeros(3,1,CV_64F),X))
            continue;
         if (cv::norm(X) > 20) continue;
         // set up 3d keypoints and 3d-2d correspondence
         KeyPoint3d kp(X.at<double>(0),X.at<double>(1),X.at<double>(2));
         kp.gid = keyPoints.size();
         view0.featurePoints[pairIdx[i][0]].gid = kp.gid;
         view1.featurePoints[pairIdx[i][1]].gid = kp.gid;
         vector<int> view_point;
         view_point.push_back(view0.id);
         view_point.push_back(pairIdx[i][0]);
         kp.viewId_ptLid.push_back(view_point);
         view_point.clear();
         view_point.push_back(view1.id);
         view_point.push_back(pairIdx[i][1]);
         kp.viewId_ptLid.push_back(view_point);
         kp.is3D = true;
         kp.estViewId = 1;
         keyPoints.push_back(kp);
         numNew3dPt++;
#ifndef HIGH_SPEED_NO_GRAPHICS
         cv::circle(canv2, featPtMatches[i][1], 2, cv::Scalar(0,0,0,1), 1);
         cv::putText(canv2, " "+num2str(parallax)+","+num2str(cv::norm(X)), featPtMatches[i][1], 1, 1, cv::Scalar(0,0,0,1));
#endif
      }
   }

   view0.R = cv::Mat::eye(3,3,CV_64F);
   view0.t = cv::Mat::zeros(3,1,CV_64F);
   view1.R = R.clone();
   view1.t = t.clone();

   // ----- set up 3D vanishing points -----
   for(int i=0; i < vpPairIdx.size(); ++i) {
      cv::Mat tmpvp = K.inv() * view0.vanishPoints[vpPairIdx[i][0]].mat();
      tmpvp = tmpvp/cv::norm(tmpvp);
      VanishPnt3d vp(tmpvp.at<double>(0),	tmpvp.at<double>(1),
            tmpvp.at<double>(2));
      vp.gid = vanishingPoints.size();
      vanishingPoints.push_back(vp);
      view0.vanishPoints[vpPairIdx[i][0]].gid = vp.gid;
      view1.vanishPoints[vpPairIdx[i][1]].gid = vp.gid;
      vector<int> vid_vpid;
      vid_vpid.push_back(view0.id);
      vid_vpid.push_back(vpPairIdx[i][0]);
      vanishingPoints.back().viewId_vpLid.push_back(vid_vpid);
      vid_vpid.clear();
      vid_vpid.push_back(view1.id);
      vid_vpid.push_back(vpPairIdx[i][1]);
      vanishingPoints.back().viewId_vpLid.push_back(vid_vpid);

   }

   matchIdealLines(view0, view1, vpPairIdx, featPtMatches, F, ilinePairIdx, 1);
   //	drawLineMatches(view0.img,view1.img, view0.idealLines, view1.idealLines, ilinePairIdx);
   //	detectPlanes_2Views (view0, view1, R, t, vpPairIdx, ilinePairIdx, primaryPlanes);

   for(int i=0; i < keyPoints.size(); ++i) {
      if(! keyPoints[i].is3D || keyPoints[i].gid<0) continue;
      FeatPoint2d p0 = view0.featurePoints[keyPoints[i].viewId_ptLid[0][1]];
      FeatPoint2d p1 = view1.featurePoints[keyPoints[i].viewId_ptLid[1][1]];
      for(int j=0; j < primaryPlanes.size(); ++j) {
         cv::Mat H = K*(R-t*primaryPlanes[j].n.t()/primaryPlanes[j].d)*K.inv();
         double dist = 0.5* cv::norm(mat2cvpt(H*p0.mat()) - mat2cvpt(p1.mat()))
            + 0.5*cv::norm(mat2cvpt(H.inv()*p1.mat()) - mat2cvpt(p0.mat()));
         if (dist < distThresh_PtHomography ) {
            keyPoints[i].pGid = primaryPlanes[j].gid;
            break;
         }
      }
   }

   // ----- set up 3D lines (only coplanar ones) -----

   for(int i=0; i < ilinePairIdx.size(); ++i) {
      double prlx = compParallax(view0.idealLines[ilinePairIdx[i][0]],
            view1.idealLines[ilinePairIdx[i][1]], K, view0.R, view1.R);
      if(prlx < THRESH_PARALLAX ||  view0.idealLines[ilinePairIdx[i][0]].pGid < 0) { // set up 2d line track
         IdealLine3d line;
         line.is3D = false;
         line.gid = idealLines.size();
         line.vpGid = view0.vanishPoints[view0.idealLines[ilinePairIdx[i][0]].vpLid].gid; // assign vpGid to line
         view0.idealLines[ilinePairIdx[i][0]].gid = line.gid;
         view1.idealLines[ilinePairIdx[i][1]].gid = line.gid;
         line.pGid = view0.idealLines[ilinePairIdx[i][0]].pGid;
         view1.idealLines[ilinePairIdx[i][1]].pGid = line.pGid;
         vector<int> pair;
         pair.push_back(view0.id);
         pair.push_back(ilinePairIdx[i][0]);
         line.viewId_lnLid.push_back(pair);
         pair.clear();
         pair.push_back(view1.id);
         pair.push_back(ilinePairIdx[i][1]);
         line.viewId_lnLid.push_back(pair);
         idealLines.push_back(line);

      } else {
         IdealLine2d a = view0.idealLines[ilinePairIdx[i][0]],
                     b = view1.idealLines[ilinePairIdx[i][1]];
         IdealLine3d line = triangluateLine(cv::Mat::eye(3,3,CV_64F),
               cv::Mat::zeros(3,1,CV_64F), R, t, K, a,  b );
         int vpGid = view0.vanishPoints[view0.idealLines[ilinePairIdx[i][0]].vpLid].gid;
         if ( abs(vanishingPoints[vpGid].mat().dot(line.direct)/cv::norm(line.direct))
               < cos(vpLnAngleThrseh*PI/180)) {
            // 3d line and vp angle too large, invalid 3d line
            continue;
         }
         line.is3D = true;
         line.gid = idealLines.size();
         line.vpGid = view0.vanishPoints[view0.idealLines[ilinePairIdx[i][0]].vpLid].gid;
         view0.idealLines[ilinePairIdx[i][0]].gid = line.gid;
         view1.idealLines[ilinePairIdx[i][1]].gid = line.gid;
         line.pGid = view0.idealLines[ilinePairIdx[i][0]].pGid;
         view1.idealLines[ilinePairIdx[i][1]].pGid = line.pGid;
         vector<int> pair;
         pair.push_back(view0.id);
         pair.push_back(ilinePairIdx[i][0]);
         line.viewId_lnLid.push_back(pair);
         pair.clear();
         pair.push_back(view1.id);
         pair.push_back(ilinePairIdx[i][1]);
         line.viewId_lnLid.push_back(pair);
         idealLines.push_back(line);
         cout<<line.gid<<'\t';
         cv::Scalar color(rand()%255,rand()%255,rand()%255,0);
         cv::line(canv1, a.extremity1, a.extremity2, color, 2);
         cv::line(canv2, b.extremity1, b.extremity2, color, 2);
      }
   }
   //	drawLineMatches(view0.img,view1.img, view0.idealLines, view1.idealLines, ilinePairIdx);
   //	showImage("view0", &canv1);
   //	showImage("view1", &canv2);

   //cv::waitKey(2000);

   Matrix3d Rx;
   Rx << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
      R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
      R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2);
   Quaterniond q(Rx);
   angVel = 2*acos(q.w()) * 180/PI
      / ((views[views.size()-1].frameId - views[views.size()-2].frameId)/fps);

   view0.angVel = 0;
   view1.angVel = angVel;


   updatePrimPlane();
   cout<<endl<<"Mfg initialization done <<<<<<<<<<<<<<"<<endl<<endl;

   return;
}

void Mfg::expand(View& nview, int fid)
{
   nview.id = views.size();
   nview.frameId = fid;

   views.push_back(nview);
   expand_keyPoints (views[views.size()-2], views[views.size()-1]);

   // ----- remove some obvious outliers -----
   detectPtOutliers(5* IDEAL_IMAGE_WIDTH/640.0);
   detectLnOutliers(10* IDEAL_IMAGE_WIDTH/640.0);

   adjustBundle();

   // ----- further clear outliers -----
   detectPtOutliers(2 * IDEAL_IMAGE_WIDTH/640.0);
   detectLnOutliers(3* IDEAL_IMAGE_WIDTH/640.0);

}

void Mfg::expand_keyPoints (View& prev, View& nview)
{
   // ----- parameter setting -----
   double numPtpairLB = 30; // lower bound of pair number to trust epi than vp
   double reprjThresh = 5 * IDEAL_IMAGE_WIDTH/640.0; // projected point distance threshold
   double scale = -1;  // relative scale of baseline
   double vpLnAngleThrseh = 50; // degree, tolerance angle between 3d line direction and vp direction
   double ilineLenLimit = 10;  // length limit of 3d line, ignore too long lines
   double maxNumNew3dPt = 200;
   double maxNumNew2d_3dPt = 100;
   bool   sortbyPrlx = true;
   double parallaxThresh = THRESH_PARALLAX;
   double parallaxDegThresh = THRESH_PARALLAX_DEGREE;

   vector<vector<cv::Point2d>> featPtMatches;
   vector<vector<int>> pairIdx;

   if(mfgSettings->getKeypointAlgorithm() <3) // sift surf
      pairIdx = matchKeyPoints (prev.featurePoints, nview.featurePoints, featPtMatches);
   else {
      bool found_in_track = false;
      Frame frm_same_nview;
      for(int i=0; i<trackFrms.size(); ++i) {
         if(trackFrms[i].filename == nview.filename) {
            found_in_track = true;
            frm_same_nview = trackFrms[i];
            break;
         }
      }
      if(!found_in_track) {// not found in track, then do opticalflow track
         // wrt last one in track
         //		cout<<"ops, need to track for selected key frame....\n";
         vector<cv::Point2f> curr_pts;
         vector<uchar> status;
         vector<float> err;
         cv::calcOpticalFlowPyrLK(
               trackFrms.back().image, nview.grayImg, // 2 consecutive images
               trackFrms.back().featpts, // input point positions in first im
               curr_pts, // output point positions in the 2nd
               status,    // tracking success
               err,      // tracking error
               cv::Size(mfgSettings->getOflkWindowSize(),mfgSettings->getOflkWindowSize()),
               3,
               cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01),
               0,
               mfgSettings->getOflkMinEigenval()  // minEignVal threshold for the 2x2 spatial motion matrix, to eleminate bad points
               );
         for(int i=0; i<status.size(); ++i) {
            if(status[i]
                  //	  &&(trackFrms.back().featpts[i].x>0 && trackFrms.back().featpts[i].y >0
                  //		  && trackFrms.back().featpts[i].x<nview.img.cols-1 && trackFrms.back().featpts[i].y<nview.img.rows-1 )
                  //      &&(curr_pts[i].x >0 &&  curr_pts[i].y>0 && curr_pts[i].x<nview.img.cols-1 &&curr_pts[i].y<nview.img.rows-1)
              ){ //found correspond
               frm_same_nview.featpts.push_back(curr_pts[i]);
               frm_same_nview.pt_lid_in_last_view.push_back(trackFrms.back().pt_lid_in_last_view[i]);
            }
         }
      }
      // === build a map ===
      std::map<cv::Point2d, int, cvpt2dCompare> pt_idx;
      vector<cv::KeyPoint> kpts;
      for(int j=0; j < frm_same_nview.featpts.size(); ++j) {//build a map
         pt_idx[frm_same_nview.featpts[j]] = frm_same_nview.pt_lid_in_last_view[j];
         kpts.push_back(cv::KeyPoint(frm_same_nview.featpts[j], mfgSettings->getFeatureDescriptorRadius()));
      }

      for(int j=0; j < nview.featurePoints.size(); ++j) {
         bool pt_exist = false;
         // change to flann later
         for(int k=0; k <frm_same_nview.featpts.size(); ++k) {
            float dx = frm_same_nview.featpts[k].x - nview.featurePoints[j].x;
            float dy = frm_same_nview.featpts[k].y - nview.featurePoints[j].y;
            if (sqrt(dx*dx+dy*dy) < mfgSettings->getGfttMinimumPointDistance()) {
               pt_exist = true;
               break;
            }
         }
         if(!pt_exist) {
            kpts.push_back(cv::KeyPoint(nview.featurePoints[j].cvpt(), mfgSettings->getFeatureDescriptorRadius()));
         }
      }

      // ==== compute descriptors for feature points, using SURF ====
      cv::SurfDescriptorExtractor surfext;
      cv::Mat descs;
      surfext.compute(nview.grayImg, kpts, descs); // note that some ktps can be removed

      nview.featurePoints.clear();
      nview.featurePoints.reserve(kpts.size());
      for(int i = 0; i < kpts.size(); ++i) {
         int lid = nview.featurePoints.size();
         nview.featurePoints.push_back(FeatPoint2d(kpts[i].pt.x, kpts[i].pt.y, descs.row(i).t(), lid,-1));
         if( pt_idx.find(kpts[i].pt) != pt_idx.end()) // matched with last keyframe
         {
            vector<int> pair(2);
            vector<cv::Point2d> match(2);
            pair[0] = pt_idx[kpts[i].pt];
            pair[1] = lid;
            match[0] = prev.featurePoints[pair[0]].cvpt();
            match[1] = kpts[i].pt;
            pairIdx.push_back(pair);
            featPtMatches.push_back(match);
         }
      }

   }
#ifndef HIGH_SPEED_NO_GRAPHICS
   cv::Mat canv1 = prev.img.clone(), canv2=nview.img.clone();
   for(int i=0; i<featPtMatches.size(); ++i) {
      cv::circle(canv1, featPtMatches[i][0], 2, cv::Scalar(100,200,200), 2);
      cv::circle(canv2, featPtMatches[i][1], 2, cv::Scalar(100,200,200), 2);
   }
   //	showImage("1",&canv1);
   //	showImage("2",&canv2);
   //	cv::waitKey();
#endif
   trackFrms.clear();

   //cv::Mat t_prev = prev.t - (prev.R * views[views.size()-3].R.t()) * views[views.size()-3].t;
   cv::Mat t_prev = views[views.size()-2].t_loc;
   cout<<t_prev<<endl;

   cv::Mat F, R, E, t; // relative pose between last two views

   // compute all potential relative poses from 5-point ransac algorithm
   vector<cv::Mat> Fs, Es, Rs, ts;

   MyTimer timer;	timer.start();
   computePotenEpipolar (featPtMatches,pairIdx,K, Fs, Es, Rs, ts, false, t_prev);
   timer.end();	cout<<pairIdx.size()<<"\t"<<"computePotenEpipolar time:"<<timer.time_ms<<"ms\n";
   // ---- find observed 3D points (and plot)  ------
   vector<cv::Point3d> pt3d, pt3d_old;
   vector<cv::Point2d> pt2d, pt2d_old;
   //	cv::Mat canv1 = prev.img.clone(), canv2 = nview.img.clone();
   for(int i=0; i < featPtMatches.size(); ++i) {
      int gid = prev.featurePoints[pairIdx[i][0]].gid;
      cv::Scalar color(rand()%255,rand()%255,rand()%255,0);
#ifndef HIGH_SPEED_NO_GRAPHICS

      if (gid >= 0 && keyPoints[gid].is3D) { // observed 3d pts
         cv::circle(canv1, featPtMatches[i][0], 4, color, 2);
         cv::circle(canv2, featPtMatches[i][1], 4, color, 2);
      } else {
         cv::circle(canv1, featPtMatches[i][0], 1, color, 1);
         cv::circle(canv2, featPtMatches[i][1], 1, color, 1);
      }
#endif
      if( gid >= 0 && keyPoints[gid].is3D) {
         pt3d.push_back(cv::Point3d(keyPoints[gid].x,keyPoints[gid].y,keyPoints[gid].z));
         pt2d.push_back(featPtMatches[i][1]);
         if(keyPoints[gid].viewId_ptLid.size() >=3) { // those observed at least 3 times
            pt3d_old.push_back(keyPoints[gid].cvpt());
            pt2d_old.push_back(featPtMatches[i][1]);
         }
      }
   }


#ifdef PREDICT_POSE_USE_R_AND_T
   vector <int> maxInliers_Rt;
   double bestScale = -100;
   vector<KeyPoint3d> localKeyPts;
   vector<double> sizes;
   //	cout<<"Observed 3d points: "<<pt3d.size() <<endl;
   for(int trial=0; trial < 20; ++trial) {
      maxInliers_Rt.clear();
      localKeyPts.clear();
      sizes.clear();
      for(int iter = 0; iter < Rs.size(); ++iter) {
         cv::Mat Ri = Rs[iter],	ti = ts[iter];
         bool isRgood = true;
         vector<vector<int>> vpPairIdx = matchVanishPts_withR(prev, nview, Ri, isRgood);
         if (!isRgood && featPtMatches.size() < numPtpairLB) {
            // when point matches are inadequate, E (R,t) are not reliable, even errant
            // if R is not consistent with VPs, reestimate with VPs)
            cout<<"R is inconsistent with vanishing point correspondences****"<<endl;
            vector<vector<cv::Mat>> vppairs;
            for(int i = 0; i < vpPairIdx.size(); ++i) {
               vector<cv::Mat> pair;
               pair.push_back(prev.vanishPoints[vpPairIdx[i][0]].mat());
               pair.push_back(nview.vanishPoints[vpPairIdx[i][1]].mat());
               vppairs.push_back(pair);
            }
            optimizeRt_withVP (K, vppairs, 1000, featPtMatches,Ri, ti);
            matchVanishPts_withR(prev, nview, Ri, isRgood);
            cout<<"rectified:"<<R<<endl<<t<<endl;
         }
         // triangulate points for scale estimation
         vector<KeyPoint3d> tmpkp; // temporary keypoints
         int num_usable_pts = 0;
         for(int i=0; i < featPtMatches.size(); ++i) {
            double parallax = compParallax (featPtMatches[i][0], featPtMatches[i][1], K, cv::Mat::eye(3,3,CV_64F), Ri);
            double parallaxDeg = compParallaxDeg (featPtMatches[i][0], featPtMatches[i][1], K, cv::Mat::eye(3,3,CV_64F), Ri);
            int iGid = prev.featurePoints[pairIdx[i][0]].gid;
            //		if ((parallax > parallaxThresh * 0.8) && iGid >= 0 && keyPoints[iGid].is3D) {
            if ((parallaxDeg > parallaxDegThresh * 0.8) && iGid >= 0 && keyPoints[iGid].is3D) {
               cv::Mat X = triangulatePoint_nonlin (cv::Mat::eye(3,3,CV_64F), cv::Mat::zeros(3,1,CV_64F),
                     Ri, ti, K, featPtMatches[i][0],	featPtMatches[i][1]);
               tmpkp.push_back(KeyPoint3d(X.at<double>(0),X.at<double>(1),X.at<double>(2)));
               tmpkp.back().is3D = true;
               ++num_usable_pts;
            } else { // not usable for estimating scale
               tmpkp.push_back(KeyPoint3d(0,0,0));
               tmpkp.back().is3D = false;
            }
         }
         cout<<"usable local 3d pts for scale estimate: "<<num_usable_pts<<endl;
         if(num_usable_pts < 1)
            continue; // abandon this pair of Ri ti

         cv::Mat Rn = Ri *prev.R;
         // ----- find best R and t based on existent 3D points------
         int maxIter = 100;
         vector<int> maxInliers;
         double best_s = -1;
         for (int it = 0; it < maxIter; ++it) {
            int i = rand()%featPtMatches.size();
            int iGid = prev.featurePoints[pairIdx[i][0]].gid;
            if (iGid < 0) {--it; continue;} // not observed before
            if (!keyPoints[iGid].is3D||keyPoints[iGid].gid<0) {--it; continue;}  // not 3D point yet
            if (!tmpkp[i].is3D) {--it; continue;}
            //-- guess scale (minimum solution)
            double s = cv::norm(keyPoints[iGid].mat(0)+(prev.R.t()*prev.t))/cv::norm(tmpkp[i].mat());

            cv::Mat tn = Ri *prev.t + ti*s;
            vector<int> inliers;
            for(int j=0; j<featPtMatches.size(); ++j) {
               int jGid = prev.featurePoints[pairIdx[j][0]].gid;
               if (jGid < 0) continue;
               if (!keyPoints[jGid].is3D ||keyPoints[jGid].gid<0) continue;
               if (!tmpkp[j].is3D) continue;
               // project 3d to n-th view
               cv::Mat pt = K*Rn*(keyPoints[jGid].mat(0)+Rn.t()*tn);
               //		cout<<cv::norm(featPtMatches[j][1] - mat2cvpt(pt))<<'\t';
               if (cv::norm(featPtMatches[j][1] - mat2cvpt(pt)) < reprjThresh )
                  inliers.push_back(j);
            }
            if (maxInliers.size() < inliers.size()) {
               maxInliers = inliers;
               best_s = s;
            }
         }
         cout<<ti<<" "<<maxInliers.size()<<endl;

         if(cv::norm(t_prev)>0.1 && abs(ti.dot(t_prev)) < cos(50*PI/180.0) ) { // larger than 45 deg
            cout<<"... smooth constraint violated...\n";
            continue;
         }

         sizes.push_back(maxInliers.size());
         if (maxInliers.size() > maxInliers_Rt.size()) {
            maxInliers_Rt = maxInliers;
            bestScale = best_s;
            R = Ri;
            t = ti;
            F = Fs[iter];
            localKeyPts = tmpkp;
         }
      }
      sort(sizes.begin(), sizes.end());
      if(sizes.size()>1 && (sizes[sizes.size()-2] >= sizes[sizes.size()-1] * 0.9
               || sizes[sizes.size()-2] >= sizes[sizes.size()-1]-1 )) {
         // ----- use motion smooth constraint ------
         if (1 || sizes[sizes.size()-1] < 5) {
            cout<<"t_prev = "<< t_prev<<endl;
            // filter out some R and t
            double minVal=1, minIdx=0;
            for(int i = 0; i < Rs.size(); ++i) {
               if (cv::norm(t_prev)>0.1 && ts[i].dot(t_prev) < minVal) {
                     minVal = ts[i].dot(t_prev);
                     minIdx = i;
               }
            }
            // remove the one most different from t_pnp
            Rs.erase(Rs.begin()+minIdx);
            ts.erase(ts.begin()+minIdx);
         } else
            reprjThresh = reprjThresh * 0.8;
      }
      else {
         break;
      }
   }
   if(maxInliers_Rt.size() > 0) {
      // ------ re-compute scale on largest concensus set --------
      nview.R = R*prev.R;
      cv::Scalar color(rand()%255,rand()%255,rand()%255,0);
      double wsum = 0, tw = 0, sum3 =0;
      vector<double> scales, scales3;
      for(int i=0; i < maxInliers_Rt.size(); ++i) {
         int j = maxInliers_Rt[i];
         int gid = prev.featurePoints[pairIdx[j][0]].gid;
         if(!localKeyPts[j].is3D) continue;
         double s = cv::norm(keyPoints[gid].mat(0)+(prev.R.t()*prev.t))/cv::norm(localKeyPts[j].mat());

         if(maxInliers_Rt.size() > 5 &&
               (abs(s-bestScale) > 0.3 || (bestScale > 0.2 && max(s/bestScale, bestScale/s) > 1.5 ))
            )
            continue ; // could be outliers

         int obsTimesSinceEst = 0;  // observed times since Established
         for(int k=0; k < keyPoints[gid].viewId_ptLid.size(); ++k) {
            if(keyPoints[gid].viewId_ptLid[k][0] >= keyPoints[gid].estViewId)
               obsTimesSinceEst ++;
         }
#ifndef HIGH_SPEED_NO_GRAPHICS
         cv::putText(canv2, "  "+num2str(s), featPtMatches[j][1], 1, 1, color);
#endif
         wsum = wsum + s * pow((double)obsTimesSinceEst+1, 3);
         tw = tw + pow((double)obsTimesSinceEst+1, 3);

         scales.push_back(s);

         if(keyPoints[gid].viewId_ptLid.size() >= 3 && obsTimesSinceEst >=2 ) { // point observed at 3 times
            sum3 = sum3 + s;
            scales3.push_back(s);
         }
      }
      if (scales.size() > 0) {
         scale = wsum/tw;
         cout<<nview.frameId-prev.frameId<<" frames, bestScale = " << bestScale
            <<", w-ave="<< wsum/tw<<", median=" << vecMedian(scales) <<endl;
         if (scales3.size() >= 15) {
            scale = vecMedian(scales3);
            cout<<nview.frameId-prev.frameId<<" frames,"<<"ave="<<vecSum(scales3)/scales3.size()
               <<", median="<<vecMedian(scales3)<< ", ["<< scales3.size()<<"]"<<endl;
         }
         scale = (scale + bestScale + vecMedian(scales))/3;
         cout<<nview.id<<": mixed scale = "<<scale<<endl;
      } else {
         scale = bestScale;
      }
      nview.t = R*prev.t + t*scale;
      nview.t_loc = t;
   } else { // small movement, use R-PnP
      vector <int> maxInliers_R;
      cv::Mat best_tn;
      vector<double> sizes;
      for(int trial=0; trial < 10; ++trial) {
         maxInliers_R.clear();
         sizes.clear();
         for(int iter = 0; iter < Rs.size(); ++iter) {
            cv::Mat Ri = Rs[iter],	ti = ts[iter];
            bool isRgood = true;
            vector<vector<int>> vpPairIdx = matchVanishPts_withR(prev, nview, Ri, isRgood);
            if (!isRgood && featPtMatches.size() < numPtpairLB) {
               // when point matches are inadequate, E (R,t) are not reliable, even errant
               // if R is not consistent with VPs, reestimate with VPs)
               cout<<"R is inconsistent with vanishing point correspondences****"<<endl;
               vector<vector<cv::Mat>> vppairs;
               for(int i = 0; i < vpPairIdx.size(); ++i) {
                  vector<cv::Mat> pair;
                  pair.push_back(prev.vanishPoints[vpPairIdx[i][0]].mat());
                  pair.push_back(nview.vanishPoints[vpPairIdx[i][1]].mat());
                  vppairs.push_back(pair);
               }
               optimizeRt_withVP (K, vppairs, 1000, featPtMatches,Ri, ti);
               matchVanishPts_withR(prev, nview, Ri, isRgood);
               cout<<"rectified:"<<R<<endl<<t<<endl;
            }

            cv::Mat Rn = Ri *prev.R;

            // ----- find best R and t based on existent 3D points------
            int maxIter = 100;
            vector<int> maxInliers;
            cv::Mat good_tn;
            double best_s = -1;
            for (int it = 0; it < maxIter; ++it) {
               int i = rand() % pt3d.size();
               int j = rand() % pt3d.size();
               if(i==j) {--it; continue;}
               vector<cv::Point3d> samplePt3d;
               samplePt3d.push_back(pt3d[i]);
               samplePt3d.push_back(pt3d[j]);
               vector<cv::Point2d> samplePt2d;
               samplePt2d.push_back(pt2d[i]);
               samplePt2d.push_back(pt2d[j]);

               cv::Mat tn = pnp_withR (samplePt3d, samplePt2d, K, Rn);
               cv::Mat t_rel = tn - Ri*prev.t;
               t_rel = t_rel/cv::norm(t_rel);
               if(cv::norm(t_prev)>0.1 && abs(t_rel.dot(t_prev) < cos(40*PI/180)))// motion smooth
                  continue;
               vector<int> inliers;
               for(int j=0; j < pt3d.size(); ++j) {
                  // project 3d to n-th view
                  cv::Mat pt = K*(Rn * cvpt2mat(pt3d[j],0) + tn);
                  if (cv::norm(pt2d[j] - mat2cvpt(pt)) < reprjThresh )
                     inliers.push_back(j);
               }
               if (maxInliers.size() < inliers.size()) {
                  maxInliers = inliers;
                  good_tn = tn;
               }
            }
            cout<<ti<<",, "<<maxInliers.size()<<endl;
            sizes.push_back(maxInliers.size());
            if (maxInliers.size() > maxInliers_R.size()) {
               maxInliers_R = maxInliers;
               best_tn = good_tn;
               R = Ri;
               t = ti;
               F = Fs[iter];
            }
         }
         sort(sizes.begin(), sizes.end());
         if(sizes.size()>1 && sizes[sizes.size()-2] >= sizes[sizes.size()-1] * 0.95)
            reprjThresh = reprjThresh * 0.9;
         else
            break;
      }
      // ------ re-compute scale on largest concensus set --------
      cv::Scalar color(rand()%255,rand()%255,rand()%255,0);

      vector<cv::Point3d> inlierPt3d;
      vector<cv::Point2d> inlierPt2d;
      for(int i=0; i < maxInliers_R.size(); ++i) {
         inlierPt3d.push_back(pt3d[maxInliers_R[i]]);
         inlierPt2d.push_back(pt2d[maxInliers_R[i]]);
      }
      nview.R = R*prev.R;
      nview.t = pnp_withR (inlierPt3d, inlierPt2d, K, nview.R);
      nview.t_loc = cv::Mat::zeros(3,1,CV_64F);
      cout<<nview.frameId-prev.frameId<<" frames,"<<"Scale (2pt_alg)="
         <<cv::norm(-nview.R.t()*nview.t + prev.R.t()*prev.t)<<", "
         <<maxInliers_R.size()<<"/"<<pt3d.size()<<endl;
   }
#endif

#ifdef PREDICT_POSE_USE_R_NOT_T
   vector <int> maxInliers_R;
   cv::Mat best_tn;
   vector<double> sizes;
   for(int trial=0; trial < 10; ++trial) {
      maxInliers_R.clear();
      sizes.clear();

      for(int iter = 0; iter < Rs.size(); ++iter) {
         cv::Mat Ri = Rs[iter],	ti = ts[iter];
         bool isRgood = true;
         vector<vector<int>> vpPairIdx = matchVanishPts_withR(prev, nview, Ri, isRgood);
         if (!isRgood && featPtMatches.size() < numPtpairLB) {
            // when point matches are inadequate, E (R,t) are not reliable, even errant
            // if R is not consistent with VPs, reestimate with VPs)
            cout<<"R is inconsistent with vanishing point correspondences****"<<endl;
            vector<vector<cv::Mat>> vppairs;
            for(int i = 0; i < vpPairIdx.size(); ++i) {
               vector<cv::Mat> pair;
               pair.push_back(prev.vanishPoints[vpPairIdx[i][0]].mat());
               pair.push_back(nview.vanishPoints[vpPairIdx[i][1]].mat());
               vppairs.push_back(pair);
            }
            optimizeRt_withVP (K, vppairs, 1000, featPtMatches,Ri, ti);
            matchVanishPts_withR(prev, nview, Ri, isRgood);
            cout<<"rectified:"<<R<<endl<<t<<endl;
         }

         cv::Mat Rn = Ri *prev.R;

         // ----- find best R and t based on existent 3D points------
         int maxIter = 100;
         vector<int> maxInliers;
         cv::Mat good_tn;
         double best_s = -1;
         for (int it = 0; it < maxIter; ++it) {
            int i = rand() % pt3d.size();
            int j = rand() % pt3d.size();
            if(i==j) {--it; continue;}
            vector<cv::Point3d> samplePt3d;
            samplePt3d.push_back(pt3d[i]);
            samplePt3d.push_back(pt3d[j]);
            vector<cv::Point2d> samplePt2d;
            samplePt2d.push_back(pt2d[i]);
            samplePt2d.push_back(pt2d[j]);

            cv::Mat tn = pnp_withR (samplePt3d, samplePt2d, K, Rn);

            vector<int> inliers;
            for(int j=0; j < pt3d.size(); ++j) {
               // project 3d to n-th view
               cv::Mat pt = K*(Rn * cvpt2mat(pt3d[j],0) + tn);
               if (cv::norm(pt2d[j] - mat2cvpt(pt)) < reprjThresh )
                  inliers.push_back(j);
            }
            if (maxInliers.size() < inliers.size()) {
               maxInliers = inliers;
               good_tn = tn;
            }
         }
         cout<<ti<<" "<<maxInliers.size()<<endl;
         sizes.push_back(maxInliers.size());
         if (maxInliers.size() > maxInliers_R.size()) {
            maxInliers_R = maxInliers;
            best_tn = good_tn;
            R = Ri;
            t = ti;
            F = Fs[iter];
         }
      }
      sort(sizes.begin(), sizes.end());
      if(sizes.size()>1 && sizes[sizes.size()-2] >= sizes[sizes.size()-1] * 0.95)
         reprjThresh = reprjThresh * 0.9;
      else
         break;
   }
   // ------ re-compute scale on largest concensus set --------
   cv::Scalar color(rand()%255,rand()%255,rand()%255,0);

   vector<cv::Point3d> inlierPt3d;
   vector<cv::Point2d> inlierPt2d;
   for(int i=0; i < maxInliers_R.size(); ++i) {
      inlierPt3d.push_back(pt3d[maxInliers_R[i]]);
      inlierPt2d.push_back(pt2d[maxInliers_R[i]]);
   }
   nview.R = R*prev.R;
   nview.t = pnp_withR (inlierPt3d, inlierPt2d, K, nview.R);
   cout<<nview.frameId-prev.frameId<<" frames,"<<"Scale (2pt_alg)="
      <<cv::norm(-nview.R.t()*nview.t + prev.R.t()*prev.t)<<", "
      <<maxInliers_R.size()<<"/"<<pt3d.size()<<endl;
#endif

   // ---- plot epipoles ----
   cv::Point2d ep1 = mat2cvpt(K*R.t()*t);
   cv::Point2d ep2 = mat2cvpt(K*t);

   prev.epipoleA = ep1;
   nview.epipoleB = ep2;
   double epNeibRds = 0.0 * IDEAL_IMAGE_WIDTH/640;
#ifndef HIGH_SPEED_NO_GRAPHICS
   cv::circle(canv1, ep1, epNeibRds, cv::Scalar(0,0,100), 1);
   cv::circle(canv2, ep2, epNeibRds, cv::Scalar(0,0,100), 1);
#endif


   Matrix3d Rx;
   Rx << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
      R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
      R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2);
   Quaterniond q(Rx);
   angVel = 2*acos(q.w()) * 180/PI / ((nview.frameId - prev.frameId)/fps);
   nview.angVel = angVel;

   if (rotateMode()) {
      SIFT_THRESH = SIFT_THRESH_LOW;
      sortbyPrlx = false; // because of more sift pts, new 3d pts should distribute more evenly
   } else if(2*acos(q.w()) * 180/PI > 10) {
      parallaxThresh = parallaxThresh * 1;
      parallaxDegThresh = parallaxDegThresh * 1;
      SIFT_THRESH = SIFT_THRESH_HIGH;
   } else {
      SIFT_THRESH = SIFT_THRESH_HIGH;
   }

   // ----- sort point matches by parallax -----
   if(sortbyPrlx) {
      vector<valIdxPair> prlxVec;
      for(int i=0; i<featPtMatches.size(); ++i) {
         valIdxPair pair;
         pair.first = compParallax (featPtMatches[i][0], featPtMatches[i][1], K,
               cv::Mat::eye(3,3,CV_64F), R);
         pair.second = i;
         prlxVec.push_back(pair);
      }
      sort(prlxVec.begin(), prlxVec.end(), comparator_valIdxPair);
      vector<vector<int>>  copyPairIdx = pairIdx;
      vector<vector<cv::Point2d>> copyFeatPtMatches = featPtMatches;
      pairIdx.clear();
      featPtMatches.clear();
      for(int i=prlxVec.size()-1; i >= 0; --i) {
         pairIdx.push_back(copyPairIdx[prlxVec[i].second]);
         featPtMatches.push_back(copyFeatPtMatches[prlxVec[i].second]);
      }
   }

   // --- set up 3d keypoints and point tracks ---
   vector<int> view_point;
   for (int i=0; i < featPtMatches.size(); ++i) {
      view_point.clear();
      int ptGid = prev.featurePoints[pairIdx[i][0]].gid;
      nview.featurePoints[pairIdx[i][1]].gid = ptGid;
      if ( ptGid >= 0 && keyPoints[ptGid].is3D ) { // points represented in 3D form
         view_point.push_back(nview.id);
         view_point.push_back(pairIdx[i][1]);
         keyPoints[ptGid].viewId_ptLid.push_back(view_point);
      }
   }

   int new3dPtNum = 0, new2_3dPtNum = 0;
   for (int i=0; i < featPtMatches.size(); ++i) {
      view_point.clear();
      int ptGid = prev.featurePoints[pairIdx[i][0]].gid;
      nview.featurePoints[pairIdx[i][1]].gid = ptGid;
      if ( ptGid >= 0 && keyPoints[ptGid].is3D ) { // points represented in 3D form
      } else if (ptGid >= 0 && !keyPoints[ptGid].is3D) {// point exist as 2D track
      } else { // newly found points
         double parallax = compParallax (featPtMatches[i][0], featPtMatches[i][1], K,cv::Mat::eye(3,3,CV_64F), R);
         double parallaxDeg = compParallaxDeg (featPtMatches[i][0], featPtMatches[i][1], K,cv::Mat::eye(3,3,CV_64F), R);
         cv::Mat Xw =  triangulatePoint_nonlin (prev.R, prev.t, nview.R, nview.t, K,
               featPtMatches[i][0], featPtMatches[i][1]);
         cv::Mat Xc = triangulatePoint (cv::Mat::eye(3,3,CV_64F), cv::Mat::zeros(3,1,CV_64F),
               R, t, K,	featPtMatches[i][0], featPtMatches[i][1]);

         cv::Scalar color(rand()%255,rand()%255,rand()%255,0);
#ifndef HIGH_SPEED_NO_GRAPHICS
         if (parallaxDeg > parallaxDegThresh) {
            cv::circle(canv1, featPtMatches[i][0], 2, color, 2);
            cv::circle(canv2, featPtMatches[i][1], 2, color, 2);
         }
#endif
         if(parallaxDeg < parallaxDegThresh
               || (cv::norm(Xw+nview.R.t()*nview.t) > mfgSettings->getDepthLimit() &&
                  cv::norm(Xc) > mfgSettings->getDepthLimit() * 1) // depth too large
               || new3dPtNum > maxNumNew3dPt
               || cv::norm(featPtMatches[i][1] - ep2) < epNeibRds
           )
         {
            // add as 2d track
            KeyPoint3d kp;
            kp.gid = keyPoints.size();
            prev.featurePoints[pairIdx[i][0]].gid = kp.gid;
            nview.featurePoints[pairIdx[i][1]].gid = kp.gid;
            view_point.clear();
            view_point.push_back(prev.id);
            view_point.push_back(pairIdx[i][0]);
            kp.viewId_ptLid.push_back(view_point);
            view_point.clear();
            view_point.push_back(nview.id);
            view_point.push_back(pairIdx[i][1]);
            kp.viewId_ptLid.push_back(view_point);
            kp.is3D = false;
            keyPoints.push_back(kp);
         } else { // add to MFG as 3D point
            if(!checkCheirality(prev.R, prev.t,Xw) || !checkCheirality(nview.R, nview.t,Xw))
               continue;
            // set up 3d keypoints and 3d-2d correspondence
            KeyPoint3d kp(Xw.at<double>(0),Xw.at<double>(1),Xw.at<double>(2));
            kp.gid = keyPoints.size();
            kp.is3D = true;
            kp.estViewId = nview.id;
            prev.featurePoints[pairIdx[i][0]].gid = kp.gid;
            nview.featurePoints[pairIdx[i][1]].gid = kp.gid;
            view_point.clear();
            view_point.push_back(prev.id);
            view_point.push_back(pairIdx[i][0]);
            kp.viewId_ptLid.push_back(view_point);
            view_point.clear();
            view_point.push_back(nview.id);
            view_point.push_back(pairIdx[i][1]);
            kp.viewId_ptLid.push_back(view_point);
            keyPoints.push_back(kp);
#ifndef HIGH_SPEED_NO_GRAPHICS
            cv::circle(canv1, featPtMatches[i][0], 7, color, 1);
            cv::circle(canv2, featPtMatches[i][1], 7, color, 1);
            cv::putText(canv1, " "+num2str(cv::norm(Xw+nview.R.t()*nview.t)), featPtMatches[i][0],
                  1, 1, color);
#endif
            // to control pt feature set size
            new3dPtNum++;
         }
      }
   }
   MyTimer tm; tm.start();

   for (int i=0; i < featPtMatches.size(); ++i) {
      view_point.clear();
      int ptGid = prev.featurePoints[pairIdx[i][0]].gid;
      nview.featurePoints[pairIdx[i][1]].gid = ptGid;
      if ( ptGid >= 0 && keyPoints[ptGid].is3D ) { // points represented in 3D form
      }
      else if (ptGid >= 0 && !keyPoints[ptGid].is3D) {// point exist as 2D track
         cv::Scalar color(rand()%255,rand()%255,rand()%255,0);
         double parallax = compParallax (featPtMatches[i][0], featPtMatches[i][1], K, cv::Mat::eye(3,3,CV_64F), R);
         double parallaxDeg = compParallaxDeg(featPtMatches[i][0], featPtMatches[i][1], K, cv::Mat::eye(3,3,CV_64F), R);
#ifndef HIGH_SPEED_NO_GRAPHICS
         if (parallaxDeg > parallaxDegThresh) {
            cv::circle(canv1, featPtMatches[i][0], 2, color, 2);
            cv::circle(canv2, featPtMatches[i][1], 2, color, 2);
         }
#endif
         // --- 1) add current observation into map ---
         view_point.push_back(nview.id);
         view_point.push_back(pairIdx[i][1]);
         keyPoints[ptGid].viewId_ptLid.push_back(view_point);

         if(new2_3dPtNum >= maxNumNew2d_3dPt) continue;  //

         if( cv::norm(featPtMatches[i][0] - ep1) < epNeibRds   // too close to epipole
               || cv::norm(featPtMatches[i][1] - ep2) < epNeibRds )
            continue;

         // --- 2) check if a 2d track is ready to establish 3d pt ---
         // -- start a voting/ransac --
         cv::Mat bestKeyPt;
         vector<int> maxInlier;
         int obs_size = keyPoints[ptGid].viewId_ptLid.size();
         for (int p = 0; p < obs_size-1; ++p) {
            for(int q = obs_size - 1; q >= p+1 && q>obs_size-3; --q) {
               // try to triangulate every pair of 2d line
               int pVid = keyPoints[ptGid].viewId_ptLid[p][0],
                   pLid = keyPoints[ptGid].viewId_ptLid[p][1],
                   qVid = keyPoints[ptGid].viewId_ptLid[q][0],
                   qLid = keyPoints[ptGid].viewId_ptLid[q][1];
               cv::Point2d ppt = views[pVid].featurePoints[pLid].cvpt(),
                  qpt = views[qVid].featurePoints[qLid].cvpt();
               double pqPrlxDeg = compParallaxDeg(ppt, qpt, K,	views[pVid].R, views[qVid].R);
               if (pqPrlxDeg < parallaxDegThresh * 1.5) 	continue; // not usable
               cv::Mat ptpq = triangulatePoint_nonlin (views[pVid].R, views[pVid].t,
                     views[qVid].R, views[qVid].t, K, ppt, qpt);
               // -- check Cheirality
               if(!checkCheirality(views[pVid].R, views[pVid].t, ptpq))
                  continue;
               vector<int> inlier;
               // check if every 2d observation agrees with this 3d point
               for (int j=0; j < keyPoints[ptGid].viewId_ptLid.size(); ++j) {
                  int vid = keyPoints[ptGid].viewId_ptLid[j][0];
                  int lid = keyPoints[ptGid].viewId_ptLid[j][1];
                  cv::Point2d pt_j = mat2cvpt(K*(views[vid].R * ptpq + views[vid].t));
                  double dist = cv::norm(views[vid].featurePoints[lid].cvpt() - pt_j);
                  if( dist < 1.5) // inlier;
                  inlier.push_back(j);
               }
               if ( maxInlier.size() < inlier.size() ) {
                  maxInlier = inlier;
                  bestKeyPt = ptpq;
               }
            }
         }
         if ( maxInlier.size() < 3) {
            // due to outlier(s), not ready to establish 3d , append to 2d track
            continue;
         }
         // --- nonlinear estimation of 3d pt ---
         vector<cv::Mat> Rs, ts;
         vector<cv::Point2d> pt;
         for(int a=0; a < maxInlier.size(); ++a) {
            int vid = keyPoints[ptGid].viewId_ptLid[maxInlier[a]][0];
            int lid = keyPoints[ptGid].viewId_ptLid[maxInlier[a]][1];
            Rs.push_back(views[vid].R);
            ts.push_back(views[vid].t);
            pt.push_back(views[vid].featurePoints[lid].cvpt());
         }
         est3dpt (Rs, ts, K, pt, bestKeyPt); // nonlinear estimation
         //est3dpt_g2o (Rs, ts, K, pt, bestKeyPt); // nonlinear estimation
         if(cv::norm(bestKeyPt+nview.R.t()*nview.t) > mfgSettings->getDepthLimit() ) continue;

         // --- 3) establish 3d pt ---
         keyPoints[ptGid].is3D = true;
         keyPoints[ptGid].estViewId = nview.id;
         keyPoints[ptGid].x = bestKeyPt.at<double>(0);
         keyPoints[ptGid].y = bestKeyPt.at<double>(1);
         keyPoints[ptGid].z = bestKeyPt.at<double>(2);
#ifndef HIGH_SPEED_NO_GRAPHICS
         cv::circle(canv1, featPtMatches[i][0], 7, color, 2);
         cv::circle(canv2, featPtMatches[i][1], 7, color, 2);
         cv::putText(canv1, " "+num2str(cv::norm(bestKeyPt+nview.R.t()*nview.t)), featPtMatches[i][0],
               1, 1, color);
#endif
         new2_3dPtNum ++;
      }
   }
   tm.end(); cout<<"2d->3d: "<<tm.time_ms<<endl;
   cout<<"2d->3d pt: "<<new2_3dPtNum<<", add new 3d pt:"<<new3dPtNum<<endl;
#ifdef PREDICT_POSE_USE_R_AND_T
   // remove outliers
   for(int i=0; i < maxInliers_Rt.size(); ++i) {
      int j = maxInliers_Rt[i];
      int gid = prev.featurePoints[pairIdx[j][0]].gid;
      if ( gid < 0 ) continue;
      double s = cv::norm(keyPoints[gid].mat(0)+(prev.R.t()*prev.t))/cv::norm(localKeyPts[j].mat());

      if( abs(s - cv::norm(-nview.R.t()*nview.t + prev.R.t()*prev.t)) > 0.15 ) {
         // delete outliers
         keyPoints[gid].gid = -1;
         keyPoints[gid].is3D = false;
         for (int k=0; k < keyPoints[gid].viewId_ptLid.size(); ++k) {
            int vid = keyPoints[gid].viewId_ptLid[k][0];
            int lid = keyPoints[gid].viewId_ptLid[k][1];
            views[vid].featurePoints[lid].gid = -1;
         }
         keyPoints[gid].viewId_ptLid.clear();
      }
   }
#endif

   // ----- setup/update vanishing points -----
   bool isRgood;
   vector<vector<int>> vpPairIdx
      = matchVanishPts_withR(prev, nview, nview.R*prev.R.t(), isRgood);
   vector<int> vid_vpid;
   for(int i=0; i < vpPairIdx.size(); ++i) {
      vid_vpid.clear();
      if(prev.vanishPoints[vpPairIdx[i][0]].gid >= 0) { // pass correspondence on
         nview.vanishPoints[vpPairIdx[i][1]].gid = prev.vanishPoints[vpPairIdx[i][0]].gid;
         vid_vpid.push_back(nview.id);
         vid_vpid.push_back(vpPairIdx[i][1]);
         vanishingPoints[prev.vanishPoints[vpPairIdx[i][0]].gid].viewId_vpLid.push_back(vid_vpid);
      } else { // establish new vanishing point in 3d
         cv::Mat vp = prev.R.t() * K.inv() * prev.vanishPoints[vpPairIdx[i][0]].mat();
         vp = vp/cv::norm(vp);
         vanishingPoints.push_back(VanishPnt3d(vp.at<double>(0),vp.at<double>(1),vp.at<double>(2)));
         vanishingPoints.back().gid = vanishingPoints.size()-1;
         prev.vanishPoints[vpPairIdx[i][0]].gid = vanishingPoints.back().gid;
         nview.vanishPoints[vpPairIdx[i][1]].gid = vanishingPoints.back().gid;
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
   // now check if vp observed before previous frame
   for(int i=0; i < nview.vanishPoints.size(); ++i){
      if(nview.vanishPoints[i].gid >= 0) continue;
      cv::Mat vpi = nview.R.t()*K.inv() * nview.vanishPoints[i].mat(); // in world coord
      vpi = vpi*(1/cv::norm(vpi));
      // try matching with existing 3d vp
      for (int j=0; j < vanishingPoints.size(); ++j) {// existing vp 3d
         cv::Mat vpj = vanishingPoints[j].mat(0)/cv::norm(vanishingPoints[j].mat(0));
         if(abs(vpi.dot(vpj)) > cos(mfgSettings->getVPointAngleThresh() *PI/180) ) {// degree
            // matched
            nview.vanishPoints[i].gid = vanishingPoints[j].gid;
            vector<int> vid_vpid;
            vid_vpid.push_back(nview.id);
            vid_vpid.push_back(nview.vanishPoints[i].lid);
            vanishingPoints[nview.vanishPoints[i].gid].viewId_vpLid.push_back(vid_vpid);
            //		cout<<"local vp "<<i<<" matched to 3d vp "<<j<<endl;
            break;
         }
      }
   }

   // ----- match ideal lines -----
   vector<vector<int>>			ilinePairIdx;
   matchIdealLines(prev, nview, vpPairIdx, featPtMatches, F, ilinePairIdx, 1);
   update3dIdealLine(ilinePairIdx, nview);
   updatePrimPlane();
   // ---- write to images for debugging ----
#ifndef HIGH_SPEED_NO_GRAPHICS
   cv::imwrite("./tmpimg/"+num2str(views.back().id)+"_pt1.jpg", canv1);
   cv::imwrite("./tmpimg/"+num2str(views.back().id)+"_pt2.jpg", canv2);
#endif
   views.back().drawAllLineSegments(true);

   //	showImage("match1", &canv1);
   //	showImage("match2" ,&canv2);

   //	cv::waitKey();
}

void Mfg::update3dIdealLine(vector<vector<int>> ilinePairIdx, View& nview)
   // ONLY establish a 3d line when it's been seen in 3 or more views
{
   View& prev = views[views.size()-2];
   double ilineLenLimit	= 10;
   double ilineDepthLimit	= 10;
   double threshPt2LnDist  = 2;  // endpt to reprojected line dist
   double epNeibRds = 60.0 * IDEAL_IMAGE_WIDTH/640;
   double parallaxThresh = THRESH_PARALLAX;
   double parallaxAngleThresh = THRESH_PARALLAX_DEGREE;

   if (rotateMode()) {
      parallaxThresh = parallaxThresh * 1.5;
   }
#ifndef HIGH_SPEED_NO_GRAPHICS
   cv::Mat canv1 = prev.img.clone(), canv2 = nview.img.clone();
#endif
   vector<int> vid_lid;
   for(int i=0; i < ilinePairIdx.size(); ++i) {
      int lnGid = prev.idealLines[ilinePairIdx[i][0]].gid;  // line gid from prev view
      nview.idealLines[ilinePairIdx[i][1]].gid = lnGid;     // pass to current view
      nview.idealLines[ilinePairIdx[i][1]].pGid = prev.idealLines[ilinePairIdx[i][0]].pGid;

      cv::Scalar color(rand()%255,rand()%255,rand()%255,0);
      int linewidth = 0;

      if (lnGid < 0)	{ // not existent in map, setup a new 2d track, not a 3d line yet
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
      } else { //existent 3D line or 2D line track
         // --- 1) add current observation into map ---
         vid_lid.clear();
         vid_lid.push_back(nview.id);
         vid_lid.push_back(ilinePairIdx[i][1]);
         idealLines[lnGid].viewId_lnLid.push_back(vid_lid);

         if ( idealLines[lnGid].is3D )
            linewidth = 1; // for debugging plotting purpose

         //--- avoid those close to epipole ---
         if( point2LineDist(prev.idealLines[ilinePairIdx[i][0]].lineEq(), prev.epipoleA) < epNeibRds
               || point2LineDist(nview.idealLines[ilinePairIdx[i][1]].lineEq(), nview.epipoleB) < epNeibRds)
            continue;
         // --- 2) check if a 2d track is ready to establish 3d line ---
         if (!idealLines[lnGid].is3D) {
            // -- start a voting/ransac --
            IdealLine3d bestLine;
            int maxNumInlier = 0;
            for (int p = 0; p < idealLines[lnGid].viewId_lnLid.size()-1; ++p) {
               for(int q = p+1; q < idealLines[lnGid].viewId_lnLid.size(); ++q) {
                  // try to triangulate every pair of 2d line
                  int pVid = idealLines[lnGid].viewId_lnLid[p][0],
                        pLid = idealLines[lnGid].viewId_lnLid[p][1],
                        qVid = idealLines[lnGid].viewId_lnLid[q][0],
                        qLid = idealLines[lnGid].viewId_lnLid[q][1];
                  IdealLine2d pln = views[pVid].idealLines[pLid],
                              qln = views[qVid].idealLines[qLid];
                  double pqPrlx = compParallax(pln, qln, K, views[pVid].R, views[qVid].R);
                  if (pqPrlx < parallaxThresh * 1.5) 	continue; // not usable
                  // triangulate a 3d line
                  IdealLine3d lnpq = triangluateLine(views[pVid].R, views[pVid].t,
                        views[qVid].R, views[qVid].t, K, pln, qln);
                  // check Cheirality
                  if(!checkCheirality(views[pVid].R, views[pVid].t, lnpq) ||
                        !checkCheirality(views[qVid].R, views[qVid].t, lnpq))
                     continue;
                  // ignore too-long line ( high possible false matching)
                  if( lnpq.length > ilineLenLimit )
                     continue;
                  int numInlier = 0;
                  //check if every 2d observation agrees with this 3d line
                  for (int j=0; j < idealLines[lnGid].viewId_lnLid.size(); ++j) {
                     int vid = idealLines[lnGid].viewId_lnLid[j][0];
                     int lid = idealLines[lnGid].viewId_lnLid[j][1];
                     cv::Mat ln2d = projectLine(lnpq, views[vid].R, views[vid].t, K);
                     double sumDist = 0;
                     for(int k=0; k < views[vid].idealLines[lid].lsEndpoints.size(); ++k) {
                        sumDist += point2LineDist(ln2d, views[vid].idealLines[lid].lsEndpoints[k]);
                     }
                     if(sumDist/views[vid].idealLines[lid].lsEndpoints.size() < threshPt2LnDist) // inlier;
                     numInlier++;
                  }
                  if ( maxNumInlier < numInlier ) {
                     maxNumInlier = numInlier;
                     bestLine = lnpq;
                  }
               }
            }
            if ( maxNumInlier < 3) { // due to outlier(s), not ready to establish 3d line, append to 2d track
               continue;
            }
            cv::Point3d curCamPos = mat2cvpt3d(-views.back().R.t()*views.back().t);
            if (cv::norm(projectPt3d2Ln3d (bestLine, curCamPos) - curCamPos) > ilineDepthLimit)
               continue;

            // --- 3) establish a new 3d line in mfg ---
            idealLines[lnGid].midpt = bestLine.midpt;
            idealLines[lnGid].direct = bestLine.direct.clone();
            idealLines[lnGid].length = bestLine.length;
            idealLines[lnGid].is3D = true;
            idealLines[lnGid].pGid = prev.idealLines[ilinePairIdx[i][0]].pGid;
            idealLines[lnGid].estViewId = nview.id;
            linewidth = 2; // 2d to 3d conversion
         }
      }
#ifndef HIGH_SPEED_NO_GRAPHICS
      if(linewidth > 0) {
         cv::line(canv1, prev.idealLines[ilinePairIdx[i][0]].extremity1, prev.idealLines[ilinePairIdx[i][0]].extremity2, color, linewidth);
         cv::line(canv2, nview.idealLines[ilinePairIdx[i][1]].extremity1, nview.idealLines[ilinePairIdx[i][1]].extremity2, color, linewidth);
      }
#endif
   }
#ifndef HIGH_SPEED_NO_GRAPHICS
#ifndef USE_PT_ONLY_FOR_LBA
   cv::imwrite("./tmpimg/ln_"+num2str(views.back().id)+"_1.jpg", canv1);
   cv::imwrite("./tmpimg/ln_"+num2str(views.back().id)+"_2.jpg", canv2);
#endif
#endif
}

// primary plane update:
// 1. assosciate new points or lines to exiting planes
// 2. use 3d key points and ideal lines to discover new planes
void Mfg::updatePrimPlane()
{
   // ======== 0. set prameters/thresholds ========
   double pt2PlaneDistThresh = mfgSettings->getMfgPointToPlaneDistance();
   // ======== 1. check if any newly added points/lines belong to existing planes =======
   // ---- 1.1 check points ----
   for (int i=0; i<keyPoints.size(); ++i) {
      if (!keyPoints[i].is3D || keyPoints[i].gid < 0) continue;
      if (keyPoints[i].estViewId < views.back().id) continue;// not newly added feature
      double	minDist = 1e6; //initialized to be very large
      int		minIdx = -1;
      for (int j=0; j<primaryPlanes.size(); ++j) {
         if(views.back().id - primaryPlanes[j].recentViewId > 5) continue; // obsolete, not use
         double dist = abs(keyPoints[i].mat(0).dot(primaryPlanes[j].n)+primaryPlanes[j].d)
            /cv::norm(primaryPlanes[j].n);
         if (dist < minDist) {
            minDist = dist;
            minIdx = j;
         }
      }
      if ( minIdx >=0 && minDist <= pt2PlaneDistThresh) {// pt-plane distance under threshold
         keyPoints[i].pGid = primaryPlanes[minIdx].gid;
         primaryPlanes[minIdx].kptGids.push_back(keyPoints[i].gid); // connect pt and plane
         primaryPlanes[minIdx].recentViewId = views.back().id;
      }
   }
   // ---- 1.2 check lines ----
   for (int i=0; i<idealLines.size(); ++i) {
      if(!idealLines[i].is3D || idealLines[i].gid < 0) continue;
      if(idealLines[i].estViewId < views.back().id) continue;
      double	minDist = 1e6; //initialized to be very large
      int		minIdx = -1;
      for (int j=0; j<primaryPlanes.size(); ++j) {
         if(views.back().id - primaryPlanes[j].recentViewId > 5) continue; // obsolete
         double dist = (abs(cvpt2mat(idealLines[i].extremity1(),0).dot(primaryPlanes[j].n)+primaryPlanes[j].d)
               + abs(cvpt2mat(idealLines[i].extremity2(),0).dot(primaryPlanes[j].n)+primaryPlanes[j].d))
            /(2*cv::norm(primaryPlanes[j].n));
         if (dist < minDist) {
            minDist = dist;
            minIdx = j;
         }
      }
      if ( minIdx >=0 && minDist <= pt2PlaneDistThresh) {// line-plane distance under threshold
         idealLines[i].pGid = primaryPlanes[minIdx].gid;
         primaryPlanes[minIdx].ilnGids.push_back(idealLines[i].gid); // connect line and plane
         primaryPlanes[minIdx].recentViewId = views.back().id;
      }
   }

   // ========== 2. discover new planes (using seq-ransac) =========
   vector<KeyPoint3d> lonePts; // used for finding new planes
   int ptNumLimit = mfgSettings->getMfgNumRecentPoints();
   for(int i=keyPoints.size()-1; i>=0; --i) {
      if (!keyPoints[i].is3D || keyPoints[i].gid < 0 || keyPoints[i].pGid >= 0) continue;
      lonePts.push_back(keyPoints[i]);
      if(lonePts.size()>ptNumLimit) break;
   }
   vector<IdealLine3d> loneLns;
   int lnNumLimit = mfgSettings->getMfgNumRecentLines();
   for (int i=idealLines.size()-1; i>=0; --i) {
      if(!idealLines[i].is3D || idealLines[i].gid<0 || idealLines[i].pGid >= 0) continue;
      loneLns.push_back(idealLines[i]);
      if(loneLns.size() > lnNumLimit) break;
   }

   vector<vector<int>> ptIdxGroups, lnIdxGroups;
   vector<cv::Mat> planeVecs;
   //	find3dPlanes_pts (lonePts, ptIdxGroups,planeVecs);
   find3dPlanes_pts_lns_VPs (lonePts, loneLns, vanishingPoints, ptIdxGroups, lnIdxGroups, planeVecs);
   for(int i=0; i<ptIdxGroups.size(); ++i) {
      int newPlaneGid = primaryPlanes.size();
      for(int j=0; j<ptIdxGroups[i].size(); ++j) {
         keyPoints[ptIdxGroups[i][j]].pGid = newPlaneGid;
      }
      if(lnIdxGroups.size() > i) {
         for(int j=0; j<lnIdxGroups[i].size(); ++j) {
            idealLines[lnIdxGroups[i][j]].pGid = newPlaneGid;
         }
         cout<<"Add plane "<<newPlaneGid<<" with "<<ptIdxGroups[i].size()<<'\t'<<lnIdxGroups[i].size()<<endl;
      }
      primaryPlanes.push_back(PrimPlane3d(planeVecs[i],newPlaneGid));// need compute plane equation
      primaryPlanes.back().kptGids = ptIdxGroups[i];
      primaryPlanes.back().ilnGids = lnIdxGroups[i];
      primaryPlanes.back().estViewId = views.back().id;
      primaryPlanes.back().recentViewId = views.back().id;
   }

}

void Mfg::draw3D() const
{
   // plot first camera, small
   glLineWidth(1);
   glBegin(GL_LINES);
   glColor3f(1,0,0); // x-axis
   glVertex3f(0,0,0);
   glVertex3f(1,0,0);
   glColor3f(0,1,0);
   glVertex3f(0,0,0);
   glVertex3f(0,1,0);
   glColor3f(0,0,1);// z axis
   glVertex3f(0,0,0);
   glVertex3f(0,0,1);
   glEnd();

   cv::Mat xw = (cv::Mat_<double>(3,1)<< 0.5,0,0),
      yw = (cv::Mat_<double>(3,1)<< 0,0.5,0),
      zw = (cv::Mat_<double>(3,1)<< 0,0,0.5);

   for (int i=1; i<views.size(); ++i) {
      if(!(views[i].R.dims==2)) continue; // handle the in-process view
      cv::Mat c = -views[i].R.t()*views[i].t;
      cv::Mat x_ = views[i].R.t() * (xw-views[i].t),
         y_ = views[i].R.t() * (yw-views[i].t),
         z_ = views[i].R.t() * (zw-views[i].t);
      glBegin(GL_LINES);

      glColor3f(1,0,0);
      glVertex3f(c.at<double>(0),c.at<double>(1),c.at<double>(2));
      glVertex3f(x_.at<double>(0),x_.at<double>(1),x_.at<double>(2));
      glColor3f(0,1,0);
      glVertex3f(c.at<double>(0),c.at<double>(1),c.at<double>(2));
      glVertex3f(y_.at<double>(0),y_.at<double>(1),y_.at<double>(2));
      glColor3f(0,0,1);
      glVertex3f(c.at<double>(0),c.at<double>(1),c.at<double>(2));
      glVertex3f(z_.at<double>(0),z_.at<double>(1),z_.at<double>(2));
      glEnd();
   }

   /*	// plot mfg planes
      glBegin(GL_LINES);
      glColor3f((rand()%100)/100.0,(rand()%100)/100.0,(rand()%100)/100.0);

      for(int i =0; i<views[0].idealLines.size(); ++i)
      {
      if (views[0].idealLines[i].pGid <0) continue;
      cv::Mat edp1, edp2;
      projectImgPt2Plane(cvpt2mat(views[0].idealLines[i].extremity1),
      primaryPlanes[views[0].idealLines[i].pGid], K, edp1);
      projectImgPt2Plane(cvpt2mat(views[0].idealLines[i].extremity2),
      primaryPlanes[views[0].idealLines[i].pGid], K, edp2);
      glVertex3f(edp1.at<double>(0),edp1.at<double>(1),edp1.at<double>(2));
      glVertex3f(edp2.at<double>(0),edp2.at<double>(1),edp2.at<double>(2));
      }
      glEnd();
      */

   glPointSize(3.0);
   glBegin(GL_POINTS);
   for (int i=0; i<keyPoints.size(); ++i){
      if(!keyPoints[i].is3D || keyPoints[i].gid<0) continue;
      if(keyPoints[i].pGid < 0) // red
         glColor3f(0.6, 0.6, 0.6);
      else { // coplanar green
         glColor3f(planeColors[keyPoints[i].pGid][0],
               planeColors[keyPoints[i].pGid][1],planeColors[keyPoints[i].pGid][2]);
      }
      glVertex3f(keyPoints[i].x, keyPoints[i].y, keyPoints[i].z);
   }
   glEnd();

#ifndef USE_PT_ONLY_FOR_LBA
   glColor3f(0,1,1);
   glLineWidth(2);
   glBegin(GL_LINES);
   for(int i=0; i<idealLines.size(); ++i) {
      if(!idealLines[i].is3D || idealLines[i].gid<0) continue;
      /*	if(idealLines[i].vpGid == 0)
         glColor3f(1,0,0);
         else if (idealLines[i].vpGid == 1)
         glColor3f(0,1,0);
         else if (idealLines[i].vpGid == 2)
         glColor3f(0,0,1);
         else
         glColor3f(0,0,0);
         */
      if(idealLines[i].pGid < 0) {
         glColor3f(0,0,0);
      } else {
         glColor3f(planeColors[idealLines[i].pGid][0],
               planeColors[idealLines[i].pGid][1],planeColors[idealLines[i].pGid][2]);
      }

      glVertex3f(idealLines[i].extremity1().x,idealLines[i].extremity1().y,idealLines[i].extremity1().z);
      glVertex3f(idealLines[i].extremity2().x,idealLines[i].extremity2().y,idealLines[i].extremity2().z);
   }
   glEnd();
#endif
}

void Mfg::adjustBundle()
{
   int numPos = 8, numFrm = 10;
   if (views.size() <=numFrm ){
      numPos = numFrm;
   }
#ifdef	USE_PT_ONLY_FOR_LBA
   //adjustBundle_Pt_Rel(numPos, numFrm);
   adjustBundle_Pt_G2O(numPos, numFrm);
#else
   //adjustBundle_PtLnVp_Rel(numPos, numFrm);
   adjustBundle_G2O(numPos, numFrm);

#endif
   // write to file
   exportCamPose (*this, "camPose.txt");
   exportMfgNode (*this, "mfgNode.txt");
}


void Mfg::detectLnOutliers(double threshPt2LnDist )
   //
{
   double ilineLenLimit	= 10;
   // delete too-far and too-long lines
   for (int i=0; i < idealLines.size(); ++i) {
      if(!idealLines[i].is3D || idealLines[i].gid<0) continue;   // only handle 3d lines
      if(idealLines[i].length > ilineLenLimit)			// delete 3d
      {
         idealLines[i].gid = -1;
         idealLines[i].is3D = false;
         // remove 2d feature's info
         for (int j=0; j < idealLines[i].viewId_lnLid.size(); ++j) {
            int vid = idealLines[i].viewId_lnLid[j][0];
            int lid = idealLines[i].viewId_lnLid[j][1];
            views[vid].idealLines[lid].gid = -1;
         }
         idealLines[i].viewId_lnLid.clear();
         //	cout<< "3d line deleted, reason: too long" <<endl;
      }
   }

   // ----- line reprojection -----

   for (int i=0; i < idealLines.size(); ++i) {
      if(!idealLines[i].is3D || idealLines[i].gid<0) continue;
      if(idealLines[i].viewId_lnLid.size() == 3 && views.back().id == idealLines[i].viewId_lnLid.back()[0]) continue;
      for(int j=0; j < idealLines[i].viewId_lnLid.size(); ++j) { // for each 2d line
         int vid = idealLines[i].viewId_lnLid[j][0];
         int lid = idealLines[i].viewId_lnLid[j][1];
         cv::Mat lneq = projectLine(idealLines[i], views[vid].R, views[vid].t, K);
         double sumDist = 0;
         for(int k=0; k < views[vid].idealLines[lid].lsEndpoints.size(); ++k) {
            sumDist += point2LineDist(lneq, views[vid].idealLines[lid].lsEndpoints[k]);
         }
         if(sumDist/views[vid].idealLines[lid].lsEndpoints.size() > threshPt2LnDist) { // outlier;
            views[vid].idealLines[lid].gid = -1;
            idealLines[i].viewId_lnLid.erase(idealLines[i].viewId_lnLid.begin()+j); // delete
            --j;
            //		cout<<"2d line outlier found:"<< sumDist/views[vid].idealLines[lid].lsEndpoints.size()
            //			<<", fid="<<vid<<", lngid=" << i<<endl;
         }
      }
      if (idealLines[i].viewId_lnLid.size() < 3 // delete from 3d map/mfg
            || ( idealLines[i].viewId_lnLid.size() == 3 &&
               abs(views.back().id - idealLines[i].viewId_lnLid.back()[0]) >= 1) ) {
         idealLines[i].gid = -1;
         idealLines[i].is3D = false;
         for (int j=0; j < idealLines[i].viewId_lnLid.size(); ++j) {
            int vid = idealLines[i].viewId_lnLid[j][0];
            int lid = idealLines[i].viewId_lnLid[j][1];
            views[vid].idealLines[lid].gid = -1;
         }
         idealLines[i].viewId_lnLid.clear();
         //	cout<< "3d line deleted, reason: too few observations" <<endl;
      }
   }

}

void Mfg::detectPtOutliers(double threshPt2PtDist)
   //
{
   //	double threshPt2PtDist = 2;
   double rangeLimit = 30;
   for (int i=0; i < keyPoints.size(); ++i) {
      if(!keyPoints[i].is3D || keyPoints[i].gid<0) continue;
      // delete too-far points
      double minD = 100;
      for(int j=0; j < keyPoints[i].viewId_ptLid.size(); ++j) {
         int vid = keyPoints[i].viewId_ptLid[j][0];
         double dist = cv::norm(keyPoints[i].mat(0) + views[vid].R.t()*views[vid].t);
         if (dist < minD)
            minD = dist;
      }
      if (minD > rangeLimit) { //delete
         keyPoints[i].is3D = false;
         keyPoints[i].gid = -1;
         for (int j=0; j < keyPoints[i].viewId_ptLid.size(); ++j) {
            int vid = keyPoints[i].viewId_ptLid[j][0];
            int lid = keyPoints[i].viewId_ptLid[j][1];
            views[vid].featurePoints[lid].gid = -1;
         }
         keyPoints[i].viewId_ptLid.clear();
         //	cout<< "3d point deleted (too far):" << i <<"'''''''''''''''" <<endl;
      }
   }
   int n_del = 0;
   for (int i=0; i < keyPoints.size(); ++i) {
      if(!keyPoints[i].is3D || keyPoints[i].gid<0) continue;
      //	if (keyPoints[i].viewId_ptLid.size()<5) continue;  // allow some time to ajust position via lba
      cv::Mat pt3d = keyPoints[i].mat(0);

      for(int j=0; j < keyPoints[i].viewId_ptLid.size(); ++j) { // for each
         int vid = keyPoints[i].viewId_ptLid[j][0];
         int lid = keyPoints[i].viewId_ptLid[j][1];
         double dist = cv::norm(mat2cvpt(K*(views[vid].R*pt3d + views[vid].t))-views[vid].featurePoints[lid].cvpt());
         if(dist > threshPt2PtDist) { // outlier;
            views[vid].featurePoints[lid].gid = -1;
            keyPoints[i].viewId_ptLid.erase(keyPoints[i].viewId_ptLid.begin()+j); // delete
            --j;
            //	cout<<"2d pt outlier found:"<< dist <<", remaining "<<keyPoints[i].viewId_ptLid.size()<<endl;
         }
      }
      if (keyPoints[i].viewId_ptLid.size() < 2
            || ( keyPoints[i].viewId_ptLid.size() == 22 &&
               abs(views.back().id - keyPoints[i].viewId_ptLid.back()[0]) >= 1) ) { // delete from 3d map/mfg
         keyPoints[i].gid = -1;
         keyPoints[i].is3D = false;
         for (int j=0; j < keyPoints[i].viewId_ptLid.size(); ++j) {
            int vid = keyPoints[i].viewId_ptLid[j][0];
            int lid = keyPoints[i].viewId_ptLid[j][1];
            views[vid].featurePoints[lid].gid = -1;
         }
         keyPoints[i].viewId_ptLid.clear();
         n_del++;
         //	cout<< "3d point deleted (too few observations):" << i <<"'''''''''''''''" <<endl;
      }
   }
   // detect newly established points when baseline too short
   for (int i=0; i < keyPoints.size(); ++i) {
      if(!keyPoints[i].is3D || keyPoints[i].gid<0) continue;
      if(views.size()<2) continue;
      for(int j=1; j<3; ++j) {
         if (views.size()-j-1 < 0) break;
         if(cv::norm(-views[views.size()-j].R.t()*views[views.size()-j].t
                  + views[views.size()-j-1].R.t()*views[views.size()-j-1].t) < 0.02
               && keyPoints[i].estViewId == views[views.size()-j].id) { // cancel these new 3d pts
            keyPoints[i].is3D = false;
            n_del++;
         }
      }
   }
   cout<<"delete 3d pts : "<<n_del<<endl;
}

bool Mfg::rotateMode ()
{
   double avThresh = 10; // angular velo degree/sec
   int	   winLen = 5;
   bool mode = false;
   for(int i=0; i< winLen; ++i) {
      int id = views.size()-1-i;
      if (id<1)
         break;
      else {
         cv::Mat R = views[id-1].R.inv()*views[id].R;
         double angle = acos(abs((R.at<double>(0,0) + R.at<double>(1,1) + R.at<double>(2,2) - 1)/2));
         if(views[id].angVel > avThresh ||
               angle > 10*PI/180) {
            mode = true;
            break;
         }
      }
   }

   return mode;
}
