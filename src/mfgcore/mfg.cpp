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

#define THRESH_PARALLAX 7 //(12.0*IDEAL_IMAGE_WIDTH/1000.0)
#define THRESH_PARALLAX_DEGREE (0.9)


extern double THRESH_POINT_MATCH_RATIO, SIFT_THRESH, SIFT_THRESH_HIGH, SIFT_THRESH_LOW;
extern int IDEAL_IMAGE_WIDTH;
extern MfgSettings* mfgSettings;
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

Mfg::Mfg(View v0, int ini_incrt, cv::Mat dc) 
{
   views.push_back(v0);   
   // tracking 
   K = v0.K;
   vector<int> v0_idx; //the local id of pts in last keyframe v0 that has been tracked using LK
   vector<cv::Point2f> prev_pts;

   for(int i=0; i < v0.featurePoints.size(); ++i) {
      prev_pts.push_back(cv::Point2f(v0.featurePoints[i].x,v0.featurePoints[i].y));
      v0_idx.push_back(v0.featurePoints[i].lid);
   }
   cv::Mat prev_img = v0.grayImg;

   string imgName = v0.filename;
   for(int idx = 0; idx < ini_incrt; ++idx) {
      imgName = nextImgName(imgName, 5, 1);
      cv::Mat oriImg = cv::imread(imgName,1);      
      if ( cv::norm(dc) > 1e-6) {
         cv::undistort(oriImg, oriImg, K, dc);
      }
   // resize image and ajust camera matrix
      if (mfgSettings->getImageWidth() > 1) {
         double scl = mfgSettings->getImageWidth()/double(oriImg.cols);
         cv::resize(oriImg,oriImg,cv::Size(),scl,scl,cv::INTER_AREA);     
      }
      cv::Mat grayImg;
      if (oriImg.channels()==3)
         cv::cvtColor(oriImg, grayImg, CV_RGB2GRAY);
      else
         grayImg = oriImg;

      vector<cv::Point2f> curr_pts, tracked_pts;
      vector<int> tracked_idx;
      vector<uchar> status;
      vector<float> err;

      cv::calcOpticalFlowPyrLK(
              prev_img, grayImg, // 2 consecutive images
              prev_pts, // input point positions in first im
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
         if(status[i]) { // match found
            tracked_idx.push_back(v0_idx[i]);
            tracked_pts.push_back(curr_pts[i]);
         }
      }
      prev_pts = tracked_pts;
      v0_idx = tracked_idx;
      prev_img = grayImg.clone();
      
   } 

   vector<vector<cv::Point2d>> featPtMatches;
   vector<vector<int>> pairIdx;
    // ----- setting parameters -----
   double distThresh_PtHomography = 1;
   double vpLnAngleThrseh = 50; // degree, tolerance angle between 3d line direction and vp direction
   double maxNoNew3dPt = 400;
    cv::Mat F, E, R, t;

   View v1(imgName, K, dc, 1, mfgSettings);
   v1.featurePoints.clear();

   for(int i=0; i<prev_pts.size();++i) {
      v1.featurePoints.push_back(FeatPoint2d(prev_pts[i].x,prev_pts[i].y, i));
      vector<cv::Point2d> match;
      match.push_back(v0.featurePoints[v0_idx[i]].cvpt());
      match.push_back(v1.featurePoints.back().cvpt());
      featPtMatches.push_back(match);
      vector<int> pair;
      pair.push_back(v0_idx[i]);
      pair.push_back(i);
      pairIdx.push_back(pair);
   }
   views.push_back(v1);
   View& view1 = views[1];
   View& view0 = views[0];
   view1.frameId = atoi (imgName.substr(imgName.size()-5-4, 5).c_str());
  
   computeEpipolar (featPtMatches, pairIdx, K, F, R, E, t, true);
   cout<<"R="<<R<<endl<<"t="<<t<<endl;
   v1.t_loc = t;
   bool isRgood = true;
   vector<vector<int>>        vpPairIdx;
   vector<vector<int>>        ilinePairIdx;
   vpPairIdx = matchVanishPts_withR(v0, v1, R, isRgood);

   vector<KeyPoint3d> tmpkp; // temporary keypoints
   for(int i=0; i < featPtMatches.size(); ++i) {
      cv::Mat X = triangulatePoint_nonlin (cv::Mat::eye(3,3,CV_64F), cv::Mat::zeros(3,1,CV_64F),
            R, t, K, featPtMatches[i][0], featPtMatches[i][1]);
      tmpkp.push_back(KeyPoint3d(X.at<double>(0),X.at<double>(1),X.at<double>(2)));
   }
   cv::Point2d ep1 = mat2cvpt(K*R.t()*t);
   cv::Point2d ep2 = mat2cvpt(K*t);

   view0.epipoleA = ep1;
   view1.epipoleB = ep2;

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
      VanishPnt3d vp(tmpvp.at<double>(0), tmpvp.at<double>(1),
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

   cv::Mat canv1, canv2;
   // ----- set up 3D lines (only coplanar ones) -----
   for(int i=0; i < ilinePairIdx.size(); ++i) {
      double prlx = compParallax(view0.idealLines[ilinePairIdx[i][0]],
            view1.idealLines[ilinePairIdx[i][1]], K, view0.R, view1.R);
      if(prlx < THRESH_PARALLAX || view0.idealLines[ilinePairIdx[i][0]].pGid < 0) { // set up 2d line track
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
#ifdef PLOT_MID_RESULTS
         cv::Scalar color(rand()%255,rand()%255,rand()%255,0);
         cv::line(canv1, a.extremity1, a.extremity2, color, 2);
         cv::line(canv2, b.extremity1, b.extremity2, color, 2);
#endif         
      }
   }

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
   cout<<endl<<" >>>>>>>>>> MFG initialized using two views <<<<<<<<<<"<<endl;

#ifdef USE_GROUND_PLANE
   need_scale_to_real = true;
   scale_since.push_back(1);
   camera_height = 1.65; // meter
#endif
   return;
}

// build up a mfg with first two views
void Mfg::initialize()
{
   // ----- setting parameters -----
   double distThresh_PtHomography = 1;
   double vpLnAngleThrseh = 50; // degree, tolerance angle between 3d line direction and vp direction
   double maxNoNew3dPt = 400;

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
   computeEpipolar (featPtMatches, pairIdx, K, F, R, E, t, true);
   cout<<"R="<<R<<endl<<"t="<<t<<endl;
   view1.t_loc = t;
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

#ifdef PLOT_MID_RESULTS
   cv::Mat canv1 = view0.img.clone();
   cv::Mat canv2 = view1.img.clone();
#endif   
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
#ifdef PLOT_MID_RESULTS
         cv::Scalar color(rand()%255,rand()%255,rand()%255,0);
         cv::line(canv1, a.extremity1, a.extremity2, color, 2);
         cv::line(canv2, b.extremity1, b.extremity2, color, 2);
#endif         
      }
   }

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
   cout<<endl<<" >>>>>>>>>> MFG initialized using two views <<<<<<<<<<"<<endl;

#ifdef USE_GROUND_PLANE
   need_scale_to_real = true;
   scale_since.push_back(1);
   camera_height = 1.65; // meter
#endif
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

   cleanup(50);

}

void Mfg::expand_keyPoints (View& prev, View& nview)
{
      MyTimer tm; 
   // ----- parameter setting -----
   double numPtpairLB = 30; // lower bound of pair number to trust epi than vp
   double reprjThresh = 5 * IDEAL_IMAGE_WIDTH/640.0; // projected point distance threshold
   double scale = -1;  // relative scale of baseline
   double vpLnAngleThrseh = 50; // degree, tolerance angle between 3d line direction and vp direction
   double ilineLenLimit = 10;  // length limit of 3d line, ignore too long lines
   double maxNumNew3dPt = 200;
   double maxNumNew2d_3dPt = 500;100;
   bool   sortbyPrlx = false;
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
         vector<cv::Point2f> curr_pts;
         vector<uchar> status;
         vector<float> err;
         cout<<trackFrms.back().filename<<endl;

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
#ifdef PLOT_MID_RESULTS
   cv::Mat canv1 = prev.img.clone(), canv2=nview.img.clone();
   for(int i=0; i<featPtMatches.size(); ++i) {
      cv::circle(canv1, featPtMatches[i][0], 2, cv::Scalar(100,200,200), 2);
      cv::circle(canv2, featPtMatches[i][1], 2, cv::Scalar(100,200,200), 2);
   }
#endif
   trackFrms.clear();

   //cv::Mat t_prev = prev.t - (prev.R * views[views.size()-3].R.t()) * views[views.size()-3].t;
   cv::Mat t_prev = views[views.size()-2].t_loc;
//   cout<<t_prev.t()<<endl;

   cv::Mat F, R, E, t; // relative pose between last two views
   vector<cv::Mat> Fs, Es, Rs, ts;
   
//   computePotenEpipolar (featPtMatches,pairIdx,K, Fs, Es, Rs, ts, false, t_prev);
   computeEpipolar(featPtMatches,pairIdx,K, Fs, Es, Rs, ts);
//   tm.end(); cout<<"computePotenEpipolar time "<<tm.time_ms<<" ms.\n";
   R = Rs[0]; t = ts[0];
   // ---- find observed 3D points (and plot)  ------
   vector<cv::Point3d> pt3d, pt3d_old;
   vector<cv::Point2d> pt2d, pt2d_old;
   for(int i=0; i < featPtMatches.size(); ++i) {
      int gid = prev.featurePoints[pairIdx[i][0]].gid;      
#ifdef PLOT_MID_RESULTS
      cv::Scalar color(rand()%255,rand()%255,rand()%255,0);
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

   ///// apply pnp //////
   cout<<"observed 3d landmarks "<<pt3d.size()<<endl;
   cv::Mat Rn_pnp, tn_pnp, R_pnp, t_pnp;
   if (pt3d.size()>3) {
  //    computePnP(pt3d, pt2d, K, Rn_pnp, tn_pnp);  
      computePnP_ransac (pt3d, pt2d, K, Rn_pnp, tn_pnp, 100);
      R_pnp = Rn_pnp * prev.R.t();
      t_pnp = tn_pnp - R_pnp * prev.t; // relative t, with scale

   }

   bool use_const_vel_model = false; // when no enought features or no good estimation
   vector <int> maxInliers_Rt;
   vector<KeyPoint3d> localKeyPts;   
   double bestScale = -100;   
   vector<double> sizes;
   for(int trial=0; trial < 10; ++trial) {
      maxInliers_Rt.clear();
      localKeyPts.clear();
      sizes.clear();
      for(int iter = 0; iter < Rs.size(); ++iter) {
         cv::Mat Ri = Rs[iter],	ti = ts[iter];
         bool isRgood = true;
         vector<vector<int>> vpPairIdx = matchVanishPts_withR(prev, nview, Ri, isRgood);
         
         // triangulate points for scale estimation
         vector<KeyPoint3d> tmpkp; // temporary keypoints
         int num_usable_pts = 0;
         for(int i=0; i < featPtMatches.size(); ++i) {
            double parallaxDeg = compParallaxDeg (featPtMatches[i][0], featPtMatches[i][1], K, cv::Mat::eye(3,3,CV_64F), Ri);
            int iGid = prev.featurePoints[pairIdx[i][0]].gid;
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
         if(num_usable_pts < 1 || (num_usable_pts < 10 && pt3d.size() > 50))
            continue; // abandon this pair of Ri ti

         cv::Mat Rn = Ri *prev.R;
         // ----- find best R and t based on existent 3D points------
         int maxIter = 100;
         vector<int> maxInliers;
         double best_s = -1;
         for (int it = 0; it < maxIter; ++it) {
            int i = xrand()%featPtMatches.size();
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
               if (cv::norm(featPtMatches[j][1] - mat2cvpt(pt)) < reprjThresh )
                  inliers.push_back(j);
            }
            if (maxInliers.size() < inliers.size()) {
               maxInliers = inliers;
               best_s = s;
            }
         }
         cout<<ti.t()<<" "<<maxInliers.size()<<endl;

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
#ifdef PLOT_MID_RESULTS
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
   //      cout<<nview.frameId-prev.frameId<<" frames, bestScale = " << bestScale <<", w-ave="<< wsum/tw<<", median=" << vecMedian(scales) <<endl;
         if (scales3.size() >= 15) {
            scale = vecMedian(scales3);
   //         cout<<nview.frameId-prev.frameId<<" frames,"<<"ave="<<vecSum(scales3)/scales3.size() <<", median="<<vecMedian(scales3)<< ", ["<< scales3.size()<<"]"<<endl;
         }
         scale = (scale + bestScale + vecMedian(scales))/3;
         cout<<"keyframe "<<nview.id<<", "<<nview.frameId-prev.frameId<<" frames: mixed scale = "<<scale;
      } else {
         scale = bestScale;
      }
      ///// compare with pnp result /////   
      if(pt3d.size()>3) {
         cout<<", "<<cv::norm(t_pnp)<<"("<<pt3d.size()<<")\n";
      } 

      if(maxInliers_Rt.size() < 5 
         && pt3d.size() > 7
         && (abs(scale - cv::norm(t_pnp)) > 0.7 * scale || abs(scale - cv::norm(t_pnp)) > 0.7 * cv::norm(t_pnp)) 
        ) {
         cout<<"Inconsistent pnp and 5-pt: "<< cv::norm(t_pnp) <<", "<<scale<<endl;
         cout<<"use pnp scale instead\n";
         scale = cv::norm(t_pnp);
      }

      nview.t = R*prev.t + t*scale;
      nview.R_loc = R;
      nview.t_loc = t;
   } else { // small movement, use R-PnP
   /*   vector <int> maxInliers_R;
      cv::Mat best_tn;
      vector<double> sizes;
      for(int trial=0; trial < 10; ++trial) {
         maxInliers_R.clear();
         sizes.clear();
         for(int iter = 0; iter < Rs.size(); ++iter) {
            cv::Mat Ri = Rs[iter],	ti = ts[iter];
            bool isRgood = true;
            vector<vector<int>> vpPairIdx = matchVanishPts_withR(prev, nview, Ri, isRgood);
            cv::Mat Rn = Ri *prev.R;

            // ----- find best R and t based on existent 3D points------
            int maxIter = 100;
            vector<int> maxInliers;
            cv::Mat good_tn;
            double best_s = -1;
            for (int it = 0; it < maxIter; ++it) {
               if(pt3d.size()==0) break;
               int i = xrand() % pt3d.size();
               int j = xrand() % pt3d.size();
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
            cout<<ti.t()<<",, "<<maxInliers.size()<<endl;
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
    #ifdef PLOT_MID_RESULTS
      cv::Scalar color(rand()%255,rand()%255,rand()%255,0);
#endif
      if(maxInliers_R.size() >= 5) {
         vector<cv::Point3d> inlierPt3d;
         vector<cv::Point2d> inlierPt2d;
         for(int i=0; i < maxInliers_R.size(); ++i) {
            inlierPt3d.push_back(pt3d[maxInliers_R[i]]);
            inlierPt2d.push_back(pt2d[maxInliers_R[i]]);
         }      
         nview.R = R*prev.R;
         nview.t = pnp_withR (inlierPt3d, inlierPt2d, K, nview.R);
         nview.R_loc = R;
         nview.t_loc = cv::Mat::zeros(3,1,CV_64F);
         cout<<nview.frameId-prev.frameId<<" frames,"<<"Scale (2pt_alg)="
            <<cv::norm(-nview.R.t()*nview.t + prev.R.t()*prev.t)<<", "
            <<maxInliers_R.size()<<"/"<<pt3d.size()<<endl;
      } else { 
      */   cout <<"Insufficient feature info for relative pose estimation...\n";
         // use pnp or const vel
         ///// compare with pnp result /////
         if(pt3d.size() > 7) {
            cout<<"fallback to pnp : "<< cv::norm(t_pnp) <<endl;            
            nview.R = Rn_pnp;
            nview.t = tn_pnp;  
            R = R_pnp;
            t = t_pnp/cv::norm(t_pnp);
            nview.R_loc = R;
            nview.t_loc = t;
         } else {
            cout<<"fallback to const-vel model ...press 1 to continue\n";
            int tmp; cin>>tmp;            
            use_const_vel_model = true;                           
         }
   //   }
   }


#ifdef USE_GROUND_PLANE
   int n_gp_pts;
   cv::Point3f gp_norm_vec;
   double gp_depth;
   double gp_quality;
   double scale_to_real = -1;
   cv::Mat gp_roi;
   double gp_qual_thres = 0.53;
   double baseline = -1;
   if(!need_scale_to_real && !use_const_vel_model) { 
      baseline = cv::norm(nview.t - R*prev.t);
   }
   if(detectGroundPlane(prev.grayImg, nview.grayImg,R,t,K, n_gp_pts, gp_depth, gp_norm_vec, gp_quality, gp_roi, baseline)) {
      if(!use_const_vel_model) // translation scale obtained from epipolor or pnp
      {
         scale_to_real = camera_height/abs((cv::norm(nview.t - R*prev.t) * gp_depth));
         if(need_scale_to_real) {  
            if(gp_quality > gp_qual_thres)
               scale_vals.push_back(scale_to_real);
            if(scale_vals.size()>=3 || nview.id - scale_since.back() > 50) {
               if(scale_vals.size()==0) {
                  scale_vals.push_back(scale_to_real);
               }
               cout<<"Scale local map by "<< vecSum(scale_vals)/scale_vals.size()<<endl; //int tmp; cin>>tmp; 
               vector<double> scale_constraint(4);
               scale_constraint[0] = prev.id; scale_constraint[1] = nview.id; 
               scale_constraint[2] = cv::norm(nview.t - R*prev.t) * vecSum(scale_vals)/scale_vals.size();
               scale_constraint[3] = 1e6 * gp_quality*gp_quality*gp_quality;
               camdist_constraints.push_back(scale_constraint);

               scaleLocalMap(scale_since.back(), nview.id, vecSum(scale_vals)/scale_vals.size(), false);
               bundle_adjust_between(scale_since.back(), nview.id, scale_since.back()+1);// when window large, not convergent
               need_scale_to_real = false;
               scale_since.push_back(nview.id);
               scale_vals.clear();
            } 
         } else { // already in real scale, check consistency
            double est_ch = cv::norm(nview.t - R*prev.t)* abs(gp_depth);
            cout<<"estimated camera height "<<est_ch<<" m, qual = "<<gp_quality<<endl;
            if ((abs(est_ch - camera_height) > 0.02 && gp_quality > gp_qual_thres) ||
                (abs(est_ch - camera_height) > 0.1 && gp_quality > gp_qual_thres ) || 
            //    (abs(est_ch - camera_height) > 0.2 && gp_quality > gp_qual_thres - 0.05) ||
                (abs(scale_since.back()-nview.id)>20 && abs(est_ch - camera_height) > 0.1 && 
                  abs(est_ch - camera_height) < 0.3 && gp_quality > gp_qual_thres-0.02)
               ) { // inconsistence
                  cout<<"Scale inconsistence, do scaling "<<scale_to_real<<" from "<<scale_since.back()<<endl; //int tmp; cin>>tmp;
               if(nview.id - scale_since.back() == 1 && scale_since.size()>2)
                  scaleLocalMap(scale_since[scale_since.size()-2], nview.id, scale_to_real, false);
               if(nview.id - scale_since.back() > 1 && nview.frameId - views[scale_since.back()].frameId < 100)   
                  scaleLocalMap(scale_since.back(), nview.id, scale_to_real, false); 
               else if (nview.frameId - views[scale_since.back()].frameId > 60) 
                  scaleLocalMap(max(0, nview.id-20), nview.id, scale_to_real, false);

               vector<double> scale_constraint(4);
               scale_constraint[0] = prev.id; scale_constraint[1] = nview.id; 
               scale_constraint[2] = cv::norm(nview.t - R*prev.t);
               scale_constraint[3] = 1e6 * gp_quality*gp_quality*gp_quality;
               camdist_constraints.push_back(scale_constraint);
               if(nview.id - scale_since.back() == 1 && scale_since.size()>2)
                  bundle_adjust_between(scale_since[scale_since.size()-2], nview.id, scale_since[scale_since.size()-2]+1);
               if(nview.id - scale_since.back() > 1 && nview.frameId - views[scale_since.back()].frameId < 100)               
                  bundle_adjust_between(scale_since.back()-1, nview.id, scale_since.back()+1);
               else if (nview.frameId - views[scale_since.back()].frameId > 60) 
                  bundle_adjust_between(max(0, nview.id-20), nview.id, max(0, nview.id-20)+1);

               cv::putText(gp_roi, "scale since "+num2str(scale_since.back())+" by "+num2str(scale_to_real), cv::Point2f(10,200), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(200,0,0));
               scale_since.push_back(nview.id);
            } else if(gp_quality >= gp_qual_thres) { // good so far
               vector<double> scale_constraint(4);
               scale_constraint[0] = prev.id; scale_constraint[1] = nview.id; 
               scale_constraint[2] = cv::norm(nview.t - R*prev.t);
               scale_constraint[3] = 1e6 * gp_quality*gp_quality*gp_quality;
               camdist_constraints.push_back(scale_constraint);
               scale_since.push_back(nview.id);
            }              
         }
         cv::imwrite("./tmpimg/gp/gp_"+num2str(nview.id)+".jpg", gp_roi);
      } else { // use_const_vel_model, t-scale not available from epipolar/pnp
         scale_to_real = camera_height/abs(gp_depth);        
         if(!need_scale_to_real) { // already in real scale, and R,t computed
            R = Rs[0];
            t = ts[0];
            nview.R = R * prev.R;
            nview.t = R*prev.t + scale_to_real*t; 
            nview.R_loc = R;            
            nview.t_loc = t;
            use_const_vel_model = false;     // avoid using const-vel via gp info 
            cout<<"t-scale unavailable ... GP recovers real scale by scaling "<<scale_to_real<<endl; //int tmp; cin>>tmp;
         }  
      }
   } else {//no ground plane detected 
      if(use_const_vel_model && !need_scale_to_real) {// start a new local scale
         need_scale_to_real = true;
         scale_since.push_back(nview.id);
         R = Rs[0];
         t = ts[0];
         nview.R = R * prev.R;
         nview.t = R*prev.t + t;
         nview.R_loc = R;            
         nview.t_loc = t;
         use_const_vel_model = false;
         cout<<"started a new local system since view "<<nview.id<<endl; //int tmp; cin>>tmp;
      }
   }
#endif


   ////// use const-vel to predict R t ///////
   cv::Mat R_const, t_const;
   if(nview.id>2) {
      Eigen::Quaterniond q_prev = r2q (prev.R_loc);
      double theta_prev = acos(q_prev.w()) * 2;
      double theta_curt = theta_prev * (nview.frameId - prev.frameId)/(prev.frameId - views[(prev.id - 1)].frameId);
      Eigen::Vector3d xyz(q_prev.x(),q_prev.y(),q_prev.z());
      xyz = xyz/xyz.norm();
      double q_curt[4] = {cos(theta_curt/2),
                                xyz(0)*sin(theta_curt/2),
                                xyz(1)*sin(theta_curt/2),
                                xyz(2)*sin(theta_curt/2)};
      R_const =  q2r(q_curt);
      t_const = (prev.t - (prev.R * views[views.size()-3].R.t()) * views[views.size()-3].t)
                   * (nview.frameId - prev.frameId)/(prev.frameId - views[(prev.id - 1)].frameId);
   //   cout<<"t_const="<<t_const.t()/cv::norm(t_const)<<cv::norm(t_const)<<endl;
   }

   if(use_const_vel_model) {
      if(R_const.cols<3) {
         cout<<"R_const not available. Terminated.\n";
         exit(0);
      }
      if(R.cols==3) { // R computed
         nview.R = R * prev.R;
         nview.t = R*prev.t + cv::norm(t_const)*t;      
      } else {
         nview.R = R_const * prev.R;
         nview.t = R_const*prev.t + t_const;
         R = R_const;
         t = t_const/cv::norm(t_const);  

      }
      nview.R_loc = R;
      nview.t_loc = cv::Mat::zeros(3,1,CV_64F);
   }

   

   // ---- plot epipoles ----
   cv::Point2d ep1 = mat2cvpt(K*R.t()*t);
   cv::Point2d ep2 = mat2cvpt(K*t);

   prev.epipoleA = ep1;
   nview.epipoleB = ep2;
   double epNeibRds = 0.0 * IDEAL_IMAGE_WIDTH/640;
#ifdef PLOT_MID_RESULTS
   cv::circle(canv1, ep1, epNeibRds, cv::Scalar(0,0,100), 1);
   cv::circle(canv2, ep2, epNeibRds, cv::Scalar(0,0,100), 1);
#endif


   Matrix3d Rx;
   Rx << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
      R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
      R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2);
   Quaterniond q(Rx);
   angVel = 2*acos(q.w()) * 180/PI / ((nview.frameId - prev.frameId)/fps);
//   cout<<"angVel="<<angVel<<" deg/sec\n";
   nview.angVel = angVel;

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
         double parallaxDeg = compParallaxDeg (featPtMatches[i][0], featPtMatches[i][1], K,cv::Mat::eye(3,3,CV_64F), R);
         cv::Mat Xw =  triangulatePoint_nonlin (prev.R, prev.t, nview.R, nview.t, K,
               featPtMatches[i][0], featPtMatches[i][1]);
         cv::Mat Xc = triangulatePoint (cv::Mat::eye(3,3,CV_64F), cv::Mat::zeros(3,1,CV_64F),
               R, t, K,	featPtMatches[i][0], featPtMatches[i][1]);

#ifdef PLOT_MID_RESULTS
         cv::Scalar color(rand()%255,rand()%255,rand()%255,0);
         if (parallaxDeg > parallaxDegThresh) {
            cv::circle(canv1, featPtMatches[i][0], 2, color, 2);
            cv::circle(canv2, featPtMatches[i][1], 2, color, 2);
         }
#endif
         if(parallaxDeg < parallaxDegThresh
               || (//cv::norm(Xw+nview.R.t()*nview.t) > mfgSettings->getDepthLimit() &&
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
#ifdef PLOT_MID_RESULTS
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
   
   
   #pragma omp parallel for
   for (int i=0; i < featPtMatches.size(); ++i) {
      vector<int> view_point;
      int ptGid = prev.featurePoints[pairIdx[i][0]].gid;
      nview.featurePoints[pairIdx[i][1]].gid = ptGid;
      if ( ptGid >= 0 && keyPoints[ptGid].is3D ) { // points represented in 3D form
      }
      else if (ptGid >= 0 && !keyPoints[ptGid].is3D) {// point exist as 2D track
         cv::Scalar color(rand()%255,rand()%255,rand()%255,0);
         double parallaxDeg = compParallaxDeg(featPtMatches[i][0], featPtMatches[i][1], K, cv::Mat::eye(3,3,CV_64F), R);
#ifdef PLOT_MID_RESULTS
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
         

         // --- 2) check if a 2d track is ready to establish 3d pt ---
         // -- start a voting/ransac --
         cv::Mat bestKeyPt;
         vector<int> maxInlier;
         int obs_size = keyPoints[ptGid].viewId_ptLid.size();
         for (int p = 0; p < obs_size-1 && p<2; ++p) {
            int pVid = keyPoints[ptGid].viewId_ptLid[p][0],
                   pLid = keyPoints[ptGid].viewId_ptLid[p][1];
               if(!views[pVid].matchable) break;    
#ifdef USE_GROUND_PLANE
            if(need_scale_to_real && scale_since.back()!=1 && pVid < scale_since.back())
               continue;
#endif
            for(int q = obs_size - 1; q >= p+1 && q>obs_size-3; --q) {
               // try to triangulate every pair of 2d pt
               int qVid = keyPoints[ptGid].viewId_ptLid[q][0],
                   qLid = keyPoints[ptGid].viewId_ptLid[q][1];
               if(!views[qVid].matchable) break;
               cv::Point2d ppt = views[pVid].featurePoints[pLid].cvpt(),
                  qpt = views[qVid].featurePoints[qLid].cvpt();
               double pqPrlxDeg = compParallaxDeg(ppt, qpt, K,	views[pVid].R, views[qVid].R);
               if (pqPrlxDeg < parallaxDegThresh * 1) 	break; // not usable
               
            //   cv::Mat ptpq = triangulatePoint_nonlin (views[pVid].R, views[pVid].t,views[qVid].R, views[qVid].t, K, ppt, qpt);
               cv::Mat ptpq = triangulatePoint (views[pVid].R, views[pVid].t,views[qVid].R, views[qVid].t, K, ppt, qpt);
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
                  if( dist < 2) // inlier;
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
            if(!views[vid].matchable) continue;
            Rs.push_back(views[vid].R);
            ts.push_back(views[vid].t);
            pt.push_back(views[vid].featurePoints[lid].cvpt());
         }
         est3dpt_g2o (Rs, ts, K, pt, bestKeyPt); // nonlinear estimation

         // --- 3) establish 3d pt ---
         keyPoints[ptGid].is3D = true;
         keyPoints[ptGid].estViewId = nview.id;
         keyPoints[ptGid].x = bestKeyPt.at<double>(0);
         keyPoints[ptGid].y = bestKeyPt.at<double>(1);
         keyPoints[ptGid].z = bestKeyPt.at<double>(2);
#ifdef PLOT_MID_RESULTS
         cv::circle(canv1, featPtMatches[i][0], 7, color, 2);
         cv::circle(canv2, featPtMatches[i][1], 7, color, 2);
         cv::putText(canv1, " "+num2str(cv::norm(bestKeyPt+nview.R.t()*nview.t)), featPtMatches[i][0],
               1, 1, color);
#endif
         new2_3dPtNum ++;
      }
   }
   cout<<"new 3d pts "<<new3dPtNum<<"\t"<<new2_3dPtNum<<endl;
   // remove outliers
   if(!use_const_vel_model) {
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
            if(views[vid].matchable)            
               views[vid].featurePoints[lid].gid = -1;
         }
         keyPoints[gid].viewId_ptLid.clear();
      }
      }
   }

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

   if(F.empty()) {
      F = K.t().inv() * R * vec2SkewMat(t) * K.inv();
   }
   vector<vector<int>>			ilinePairIdx;
   matchIdealLines(prev, nview, vpPairIdx, featPtMatches, F, ilinePairIdx, 1);
   update3dIdealLine(ilinePairIdx, nview);

#ifdef USE_GROUND_PLANE
   if(!need_scale_to_real)
#endif      
      updatePrimPlane();
   // ---- write to images for debugging ----
#ifdef PLOT_MID_RESULTS
   cv::imwrite("./tmpimg/"+num2str(views.back().id)+"_pt1.jpg", canv1);
   cv::imwrite("./tmpimg/"+num2str(views.back().id)+"_pt2.jpg", canv2);
   views.back().drawAllLineSegments(true);
#endif
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
#ifdef PLOT_MID_RESULTS
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
                int pVid = idealLines[lnGid].viewId_lnLid[p][0],
                        pLid = idealLines[lnGid].viewId_lnLid[p][1];
                 if(!views[pVid].matchable) break;
#ifdef USE_GROUND_PLANE
            if(need_scale_to_real && scale_since.back()!=1 && pVid < scale_since.back())
               continue;
#endif
               for(int q = p+1; q < idealLines[lnGid].viewId_lnLid.size(); ++q) {
                  // try to triangulate every pair of 2d line                 
                     int   qVid = idealLines[lnGid].viewId_lnLid[q][0],
                        qLid = idealLines[lnGid].viewId_lnLid[q][1];
                        if(!views[qVid].matchable) break;
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
#ifdef PLOT_MID_RESULTS
      if(linewidth > 0) {
         cv::line(canv1, prev.idealLines[ilinePairIdx[i][0]].extremity1, prev.idealLines[ilinePairIdx[i][0]].extremity2, color, linewidth);
         cv::line(canv2, nview.idealLines[ilinePairIdx[i][1]].extremity1, nview.idealLines[ilinePairIdx[i][1]].extremity2, color, linewidth);
      }
#endif
   }
#ifdef PLOT_MID_RESULTS
   cv::imwrite("./tmpimg/ln_"+num2str(views.back().id)+"_1.jpg", canv1);
   cv::imwrite("./tmpimg/ln_"+num2str(views.back().id)+"_2.jpg", canv2);
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

   glPointSize(3.0);
   glBegin(GL_POINTS);
   for (int i=0; i<keyPoints.size(); ++i){
      if(!keyPoints[i].is3D || keyPoints[i].gid<0) continue;
      if(keyPoints[i].pGid < 0) // red
         glColor3f(0.6, 0.6, 0.6);
      else { // coplanar green
         glColor3f(0.0, 1.0, 0.0);
      }
      glVertex3f(keyPoints[i].x, keyPoints[i].y, keyPoints[i].z);
   }
   glEnd();

   glColor3f(0,1,1);
   glLineWidth(2);
   glBegin(GL_LINES);
   for(int i=0; i<idealLines.size(); ++i) {
      if(!idealLines[i].is3D || idealLines[i].gid<0) continue;
      if(idealLines[i].pGid < 0) {
         glColor3f(0,0,0);
      } else {
         glColor3f(0.0,1.0,0.0);
      }

      glVertex3f(idealLines[i].extremity1().x,idealLines[i].extremity1().y,idealLines[i].extremity1().z);
      glVertex3f(idealLines[i].extremity2().x,idealLines[i].extremity2().y,idealLines[i].extremity2().z);
   }
   glEnd();
}

void Mfg::adjustBundle()
{
   int numPos = 10, numFrm = 15;
   if (views.size() <=numFrm ){
      numPos = numFrm;
   }
   
   adjustBundle_G2O(numPos, numFrm);
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
            if(views[vid].matchable)            
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
         if(!views[vid].matchable) continue;
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
            if(views[vid].matchable)
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
            if(views[vid].matchable)
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
         if(!views[vid].matchable) continue;
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
            if(views[vid].matchable)            
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
 //  cout<<"delete 3d pts : "<<n_del<<endl;
}

bool Mfg::rotateMode ()
{
   double avThresh = 15; // angular velo degree/sec
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

void Mfg::scaleLocalMap(int from_view_id, int to_view_id, double scale, bool interpolate)
// transform all campose and landmarks to the local one, scale, then transform back to global 
{// do incremental scaling, not all the same

   cv::Mat R0 = views[from_view_id-1].R.clone(), t0 = views[from_view_id-1].t.clone();
   for(int i=from_view_id; i <= to_view_id; ++i) {
     // views[i].t = views[i].t * scale;    
      double interp_scale = scale; 
      if(interpolate)
         interp_scale = 1 + (i-from_view_id)*(scale-1)/(to_view_id-from_view_id);
      views[i].t = views[i].R * R0.t()*t0 + interp_scale *(views[i].t - views[i].R * R0.t()*t0);
   }
   for(int i=0; i<keyPoints.size();++i) {      
      if (keyPoints[i].is3D && keyPoints[i].gid >= 0
          && keyPoints[i].estViewId >= from_view_id) {
         double interp_scale = scale; 
         if(interpolate)
            interp_scale  = 1 + (keyPoints[i].estViewId-from_view_id)*(scale-1)/(to_view_id-from_view_id) ;
            cv::Mat X = R0.t()*((R0*keyPoints[i].mat(false)+t0)*interp_scale - t0);                    
            keyPoints[i].x = X.at<double>(0);
            keyPoints[i].y = X.at<double>(1);
            keyPoints[i].z = X.at<double>(2);
      }
   }
   for(int i=0; i<idealLines.size();++i) {
      if(idealLines[i].is3D && idealLines[i].gid >= 0
         && idealLines[i].estViewId >= from_view_id) {
         double interp_scale = scale; 
         if(interpolate)
            interp_scale  = 1 + (idealLines[i].estViewId-from_view_id)*(scale-1)/(to_view_id-from_view_id) ;
         idealLines[i].length = idealLines[i].length * interp_scale;
         cv::Mat mp = R0.t()*((R0*cvpt2mat(idealLines[i].midpt,false)+t0)*interp_scale - t0); 
         idealLines[i].midpt.x = mp.at<double>(0);
         idealLines[i].midpt.y = mp.at<double>(1);
         idealLines[i].midpt.z = mp.at<double>(2);
      }
   }
   // when need_scale_to_real, don't update planes
   /*for(int i=0; i<primaryPlanes.size(); ++i) {
      primaryPlanes[i].d = primaryPlanes[i].d * scale;
   }*/
}

void Mfg::cleanup(int n_kf_keep) 
{
   for(int i=0; i < (int)views.size()-n_kf_keep; ++i) {
      std::vector <FeatPoint2d> ().swap(views[i].featurePoints);
      std::vector <LineSegmt2d> ().swap(views[i].lineSegments);
      std::vector <VanishPnt2d> ().swap(views[i].vanishPoints);
      std::vector <IdealLine2d> ().swap(views[i].idealLines);
      views[i].img.release();
      views[i].grayImg.release();
      views[i].matchable = false;
   }



}