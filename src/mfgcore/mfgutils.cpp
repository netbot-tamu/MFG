
#include "mfgutils.h"

#include <iostream>
#include <vector>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "view.h"
#include "utils.h"
#include "consts.h"
#include "settings.h"
#include "features2d.h"
#include "features3d.h"

// TODO: FIXME: circular dependency
#include "mfg.h"

using namespace std;
extern MfgSettings* mfgSettings;


bool isKeyframe (Mfg& map, const View& v1, int th_pair, int th_overlap)
// determine if v1 is a keyframe, given v0 is the last keyframe
// point matches, overlap
{
	const View& v0 = map.views.back();
	int ThreshPtPair = th_pair;
	vector<vector<cv::Point2d>> ptmatches;
	vector<vector<int>> pairIdx;
	vector<int> tracked_idx; //the local id of pts in last keyframe v0 that has been tracked using LK
	vector<cv::Point2f> add_curr_pts;
	vector<int> add_curr_idx_in_lastview;

	if (mfgSettings->getKeypointAlgorithm() < 3) // sift/surf
		pairIdx = matchKeyPoints (v0.featurePoints, v1.featurePoints, ptmatches);
	else {
		vector<cv::Point2f> prev_pts, curr_pts;
		vector<uchar> status;
		vector<float> err;
		cv::Mat prev_img;
		if(map.trackFrms.size()==0) {
			prev_img = v0.grayImg;
			for(int i=0; i<v0.featurePoints.size();++i)
				prev_pts.push_back(cv::Point2f(v0.featurePoints[i].x,v0.featurePoints[i].y));
		}
		else {
			prev_img = map.trackFrms.back().image;
			prev_pts = map.trackFrms.back().featpts;
		}
		cv::calcOpticalFlowPyrLK(
      prev_img, v1.grayImg, // 2 consecutive images
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
		vector<cv::Point2f> prev_pts_rvs = prev_pts, curr_pts_rvs = curr_pts;
		vector<uchar> status_reverse;
		cv::calcOpticalFlowPyrLK(
      v1.grayImg, prev_img,// 2 consecutive images
              curr_pts_rvs, //  point positions in the 2nd
              prev_pts_rvs,
              status_reverse,    // tracking success
              err,      // tracking error
              cv::Size(mfgSettings->getOflkWindowSize(),mfgSettings->getOflkWindowSize()),
              3,
              cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01),
              0,
              mfgSettings->getOflkMinEigenval()  // minEignVal threshold for the 2x2 spatial motion matrix, to eleminate bad points
      );

		for(int i=0; i<status.size(); ++i) {
			if(status[i] && status_reverse[i] && cv::norm(prev_pts_rvs[i]-prev_pts[i])<0.5
                 // &&(curr_pts[i].x >0 &&  curr_pts[i].y>0 && curr_pts[i].x<v1.img.cols-1 &&curr_pts[i].y<v1.img.rows-1)
                 // &&(prev_pts[i].x>0 && prev_pts[i].y >0 && prev_pts[i].x<v1.img.cols-1 && prev_pts[i].y<v1.img.rows-1 )
                 ) { // match found
				// todo: verify by ORB descriptor
				vector<int> pair(2);
				pair[0] = i; pair[1] = i;
				pairIdx.push_back(pair);
				vector<cv::Point2d> match(2);
				match[0] = prev_pts[i];
				match[1] = curr_pts[i];
				ptmatches.push_back(match);
				if(map.trackFrms.size()==0) { // previous image is keyframe
					tracked_idx.push_back(i);
				} else {
					tracked_idx.push_back(map.trackFrms.back().pt_lid_in_last_view[i]);
				}
			}
		}
	}
	cout<<ptmatches.size()<<'\t';
	// === filter out false matches by F matrix ===
	if (pairIdx.size() > 7) {
		int nPts = ptmatches.size();
		cv::Mat pts1(2, nPts, CV_32F), pts2(2, nPts ,CV_32F);
		for (int i=0; i<ptmatches.size(); ++i) {
			pts1.at<float>(0,i) = ptmatches[i][0].x;
			pts1.at<float>(1,i) = ptmatches[i][0].y;
			pts2.at<float>(0,i) = ptmatches[i][1].x;
			pts2.at<float>(1,i) = ptmatches[i][1].y;
		}
		vector<uchar> inSetF;
		cv::Mat F = cv::findFundamentalMat(pts1.t(), pts2.t(), 8, 2, 0.99, inSetF);
		if(cv::norm(F) < 1e-10) { // F is invalid (zero matrix)
		//	cout<<" 0F ";
		} else {
			for(int j = inSetF.size()-1; j>=0; --j) {
				if (inSetF[j] == 0) {
					ptmatches.erase(ptmatches.begin()+j);
					pairIdx.erase(pairIdx.begin()+j);
					tracked_idx.erase(tracked_idx.begin()+j);
				}
			}
		}
	}
	cout<<ptmatches.size()<<'\t';
	Frame frm; // probably added to track_frms
	frm.filename = v1.filename;
	frm.image = v1.grayImg;
	for (int i=0; i<pairIdx.size(); ++i) {
		frm.featpts.push_back(ptmatches[i][1]);
		if(map.trackFrms.size()==0) { // previous image is keyframe
			frm.pt_lid_in_last_view.push_back(pairIdx[i][0]);
		} else {
			frm.pt_lid_in_last_view.push_back(map.trackFrms.back().pt_lid_in_last_view[pairIdx[i][0]]);
		}
	}

	// compute current pose, if rotation large, drop key frame
	vector<cv::Point3d> X;
	vector<cv::Point2d> x;
	for(int i=0; i<pairIdx.size(); ++i) {
		int gid = -1;
		if (map.trackFrms.size()==0) {
			gid = v0.featurePoints[pairIdx[i][0]].gid;
		} else {
			int lid = map.trackFrms.back().pt_lid_in_last_view[pairIdx[i][0]];
			gid = map.views.back().featurePoints[lid].gid;
		}
		if (gid < 0 || !map.keyPoints[gid].is3D) continue;
		X.push_back(map.keyPoints[gid].cvpt());
		x.push_back(ptmatches[i][1]);
	}

	int min3Dtrack = 30; // min num of tracked 3d pts before reprojecting 3d to 2d to find more or use prev-prev frame to track

	if(X.size() > 7) {
		cv::Mat Rn, tn;
		// need a ransac pnp
		computePnP(X,x,map.K,Rn,tn); //current pose Rn, tn
		cv::Mat R = Rn*map.views.back().R.inv();
		double angle = acos(abs((R.at<double>(0,0) + R.at<double>(1,1) + R.at<double>(2,2) - 1)/2));
		cout<<angle*180/PI<<'\t';
      // TODO: needs tr1 namespace???
		unordered_map<int,int> tracked_lid_id;
      for (int i=0; i<tracked_idx.size();++i) {
         tracked_lid_id[tracked_idx[i]] = i;
      }

		if(mfgSettings->getKeypointAlgorithm()>=3 &&
              (X.size() < min3Dtrack || angle > 10*PI/180) &&
              X.size() > 8 // reliable pnp estimate
              ) {
			/////// reproject 3d pts to find more tracking points
			double radius = 30 * angle;  // search radius
			double desc_dist_thresh = 0.4;
			for(int i=0; i < map.keyPoints.size(); ++i)	{
				if ( !map.keyPoints[i].is3D ||map.keyPoints[i].gid<0 ) continue;
				if( map.keyPoints[i].estViewId < 2) continue;
				int last_view_id = map.keyPoints[i].viewId_ptLid.back()[0];
				int last_view_ptlid = map.keyPoints[i].viewId_ptLid.back()[1];
				if (last_view_id != v0.id) continue; // FOR NOW, only consider those 3d pts observable in last keyframe
				if (last_view_id == v0.id &&  // already tracked pts
                    tracked_lid_id.find(last_view_ptlid) != tracked_lid_id.end() ) continue;
				if(!checkCheirality (Rn, tn, map.keyPoints[i].mat())) continue;
				cv::Point2d pt = mat2cvpt(map.K*(Rn*map.keyPoints[i].mat(0)+tn));
				if ( int(pt.x - radius)<0 || (pt.x + radius) > v1.img.cols-1 ||
                    int(pt.y - radius)<0 || (pt.y + radius) > v1.img.rows-1) continue;
				// detect pts
				cv::Mat mask = cv::Mat::zeros(v1.grayImg.size(), CV_8UC1);
				cv::Mat roi = cv::Mat(mask, cv::Rect(pt.x-radius, pt.y-radius, 2*radius, 2*radius));
				roi = 1;
				vector<cv::Point2f> partpts;
				for(float j =-radius; j<radius; j=j+0.5) {
					for(float k =-radius; k<radius; k=k+0.5) {
						partpts.push_back(cv::Point2f(j+pt.x,k+pt.y));
					}
				}
				// extract descriptors
				vector<cv::KeyPoint> kpts;
				for(int j=0; j<partpts.size(); ++j) {
					kpts.push_back(cv::KeyPoint(partpts[j], 21));// no angle info provided
				}
				cv::SurfDescriptorExtractor surfext;
				cv::Mat descs;
				surfext.compute(v1.grayImg, kpts, descs);
				cv::Mat desc_3dpt = map.views[last_view_id].featurePoints[last_view_ptlid].siftDesc;
				double mindist = 2;
				int minIdx = -1;
				for(int j=0; j<kpts.size();++j) {
					if(cv::norm(desc_3dpt - descs.row(j).t()) < mindist) {
						mindist = cv::norm(desc_3dpt - descs.row(j).t());
						minIdx = j;
					}
				}
				if(minIdx>=0 && mindist < desc_dist_thresh) {
					if (last_view_id == v0.id) {
						add_curr_idx_in_lastview.push_back(last_view_ptlid);
						add_curr_pts.push_back(kpts[minIdx].pt);
						tracked_lid_id[last_view_ptlid] = -1; // to avoid re-tracking in next step
					}
				}
			}
		}
		if(mfgSettings->getKeypointAlgorithm()>=3 &&
              map.trackFrms.size() > 0 &&
              (X.size() < min3Dtrack || map.rotateMode())
              ) {
			// === use prev prev image to track, deal with blur ===
			cv::Mat pp_img;
			vector<cv::Point2f> pp_pts, pp_pts2, curr_pts;
			vector<uchar> status, status2;
			vector<float> err;
			vector<int> pp_idx; // the local lid of points in last keyframe that are already tracked
			if(map.trackFrms.size()==1) {
				pp_img = v0.grayImg;
				for(int j=0; j<v0.featurePoints.size();++j) {
					pp_pts.push_back(cv::Point2f(v0.featurePoints[j].x,v0.featurePoints[j].y));
					pp_idx.push_back(j);
				}
			} else {
				pp_img = map.trackFrms[map.trackFrms.size()-2].image;
				pp_pts = map.trackFrms[map.trackFrms.size()-2].featpts;
				pp_idx = map.trackFrms[map.trackFrms.size()-2].pt_lid_in_last_view;
			}
			cv::calcOpticalFlowPyrLK(
         pp_img, v1.grayImg, // 2 consecutive images
                 pp_pts, // input point positions in first im
                 curr_pts, // output point positions in the 2nd
                 status,    // tracking success
                 err,      // tracking error
                 cv::Size(mfgSettings->getOflkWindowSize(),mfgSettings->getOflkWindowSize()),
                 3,
                 cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01),
                 0,
                 mfgSettings->getOflkMinEigenval()  // minEignVal threshold for the 2x2 spatial motion matrix, to eleminate bad points
         );
			cv::calcOpticalFlowPyrLK(
         v1.grayImg, pp_img,  // 2 consecutive images
                 curr_pts, // input point positions in first im
                 pp_pts2, // output point positions in the 2nd
                 status2,    // tracking success
                 err,      // tracking error
                 cv::Size(mfgSettings->getOflkWindowSize(),mfgSettings->getOflkWindowSize()),
                 3,
                 cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01),
                 0,
                 mfgSettings->getOflkMinEigenval()  // minEignVal threshold for the 2x2 spatial motion matrix, to eleminate bad points
         );

			for(int i=0; i<status.size(); ++i) {
				if(status[i] && status2[i] && cv::norm(pp_pts[i]-pp_pts2[i])<0.5
                    //    &&( curr_pts[i].x >0 &&  curr_pts[i].y>0 && curr_pts[i].x<v1.img.cols-1 &&curr_pts[i].y<v1.img.rows-1)
                    //    &&(pp_pts[i].x>0 && pp_pts[i].y >0 && pp_pts[i].x<v1.img.cols-1 && pp_pts[i].y<v1.img.rows-1 )
                    ) { // match found
					// check if already tracked
					if(tracked_lid_id.find(pp_idx[i]) == tracked_lid_id.end()) {
						add_curr_idx_in_lastview.push_back(pp_idx[i]);
						add_curr_pts.push_back(curr_pts[i]);
					}
				}
			}
		}

		for (int i=0; i<add_curr_idx_in_lastview.size(); ++i) {
			frm.featpts.push_back(add_curr_pts[i]);
			frm.pt_lid_in_last_view.push_back(add_curr_idx_in_lastview[i]);
		}

		map.angleSinceLastKfrm = angle;
		if (angle > 15 * PI/180
              || cv::norm(-Rn.t()*tn + map.views.back().R.t()*map.views.back().t) > 1.2  // large translation
              ) {
			cout<<" ,rotation angle large, drop keyframe!!!\n";
			if(map.trackFrms.size()==0) {
				map.trackFrms.push_back(frm);
			}
			return true;
		}
	}
	int count=0;
	for(int i=0; i<pairIdx.size(); ++i) {
		int gid;
		if(mfgSettings->getKeypointAlgorithm() <3) {
			gid = v0.featurePoints[pairIdx[i][0]].gid;
		} else {
			if (map.trackFrms.size()==0) {
				gid = v0.featurePoints[pairIdx[i][0]].gid;
			} else {
				gid = v0.featurePoints[
                    map.trackFrms.back().pt_lid_in_last_view[pairIdx[i][0]]].gid;
			}
		}
		if (gid >= 0 && map.keyPoints[gid].is3D) {
			count++;
		}
	}

	cout<<"Keypoint Matches: "<<ptmatches.size()<<"/"<<v1.featurePoints.size()
           <<"   overlap="<<count<<endl;

	if (ptmatches.size() < min((double)th_pair, v0.featurePoints.size()/50.0) ||
           count < min((double)th_overlap, ptmatches.size()/3.0)) {
      if(map.trackFrms.size()==0) {
         map.trackFrms.push_back(frm);
      }
      return true;
	} else {
		if(mfgSettings->getKeypointAlgorithm() >=3) {
			map.trackFrms.push_back(frm);
		}
		return false;
	}
}

void drawFeatPointMatches(View& view1, View& view2, vector<vector<cv::Point2d>> featPtMatches)
{
	cv::Mat canv1 = view1.img.clone(),
           canv2 = view2.img.clone();
	for (int i=0; i<featPtMatches.size(); ++i)
	{
		cv::Scalar color(rand()%255,rand()%255,rand()%255,0);
		cv::circle(canv1, featPtMatches[i][0], 2,color, 2);
		cv::circle(canv2, featPtMatches[i][1], 2,color, 2);
		cv::putText(canv1, num2str(i),  featPtMatches[i][0], cv::FONT_HERSHEY_COMPLEX, 0.5,color);
		cv::putText(canv2, num2str(i),  featPtMatches[i][1], cv::FONT_HERSHEY_COMPLEX, 0.5,color);
	}
	showImage("img1-point,"+num2str(featPtMatches.size()), &canv1);
	showImage("img2-point,"+num2str(featPtMatches.size()), &canv2);

	cv::waitKey();
}






vector<vector<int>> matchVanishPts_withR(View& view1, View& view2, cv::Mat R, bool& goodR)
{
	cv::Mat score(view1.vanishPoints.size(), view2.vanishPoints.size(), CV_64F);
	for(int i=0; i<view1.vanishPoints.size(); ++i) {
		cv::Mat vp1 = view1.K.inv() * view1.vanishPoints[i].mat(); // in world coord
		vp1 = vp1*(1/cv::norm(vp1));
		for (int j=0; j < view2.vanishPoints.size(); ++j)	{
			cv::Mat vp2 = R.t() * view2.K.inv() * view2.vanishPoints[j].mat();
			vp2 = vp2*(1/cv::norm(vp2));
			score.at<double>(i,j) = vp1.dot(vp2);
		}
	}
	//cout<<score<<endl;
	vector<vector<int>> pairIdx;
	for (int i=0; i<score.rows; ++i) {
		vector<int> onePairIdx;
		double maxVal;
		cv::Point maxPos;
		cv::minMaxLoc(score.row(i),NULL,&maxVal,NULL,&maxPos);
		if (maxVal > cos(10*PI/180)) { // angle should be smaller than 10deg
			double maxV;
			cv::Point maxP;
			cv::minMaxLoc(score.col(maxPos.x),NULL,&maxV,NULL,&maxP);
			if (i==maxP.y) {
				onePairIdx.push_back(i);
				onePairIdx.push_back(maxPos.x);
				pairIdx.push_back(onePairIdx);
				//		cout<<i<<","<<maxPos.x<<"\t"<<acos(maxV)*180/PI<<endl;
				////// need a better metric to determine if R is good or not.................
				if( view1.vanishPoints[i].idlnLids.size() > 15 &&
                    view2.vanishPoints[maxPos.x].idlnLids.size() > 15 &&
                    acos(maxV)*180/PI > 5) { // 3 degree
               if (pairIdx.size() >2) 	{
                  cv::Mat vp1_1 = view1.K.inv() * view1.vanishPoints[1].mat();
                  cv::Mat vp2_1 = view1.K.inv() * view1.vanishPoints[2].mat();
                  cv::Mat vp1_2 = view2.K.inv() * view2.vanishPoints[1].mat();
                  cv::Mat vp2_2 = view2.K.inv() * view2.vanishPoints[2].mat();

                  double ang1 = acos(abs(vp1_1.dot(vp2_1))/cv::norm(vp1_1)/cv::norm(vp2_1))*180/PI;
                  double ang2 = acos(abs(vp1_2.dot(vp2_2))/cv::norm(vp1_2)/cv::norm(vp2_2))*180/PI;
                  if (abs(ang1 - ang2) < 5) // inner angle is consistent
                     goodR = false;
               } else
                  goodR = false;
				}
			}
		}
	}
	return pairIdx;
}
