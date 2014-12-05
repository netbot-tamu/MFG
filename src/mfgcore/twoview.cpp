
#include "twoview.h"

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

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

#include "utils.h"
#include "consts.h"
#include "mfgutils.h"

#include "levmar.h"

//#define DEBUG

using namespace std;

extern int IDEAL_IMAGE_WIDTH;
extern double THRESH_POINT_MATCH_RATIO;


TwoView::TwoView(View& v1, View& v2)
{
	view1 = v1;
	view2 = v2;
	view1.R = cv::Mat::eye(3,3,CV_64F);
	MyTimer t;
	// 0. point match
	t.start();
	matchKeyPoints (view1.featurePoints, view2.featurePoints, featPtMatches);
	t.end(); cout<<"point matching : "<<t.time_ms<<"ms"<<endl;

//	drawFeatPointMatches();
//	computeEpipolar (featPtMatches, view1.K, F, R, E, this->t);

	vector<cv::Mat> Fs, Es, Rs, ts;
	vector<vector<int>> pairIdx;
	computePotenEpipolar (featPtMatches,pairIdx, view1.K, Fs, Es, Rs, ts);


	triangulateFeatPoints();
	drawFeatPointMatches();

	view2.R = R;
	vpPairIdx = matchVanishPts();

	t.start();
	matchIdealLines(1);
	t.end(); cout<<"IdealLine matching : "<<t.time_ms<<"ms"<<endl;

	drawLineMatches(view1.img,view2.img, view1.idealLines, view2.idealLines, ilinePairIdx);

	detectPlanes ();
	//	triangulateIdealLines();
	//	optimize();
	//	triangulateFeatPoints();
	cout<<this->t<<endl<<R<<endl;
}

vector<vector<int>> TwoView::matchVanishPts()
{
	cv::Mat score(view1.vanishPoints.size(), view2.vanishPoints.size(), CV_64F);
	for(int i=0; i<view1.vanishPoints.size(); ++i) {
		cv::Mat vp1 = view1.R.t() * view1.K.inv() * view1.vanishPoints[i].mat(); // in world coord
		vp1 = vp1*(1/cv::norm(vp1));
		for (int j=0; j< view2.vanishPoints.size(); ++j)	{
			cv::Mat vp2 = view2.R.t() * view2.K.inv() * view2.vanishPoints[j].mat();
			vp2 = vp2*(1/cv::norm(vp2));
			score.at<double>(i,j) = vp1.dot(vp2);
		}
	}
	cout<<"R="<<view2.R<<endl;
	cout<<score<<endl;
	vector<vector<int>> pairIdx;
	for (int i=0; i<score.rows; ++i) {
		vector<int> onePairIdx;
		double maxVal;
		cv::Point maxPos;
		cv::minMaxLoc(score.row(i),NULL,&maxVal,NULL,&maxPos);
		if (maxVal > 0.8) {
			double maxV;
			cv::Point maxP;
			cv::minMaxLoc(score.col(maxPos.x),NULL,&maxV,NULL,&maxP);
			if (i==maxP.y) {
				onePairIdx.push_back(i);
				onePairIdx.push_back(maxPos.x);
				pairIdx.push_back(onePairIdx);
				cout<<i<<","<<maxPos.x<<"\t"<<acos(maxV)*180/PI<<endl;
			}
		}
	}
	return pairIdx;
}

void TwoView::matchIdealLines(bool usePtMatch)
	// when point matching is available,
	//
{
	//== gather lines into groups by vp ==
	vector<vector<IdealLine2d>> grpLines1(view1.vpGrpIdLnIdx.size()),
		grpLines2(view2.vpGrpIdLnIdx.size());

	for (int i =0; i<view1.vpGrpIdLnIdx.size(); ++i){
		for (int j=0; j<view1.vpGrpIdLnIdx[i].size(); ++j){
			grpLines1[i].push_back(view1.idealLines[view1.vpGrpIdLnIdx[i][j]]);
		}
	}
	for (int i =0; i<view2.vpGrpIdLnIdx.size(); ++i){
		for (int j=0; j<view2.vpGrpIdLnIdx[i].size(); ++j){
			grpLines2[i].push_back(view2.idealLines[view2.vpGrpIdLnIdx[i][j]]);
		}
	}

	MyTimer t;
	t.start();
	//==  point based line matching ==
	if (usePtMatch)	{
		for (int i=0; i<vpPairIdx.size(); ++i)	{
			int vpIdx1 = vpPairIdx[i][0], vpIdx2 = vpPairIdx[i][1];
			matchLinesByPointPairs (view1.img.cols,	grpLines1[vpIdx1],
				grpLines2[vpIdx2],	featPtMatches, ilinePairIdx);
		}
		// check gradient consistency and msld similarity
		for (int i=0; i<ilinePairIdx.size(); ++i) {
			if((view1.idealLines[ilinePairIdx[i][0]].gradient.dot(
				view2.idealLines[ilinePairIdx[i][1]].gradient) < 0)
				||(compMsldDiff(view1.idealLines[ilinePairIdx[i][0]],
				view2.idealLines[ilinePairIdx[i][1]]) > 0.8) )
			{
				ilinePairIdx.erase(ilinePairIdx.begin()+i);
				i--;
			}
		}
		t.end();
		cout<<"point-based = "<<t.time_ms<<"\t";
	}
	// assign gid of matched lines
	for (int i=0; i<ilinePairIdx.size(); ++i) {
		view1.idealLines[ilinePairIdx[i][0]].gid = i;
		view2.idealLines[ilinePairIdx[i][1]].gid = i;
	}
	//	drawLineMatches(view1.img,view2.img, view1.idealLines,
	//		view2.idealLines, ilinePairIdx);

	//== again, gather lines into groups by vp, excluding matched ones ==

	for (int i =0; i<view1.vpGrpIdLnIdx.size(); ++i){
		grpLines1[i].clear();
		for (int j=0; j<view1.vpGrpIdLnIdx[i].size(); ++j){
			if (view1.idealLines[view1.vpGrpIdLnIdx[i][j]].gid >= 0)
				continue;
			grpLines1[i].push_back(view1.idealLines[view1.vpGrpIdLnIdx[i][j]]);
		}
	}
	for (int i =0; i<view2.vpGrpIdLnIdx.size(); ++i){
		grpLines2[i].clear();
		for (int j=0; j<view2.vpGrpIdLnIdx[i].size(); ++j){
			if (view2.idealLines[view2.vpGrpIdLnIdx[i][j]].gid >= 0)
				continue;
			grpLines2[i].push_back(view2.idealLines[view2.vpGrpIdLnIdx[i][j]]);
		}
	}

	t.start();
	//== F-guided linematching ==
	vector<vector<int>> ilinePairIdx_F;
	for (int i=0; i<vpPairIdx.size(); ++i)	{
		int vpIdx1 = vpPairIdx[i][0], vpIdx2 = vpPairIdx[i][1];
		vector<vector<int>> tmpPairs;
		tmpPairs = F_guidedLinematch (F, grpLines1[vpIdx1], grpLines2[vpIdx2],
			view1.img, view2.img);
		ilinePairIdx_F.insert(ilinePairIdx_F.end(),tmpPairs.begin(),tmpPairs.end());
	}
	t.end();
	cout<<"F-guided = "<<t.time_ms<<endl;

	//	drawLineMatches(view1.img,view2.img, view1.idealLines,
	//					view2.idealLines, ilinePairIdx_F);
	ilinePairIdx.insert(ilinePairIdx.end(),ilinePairIdx_F.begin(),ilinePairIdx_F.end());
	//	if (usePtMatch)
	//		drawLineMatches(view1.img,view2.img, view1.idealLines,
	//		view2.idealLines, ilinePairIdx);
}

void  matchIdealLines(View& view1, View& view2, vector<vector<int>> vpPairIdx,
	vector<vector<cv::Point2d>> featPtMatches, cv::Mat F, vector<vector<int>>& ilinePairIdx,
	bool usePtMatch)
{
	//== gather lines into groups by vp ==
	vector<vector<IdealLine2d>> grpLines1(view1.vpGrpIdLnIdx.size()),
		grpLines2(view2.vpGrpIdLnIdx.size());

	for (int i =0; i<view1.vpGrpIdLnIdx.size(); ++i){
		for (int j=0; j<view1.vpGrpIdLnIdx[i].size(); ++j){
			grpLines1[i].push_back(view1.idealLines[view1.vpGrpIdLnIdx[i][j]]);
		}
	}
	for (int i =0; i<view2.vpGrpIdLnIdx.size(); ++i){
		for (int j=0; j<view2.vpGrpIdLnIdx[i].size(); ++j){
			grpLines2[i].push_back(view2.idealLines[view2.vpGrpIdLnIdx[i][j]]);
		}
	}

	MyTimer t;
	t.start();
	//==  point based line matching ==
	if (usePtMatch)	{
		for (int i=0; i<vpPairIdx.size(); ++i)	{
			int vpIdx1 = vpPairIdx[i][0], vpIdx2 = vpPairIdx[i][1];
			matchLinesByPointPairs (view1.img.cols,	grpLines1[vpIdx1],
				grpLines2[vpIdx2],	featPtMatches, ilinePairIdx);
		}
		// check gradient consistency and msld similarity
		for (int i=0; i<ilinePairIdx.size(); ++i) {
			if((view1.idealLines[ilinePairIdx[i][0]].gradient.dot(
				view2.idealLines[ilinePairIdx[i][1]].gradient) < 0)
				||(compMsldDiff(view1.idealLines[ilinePairIdx[i][0]],
				view2.idealLines[ilinePairIdx[i][1]]) > 0.8) )
			{
				ilinePairIdx.erase(ilinePairIdx.begin()+i);
				i--;
			}
		}
		t.end();
//		cout<<"point-based = "<<t.time_ms<<"\t";
	}

	// assign gid of matched lines
	for (int i=0; i<ilinePairIdx.size(); ++i) {
		view1.idealLines[ilinePairIdx[i][0]].lid = -1;
		view2.idealLines[ilinePairIdx[i][1]].lid = -1;
	}

	//	drawLineMatches(view1.img,view2.img, view1.idealLines,
	//		view2.idealLines, ilinePairIdx);

	//== again, gather lines into groups by vp, excluding matched ones ==

	for (int i =0;  i < view1.vpGrpIdLnIdx.size(); ++i){
		grpLines1[i].clear();
		for (int j=0; j<view1.vpGrpIdLnIdx[i].size(); ++j){
			if (view1.idealLines[view1.vpGrpIdLnIdx[i][j]].lid < 0)
				continue;
			grpLines1[i].push_back(view1.idealLines[view1.vpGrpIdLnIdx[i][j]]);
		}
	}
	for (int i =0; i<view2.vpGrpIdLnIdx.size(); ++i){
		grpLines2[i].clear();
		for (int j=0; j<view2.vpGrpIdLnIdx[i].size(); ++j){
			if (view2.idealLines[view2.vpGrpIdLnIdx[i][j]].lid < 0)
				continue;
			grpLines2[i].push_back(view2.idealLines[view2.vpGrpIdLnIdx[i][j]]);
		}
	}

	t.start();
	//== F-guided linematching ==
	vector<vector<int>> ilinePairIdx_F;
	for (int i=0; i<vpPairIdx.size(); ++i)	{
		int vpIdx1 = vpPairIdx[i][0], vpIdx2 = vpPairIdx[i][1];
		if(grpLines1[vpIdx1].size()==0 || grpLines2[vpIdx2].size()==0) continue;
		vector<vector<int>> tmpPairs;
		tmpPairs = F_guidedLinematch (F, grpLines1[vpIdx1], grpLines2[vpIdx2],
			view1.img, view2.img);
		ilinePairIdx_F.insert(ilinePairIdx_F.end(),tmpPairs.begin(),tmpPairs.end());
	}
	t.end();
//	cout<<"F-guided = "<<t.time_ms<<endl;

	//	drawLineMatches(view1.img,view2.img, view1.idealLines,
	//					view2.idealLines, ilinePairIdx_F);
	ilinePairIdx.insert(ilinePairIdx.end(),ilinePairIdx_F.begin(),ilinePairIdx_F.end());
	//	if (usePtMatch)
	//		drawLineMatches(view1.img,view2.img, view1.idealLines,
	//		view2.idealLines, ilinePairIdx);

	// ----- restore line lid ------
	for (int i=0; i < view1.idealLines.size(); ++i) {
		view1.idealLines[i].lid = i;
	}
	for (int i=0; i < view2.idealLines.size(); ++i) {
		view2.idealLines[i].lid = i;
	}

}

void TwoView::drawFeatPointMatches()
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

double compPlaneDepth (IdealLine2d a, IdealLine2d b, cv::Mat K, cv::Mat R,
	cv::Mat t, cv::Mat n)
	// given that a = H^T b, H = K(R-tn^T /d)K^(-1)
	// compute d
	// Ad = y
{
	cv::Mat A = vec2SkewMat(K.t()*a.lineEq())*R.t()*K.t()*b.lineEq(),
		y = vec2SkewMat(K.t()*a.lineEq())*n*t.t()*K.t()*b.lineEq();
	cv::Mat d = (A.t()*A).inv()*A.t()*y;
	return d.at<double>(0);
}

double compPlaneDepth (vector<IdealLine2d> a, vector<IdealLine2d> b,
	cv::Mat K, cv::Mat R, cv::Mat t, cv::Mat n)
{
	cv::Mat A(3*a.size(), 1, CV_64F), y(3*a.size(), 1, CV_64F);
	for(int i=0; i<a.size(); ++i) {
		cv::Mat tmp = vec2SkewMat(K.t()*a[i].lineEq())*R.t()*K.t()*b[i].lineEq();
		tmp.row(0).copyTo(A.row(i));
		tmp.row(1).copyTo(A.row(i+a.size()));
		tmp.row(2).copyTo(A.row(i+2*a.size()));
		cv::Mat tmpY = vec2SkewMat(K.t()*a[i].lineEq())*n*t.t()*K.t()*b[i].lineEq();
		y.at<double>(i) = tmpY.at<double>(0);
		y.at<double>(i+a.size()) = tmpY.at<double>(1);
		y.at<double>(i+2*a.size()) = tmpY.at<double>(2);
	}
	cv::Mat d =  (A.t()*A).inv()*A.t()*y;
	return d.at<double>(0);
}

cv::Mat compPlaneNormDepth (vector<IdealLine2d> a, vector<IdealLine2d> b,
	cv::Mat K, cv::Mat R, cv::Mat t)
	// size of a should be not less than 2
	// compute plane x = n/d
{
	cv::Mat A(2*a.size(), 3, CV_64F), y(2*a.size(), 1, CV_64F);
	for(int i=0; i<a.size(); ++i) {
		cv::Mat tmp = vec2SkewMat(K.t()*a[i].lineEq());
		tmp.row(0).copyTo(A.row(i));
		tmp.row(1).copyTo(A.row(i+a.size()));
		cv::Mat tmpY =
			tmp*R.t()*K.t()*b[i].lineEq()*(t.t()*K.t()*b[i].lineEq()).inv();
		y.at<double>(i) = tmpY.at<double>(0);
		y.at<double>(i+a.size()) = tmpY.at<double>(1);
	}

	cv::Mat n =  (A.t()*A).inv()*A.t()*y;
	return n;
}

struct DataOptPlaneDepth
{
	vector<IdealLine2d> ls1;
	vector<IdealLine2d> ls2;
	cv::Mat K, R, t, n;
};

void costFun_optPlaneDepth  (double *p, double *error, int m, int N, void *adata)
{
	struct DataOptPlaneDepth* dptr;
	dptr = (struct DataOptPlaneDepth *) adata;
	cv::Mat H = dptr->K*(dptr->R - dptr->t * dptr->n.t()/p[0])*dptr->K.inv();
	double cost=0;
	for (int i=0; i<N; ++i) {
		error[i] = ln2LnDist_H(dptr->ls1[i],dptr->ls2[i], H);
//		cost = cost+error[i]*error[i];
	}
//	cout<<"cost = "<<cost<<"\t"<<p[0]<<endl;
}

double optimizePlaneDepth (vector<IdealLine2d> a, vector<IdealLine2d> b,
	cv::Mat K, cv::Mat R, cv::Mat t, cv::Mat n, double d0)
	// given that a = H^T b, H = K(R-tn^T /d)K^(-1)
	// compute d
{
	int N = a.size();

	double* measurement = new double[N];
	for (int i=0; i<N; ++i)	measurement[i] = 0;
	double opts[LM_OPTS_SZ], info[LM_INFO_SZ];
	opts[0]=LM_INIT_MU*1;
	opts[1]=1E-15;
	opts[2]=1E-50;
	opts[3]=1E-20;
	opts[4]= LM_DIFF_DELTA;
	int matIter = 1000;
	double para[1] = {d0};

	DataOptPlaneDepth data;
	data.ls1 = a;
	data.ls2 = b;
	data.K = K;
	data.R = R;
	data.t = t;
	data.n = n;
	int ret = dlevmar_dif(costFun_optPlaneDepth, para, measurement, 1, N,
		matIter, opts, info, NULL, NULL, (void*)&data);
	return para[0];

}

struct DataOptPlaneNormDep
{
	vector<IdealLine2d> ls1;
	vector<IdealLine2d> ls2;
	cv::Mat K, R, t;
};

void costFun_optPlaneNormDep  (double *p, double *error, int m, int N, void *adata)
{
	struct DataOptPlaneDepth* dptr;
	dptr = (struct DataOptPlaneDepth *) adata;
	cv::Mat nd = (cv::Mat_<double>(3,1)<<p[0],p[1],p[2]);
	cv::Mat H = dptr->K*(dptr->R - dptr->t * nd.t())*dptr->K.inv();
	double cost=0;
	//	double mx=0;
	for (int i=0; i<N; ++i) {
		error[i] = ln2LnDist_H(dptr->ls1[i],dptr->ls2[i], H);
		//		if (mx < abs(ln2LnDist_H(dptr->ls1[i],dptr->ls2[i], H)))
		//			mx = ln2LnDist_H(dptr->ls1[i],dptr->ls2[i], H);
		cost = cost+error[i]*error[i];
	}
	//	error[0] = mx;
	//	cost = abs(error[0]);
//	cout<<cost<<"\t";
}

cv::Mat optimizePlaneNormDep (vector<IdealLine2d> a, vector<IdealLine2d> b,
	cv::Mat K, cv::Mat R, cv::Mat t, cv::Mat nd)
	// given that a = H^T b, H = K(R-tn^T /d)K^(-1)
	// compute n,d
{
	int N = a.size();

	double* measurement = new double[N];
	for (int i=0; i<N; ++i)	measurement[i] = 0;
	double opts[LM_OPTS_SZ], info[LM_INFO_SZ];
	opts[0]=LM_INIT_MU*1;
	opts[1]=1E-15;
	opts[2]=1E-50;
	opts[3]=1E-20;
	opts[4]= LM_DIFF_DELTA;
	int matIter = 1000;
	double para[3] = {nd.at<double>(0),nd.at<double>(1),nd.at<double>(2)};

	DataOptPlaneDepth data;
	data.ls1 = a;
	data.ls2 = b;
	data.K = K;
	data.R = R;
	data.t = t;
	int ret = dlevmar_dif(costFun_optPlaneNormDep, para, measurement, 3, N,
		matIter, opts, info, NULL, NULL, (void*)&data);

	cv::Mat result = (cv::Mat_<double>(3,1)<<para[0],para[1],para[2]);
	return result;

}

void TwoView::detectPlanes ()
	// first use rotation-homography to isolate infinite plane and lines on it
	// then use regular 3 line based ransac to find planes in near view

	// the plane normals can be determined from the horizontal vanishing points
	// With respect to first CCS
{
	double PdThresh = 20;  // plane depth threshold, to ignore far-view plane/ infinite plane
	double LnPrlxThresh = 5 * IDEAL_IMAGE_WIDTH/1000;  // parallax threshold
	double PtPrlxThresh = 2 * IDEAL_IMAGE_WIDTH/1000;

	vector<vector<int>> linePairs = ilinePairIdx;

	// =======  1. isolate infinite plane and lines/points on it ============
	cv::Mat Hp = view1.K * R * view1.K.inv();   // homography maps point from view 1 to view 2
	// ---- test homography -------
	for (int i=0; i < linePairs.size(); ++i) {
		IdealLine2d&	l1 = view1.idealLines[linePairs[i][0]];
		IdealLine2d&	l2 = view2.idealLines[linePairs[i][1]];
		IdealLine2d  l1in2 = IdealLine2d(
			LineSegmt2d(mat2cvpt(Hp * cvpt2mat(l1.extremity1)),
			mat2cvpt(Hp * cvpt2mat(l1.extremity2))) );
		if ( //abs(t.dot(ns[grp])) < 1.7
			// if vanishing point direction is near parallel to translation, parallax can't be large
			aveLine2LineDist(l1in2, l2) < LnPrlxThresh ) {
				linePairs.erase(linePairs.begin()+i);
				i--;
		}
	}
	drawLineMatches(view1.img,view2.img, view1.idealLines, view2.idealLines, linePairs);

	double lsDistThresh = 1;
	double ptDistThresh  = 1.0 * IDEAL_IMAGE_WIDTH/640.0;//3

	// ===== sequential ransac for homography ======
	int maxTotalIter = 3000, totalIter = 0;
	for (int planeId = 0; totalIter<maxTotalIter; ++planeId) {
		int maxIterNo = 500, iter=0; // for ransac
		double conf = 0.99; //confidence
		vector<int>  maxInliers;
		cv::Mat Hmax, n_max;
		double d_max;

		while (iter < maxIterNo) {
			++iter;	++totalIter;
			vector<int> curInliers;
			// --- 1. select mimimal solution set ---
			int i = rand()% linePairs.size();
			IdealLine2d&	l1 = view1.idealLines[linePairs[i][0]];
			IdealLine2d&	l2 = view2.idealLines[linePairs[i][1]];

			// --- 2. compute minimal solution(s) ---
			// --- assume plane normal is known ---
			vector<double> d;
			vector<cv::Mat> n, H;
			cv::Mat vi = view1.K.inv()*view1.vanishPoints[l1.vpLid].mat();
			for (int j=0; j < vpPairIdx.size(); ++j) {
				if (vpPairIdx[j][0] == l1.vpLid) continue;
				cv::Mat vj = view1.K.inv()*view1.vanishPoints[vpPairIdx[j][0]].mat();
				cv::Mat nv = vi.cross(vj); // plane normal
				nv = nv/cv::norm(nv);
				double depth = compPlaneDepth (l1, l2, view1.K, R, t, nv);
				cv::Mat Hm = view1.K*(R-t*nv.t()/depth)*view1.K.inv();
				n.push_back(nv);
				d.push_back(depth);
				H.push_back(Hm);
			}

			// --- 3. find consensus set ---
			vector<vector<int>> inliers(n.size());// counts for each solution

			for (int j=0; j < linePairs.size(); ++j) {
				IdealLine2d lj1 = view1.idealLines[linePairs[j][0]],
					lj2 = view2.idealLines[linePairs[j][1]];
				if(lj1.vpLid<0) continue;  // not consider
				cv::Mat vj = (view1.K.inv()*view1.vanishPoints[lj1.vpLid].mat());
				vj = vj/cv::norm(vj);
				for(int k=0; k < n.size(); ++k) {
					//not consider vp groups inconsistent with normal
					if (abs(vj.dot(n[k])) > cos(15*PI/180)) continue;
					double dist = ln2LnDist_H(lj1,lj2,H[k]);
					if (dist < lsDistThresh)
						inliers[k].push_back(j);
				}
			}
			int kk=0, mx=0;
			for(int k=0; k < n.size(); ++k) {
				if (inliers[k].size() > mx) {
					kk = k;
					mx = inliers[k].size();
				}
			}

			if (inliers[kk].size() > maxInliers.size())
			{
				maxInliers = inliers[kk];
				Hmax = H[kk];
				n_max = n[kk];
				d_max = d[kk];
//				cout<<n_max<<","<<d[kk]<<endl;
#ifdef DEBUG
				cv::Mat canv1 = view1.img.clone(),canv2 = view2.img.clone();
				cv::line(canv1,l1.extremity1,l1.extremity2,	cv::Scalar(200,0,0,0),2);
				cv::line(canv2, mat2cvpt(Hmax*cvpt2mat(l1.extremity1)),
					mat2cvpt(Hmax*cvpt2mat(l1.extremity2)),	cv::Scalar(200,200,0,0),2);
				cv::line(canv2,l2.extremity1,l2.extremity2, cv::Scalar(200,0,0,0),1);
				showImage("inlier = "+num2str(maxInliers.size()),&canv1);
				showImage("inlier-im2",&canv2);
				//			cv::waitKey(0);
				cv::destroyWindow("inlier = "+num2str(maxInliers.size()));
				cv::destroyWindow("inlier-im2");

#endif
			}
			// update maximum iteration number adaptively
		}
		vector<vector<int>> coplanarPairs;
		for (int i=0; i<maxInliers.size(); ++i)
			coplanarPairs.push_back(linePairs[maxInliers[i]]);
//		drawLineMatches(view1.img,view2.img, view1.idealLines,
//			view2.idealLines, coplanarPairs);

		// --- 4. re-optimization and guided matching ---
		vector<int> inliers;
		cv::Mat nd = n_max/d_max;
		cout<<endl<<nd/cv::norm(nd)<<","<<1/cv::norm(nd)<<endl;
		while(1) {
			inliers.clear();
			vector<IdealLine2d> a, b;
			for (int i=0; i<maxInliers.size(); ++i) {
				a.push_back(view1.idealLines[linePairs[maxInliers[i]][0]]);
				b.push_back(view2.idealLines[linePairs[maxInliers[i]][1]]);
			}
			// optimize n and d
	//		nd = optimizePlaneNormDep(a, b, view1.K, R, t, nd);
	//		cout<<endl<<nd/cv::norm(nd)<<","<<1/cv::norm(nd)<<endl;
	//		Hmax = view1.K*(R-t*nd.t())*view1.K.inv();

			// optimize depth only
					double dmax = compPlaneDepth (a, b, view1.K, R, t, n_max);
					dmax = optimizePlaneDepth(a, b, view1.K, R, t, n_max, d_max);
					Hmax = view1.K*(R-t*n_max.t()/dmax)*view1.K.inv();

			for (int j=0; j < linePairs.size(); ++j) {
				IdealLine2d lj1 = view1.idealLines[linePairs[j][0]],
					lj2 = view2.idealLines[linePairs[j][1]];
				if(lj1.vpLid<0) continue;  // not consider
				cv::Mat vj = (view1.K.inv()*view1.vanishPoints[lj1.vpLid].mat());
				vj = vj/cv::norm(vj);
				//	not consider vp groups inconsistent with normal
				if (abs(vj.dot(n_max)) > cos(20*PI/180)) continue;
				double dist = ln2LnDist_H(lj1,lj2,Hmax);
				if (dist < lsDistThresh)
					inliers.push_back(j);
			}
//			cout<<"inlier:"<<inliers.size()<<endl;
			coplanarPairs.clear();
			for (int i=0; i<inliers.size(); ++i)
				coplanarPairs.push_back(linePairs[inliers[i]]);
//			drawLineMatches(view1.img,view2.img, view1.idealLines,
//				view2.idealLines, coplanarPairs);
			if(inliers.size() > maxInliers.size())
				maxInliers = inliers;
			else
				break;
		}
		cout<<endl<<nd/cv::norm(nd)<<","<<1/cv::norm(nd)<<endl;
		if(maxInliers.size()<10)
			continue;
		drawLineMatches(view1.img,view2.img, view1.idealLines,
				view2.idealLines, coplanarPairs);
		// --- 5. remove from
		PrimPlane3d pp(nd,primaryPlanes.size());
		primaryPlanes.push_back(pp);

		for (int i= maxInliers.size()-1; i>-1; --i) {
			view1.idealLines[linePairs[maxInliers[i]][0]].pGid = pp.gid;
			view2.idealLines[linePairs[maxInliers[i]][1]].pGid = pp.gid;
			linePairs.erase(linePairs.begin()+ maxInliers[i]);
		}
//		drawLineMatches(view1.img,view2.img, view1.idealLines,
//				view2.idealLines, linePairs);

	}
}

void detectPlanes_2Views (View& view1, View& view2, cv::Mat R, cv::Mat t, vector<vector<int>> vpPairIdx,
	vector<vector<int>> ilinePairIdx, vector <PrimPlane3d>&	primaryPlanes)
	// first use rotation-homography to isolate infinite plane and lines on it
	// then use regular 3 line based ransac to find planes in near view

	// the plane normals can be determined from the horizontal vanishing points
	// With respect to first CCS
{
	double PdThresh = 20;  // plane depth threshold, to ignore far-view plane/ infinite plane
	double LnPrlxThresh = 5 * IDEAL_IMAGE_WIDTH/1000;  // parallax threshold
	double PtPrlxThresh = 2 * IDEAL_IMAGE_WIDTH/1000;

	vector<vector<int>> linePairs = ilinePairIdx;

	// =======  1. isolate infinite plane and lines/points on it ============
	cv::Mat Hp = view1.K * R * view1.K.inv();   // homography maps point from view 1 to view 2
	// ---- test homography -------
	for (int i=0; i < linePairs.size(); ++i) {
		IdealLine2d&	l1 = view1.idealLines[linePairs[i][0]];
		IdealLine2d&	l2 = view2.idealLines[linePairs[i][1]];
		double prlx = compParallax(l1, l2, view1.K, view1.R, view2.R);
		if ( //abs(t.dot(ns[grp])) < 1.7
			// if vanishing point direction is near parallel to translation, parallax can't be large
			prlx < LnPrlxThresh ) {
				linePairs.erase(linePairs.begin()+i);
				i--;
		}
	}
//	drawLineMatches(view1.img,view2.img, view1.idealLines, view2.idealLines, linePairs);

	double lsDistThresh = 1;
	double ptDistThresh  = 1.0 * IDEAL_IMAGE_WIDTH/640.0;//3

	// ===== sequential ransac for homography ======
	int maxTotalIter = 3000, totalIter = 0;
	for (int planeId = 0; totalIter<maxTotalIter; ++planeId) {
		int maxIterNo = 500, iter=0; // for ransac
		double conf = 0.99; //confidence
		vector<int>  maxInliers;
		cv::Mat Hmax, n_max;
		double d_max;

		while (iter < maxIterNo) {
			++iter;	++totalIter;
			vector<int> curInliers;
			// --- 1. select mimimal solution set ---
			int i = rand()% linePairs.size();
			IdealLine2d&	l1 = view1.idealLines[linePairs[i][0]];
			IdealLine2d&	l2 = view2.idealLines[linePairs[i][1]];

			// --- 2. compute minimal solution(s) ---
			// --- assume plane normal is known ---
			vector<double> d;
			vector<cv::Mat> n, H;
			cv::Mat vi = view1.K.inv()*view1.vanishPoints[l1.vpLid].mat();
			for (int j=0; j < vpPairIdx.size(); ++j) {
				if (vpPairIdx[j][0] == l1.vpLid) continue;
				cv::Mat vj = view1.K.inv()*view1.vanishPoints[vpPairIdx[j][0]].mat();
				cv::Mat nv = vi.cross(vj); // plane normal
				nv = nv/cv::norm(nv);
				double depth = compPlaneDepth (l1, l2, view1.K, R, t, nv);
				cv::Mat Hm = view1.K*(R-t*nv.t()/depth)*view1.K.inv();
				n.push_back(nv);
				d.push_back(depth);
				H.push_back(Hm);
			}

			// --- 3. find consensus set ---
			vector<vector<int>> inliers(n.size());// counts for each solution

			for (int j=0; j < linePairs.size(); ++j) {
				IdealLine2d lj1 = view1.idealLines[linePairs[j][0]],
					lj2 = view2.idealLines[linePairs[j][1]];
				if(lj1.vpLid<0) continue;  // not consider
				cv::Mat vj = (view1.K.inv()*view1.vanishPoints[lj1.vpLid].mat());
				vj = vj/cv::norm(vj);
				for(int k=0; k < n.size(); ++k) {
					//not consider vp groups inconsistent with normal
					if (abs(vj.dot(n[k])) > cos(15*PI/180)) continue;
					double dist = ln2LnDist_H(lj1,lj2,H[k]);
					if (dist < lsDistThresh)
						inliers[k].push_back(j);
				}
			}
			int kk=0, mx=0;
			for(int k=0; k < n.size(); ++k) {
				if (inliers[k].size() > mx) {
					kk = k;
					mx = inliers[k].size();
				}
			}

			if (inliers[kk].size() > maxInliers.size())
			{
				maxInliers = inliers[kk];
				Hmax = H[kk];
				n_max = n[kk];
				d_max = d[kk];
//				cout<<n_max<<","<<d[kk]<<endl;
#ifdef DEBUG
				cv::Mat canv1 = view1.img.clone(),canv2 = view2.img.clone();
				cv::line(canv1,l1.extremity1,l1.extremity2,	cv::Scalar(200,0,0,0),2);
				cv::line(canv2, mat2cvpt(Hmax*cvpt2mat(l1.extremity1)),
					mat2cvpt(Hmax*cvpt2mat(l1.extremity2)),	cv::Scalar(200,200,0,0),2);
				cv::line(canv2,l2.extremity1,l2.extremity2, cv::Scalar(200,0,0,0),1);
				showImage("inlier = "+num2str(maxInliers.size()),&canv1);
				showImage("inlier-im2",&canv2);
				//			cv::waitKey(0);
				cv::destroyWindow("inlier = "+num2str(maxInliers.size()));
				cv::destroyWindow("inlier-im2");

#endif
			}
			// update maximum iteration number adaptively
		}
		vector<vector<int>> coplanarPairs;
		for (int i=0; i<maxInliers.size(); ++i)
			coplanarPairs.push_back(linePairs[maxInliers[i]]);

		// --- 4. re-optimization and guided matching ---
		vector<int> inliers;
		cv::Mat nd = n_max/d_max;
		cout<<endl<<nd/cv::norm(nd)<<","<<1/cv::norm(nd)<<endl;
		while(1) {
			inliers.clear();
			vector<IdealLine2d> a, b;
			for (int i=0; i<maxInliers.size(); ++i) {
				a.push_back(view1.idealLines[linePairs[maxInliers[i]][0]]);
				b.push_back(view2.idealLines[linePairs[maxInliers[i]][1]]);
			}
			// optimize n and d
	//		nd = optimizePlaneNormDep(a, b, view1.K, R, t, nd);
	//		cout<<endl<<nd/cv::norm(nd)<<","<<1/cv::norm(nd)<<endl;
	//		Hmax = view1.K*(R-t*nd.t())*view1.K.inv();

			// optimize depth only
					double dmax = compPlaneDepth (a, b, view1.K, R, t, n_max);
					dmax = optimizePlaneDepth(a, b, view1.K, R, t, n_max, d_max);
					Hmax = view1.K*(R-t*n_max.t()/dmax)*view1.K.inv();
					nd = n_max/dmax;

			for (int j=0; j < linePairs.size(); ++j) {
				IdealLine2d lj1 = view1.idealLines[linePairs[j][0]],
					lj2 = view2.idealLines[linePairs[j][1]];
				if(lj1.vpLid<0) continue;  // not consider
				cv::Mat vj = (view1.K.inv()*view1.vanishPoints[lj1.vpLid].mat());
				vj = vj/cv::norm(vj);
				//	not consider vp groups inconsistent with normal
				if (abs(vj.dot(n_max)) > cos(20*PI/180)) continue;
				double dist = ln2LnDist_H(lj1,lj2,Hmax);
				if (dist < lsDistThresh)
					inliers.push_back(j);
			}
			coplanarPairs.clear();
			for (int i=0; i<inliers.size(); ++i)
				coplanarPairs.push_back(linePairs[inliers[i]]);
			if(inliers.size() > maxInliers.size())
				maxInliers = inliers;
			else
				break;
		}

		cout<<nd/cv::norm(nd)<<","<<1/cv::norm(nd)<<endl;

		if(maxInliers.size() < 10)
			continue;
//		drawLineMatches(view1.img,view2.img, view1.idealLines,
//				view2.idealLines, coplanarPairs);
		// --- 5. remove from
		PrimPlane3d pp(nd,primaryPlanes.size());
		primaryPlanes.push_back(pp);

		for (int i= maxInliers.size()-1; i>-1; --i) {
			view1.idealLines[linePairs[maxInliers[i]][0]].pGid = pp.gid;
			view2.idealLines[linePairs[maxInliers[i]][1]].pGid = pp.gid;
			linePairs.erase(linePairs.begin()+ maxInliers[i]);
		}
	}
}

void TwoView::triangulateIdealLines()
{
	cv::Mat P1 = (cv::Mat_<double>(3,4)<<1,0,0,0,
		0,1,0,0,
		0,0,1,0);
	cv::Mat P2(3,4,CV_64F);
	R.copyTo(P2.colRange(0,3)); 	 t.copyTo(P2.col(3));
	CvMat cvP1 = P1, cvP2 = P2; // projection matrix
	cv::Mat A1(4,1,CV_64F), A2(4,1,CV_64F);
	for(int i=0; i<ilinePairIdx.size(); ++i) {
		IdealLine2d a = view1.idealLines[ilinePairIdx[i][0]],
					b = view2.idealLines[ilinePairIdx[i][1]];

		IdealLine3d line = triangluateLine (cv::Mat::eye(3,3,CV_64F), cv::Mat::zeros(3,1,CV_64F),
									 R, t, view1.K, a, b );
		idealLines.push_back(line);
	/*	cv::Mat a1_r = cvpt2mat(mat2cvpt(view1.K.inv()*
						   (b.lineEq().cross(F*cvpt2mat(a.extremity1,1)))),0),
					a2_r = cvpt2mat(mat2cvpt(view1.K.inv()*
						   (b.lineEq().cross(F*cvpt2mat(a.extremity2,1)))),0),

					a1_l = cvpt2mat(mat2cvpt(view1.K.inv()*cvpt2mat(a.extremity1)),0),
					a2_l = cvpt2mat(mat2cvpt(view1.K.inv()*cvpt2mat(a.extremity2)),0);
		CvMat cv_a1_r = a1_r,
			  cv_a2_r = a2_r,
			  cv_a1_l = a1_l,
			  cv_a2_l = a2_l;

		CvMat cvA1 = A1, cvA2 = A2;
		cvTriangulatePoints(&cvP1, &cvP2, &cv_a1_l, &cv_a1_r, &cvA1);
		cvTriangulatePoints(&cvP1, &cvP2, &cv_a2_l, &cv_a2_r, &cvA2);
		cv::Mat A1_(&cvA1), A2_(&cvA2);
		A1 = A1_/A1_.at<double>(3);
		A2 = A2_/A2_.at<double>(3);
		idealLines.push_back(IdealLine3d(
			cv::Point3d(A1.at<double>(0),A1.at<double>(1),A1.at<double>(2)),
			cv::Point3d(A2.at<double>(0),A2.at<double>(1),A2.at<double>(2))));
   */
//		cv::line(canv1,mat2cvpt(view1.K*P1*A1),mat2cvpt(view1.K*P1*A2),color,1);
//		cv::line(canv2,mat2cvpt(view1.K*P2*A1),mat2cvpt(view1.K*P2*A2),color,1);

//		showImage("1",&canv1);showImage("2",&canv2);
//		cv::waitKey();
	}

}

void TwoView::triangulateFeatPoints()
{
	cv::Mat P1 = (cv::Mat_<double>(3,4)<<1,0,0,0,
										0,1,0,0,
										0,0,1,0);
	cv::Mat P2(3,4,CV_64F);
	R.copyTo(P2.colRange(0,3)); 	 t.copyTo(P2.col(3));
	CvMat cvP1 = P1, cvP2 = P2; // projection matrix
	cv::Mat X(4,1,CV_64F);

	for(int i=0; i<featPtMatches.size(); ++i) {
		cv::Mat x1 = cvpt2mat(mat2cvpt(view1.K.inv()*cvpt2mat(featPtMatches[i][0])),0),
		x2 = cvpt2mat(mat2cvpt(view1.K.inv()*cvpt2mat(featPtMatches[i][1])),0);
		CvMat cx1 = x1, cx2 = x2, cX = X;
		cvTriangulatePoints(&cvP1, &cvP2, &cx1, &cx2, &cX);
		cv::Mat X_(&cX);
		X = X_/X_.at<double>(3);
		keyPoints.push_back(KeyPoint3d(X.at<double>(0),X.at<double>(1),X.at<double>(2)));
	}
}

struct DataOpt
{
	vector<vector<cv::Point2d>> pointPairs;
	cv::Mat pts1, pts2;
	cv::Mat K;
	vector<IdealLine2d> lines2;
	vector<vector<int>>			vpPairIdx;
	vector <VanishPnt2d> vps1;
	vector<vector<int>> vpGrpIdLnIdx;

};

void costFun_opt(double *p, double *error, int N, int M, void *adata)
	// M measurement size
	// N para size
{
	struct DataOpt* dptr = (struct DataOpt *) adata;

	Eigen::Quaterniond q( p[0], p[1], p[2], p[3]);
	q.normalize();
	cv::Mat R = (cv::Mat_<double>(3,3)
		<< q.matrix()(0,0), q.matrix()(0,1), q.matrix()(0,2),
		q.matrix()(1,0), q.matrix()(1,1), q.matrix()(1,2),
		q.matrix()(2,0), q.matrix()(2,1), q.matrix()(2,2));

	cv::Mat t = (cv::Mat_<double>(3,1)<<p[4],p[5],p[6]);
	cv::Mat P1 = (cv::Mat_<double>(3,4)<<1,0,0,0,
										0,1,0,0,
										0,0,1,0);
	cv::Mat P2(3,4,CV_64F);
	R.copyTo(P2.colRange(0,3)); 	 t.copyTo(P2.col(3));
	P1 = dptr->K * P1;
	P2 = dptr->K * P2;
	double cost = 0;
	int idx = 0;
	cv::Mat X(4,dptr->pointPairs.size(),CV_64F), pt1(3,dptr->pointPairs.size(),CV_64F),
		pt2(3,dptr->pointPairs.size(),CV_64F);
	for (int i=0; i<dptr->pointPairs.size(); ++i)
	{
		X.at<double>(0,i) = p[7+i*3];
		X.at<double>(1,i) = p[7+i*3+1];
		X.at<double>(2,i) = p[7+i*3+2];
		X.at<double>(3,i) = 1;
/*		cv::Mat X = (cv::Mat_<double>(4,1)<<p[7+i*3],p[7+i*3+1],p[7+i*3+2],1);
		double dx1 = mat2cvpt(P1*X).x - dptr->pointPairs[i][0].x,
		dy1 = mat2cvpt(P1*X).y - dptr->pointPairs[i][0].y,
		dx2 = mat2cvpt(P2*X).x - dptr->pointPairs[i][1].x,
		dy2 = mat2cvpt(P2*X).y - dptr->pointPairs[i][1].y;
		error[i*4] = dx1;
		error[i*4+1] = dy1;
		error[i*4+2] = dx2;
		error[i*4+3] = dy2;
		cost = cost + dx1*dx1 + dy1*dy1 + dx2*dx2 + dy2*dy2;
*/
	}
	cv::Mat p1 = P1*X , p2 = P2*X - dptr->pts2;

	for(int i=0; i<dptr->pointPairs.size(); ++i)
	{
		error[i*4] = p1.at<double>(0,i)/p1.at<double>(2,i) - dptr->pointPairs[i][0].x;
		error[i*4+1] = p1.at<double>(1,i)/p1.at<double>(2,i) - dptr->pointPairs[i][0].y;
		error[i*4+2] = p2.at<double>(0,i)/p2.at<double>(2,i) - dptr->pointPairs[i][1].x;
		error[i*4+3] = p2.at<double>(1,i)/p2.at<double>(2,i) - dptr->pointPairs[i][1].y;
		cost = cost+error[i*4]*error[i*4] + error[i*4+1]*error[i*4+1] +
			error[i*4+2]*error[i*4+2] + error[i*4+3]*error[i*4+3];
	}

/*	idx = dptr->pointPairs.size()*4;
	for (int i=0; i<dptr->vpPairIdx.size(); ++i)
	{
		cv::Mat v2 = dptr->K*R*dptr->K.inv()*dptr->vps1[dptr->vpPairIdx[i][0]].mat();

		for (int j=0; j<dptr->vpGrpIdLnIdx[dptr->vpPairIdx[i][1]].size(); ++j)
		{
			cv::Mat l = dptr->lines2[dptr->vpGrpIdLnIdx[dptr->vpPairIdx[i][1]][j]].lineEq();
			error[idx] = point2LineDist(l, v2);
			cost = cost + error[idx]*error[idx];
			idx++;
		}
	}
*/
	cout<<cost<<'\t';
}

void TwoView::optimize()
// optimization over all features
{
	int M=0;
	for (int i=0; i<vpPairIdx.size(); ++i)
	{
		for (int j=0; j < view2.vpGrpIdLnIdx[vpPairIdx[i][1]].size(); ++j)
		{
	//		M++;
		}
	}

	M = M + 4*keyPoints.size(); // measurement size
	double* measurement = new double[M];
	for (int i=0; i<M; ++i) measurement[i] = 0;

	double opts[LM_OPTS_SZ], info[LM_INFO_SZ];
	opts[0]=LM_INIT_MU; //
	opts[1]=1E-15;
	opts[2]=1E-50; // original 1e-50
	opts[3]=1E-20;
	opts[4]= -LM_DIFF_DELTA;
	int maxIter = 500;

	Eigen::Matrix3d Rx;
	Rx << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
		R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
		R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2);
	Eigen::Quaterniond q(Rx);

	int N = 7+3*keyPoints.size();
	double * para = new double[N];
	para[0] = q.w();
	para[1] = q.x();
	para[2] = q.y();
	para[3] = q.z();
	para[4] = t.at<double>(0);
	para[5] = t.at<double>(1);
	para[6] = t.at<double>(2);
	for (int i=0; i < keyPoints.size(); ++i) {
		para[7+i*3]	  = keyPoints[i].x;
		para[7+i*3+1] = keyPoints[i].y;
		para[7+i*3+2] = keyPoints[i].z;
	}


	DataOpt data;
	data.pointPairs = featPtMatches;
	data.K = view1.K;
	data.lines2 = view2.idealLines;
	data.vpPairIdx = vpPairIdx;
	data.vps1 = view1.vanishPoints;
	data.vpGrpIdLnIdx = view2.vpGrpIdLnIdx;

	int ret = dlevmar_dif(costFun_opt, para, measurement, N, M,
						maxIter, opts, info, NULL, NULL, (void*)&data);
	delete[] measurement;
	q.w() = para[0];
	q.x() = para[1];
	q.y() = para[2];
	q.z() = para[3];
	q.normalize();
	for (int i=0; i<3; ++i)
		for (int j=0; j<3; ++j)
			R.at<double>(i,j) = q.matrix()(i,j);
	t.at<double>(0) = para[4];
	t.at<double>(1) = para[5];
	t.at<double>(2) = para[6];
/*
	for (int i=0; i < keyPoints.size(); ++i) {
		keyPoints[i].x = para[7+i*3];
		keyPoints[i].y = para[7+i*3+1];
		keyPoints[i].z = para[7+i*3+2];
	}
	*/
}

void TwoView::draw3D()
{
	// plot first camera, small
	glBegin(GL_LINES);
	glColor3f(1,0,0); // x-axis
	glVertex3f(0,0,0);
	glVertex3f(0.5,0,0);
	glColor3f(0,1,0);
	glVertex3f(0,0,0);
	glVertex3f(0,0.5,0);
	glColor3f(0,0,1);// z axis
	glVertex3f(0,0,0);
	glVertex3f(0,0,0.5);
	glEnd();

	// plot second camera
	cv::Mat c = -R.t()*t;
	cv::Mat xw = (cv::Mat_<double>(3,1)<< 1,0,0),
			yw = (cv::Mat_<double>(3,1)<< 0,1,0),
			zw = (cv::Mat_<double>(3,1)<< 0,0,1);
	cv::Mat x_ = R.t() * (xw-t),
			y_ = R.t() * (yw-t),
			z_ = R.t() * (zw-t);
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

	// plot mfg planes
	glBegin(GL_LINES);
	glColor3f((rand()%100)/100.0,(rand()%100)/100.0,(rand()%100)/100.0);

	for(int i =0; i<view1.idealLines.size(); ++i)
	{
		if (view1.idealLines[i].pGid <0) continue;
		cv::Mat edp1, edp2;
		projectImgPt2Plane(cvpt2mat(view1.idealLines[i].extremity1),
			primaryPlanes[view1.idealLines[i].pGid],view1.K, edp1);
		projectImgPt2Plane(cvpt2mat(view1.idealLines[i].extremity2),
			primaryPlanes[view1.idealLines[i].pGid],view1.K, edp2);
		glVertex3f(edp1.at<double>(0),edp1.at<double>(1),edp1.at<double>(2));
		glVertex3f(edp2.at<double>(0),edp2.at<double>(1),edp2.at<double>(2));
	}
	glEnd();


	glPointSize(2.0);
	glBegin(GL_POINTS);
	for (int i=0; i<keyPoints.size(); ++i)
	{
		glColor3f(1,0,0);
		glVertex3f(keyPoints[i].x, keyPoints[i].y, keyPoints[i].z);
	}
	glEnd();


	glColor3f(0,1,0);
	glBegin(GL_LINES);
	for(int i=0; i<idealLines.size(); ++i) {
		if(view1.idealLines[ilinePairIdx[i][0]].pGid <0) continue;
		if(idealLines[i].extremity1().z < 0 || idealLines[i].extremity2().z < 0)
			continue;
		glVertex3f(idealLines[i].extremity1().x,idealLines[i].extremity1().y,idealLines[i].extremity1().z);
		glVertex3f(idealLines[i].extremity2().x,idealLines[i].extremity2().y,idealLines[i].extremity2().z);
	}
	glEnd();
}


struct Data_optimizeRt_withVP
{
	vector<vector<cv::Point2d>> pointPairs;
	cv::Mat K;
	vector<vector<cv::Mat>>		vpPairs;
	double weightVP;
};

void costFun_optimizeRt_withVP (double *p, double *error, int N, int M, void *adata)
	// M measurement size
	// N para size
{
	struct Data_optimizeRt_withVP* dptr = (struct Data_optimizeRt_withVP *) adata;

	Eigen::Quaterniond q( p[0], p[1], p[2], p[3]);
	q.normalize();
	cv::Mat R = (cv::Mat_<double>(3,3)
		<< q.matrix()(0,0), q.matrix()(0,1), q.matrix()(0,2),
		q.matrix()(1,0), q.matrix()(1,1), q.matrix()(1,2),
		q.matrix()(2,0), q.matrix()(2,1), q.matrix()(2,2));

	double t_norm = sqrt(p[4]*p[4] + p[5]*p[5] + p[6]*p[6]);
	p[4] = p[4]/t_norm;
	p[5] = p[5]/t_norm;
	p[6] = p[6]/t_norm;
	cv::Mat t = (cv::Mat_<double>(3,1)<<p[4],p[5],p[6]);
	cv::Mat K = dptr->K;

	double weightVp = dptr->weightVP;
	double cost = 0;
	int	errEndIdx = 0;

	cv::Mat F = K.t().inv() * (vec2SkewMat(t) * R) * K.inv();

	for (int i=0; i<dptr->pointPairs.size(); ++i, ++errEndIdx)
	{
		error[errEndIdx] = sqrt(fund_samperr(cvpt2mat(dptr->pointPairs[i][0]),
			cvpt2mat(dptr->pointPairs[i][1]), F));
		cost += error[errEndIdx] * error[errEndIdx];
	}

	for (int i=0; i < dptr->vpPairs.size(); ++i, ++errEndIdx) {
		cv::Mat vp1 = K.inv() * dptr->vpPairs[i][0];
		cv::Mat vp2 = R.t() * K.inv() * dptr->vpPairs[i][1];
		vp1 = vp1/cv::norm(vp1);
		vp2 = vp2/cv::norm(vp2);
		error[errEndIdx] = cv::norm(vp1.cross(vp2)) *  weightVp;
		cost = cost + error[errEndIdx] * error[errEndIdx];
	}
//	cout<<cost<<'\t';
}

void optimizeRt_withVP (cv::Mat K, vector<vector<cv::Mat>> vppairs, double weightVP,
						vector<vector<cv::Point2d>>& featPtMatches,
						cv::Mat R, cv::Mat t)
// optimize relative pose (from 5-point alg) using vanishing point correspondences
// input: K, vppairs, featPtMatches
// output: R, t
{
	int numMeasure = vppairs.size() + featPtMatches.size();
	double* measurement = new double[numMeasure];
	for (int i=0; i < numMeasure; ++i) measurement[i] = 0;

	double opts[LM_OPTS_SZ], info[LM_INFO_SZ];
	opts[0]=LM_INIT_MU; //
	opts[1]=1E-15;
	opts[2]=1E-50; // original 1e-50
	opts[3]=1E-20;
	opts[4]= -LM_DIFF_DELTA;
	int maxIter = 5000;

	Eigen::Matrix3d Rx;
	Rx << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
		R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
		R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2);
	Eigen::Quaterniond q(Rx);

	int numPara = 7 ;
	double * para = new double[numPara];
	para[0] = q.w();
	para[1] = q.x();
	para[2] = q.y();
	para[3] = q.z();
	para[4] = t.at<double>(0);
	para[5] = t.at<double>(1);
	para[6] = t.at<double>(2);

	// --- pass additional data ---
	Data_optimizeRt_withVP data;
	data.pointPairs = featPtMatches;
	data.K = K;
	data.vpPairs = vppairs;
	data.weightVP =  weightVP;

	int ret = dlevmar_dif(costFun_optimizeRt_withVP, para, measurement, numPara, numMeasure,
						maxIter, opts, info, NULL, NULL, (void*)&data);
	delete[] measurement;
	q.w() = para[0];
	q.x() = para[1];
	q.y() = para[2];
	q.z() = para[3];
	q.normalize();
	for (int i=0; i<3; ++i)
		for (int j=0; j<3; ++j)
			R.at<double>(i,j) = q.matrix()(i,j);
	t.at<double>(0) = para[4];
	t.at<double>(1) = para[5];
	t.at<double>(2) = para[6];
/*
	for (int i=0; i < keyPoints.size(); ++i) {
		keyPoints[i].x = para[7+i*3];
		keyPoints[i].y = para[7+i*3+1];
		keyPoints[i].z = para[7+i*3+2];
	}
	*/
}


struct Data_optimize_t_givenR
{
	vector<vector<cv::Point2d>> pointPairs;
	cv::Mat K, R;
};

void costFun_optimize_t_givenR (double *p, double *error, int N, int M, void *adata)
	// M measurement size
	// N para size
{
	struct Data_optimize_t_givenR* dptr = (struct Data_optimize_t_givenR *) adata;

	cv::Mat R = dptr->R;

	double t_norm = sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
	p[0] = p[0]/t_norm;
	p[1] = p[1]/t_norm;
	p[2] = p[2]/t_norm;
	cv::Mat t = (cv::Mat_<double>(3,1)<<p[0],p[1],p[2]);
	cv::Mat K = dptr->K;

	double cost = 0;
	int	errEndIdx = 0;

	cv::Mat F = K.t().inv() * (vec2SkewMat(t) * R) * K.inv();

	for (int i=0; i<dptr->pointPairs.size(); ++i, ++errEndIdx)
	{
		error[errEndIdx] = sqrt(fund_samperr(cvpt2mat(dptr->pointPairs[i][0]),
			cvpt2mat(dptr->pointPairs[i][1]), F));
		cost += error[errEndIdx] * error[errEndIdx];
	}

	cout<<cost<<'\t';
}

void optimize_t_givenR (cv::Mat K, cv::Mat R, vector<vector<cv::Point2d>>& featPtMatches,
						cv::Mat t)
// optimize relative pose (from 5-point alg) using vanishing point correspondences
// input: K, R, vppairs, featPtMatches
// output: t
{
	int numMeasure = featPtMatches.size();
	double* measurement = new double[numMeasure];
	for (int i=0; i < numMeasure; ++i) measurement[i] = 0;

	double opts[LM_OPTS_SZ], info[LM_INFO_SZ];
	opts[0]=LM_INIT_MU; //
	opts[1]=1E-15;
	opts[2]=1E-50; // original 1e-50
	opts[3]=1E-20;
	opts[4]= -LM_DIFF_DELTA;
	int maxIter = 1000;

	int numPara = 3;
	double * para = new double[numPara];
	para[0] = t.at<double>(0);
	para[1] = t.at<double>(1);
	para[2] = t.at<double>(2);

	// --- pass additional data ---
	Data_optimize_t_givenR data;
	data.pointPairs = featPtMatches;
	data.K = K;
	data.R = R;

	int ret = dlevmar_dif(costFun_optimize_t_givenR, para, measurement, numPara, numMeasure,
						maxIter, opts, info, NULL, NULL, (void*)&data);
	delete[] measurement;
	t.at<double>(0) = para[0];
	t.at<double>(1) = para[1];
	t.at<double>(2) = para[2];
}

