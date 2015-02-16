#include <QtGui>
#include <QtOpenGL>
#include <gl/GLU.h>
#include <math.h>
#include <fstream>

#include "glwidget.h"
#include "mfg.h"
#include "mfg_utils.h"
#define DEBUG
extern int IDEAL_IMAGE_WIDTH, step;
extern double THRESH_POINT_MATCH_RATIO;
////////////////////////////  MfgTwoView    /////////////////////////////
MfgTwoView::MfgTwoView(string imageName1,string imageName2,cv::Mat K,
	cv::Mat dc)
{
	view1 = MfgSingleView(imageName1,K,dc);
	view2 = MfgSingleView(imageName2,K,dc);
	id = step;
	MyTimer timer;
	timer.start();  matchKeyPoints();  timer.end();
	//	drawPointMatches();

	getRotationMat ();
	//	timer.start();	matchLines();	timer.end();
	//	cout<< "Line matching time: "<<timer.time_ms<<endl;
	//	drawIdealLineMatches();

	findPlane_4Pts();	
	findPlane_3lines ();	
	drawPointMatches();
	//drawMappedLines();
}

MfgTwoView::MfgTwoView (MfgSingleView & v1, MfgSingleView v2)
{
	view1 = v1;
	view2 = v2;
	id = step;
	MyTimer timer;

	timer.start();  matchKeyPoints();  timer.end();
	cout<< "point matching time: "<<timer.time_ms<<endl;

	getRotationMat ();
	findPlane_4Pts();	
	drawPointMatches();

	//	timer.start();	matchLines();	timer.end();
	//	cout<< "Line matching time: "<<timer.time_ms<<endl;
//	drawIdealLineMatches();

	//	findPlane_3lines ();
	findPlane_3lines_hilgur();
	mergePrimaryPlanes();

	v1 = view1; // pass the changes of view1 out
}

void MfgTwoView::getRotationMat (bool isRect) // match line groups first
{
	// match vertical lines
	vector<vector<int>> lnMatches;
	matchLinesByPointPairs(&view1.mImgGray, &view2.mImgGray, 
		view1.idealLineGroups[0], view2.idealLineGroups[0],
		pointMatches, lnMatches);
	lineMatchesIdx.push_back(lnMatches);	
	cv::Mat 
		n0_1 = view1.camMat.inv()*view1.vanishPoints[0].matHomo(),
		n0_2 = view2.camMat.inv()*view2.vanishPoints[0].matHomo(),
		n1_1 = view1.camMat.inv()*view1.vanishPoints[1].matHomo(),
		n1_2 = view2.camMat.inv()*view2.vanishPoints[1].matHomo(),
		n2_1 = view1.camMat.inv()*view1.vanishPoints[2].matHomo(),
		n2_2 = view2.camMat.inv()*view2.vanishPoints[2].matHomo();
	// normalize vectors
	n0_1 = n0_1/cv::norm(n0_1);		n0_2 = n0_2/cv::norm(n0_2);
	n1_1 = n1_1/cv::norm(n1_1);		n1_2 = n1_2/cv::norm(n1_2);
	n2_1 = n2_1/cv::norm(n2_1);		n2_2 = n2_2/cv::norm(n2_2);

	if (n0_1.dot(n0_2)<0) 
		n0_2 = -n0_2;
	cv::Mat V1(3,3,CV_64F), V2(3,3,CV_64F), R;
	n0_1.copyTo(V1.col(0));
	n0_2.copyTo(V2.col(0));

	vector<vector<int>> lnMatches1,lnMatches2; 
	matchLinesByPointPairs(&view1.mImgGray, &view2.mImgGray, 
		view1.idealLineGroups[1], view2.idealLineGroups[1],
		pointMatches, lnMatches1);
	if (isRect) { // assume horizontal line groups are orthogonal
		if(abs(n1_1.dot(n2_1)) > 0.2 || abs(n1_2.dot(n2_2)) > 0.2) {
			// horizontal groups are not orthogonal to each other
			// delete group 2
			lineMatchesIdx.push_back (lnMatches1);
			if (n1_1.dot(n1_2)<0) 
				n1_2 = -n1_2;
			n1_1.copyTo(V1.col(1));
			n1_2.copyTo(V2.col(1));
			cv::Mat n2_1x = n0_1.cross(n1_1)/cv::norm(n0_1.cross(n1_1));
			cv::Mat n2_2x = n0_2.cross(n1_2)/cv::norm(n0_2.cross(n1_2));			
			n2_1x.copyTo(V1.col(2));
			n2_2x.copyTo(V2.col(2));
			// remove hori-group 2
			view1.vanishPoints.erase(view1.vanishPoints.begin()+2);
			view1.lineSegmentGroups.erase(view1.lineSegmentGroups.begin()+2);
			view1.idealLineGroups.erase(view1.idealLineGroups.begin()+2);
			view1.lineMemberSegmentIdx.erase
				(view1.lineMemberSegmentIdx.begin()+2);
			view2.vanishPoints.erase(view2.vanishPoints.begin()+2);
			view2.lineSegmentGroups.erase(view2.lineSegmentGroups.begin()+2);
			view2.idealLineGroups.erase(view2.idealLineGroups.begin()+2);
			view2.lineMemberSegmentIdx.erase
				(view2.lineMemberSegmentIdx.begin()+2);
		} else {
			// check if necessary to swap group 1 and 2		

			// always swap view 1 when needed!
			matchLinesByPointPairs(&view1.mImgGray, &view2.mImgGray, 
				view1.idealLineGroups[2], view2.idealLineGroups[1],
				pointMatches, lnMatches2);
			if (lnMatches1.size() < lnMatches2.size()) {
				swap(n1_1, n2_1);
				swap(view1.vanishPoints[1],view1.vanishPoints[2]);
				swap(view1.lineSegmentGroups[1],view1.lineSegmentGroups[2]);
				swap(view1.idealLineGroups[1],view1.idealLineGroups[2]);
				swap(view1.lineMemberSegmentIdx[1], 
					view1.lineMemberSegmentIdx[2]);
				lineMatchesIdx.push_back (lnMatches2);
				/*	NO DELETE	// always swap view 2 when needed
				matchLinesByPointPairs(&view1.mImgGray, &view2.mImgGray, 
				view1.idealLineGroups[1], view2.idealLineGroups[2],
				pointMatches, lnMatches2);
				if (lnMatches1.size() < lnMatches2.size()) {
				swap(n1_2, n2_2);
				swap(view2.vanishPoints[1],view2.vanishPoints[2]);
				swap(view2.lineSegmentGroups[1],view2.lineSegmentGroups[2]);
				swap(view2.idealLineGroups[1],view2.idealLineGroups[2]);
				swap(view2.lineMemberSegmentIdx[1], 
				view2.lineMemberSegmentIdx[2]);
				lineMatchesIdx.push_back (lnMatches2);
				*/
			} else
				lineMatchesIdx.push_back (lnMatches1);
			lnMatches2.clear();
			matchLinesByPointPairs(&view1.mImgGray, &view2.mImgGray, 
				view1.idealLineGroups[2], view2.idealLineGroups[2],
				pointMatches, lnMatches2);
			lineMatchesIdx.push_back(lnMatches2);	

			if (n1_1.dot(n1_2)<0) 
				n1_2 = -n1_2;			
			if (n2_1.dot(n2_2)<0) 
				n2_2 = -n2_2;			
			n1_1.copyTo(V1.col(1));
			n1_2.copyTo(V2.col(1));
			n2_1.copyTo(V1.col(2)); 
			n2_2.copyTo(V2.col(2)); 
		}
	} 
	R = V2*V1.inv();
	cout<<"vpR="<<R<<cv::determinant(R)<<endl;
	rotMat_vpt = R;		
}

cv::Mat get_r_pts (cv::Mat p1, cv::Mat p2, cv::Mat n, cv::Mat K, cv::Mat R)
{
	cv::Mat lhs(3*p1.cols, 3, CV_64F), rhs(3*p1.cols, 1, CV_64F);
	for (int i=0; i<p1.cols; ++i) {
		cv::Mat temp = vec2SkewMat(p2.col(i))*K*n.dot(K.inv()*p1.col(i));		
		temp.row(0).copyTo(lhs.row(i*3));
		temp.row(1).copyTo(lhs.row(i*3+1));
		temp.row(2).copyTo(lhs.row(i*3+2));
		temp = vec2SkewMat(p2.col(i))*K*R*K.inv()*p1.col(i);
		temp.row(0).copyTo(rhs.row(i*3));
		temp.row(1).copyTo(rhs.row(i*3+1));
		temp.row(2).copyTo(rhs.row(i*3+2));
	}
	cv::Mat r;
	cv::solve(lhs,rhs,r, cv::DECOMP_SVD);
	return r;
}

cv::Mat get_r_ptln (cv::Mat p1, cv::Mat p2, cv::Mat l1, cv::Mat l2, 
	cv::Mat n, cv::Mat K, cv::Mat R) 
{
	cv::Mat lhs(2*p1.cols+l1.cols, 3, CV_64F),
		rhs(2*p1.cols+l1.cols, 1, CV_64F);
	for (int i=0; i<p1.cols; ++i) {
		cv::Mat temp = vec2SkewMat(p2.col(i))*K*n.dot(K.inv()*p1.col(i));		
		temp.row(0).copyTo(lhs.row(i*2));
		temp.row(1).copyTo(lhs.row(i*2+1));
		temp = vec2SkewMat(p2.col(i))*K*R*K.inv()*p1.col(i);
		temp.row(0).copyTo(rhs.row(i*2));
		temp.row(1).copyTo(rhs.row(i*2+1));
	}
	for (int j = 0; j<l1.cols; ++j) {
		cv::Mat tmp = vec2SkewMat(l1.col(j))*K.t().inv()*n*l2.col(j).t()*K;
		tmp.row(0).copyTo(lhs.row(2*p1.cols+j));
		tmp = vec2SkewMat(l1.col(j))*K.t().inv()*R.t()*K.t()*l2.col(j);
		tmp.row(0).copyTo(rhs.row(2*p1.cols+j));	
	}
	cv::Mat r;
	cv::solve(lhs,rhs,r, cv::DECOMP_SVD);
	return r;
}

cv::Mat get_r_lns (cv::Mat l1, cv::Mat l2, cv::Mat n, cv::Mat K, cv::Mat R)
{// a line pair only provide ONE independent equation!!!
	int m=3;
	cv::Mat lhs(m*(l1.cols), 3, CV_64F), rhs(m*(l1.cols), 1, CV_64F);
	for (int i=0; i<l1.cols; ++i) {
		cv::Mat tmp = vec2SkewMat(l1.col(i))*K.t().inv()*n*l2.col(i).t()*K;
		tmp.row(0).copyTo(lhs.row(i*m));
		tmp.row(1).copyTo(lhs.row(i*m+1));
		tmp.row(2).copyTo(lhs.row(i*m+2));
		cv::Mat tmp2 = vec2SkewMat(l1.col(i))*K.t().inv()*R.t()*K.t()*l2.col(i);
		tmp2.row(0).copyTo(rhs.row(i*m));
		tmp2.row(1).copyTo(rhs.row(i*m+1));
		tmp2.row(2).copyTo(rhs.row(i*m+2));
	}
	cv::Mat r;
	cv::solve(lhs,rhs,r, cv::DECOMP_SVD);
	return r;
}

void MfgTwoView::findPlane_3lines ()
	// the plane normals can be determined from the horizontal vanishing points
	// With respect to first CCS 
{
	cv::Mat // define 3 vanishing point direction
		n0 = view1.camMat.inv()*view1.vanishPoints[0].matHomo(), 
		n1 = view1.camMat.inv()*view1.vanishPoints[1].matHomo(),
		n2 = view1.camMat.inv()*view1.vanishPoints[2].matHomo();

	cv::Mat n01 = n0.cross(n1), n02 = n0.cross(n2);
	n01 = n01/cv::norm(n01);  // plane normal determined by n0 and n1
	n02 = n02/cv::norm(n02);  // 
	double lsDistThresh = 1.5* IDEAL_IMAGE_WIDTH/640.0;//12
	double ptDistThresh  = 1.0 * IDEAL_IMAGE_WIDTH/640.0;//3
	int sz0 = lineMatchesIdx[0].size(), 
		sz1 = lineMatchesIdx[1].size(),
		sz2 = 0;
	if (lineMatchesIdx.size()>2)
		sz2 = lineMatchesIdx[2].size();
	int N = sz0 + sz1 + sz2;
	vector<vector<int>> lnMchIdx;
	vector<MfgLineSegment> allLs1, allLs2;		
	for (int i=0; i<lineMatchesIdx.size(); ++i) {
		for (int j=0; j<lineMatchesIdx[i].size(); ++j) {
			lnMchIdx.push_back(lineMatchesIdx[i][j]);
			allLs1.push_back(view1.idealLineGroups[i][lineMatchesIdx[i][j][0]]);
			allLs2.push_back(view2.idealLineGroups[i][lineMatchesIdx[i][j][1]]);			
		}
	}
	vector<vector<vector<int>>> copyLnMchIdx = lineMatchesIdx;
	FeaturePointPairs copyPtMchs = pointMatches;
	// ===== sequential ransac for homography ======
	//****************
	int maxTotalIterNo = 1500*IDEAL_IMAGE_WIDTH/640.0; // for all iterations of "findPlane_3lines"
	int totalIter = 0;
	//****************
	while (totalIter < maxTotalIterNo) {
		int nPts = pointMatches.size();	
		cv::Mat pts1(3,nPts,CV_64F), pts2(3,nPts,CV_64F);
		for (int i=0; i<pointMatches.size(); ++i) {	
			cv::Mat p1 = (cv::Mat_<double>(3,1,CV_64F)<<
				pointMatches[i][0].x, pointMatches[i][0].y, 1),
				p2 = (cv::Mat_<double>(3,1,CV_64F)<<
				pointMatches[i][1].x, pointMatches[i][1].y, 1);
			p1.copyTo(pts1.col(i));
			p2.copyTo(pts2.col(i));
		}

		sz0 = lineMatchesIdx[0].size();
		sz1 = lineMatchesIdx[1].size();
		sz2 = 0;
		if (lineMatchesIdx.size()>2)
			sz2 = lineMatchesIdx[2].size();
		N = sz0 + sz1 + sz2;		
		int maxIterNo = 500, iter=0; // for ransac
		double conf = 0.99; //confidence
		vector<int>  maxInliers, maxPtInliers;
		cv::Mat Hmax, rmax;
		cv::Mat n_pi, n_pi_max;
		bool gotMiniSol = false;
		while (iter < maxIterNo && totalIter < maxTotalIterNo) {
			++iter;	++totalIter;
			gotMiniSol = false;
			vector<int> curInliers;
			// --- 1. select mimimal solution set (mss) ---
			int i,j,k;
			MfgLineSegment li1, li2, lj1, lj2, lk1, lk2;
			int vert, hori;
			if (sz1 * sz2 > 0 && sz1+sz2 > 2 )
				vert = rand() % 3;
			else
				vert = rand() % 2;			
			if ( sz1+sz2 > 0 )
				hori = rand() % (sz1+sz2);
			else
				continue;
			if (vert ==0)  {// one from group 0, two from another group
				if (sz0 < 1)
					break;
				i = rand() % sz0; // select 1 from group 0
				li1 = view1.idealLineGroups[0][lineMatchesIdx[0][i][0]];
				li2 = view2.idealLineGroups[0][lineMatchesIdx[0][i][1]];
				if (hori < sz1) { // select 2 from group 1	
					if (sz1 < 2)
						continue;
					j = rand()% sz1;
					lj1 = view1.idealLineGroups[1][lineMatchesIdx[1][j][0]];
					lj2 = view2.idealLineGroups[1][lineMatchesIdx[1][j][1]];
					for (int ii=0; ii<100; ++ii) {
						k = rand()% sz1;
						if (k!=j)
							break;
					}
					lk1 = view1.idealLineGroups[1][lineMatchesIdx[1][k][0]];
					lk2 = view2.idealLineGroups[1][lineMatchesIdx[1][k][1]];
					n_pi = n01;
					gotMiniSol = true;
				} else {   // select 2 from group 2	
					if (sz2 == 0)
						continue;
					j = rand()% sz2;
					lj1 = view1.idealLineGroups[2][lineMatchesIdx[2][j][0]];
					lj2 = view2.idealLineGroups[2][lineMatchesIdx[2][j][1]];
					for (int ii=0; ii<100; ++ii) {
						k = rand()% sz2;
						if (k!=j)
							break;
					}
					lk1 = view1.idealLineGroups[2][lineMatchesIdx[2][k][0]];
					lk2 = view2.idealLineGroups[2][lineMatchesIdx[2][k][1]];
					n_pi = n02;
					gotMiniSol = true;
				}
			} else if (vert == 1){ // two from group 0, one from another group
				if (sz0 < 2)
					continue;
				j = rand()% sz0;
				lj1 = view1.idealLineGroups[0][lineMatchesIdx[0][j][0]];
				lj2 = view2.idealLineGroups[0][lineMatchesIdx[0][j][1]];
				for (int ii=0; ii<100; ++ii) {
					i = rand()% sz0;
					if (i!=j)
						break;
				}
				li1 = view1.idealLineGroups[0][lineMatchesIdx[0][i][0]];
				li2 = view2.idealLineGroups[0][lineMatchesIdx[0][i][1]];
				if (hori < sz1) {					
					k = rand()% sz1;
					lk1 = view1.idealLineGroups[1][lineMatchesIdx[1][k][0]];
					lk2 = view2.idealLineGroups[1][lineMatchesIdx[1][k][1]];
					n_pi = n01;
					gotMiniSol = true;
				} else {
					if (sz2 == 0)
						continue;
					k = rand()% sz2;
					lk1 = view1.idealLineGroups[2][lineMatchesIdx[2][k][0]];
					lk2 = view2.idealLineGroups[2][lineMatchesIdx[2][k][1]];
					n_pi = n02;
					gotMiniSol = true;
				}
			} else {  // when vert = 2, pick only from horizontal groups
				if (hori < sz1) { // two from group 1
					if ( sz1 < 2 )
						continue;
					i = rand() % sz1; // select 1st from group 1
					li1 = view1.idealLineGroups[1][lineMatchesIdx[1][i][0]];
					li2 = view2.idealLineGroups[1][lineMatchesIdx[1][i][1]];
					for (int ii=0; ii<100; ++ii) {
						j = rand() % sz1;
						if (i!=j)
							break;
					}					
					lj1 = view1.idealLineGroups[1][lineMatchesIdx[1][j][0]];
					lj2 = view2.idealLineGroups[1][lineMatchesIdx[1][j][1]];
					k = rand()% sz2;
					lk1 = view1.idealLineGroups[2][lineMatchesIdx[2][k][0]];
					lk2 = view2.idealLineGroups[2][lineMatchesIdx[2][k][1]];
					n_pi = n0; // or n1 x n2
					gotMiniSol = true;
				}
				else {   // pick two from group 2 
					if ( sz2 < 2 )
						continue;
					i = rand() % sz2; // select 1st from group 1
					li1 = view1.idealLineGroups[2][lineMatchesIdx[2][i][0]];
					li2 = view2.idealLineGroups[2][lineMatchesIdx[2][i][1]];
					for (int ii=0; ii<100; ++ii) {
						j = rand() % sz2;
						if (i!=j)
							break;
					}					
					lj1 = view1.idealLineGroups[2][lineMatchesIdx[2][j][0]];
					lj2 = view2.idealLineGroups[2][lineMatchesIdx[2][j][1]];
					k = rand()% sz1;
					lk1 = view1.idealLineGroups[1][lineMatchesIdx[1][k][0]];
					lk2 = view2.idealLineGroups[1][lineMatchesIdx[1][k][1]];
					n_pi = n0; // or n1 x n2
					gotMiniSol = true;
				}
			}

			if ( !gotMiniSol )
				continue;
			// --- 2. compute minimal solution ---
			cv::Mat mss1(3,3,CV_64F), mss2(3,3,CV_64F);
			li1.matLnEq.copyTo (mss1.col(0));
			lj1.matLnEq.copyTo (mss1.col(1));
			lk1.matLnEq.copyTo (mss1.col(2));
			li2.matLnEq.copyTo (mss2.col(0));
			lj2.matLnEq.copyTo (mss2.col(1));
			lk2.matLnEq.copyTo (mss2.col(2));
			cv::Mat r = get_r_lns (mss1, mss2, n_pi, view1.camMat, rotMat_vpt);			
			cv::Mat H = view1.camMat*(rotMat_vpt-r*n_pi.t())*view1.camMat.inv();

			if (symmErr_LinePair(li1,li2,H)>lsDistThresh ||
				symmErr_LinePair(lj1,lj2,H)>lsDistThresh ||
				symmErr_LinePair(lk1,lk2,H)>lsDistThresh )
				//	continue;	
				;

			// --- 3. find consensus set ---
			for (int ii = 0; ii < allLs1.size(); ++ii) {
				if (cv::norm(n_pi - n01)==0 && ii>=sz0+sz1) continue;
				if (cv::norm(n_pi - n02)==0 && ii>=sz0 && ii<sz0+sz1) {
					continue;
				}
				double dis = symmErr_LinePair(allLs1[ii],allLs2[ii],H);			
				if (dis < lsDistThresh)
					curInliers.push_back(ii);
			}
			vector<int> curPtInliers;
			for (int i=0; i<nPts; ++i) {
				double dist_symm = symmErr_PtPair(pts1.col(i),pts2.col(i),H);
				if (dist_symm < ptDistThresh)
					curPtInliers.push_back(i);
			}		
			// update largest consensus set
			if (iter==0 || curInliers.size() + curPtInliers.size() >
				maxInliers.size() + maxPtInliers.size()) {
#ifdef DEBUG
					cv::Mat canv1 = view1.mImg.clone(),canv2 = view2.mImg.clone();
					cv::line(canv1,li1.ptA.cvPt(),li1.ptB.cvPt(),
						cv::Scalar(200,0,0,0),2);
					cv::line(canv1,lj1.ptA.cvPt(),lj1.ptB.cvPt(),
						cv::Scalar(200,0,0,0),2);
					cv::line(canv1,lk1.ptA.cvPt(),lk1.ptB.cvPt(),
						cv::Scalar(200,0,0,0),2);
					cv::line(canv2, Mfg2dPoint(H*li1.ptA.matHomo()).cvPt(),
						Mfg2dPoint(H*li1.ptB.matHomo()).cvPt(), 
						cv::Scalar(200,200,0,0),2);
					cv::line(canv2,li2.ptA.cvPt(),li2.ptB.cvPt(),
						cv::Scalar(200,0,0,0),1);
					cv::line(canv2, Mfg2dPoint(H*lj1.ptA.matHomo()).cvPt(),
						Mfg2dPoint(H*lj1.ptB.matHomo()).cvPt(),
						cv::Scalar(200,200,0,0),2);
					cv::line(canv2,lj2.ptA.cvPt(),lj2.ptB.cvPt(),
						cv::Scalar(200,0,0,0),1);
					cv::line(canv2, Mfg2dPoint(H*lk1.ptA.matHomo()).cvPt(),
						Mfg2dPoint(H*lk1.ptB.matHomo()).cvPt(),
						cv::Scalar(200,200,0,0),2);
					cv::line(canv2,lk2.ptA.cvPt(),lk2.ptB.cvPt(),
						cv::Scalar(200,0,0,0),1);
					showImage("inlier = "+num2str(curInliers.size())
						+","+num2str(curPtInliers.size()),&canv1);
					showImage("inlier-im2",&canv2);
					//					cv::waitKey(0);
					cv::destroyWindow("inlier = "+num2str(curInliers.size())
						+","+num2str(curPtInliers.size()));
					cv::destroyWindow("inlier-im2");
#endif
					maxInliers = curInliers;
					maxPtInliers = curPtInliers;
					Hmax = H;
					rmax = r;	
					n_pi_max = n_pi;					
			}
			// update maximum iteration number adaptively
			if (maxInliers.size() > 0)
				maxIterNo = abs(log(1-conf) 
				/log(1-pow(double(maxInliers.size())/N,mss1.cols)));//????				
		}	
		if ( !gotMiniSol )
			continue;

		// --- 4.re-estimation and guided matching for lines ---		
		double scale = 1.0;
		int ptInlierNum, maxPtNum = maxPtInliers.size();
		cv::Mat optH = Hmax, opt_r = rmax, opt_n_pi = n_pi_max;
		for (int ix = 0; ix >=0; ++ix ) {
			vector<MfgLineSegment> ls1, ls2;
			cv::Mat l1(3,maxInliers.size(),CV_64F), 
				l2(3,maxInliers.size(),CV_64F);
			for (int i=0; i<maxInliers.size(); ++i) {
				ls1.push_back(allLs1[maxInliers[i]]);
				ls2.push_back(allLs2[maxInliers[i]]);
				allLs1[maxInliers[i]].matLnEq.copyTo(l1.col(i));
				allLs2[maxInliers[i]].matLnEq.copyTo(l2.col(i));
			}
			if (ls1.size()>=9)
				mle_2dLineHomo (ls1, ls2, &Hmax); // 
			ptInlierNum = findPtsOnPlane (&Hmax, maxPtInliers);
			vector<cv::Point2d> ptInliers1, ptInliers2;
			cv::Mat pts1(3,maxPtInliers.size(), CV_64F), 
				pts2(3,maxPtInliers.size(), CV_64F)	;
			for (int i=0; i < maxPtInliers.size(); ++i) {
				ptInliers1.push_back(pointMatches[maxPtInliers[i]][0]);
				ptInliers2.push_back(pointMatches[maxPtInliers[i]][1]);
				Mfg2dPoint(ptInliers1.back().x,ptInliers1.back().y).matHomo().
					copyTo(pts1.col(i));
				Mfg2dPoint(ptInliers2.back().x,ptInliers2.back().y).matHomo().
					copyTo(pts2.col(i));
			}
			// compute initial point using DLT for optimization 
			if ( maxPtInliers.size() > 4 )
				rmax = get_r_pts (pts1, pts2, n_pi_max,view1.camMat,rotMat_epi);
			else if (ls1.size() > 3)
				//rmax = get_r_lns (l1, l2, n_pi_max, view1.camMat, rotMat_vpt);			
				optTranslate_lines 
				(ls1, ls2, rotMat_vpt, n_pi_max, view1.camMat, &rmax);

			// **** optimization ****
			if (rmax.dot(t_epi) > 0) 
				rmax = t_epi * cv::norm(rmax); // enforce rmax parallel to t_epi
			else {
				rmax = t_epi * cv::norm(rmax);
				n_pi_max = -n_pi_max;
			}
			if (ptInliers1.size()>=3) {
				//	opt_r_d_pts (ptInliers1,ptInliers2, rotMat_epi, view1.camMat, 
				//												&n_pi_max,&rmax);
				opt_r_d_ptlns (ptInliers1,ptInliers2,ls1, ls2, rotMat_epi, view1.camMat,
					&n_pi_max,&rmax);
			}

			vector<int> curInliers, curPtInliers;
			for (int i = 0; i < allLs1.size(); ++i) {
				if (cv::norm(n_pi_max - n01)<0.5 && i>=sz0+sz1) continue;
				if (cv::norm(n_pi_max - n02)<0.5 && i>=sz0 && i<sz0+sz1) {
					continue;
				}
				double dis = symmErr_LinePair(allLs1[i],allLs2[i],Hmax);				
				if ( dis < lsDistThresh * scale ) 
					curInliers.push_back(i);
			}
			if (ix==0 || 
				curInliers.size() + ptInlierNum > maxInliers.size()+maxPtNum) {
					scale = scale* pow(double(maxInliers.size())/curInliers.size(),0.5);
					maxInliers.clear();				
					maxInliers = curInliers;
					maxPtNum = ptInlierNum;
					optH = Hmax;
					opt_r = rmax;
					opt_n_pi = n_pi_max;	

			} else
				break;			
		}
		cv::Mat tmp = Hmax;
		if (ptInlierNum < 5 || maxInliers.size() < 5 )
			continue;

		primaryPlanes.push_back ( Mfg3dPlane(
			opt_n_pi.at<double>(0),	opt_n_pi.at<double>(1),	opt_n_pi.at<double>(2),
			sgn( opt_r.dot(t_epi) )/cv::norm(opt_r)));

		cout<<"n_pi = "<<opt_n_pi<< ","	<<1/cv::norm(opt_r)<<endl;

		cv::Mat canv1 = view1.mImg.clone(),canv2 = view2.mImg.clone();
		for (int i=0; i<maxInliers.size(); ++i) {
			bool iFrom0 = maxInliers[i] < sz0, 
				iFrom1 = maxInliers[i]>=sz0 && maxInliers[i]<sz0+sz1,
				iFrom2 = maxInliers[i]>=sz0+sz1;
			MfgLineSegment lk1,lk2;
			if (iFrom0) {
				lk1 = view1.idealLineGroups[0][lnMchIdx[maxInliers[i]][0]];
				lk2 = view2.idealLineGroups[0][lnMchIdx[maxInliers[i]][1]];
			}
			if (iFrom1) {
				lk1 = view1.idealLineGroups[1][lnMchIdx[maxInliers[i]][0]];
				lk2 = view2.idealLineGroups[1][lnMchIdx[maxInliers[i]][1]];
			}
			if (iFrom2) {				
				lk1 = view1.idealLineGroups[2][lnMchIdx[maxInliers[i]][0]];
				lk2 = view2.idealLineGroups[2][lnMchIdx[maxInliers[i]][1]];
			}
#ifdef DEBUG
			cv::Scalar color = cv::Scalar(rand()%255,rand()%255,rand()%255,0);
			cv::line(canv1,lk1.ptA.cvPt(),lk1.ptB.cvPt(),color,2);
			cv::line(canv2, Mfg2dPoint(optH*lk1.ptA.matHomo()).cvPt(),
				Mfg2dPoint(optH*lk1.ptB.matHomo()).cvPt(), color,2);
			cv::line(canv2,lk2.ptA.cvPt(),lk2.ptB.cvPt(),color,1);			
#endif
			primaryPlanes[primaryPlanes.size()-1].mem2dSegments.push_back(lk1);
		}

#ifdef DEBUG
		showImage("line inlier-im1",&canv1);
		showImage("line inlier-im2",&canv2);
		cv::waitKey();
		cv::destroyWindow("line inlier-im1");
		cv::destroyWindow("line inlier-im2");
#endif
		// --- 5. remove inliers from whole dataset ---
		removePlaneFeatures(allLs1, allLs2, lnMchIdx, maxInliers, maxPtInliers);
	}
	lineMatchesIdx.clear(); lineMatchesIdx = copyLnMchIdx;
	isolatePtMatches = pointMatches; // point pairs not on any plane~~~~
	pointMatches.clear();   pointMatches = copyPtMchs;
}

void MfgTwoView::findPlane_3lines_hilgur ()
	// first use rotation-homography to isolate infinite plane and lines on it
	// then use regular 3 line based ransac to find planes in near view

	// the plane normals can be determined from the horizontal vanishing points
	// With respect to first CCS 
{
	double PdThresh = 20;  // plane depth threshold, to ignore far-view plane/ infinite plane
	double LnPrlxThresh = 2 * IDEAL_IMAGE_WIDTH/1000;  // parallax threshold
	double PtPrlxThresh = 2 * IDEAL_IMAGE_WIDTH/1000;
	
	vector<vector<vector<int>>> copyLnMchIdx = lineMatchesIdx;
	FeaturePointPairs copyPtMchs = pointMatches;
	cv::Mat // define 3 vanishing point direction
		n0 = view1.camMat.inv() * view1.vanishPoints[0].matHomo(), 
		n1 = view1.camMat.inv() * view1.vanishPoints[1].matHomo(),
		n2 = view1.camMat.inv() * view1.vanishPoints[2].matHomo();
	
	vector<cv::Mat> ns;
	ns.push_back(n0/cv::norm(n0));
	ns.push_back(n1/cv::norm(n1));
	ns.push_back(n2/cv::norm(n2));

	// =======  1. isolate infinite plane and lines/points on it ============
	cv::Mat Hp = view1.camMat * rotMat_epi * view1.camMat.inv();   // homography maps point from view 1 to view 2 
	// ---- test homography -------
	for (int grp = 0; grp < lineMatchesIdx.size(); ++grp) {
		for (int i=0; i < lineMatchesIdx[grp].size(); ++i) {
			MfgLineSegment&	li1 = view1.idealLineGroups[grp][lineMatchesIdx[grp][i][0]];
			MfgLineSegment&	li2 = view2.idealLineGroups[grp][lineMatchesIdx[grp][i][1]];
			MfgLineSegment  li1in2 = 
				MfgLineSegment(Mfg2dPoint(Hp*li1.ptA.matHomo()).cvPt().x,
				Mfg2dPoint(Hp*li1.ptA.matHomo()).cvPt().y,
				Mfg2dPoint(Hp*li1.ptB.matHomo()).cvPt().x,
				Mfg2dPoint(Hp*li1.ptB.matHomo()).cvPt().y);
			//cv::Mat canv1 = view1.mImg.clone(), canv2 = view2.mImg.clone();
			//cv::line(canv1,li1.ptA.cvPt(),li1.ptB.cvPt(),
			//	cv::Scalar(200,0,0,0),2);
			//cv::line(canv2, Mfg2dPoint(Hp*li1.ptA.matHomo()).cvPt(),
			//	Mfg2dPoint(Hp*li1.ptB.matHomo()).cvPt(), 
			//	cv::Scalar(0,200,0,0),2);
			//cv::line(canv2,li1.ptA.cvPt(),li1.ptB.cvPt(),
			//	cv::Scalar(0,0,200,0),2);
			//cv::line(canv2,li2.ptA.cvPt(),li2.ptB.cvPt(),
			//	cv::Scalar(200,0,0,0),2);
			if ( abs(t_epi.dot(ns[grp])) < 1.7 
				// if vanishing point direction is near parallel to translation, parallax can't be large 
				&& aveLine2LineDist(li1in2, li2) < LnPrlxThresh ) {
					//lineMatchesIdx[grp].erase(lineMatchesIdx[grp].begin()+i);
					//i--;
					li1.lowPalx = true;
					//showImage("lowParallax1", &canv1);
					//showImage("lowParallax2", &canv2);
					//cout<<aveLine2LineDist(li1in2, li2)<<endl;
					//cv::waitKey();

			} else
				li1.lowPalx = false;			
		}		
	}
	drawIdealLineMatches();
	for (int i=0; i < pointMatches.size(); ++i) {
		cv::Point2d pi1 = pointMatches[i][0];
		cv::Point2d pi2 = pointMatches[i][1];
		cv::Point2d pi1_in2 = Mfg2dPoint(Hp* Mfg2dPoint(pi1.x,pi1.y).matHomo()).cvPt();
			if (cv::norm((pi2 - pi1_in2)) < PtPrlxThresh ) {
				pointMatches.erase(pointMatches.begin()+i);
				i--;
			}
		}
		
	cv::Mat n01 = n0.cross(n1), n02 = n0.cross(n2);
	n01 = n01/cv::norm(n01);  // plane normal determined by n0 and n1
	n02 = n02/cv::norm(n02);  // 
//	double lsDistThresh = 1.5* IDEAL_IMAGE_WIDTH/640.0;//12
	double lsDistThresh = std::tan(0.3*PI/180) ;// using dis/len ratio as threshold
	double ptDistThresh  = 1.0 * IDEAL_IMAGE_WIDTH/640.0;//3
	int sz0 = lineMatchesIdx[0].size(), 
		sz1 = lineMatchesIdx[1].size(),
		sz2 = 0;
	if (lineMatchesIdx.size()>2)
		sz2 = lineMatchesIdx[2].size();
	int N = sz0 + sz1 + sz2;
	vector<vector<int>> lnMchIdx;
	vector<MfgLineSegment> allLs1, allLs2;		
	for (int i=0; i<lineMatchesIdx.size(); ++i) {
		for (int j=0; j<lineMatchesIdx[i].size(); ++j) {
			lnMchIdx.push_back(lineMatchesIdx[i][j]);
			allLs1.push_back(view1.idealLineGroups[i][lineMatchesIdx[i][j][0]]);
			allLs2.push_back(view2.idealLineGroups[i][lineMatchesIdx[i][j][1]]);			
		}
	}

	// ===== sequential ransac for homography ======
	//****************
	int maxTotalIterNo = 2500*IDEAL_IMAGE_WIDTH/640.0; // for all iterations of "findPlane_3lines"
	int totalIter = 0;
	//****************
	while (totalIter < maxTotalIterNo) {
		int nPts = pointMatches.size();	
		cv::Mat pts1(3,nPts,CV_64F), pts2(3,nPts,CV_64F);
		for (int i=0; i<pointMatches.size(); ++i) {	
			cv::Mat p1 = (cv::Mat_<double>(3,1,CV_64F)<<
				pointMatches[i][0].x, pointMatches[i][0].y, 1),
				p2 = (cv::Mat_<double>(3,1,CV_64F)<<
				pointMatches[i][1].x, pointMatches[i][1].y, 1);
			p1.copyTo(pts1.col(i));
			p2.copyTo(pts2.col(i));
		}

		sz0 = lineMatchesIdx[0].size();
		sz1 = lineMatchesIdx[1].size();
		sz2 = 0;
		if (lineMatchesIdx.size()>2)
			sz2 = lineMatchesIdx[2].size();
		N = sz0 + sz1 + sz2;		
		int maxIterNo = 500, iter=0; // for ransac
		double conf = 0.99; //confidence
		vector<int>  maxInliers, maxPtInliers;
		cv::Mat Hmax, rmax;
		cv::Mat n_pi, n_pi_max;
		bool gotMiniSol = false;
		while (iter < maxIterNo && totalIter < maxTotalIterNo) {
			++iter;	++totalIter;
			gotMiniSol = false;
			vector<int> curInliers;
			// --- 1. select mimimal solution set (mss) ---
			int i,j,k;
			MfgLineSegment li1, li2, lj1, lj2, lk1, lk2;
			int vert, hori;
			if (sz1 * sz2 > 0 && sz1+sz2 > 2) // two horizontal line groups are available for horizontal planes
				vert = rand() % 3;
			else
				vert = rand() % 2;			
			if ( sz1+sz2 > 0 )
				hori = rand() % (sz1+sz2);
			else
				continue;
			if (vert ==0)  {// one from group 0, two from another group
				if (sz0 < 1)
					continue;
				i = rand() % sz0; // select 1 from group 0
				li1 = view1.idealLineGroups[0][lineMatchesIdx[0][i][0]];
				li2 = view2.idealLineGroups[0][lineMatchesIdx[0][i][1]];
				if (hori < sz1) { // select 2 from group 1	
					if (sz1 < 2)
						continue;
					j = rand()% sz1;
					lj1 = view1.idealLineGroups[1][lineMatchesIdx[1][j][0]];
					lj2 = view2.idealLineGroups[1][lineMatchesIdx[1][j][1]];
					for (int ii=0; ii<100; ++ii) {
						k = rand()% sz1;
						if (k!=j)
							break;
					}
					lk1 = view1.idealLineGroups[1][lineMatchesIdx[1][k][0]];
					lk2 = view2.idealLineGroups[1][lineMatchesIdx[1][k][1]];
					n_pi = n01;
					gotMiniSol = true;
				} else {   // select 2 from group 2	
					if (sz2 < 2)
						continue;
					j = rand()% sz2;
					lj1 = view1.idealLineGroups[2][lineMatchesIdx[2][j][0]];
					lj2 = view2.idealLineGroups[2][lineMatchesIdx[2][j][1]];
					for (int ii=0; ii<100; ++ii) {
						k = rand()% sz2;
						if (k!=j)
							break;
					}
					lk1 = view1.idealLineGroups[2][lineMatchesIdx[2][k][0]];
					lk2 = view2.idealLineGroups[2][lineMatchesIdx[2][k][1]];
					n_pi = n02;
					gotMiniSol = true;
				}
			} else if (vert == 1){ // two from group 0, one from another group
				if (sz0 < 2)
					continue;
				j = rand()% sz0;
				lj1 = view1.idealLineGroups[0][lineMatchesIdx[0][j][0]];
				lj2 = view2.idealLineGroups[0][lineMatchesIdx[0][j][1]];
				for (int ii=0; ii<100; ++ii) {
					i = rand()% sz0;
					if (i!=j)
						break;
				}
				li1 = view1.idealLineGroups[0][lineMatchesIdx[0][i][0]];
				li2 = view2.idealLineGroups[0][lineMatchesIdx[0][i][1]];
				if (hori < sz1) {					
					k = rand()% sz1;
					lk1 = view1.idealLineGroups[1][lineMatchesIdx[1][k][0]];
					lk2 = view2.idealLineGroups[1][lineMatchesIdx[1][k][1]];
					n_pi = n01;
					gotMiniSol = true;
				} else {
					if (sz2 == 0)
						continue;
					k = rand()% sz2;
					lk1 = view1.idealLineGroups[2][lineMatchesIdx[2][k][0]];
					lk2 = view2.idealLineGroups[2][lineMatchesIdx[2][k][1]];
					n_pi = n02;
					gotMiniSol = true;
				}
			} else {  // when vert = 2, pick only from horizontal groups
				if (hori < sz1) { // two from group 1
					if ( sz1 < 2 )
						continue;
					i = rand() % sz1; // select 1st from group 1
					li1 = view1.idealLineGroups[1][lineMatchesIdx[1][i][0]];
					li2 = view2.idealLineGroups[1][lineMatchesIdx[1][i][1]];
					for (int ii=0; ii<100; ++ii) {
						j = rand() % sz1;
						if (i!=j)
							break;
					}					
					lj1 = view1.idealLineGroups[1][lineMatchesIdx[1][j][0]];
					lj2 = view2.idealLineGroups[1][lineMatchesIdx[1][j][1]];
					k = rand()% sz2;
					lk1 = view1.idealLineGroups[2][lineMatchesIdx[2][k][0]];
					lk2 = view2.idealLineGroups[2][lineMatchesIdx[2][k][1]];
					n_pi = ns[0]; // or n1 x n2
					gotMiniSol = true;
				}
				else {   // pick two from group 2 
					if ( sz2 < 2 )
						continue;
					i = rand() % sz2; // select 1st from group 1
					li1 = view1.idealLineGroups[2][lineMatchesIdx[2][i][0]];
					li2 = view2.idealLineGroups[2][lineMatchesIdx[2][i][1]];
					for (int ii=0; ii<100; ++ii) {
						j = rand() % sz2;
						if (i!=j)
							break;
					}					
					lj1 = view1.idealLineGroups[2][lineMatchesIdx[2][j][0]];
					lj2 = view2.idealLineGroups[2][lineMatchesIdx[2][j][1]];
					k = rand()% sz1;
					lk1 = view1.idealLineGroups[1][lineMatchesIdx[1][k][0]];
					lk2 = view2.idealLineGroups[1][lineMatchesIdx[1][k][1]];
					n_pi = ns[0]; // or n1 x n2
					gotMiniSol = true;
				}
			}

			if ( !gotMiniSol )
				continue;

			// --- 2. compute minimal solution ---
			cv::Mat mss1(3,3,CV_64F), mss2(3,3,CV_64F);
			li1.matLnEq.copyTo (mss1.col(0));
			lj1.matLnEq.copyTo (mss1.col(1));
			lk1.matLnEq.copyTo (mss1.col(2));
			li2.matLnEq.copyTo (mss2.col(0));
			lj2.matLnEq.copyTo (mss2.col(1));
			lk2.matLnEq.copyTo (mss2.col(2));
			cv::Mat r = get_r_lns (mss1, mss2, n_pi, view1.camMat, rotMat_vpt);			
			cv::Mat H = view1.camMat*(rotMat_vpt-r*n_pi.t())*view1.camMat.inv();

			if (1/cv::norm(r) > PdThresh) // neglect far-view planes
			{
				gotMiniSol = false;
					continue;
			}
			// --- 3. find consensus set ---
			for (int ii = 0; ii < allLs1.size(); ++ii) {
				if (cv::norm(n_pi - n01) < 1e-5  && ii>=sz0+sz1) continue;
				if (cv::norm(n_pi - n02) < 1e-5 && ii>=sz0 && ii<sz0+sz1) continue;
				if (cv::norm(n_pi - ns[0]) < 1e-5 && ii<sz0 )	continue;

				if (allLs1[ii].lowPalx) //not consider low-parallax lines
					continue;

				double dis = symmErr_LinePair(allLs1[ii],allLs2[ii],H);			
				if (dis/(allLs1[ii].length()+allLs2[ii].length()) < lsDistThresh)
					curInliers.push_back(ii);
			}
			vector<int> curPtInliers;
			for (int i=0; i<nPts; ++i) {
				double dist_symm = symmErr_PtPair(pts1.col(i),pts2.col(i),H);
				if (dist_symm < ptDistThresh)
					curPtInliers.push_back(i);
			}		

			// update largest consensus set
			if ((Hmax.rows  < 1 )|| 
				(curInliers.size() + curPtInliers.size() > maxInliers.size() + maxPtInliers.size()
				&&  curPtInliers.size() > 2)) {
#ifdef DEBUG
					cv::Mat canv1 = view1.mImg.clone(),canv2 = view2.mImg.clone();
					cv::line(canv1,li1.ptA.cvPt(),li1.ptB.cvPt(),
						cv::Scalar(200,0,0,0),2);
					cv::line(canv1,lj1.ptA.cvPt(),lj1.ptB.cvPt(),
						cv::Scalar(200,0,0,0),2);
					cv::line(canv1,lk1.ptA.cvPt(),lk1.ptB.cvPt(),
						cv::Scalar(200,0,0,0),2);
					cv::line(canv2, Mfg2dPoint(H*li1.ptA.matHomo()).cvPt(),
						Mfg2dPoint(H*li1.ptB.matHomo()).cvPt(), 
						cv::Scalar(200,200,0,0),2);
					cv::line(canv2,li2.ptA.cvPt(),li2.ptB.cvPt(),
						cv::Scalar(200,0,0,0),1);
					cv::line(canv2, Mfg2dPoint(H*lj1.ptA.matHomo()).cvPt(),
						Mfg2dPoint(H*lj1.ptB.matHomo()).cvPt(),
						cv::Scalar(200,200,0,0),2);
					cv::line(canv2,lj2.ptA.cvPt(),lj2.ptB.cvPt(),
						cv::Scalar(200,0,0,0),1);
					cv::line(canv2, Mfg2dPoint(H*lk1.ptA.matHomo()).cvPt(),
						Mfg2dPoint(H*lk1.ptB.matHomo()).cvPt(),
						cv::Scalar(200,200,0,0),2);
					cv::line(canv2,lk2.ptA.cvPt(),lk2.ptB.cvPt(),
						cv::Scalar(200,0,0,0),1);
					showImage("inlier = "+num2str(curInliers.size())
						+","+num2str(curPtInliers.size()),&canv1);
					showImage("inlier-im2",&canv2);
//				cv::waitKey(0);
					cv::destroyWindow("inlier = "+num2str(curInliers.size())
						+","+num2str(curPtInliers.size()));
					cv::destroyWindow("inlier-im2");

#endif
					maxInliers = curInliers;
					maxPtInliers = curPtInliers;
					Hmax = H;
					rmax = r;	
					n_pi_max = n_pi;					
			}
			// update maximum iteration number adaptively
//			if (maxInliers.size() > 0)
//				maxIterNo = abs(log(1-conf) 
//				/log(1-pow(double(maxInliers.size())/N,mss1.cols)));//????	

			if (maxInliers.size()<1)
				cout<<"";
		}	
		if ( !gotMiniSol )
			continue;

		// --- 4.re-estimation and guided matching for lines ---		
		double scale = 1.0;
		int ptInlierNum, maxPtNum = maxPtInliers.size();
		cv::Mat optH = Hmax, opt_r = rmax, opt_n_pi = n_pi_max;
		for (int ix = 0; ix >=0; ++ix ) {
			vector<MfgLineSegment> ls1, ls2;
			cv::Mat l1(3,maxInliers.size(),CV_64F), 
				l2(3,maxInliers.size(),CV_64F);
			for (int i=0; i<maxInliers.size(); ++i) {
				ls1.push_back(allLs1[maxInliers[i]]);
				ls2.push_back(allLs2[maxInliers[i]]);
				allLs1[maxInliers[i]].matLnEq.copyTo(l1.col(i));
				allLs2[maxInliers[i]].matLnEq.copyTo(l2.col(i));
			}
			if (ls1.size()>=9)
				mle_2dLineHomo (ls1, ls2, &Hmax); // 

			ptInlierNum = findPtsOnPlane (&Hmax, maxPtInliers);
			
			vector<cv::Point2d> ptInliers1, ptInliers2;
			cv::Mat pts1(3,maxPtInliers.size(), CV_64F), 
				pts2(3,maxPtInliers.size(), CV_64F)	;
			for (int i=0; i < maxPtInliers.size(); ++i) {
				ptInliers1.push_back(pointMatches[maxPtInliers[i]][0]);
				ptInliers2.push_back(pointMatches[maxPtInliers[i]][1]);
				Mfg2dPoint(ptInliers1.back().x,ptInliers1.back().y).matHomo().
					copyTo(pts1.col(i));
				Mfg2dPoint(ptInliers2.back().x,ptInliers2.back().y).matHomo().
					copyTo(pts2.col(i));
			}
			// compute initial point using DLT for optimization 
			if ( maxPtInliers.size() > 4 )
				rmax = get_r_pts (pts1, pts2, n_pi_max,view1.camMat,rotMat_epi);
			else if (ls1.size() > 3)
				optTranslate_lines 
				(ls1, ls2, rotMat_vpt, n_pi_max, view1.camMat, &rmax);

			// **** optimization ****
			if (rmax.dot(t_epi) > 0) 
				rmax = t_epi * cv::norm(rmax); // enforce rmax parallel to t_epi
			else {
				rmax = t_epi * cv::norm(rmax);
				n_pi_max = -n_pi_max;
			}
			if (ptInliers1.size() >= 3) {
				opt_r_d_ptlns (ptInliers1,ptInliers2,ls1, ls2, rotMat_epi, view1.camMat,
					&n_pi_max,&rmax);
			}

			vector<int> curInliers, curPtInliers;
			for (int i = 0; i < allLs1.size(); ++i) {	
				if (abs(n_pi_max.dot(n01)) > 0.7 && i>=sz0+sz1) continue;
				if (abs(n_pi_max.dot(n02)) > 0.7 && i>=sz0 && i<sz0+sz1) continue;
				if (abs(n_pi_max.dot(ns[0])) > 0.7 && i<sz0) continue;
				if (allLs1[i].lowPalx) // not consider low-parallax lines
					continue;
				double dis = symmErr_LinePair(allLs1[i],allLs2[i],Hmax);				
				if ( dis/(allLs1[i].length()+allLs1[i].length()) < lsDistThresh * scale ) 
					curInliers.push_back(i);
			}
			if (curInliers.size() + ptInlierNum > maxInliers.size()+maxPtNum) {
					scale = scale* pow(double(maxInliers.size())/curInliers.size(),0.5);
					maxInliers.clear();				
					maxInliers = curInliers;
					maxPtNum = ptInlierNum;
					optH = Hmax;
					opt_r = rmax;
					opt_n_pi = n_pi_max;	

			} else
				break;			
		}
		cv::Mat tmp = Hmax;
		if (ptInlierNum < 4 || maxInliers.size() < 4 )
			continue;

		// Ignore far-view planes by thresholding depth for corridor scenarios
		if (1/cv::norm(opt_r) > PdThresh)
			continue;
		primaryPlanes.push_back ( Mfg3dPlane(
			opt_n_pi.at<double>(0),	opt_n_pi.at<double>(1),	opt_n_pi.at<double>(2),
			sgn( opt_r.dot(t_epi) )/cv::norm(opt_r)));

		cout<<"n_pi = "<<opt_n_pi<< ","	<< sgn( opt_r.dot(t_epi) )/cv::norm(opt_r) <<endl;


#ifdef DEBUG
		cv::Mat canv1 = view1.mImg.clone(),canv2 = view2.mImg.clone();
#endif
		cv::Mat edp1,edp2; 
		int count_behindCam = 0;
		for (int i=0; i<maxInliers.size(); ++i) {
			bool iFrom0 = maxInliers[i] < sz0, 
				iFrom1 = maxInliers[i]>=sz0 && maxInliers[i]<sz0+sz1,
				iFrom2 = maxInliers[i]>=sz0+sz1;
			MfgLineSegment lk1,lk2;
			if (iFrom0) {
				lk1 = view1.idealLineGroups[0][lnMchIdx[maxInliers[i]][0]];
				lk2 = view2.idealLineGroups[0][lnMchIdx[maxInliers[i]][1]];
			}
			if (iFrom1) {
				lk1 = view1.idealLineGroups[1][lnMchIdx[maxInliers[i]][0]];
				lk2 = view2.idealLineGroups[1][lnMchIdx[maxInliers[i]][1]];
			}
			if (iFrom2) {				
				lk1 = view1.idealLineGroups[2][lnMchIdx[maxInliers[i]][0]];
				lk2 = view2.idealLineGroups[2][lnMchIdx[maxInliers[i]][1]];
			}

#ifdef DEBUG
			cv::Scalar color = cv::Scalar(rand()%255,rand()%255,rand()%255,0);
			cv::line(canv1,lk1.ptA.cvPt(),lk1.ptB.cvPt(),color,2);
			cv::line(canv2, Mfg2dPoint(optH*lk1.ptA.matHomo()).cvPt(),
				Mfg2dPoint(optH*lk1.ptB.matHomo()).cvPt(), color,2);
			cv::line(canv2,lk2.ptA.cvPt(),lk2.ptB.cvPt(),color,1);			
#endif			
			// check if the plane and its lines are behind camera or not
			projectImgPt2Plane(lk1.ptA.matHomo(), primaryPlanes[primaryPlanes.size()-1],
				view1.camMat, edp1); // compute line endpoint in 3d
			projectImgPt2Plane(lk1.ptB.matHomo(), primaryPlanes[primaryPlanes.size()-1],
				view1.camMat, edp2);
			if (edp1.at<double>(2) <0 || edp2.at<double>(2) <0) {
				count_behindCam++;
				maxInliers.erase(maxInliers.begin()+i);
				i--;
//				cout<<"a line behind cam found..."<<endl;
#ifdef DEBUG 
				cv::line(canv2,lk2.ptA.cvPt(),lk2.ptB.cvPt(),cv::Scalar(0,0,0,0),2);
#endif
			}
			else
				primaryPlanes[primaryPlanes.size()-1].mem2dSegments.push_back(lk1);
		}
		if (double(count_behindCam)/maxInliers.size() >1
			|| maxInliers.size() < 4 ) // if more than half endpoints are behind cam
			// eliminate this plane 
		{
			primaryPlanes.pop_back();
			cout<<"plane removed due to behind camera"<<endl;
			maxInliers.clear();
			maxPtInliers.clear();
		}

#ifdef DEBUG
		showImage("line inlier-im1",&canv1);
		showImage("line inlier-im2",&canv2);
		cv::waitKey();
		cv::destroyWindow("line inlier-im1");
		cv::destroyWindow("line inlier-im2");
#endif
		// --- 5. remove inliers from whole dataset ---
		removePlaneFeatures(allLs1, allLs2, lnMchIdx, maxInliers, maxPtInliers);
	}
	lineMatchesIdx.clear(); lineMatchesIdx = copyLnMchIdx;
	isolatePtMatches = pointMatches; // point pairs not on any plane~~~~
	pointMatches.clear();   pointMatches = copyPtMchs;
}

void MfgTwoView::removePlaneFeatures(vector<MfgLineSegment>& allLs1, 
	vector<MfgLineSegment>&allLs2,vector<vector<int>>& lnMchIdx,
	vector<int>& maxLnInliers, vector<int>& maxPtInliers)
	// remove line and point features located on the found plane.
{
	vector<vector<int>> OnePlaneLnMatIdx;
	int sz0 = lineMatchesIdx[0].size(),
		sz1 = lineMatchesIdx[1].size(),
		sz2 = 0;
	if (lineMatchesIdx.size()>2)
		sz2 = lineMatchesIdx[2].size();
	double hDistMin = 1e5, hDistMax = 0, vDistMin = 1e5, vDistMax = 0;
	MfgLineSegment left1, right1, up1, down1,left2, right2, up2, down2;
	for (int i= maxLnInliers.size()-1; i>=0; --i) {			
		int grp, j;
		if (maxLnInliers[i]<sz0) {// from group 0	
			grp = 0;
			j = maxLnInliers[i];			
			double d = point2LineDist(allLs1[maxLnInliers[i]].matLnEq, 
				view1.vanishPoints[1].matHomo());
			if (d > hDistMax) {
				hDistMax = d;
				left1 = allLs1[maxLnInliers[i]];
				left2 = allLs2[maxLnInliers[i]];
			}
			if (d < hDistMin) {
				hDistMin = d;
				right1 = allLs1[maxLnInliers[i]];
				right2 = allLs2[maxLnInliers[i]];
			}
		} 
		if (maxLnInliers[i]>=sz0 && maxLnInliers[i]<sz0+sz1) {
			grp = 1;
			j = maxLnInliers[i] - sz0;
			double d = point2LineDist(allLs1[maxLnInliers[i]].matLnEq, 
				view1.vanishPoints[0].matHomo());
			if (d > vDistMax) {
				vDistMax = d;
				down1 = allLs1[maxLnInliers[i]];
				down2 = allLs2[maxLnInliers[i]];
			}
			if (d < vDistMin) {
				vDistMin = d;
				up1 = allLs1[maxLnInliers[i]];
				up2 = allLs2[maxLnInliers[i]];
			}
		} 
		if (maxLnInliers[i]>=sz0+sz1)	{
			grp = 2;
			j = maxLnInliers[i] - sz0 - sz1;
			double d = point2LineDist(allLs1[maxLnInliers[i]].matLnEq, 
				view1.vanishPoints[0].matHomo());
			if (d > vDistMax) {
				vDistMax = d;
				down1 = allLs1[maxLnInliers[i]];
				down2 = allLs2[maxLnInliers[i]];
			}
			if (d < vDistMin) {
				vDistMin = d;
				up1 = allLs1[maxLnInliers[i]];
				up2 = allLs2[maxLnInliers[i]];
			}
		}
		vector<int> tmp(3);
		tmp[0] = grp;		// parallel group number, consistent for both views
		tmp[1] = lineMatchesIdx[grp][j][0];  // ideal line idx in 1st view
		tmp[2] = lineMatchesIdx[grp][j][1];  // ideal line idx in 2nd view
		OnePlaneLnMatIdx.push_back(tmp);

		allLs1.erase(allLs1.begin() + maxLnInliers[i]);
		allLs2.erase(allLs2.begin() + maxLnInliers[i]);
		lnMchIdx.erase(lnMchIdx.begin()+maxLnInliers[i]);
		lineMatchesIdx[grp].erase(lineMatchesIdx[grp].begin()+j);
	}

	FeaturePointPairs tmpcoplanarPtPair;
	for (int i = maxPtInliers.size()-1; i>=0; --i) {
		tmpcoplanarPtPair.push_back( pointMatches[maxPtInliers[i]] );		
		pointMatches.erase(pointMatches.begin()+maxPtInliers[i]);
	}
#ifdef DEBUG
	cv::Mat canvas1 = view1.mImg.clone(), canvas2 = view2.mImg.clone();
	cv::Scalar color(rand()%255,rand()%255,rand()%255,0);
	cv::line(canvas1,left1.ptA.cvPt(),left1.ptB.cvPt(), color, 2);
	cv::line(canvas2,left2.ptA.cvPt(),left2.ptB.cvPt(), color, 2);
	cv::line(canvas1,right1.ptA.cvPt(),right1.ptB.cvPt(), color, 2);
	cv::line(canvas2,right2.ptA.cvPt(),right2.ptB.cvPt(), color, 2);
	cv::line(canvas1,up1.ptA.cvPt(),up1.ptB.cvPt(), color, 2);
	cv::line(canvas2,up2.ptA.cvPt(),up2.ptB.cvPt(), color, 2);
	cv::line(canvas1,down1.ptA.cvPt(),down1.ptB.cvPt(), color, 2);
	cv::line(canvas2,down2.ptA.cvPt(),down2.ptB.cvPt(), color, 2);
	showImage("Region boundary - view1", &canvas1);
	showImage("Region boundary - view2", &canvas2);
	cv::waitKey();
	cv::destroyWindow("Region boundary - view1");
	cv::destroyWindow("Region boundary - view2");
#endif
	coplanarPtMatches.push_back(tmpcoplanarPtPair);
	onPlaneLineMatchesIdx.push_back(OnePlaneLnMatIdx);
	if (tmpcoplanarPtPair.size() ==0 && OnePlaneLnMatIdx.size()==0){
		coplanarPtMatches.pop_back();
		onPlaneLineMatchesIdx.pop_back();
	}

}

void MfgTwoView::mergePrimaryPlanes() {	
	for (int i =0; i<primaryPlanes.size()-1; ++i) {
		for (int j =i+1; j <primaryPlanes.size(); ++j) {
			double depthDif 
				= abs(primaryPlanes[i].depth()-primaryPlanes[j].depth()),
				normPrdt = primaryPlanes[i].matNormal().dot
				(primaryPlanes[j].matNormal());
			if (depthDif < primaryPlanes[i].depth()*0.10 
				&& normPrdt > 0.9) { // planes are close enough
					coplanarPtMatches[i].insert(coplanarPtMatches[i].end(),
						coplanarPtMatches[j].begin(),
						coplanarPtMatches[j].end());
					coplanarPtMatches.erase(coplanarPtMatches.begin()+j);

					onPlaneLineMatchesIdx[i].insert(onPlaneLineMatchesIdx[i].end(),
						onPlaneLineMatchesIdx[j].begin(),
						onPlaneLineMatchesIdx[j].end());
					onPlaneLineMatchesIdx.erase(onPlaneLineMatchesIdx.begin()+j);

					primaryPlanes[i].setValues(primaryPlanes[i].n_[0],
						primaryPlanes[i].n_[1], primaryPlanes[i].n_[2],
						(primaryPlanes[i].depth()+primaryPlanes[j].depth())/2);// to improves 
					primaryPlanes[i].mem2dSegments.insert(
						primaryPlanes[i].mem2dSegments.end(),
						primaryPlanes[j].mem2dSegments.begin(),
						primaryPlanes[j].mem2dSegments.end());					
					primaryPlanes.erase(primaryPlanes.begin()+j);
					--j;
			}
		}
	}
}

int MfgTwoView::findPtsOnPlane (cv::Mat* H, vector<int>& maxInlierSet)
{
	if (H->rows<1)
		cout<<"";
	int nPts = pointMatches.size();
	double ptDistThresh  = 1.50 *IDEAL_IMAGE_WIDTH/640.0;// initial threshold 4!
	cv::Mat pts1(3,nPts,CV_64F), pts2(3,nPts,CV_64F);
	for (int i=0; i<pointMatches.size(); ++i) {			
		cv::Mat p1 = (cv::Mat_<double>(3,1,CV_64F)<<
			pointMatches[i][0].x, pointMatches[i][0].y, 1),
			p2 = (cv::Mat_<double>(3,1,CV_64F)<<
			pointMatches[i][1].x, pointMatches[i][1].y, 1);
		p1.copyTo(pts1.col(i));
		p2.copyTo(pts2.col(i));
	}

	maxInlierSet.clear();
	double scale = 1;
	int prevSize = 0;
	while (true) {
		//--- find inliers ---
		vector<int> curInlierSet;
		for (int i=0; i<nPts; ++i) {
			double dist_symm = // symmetric transfer error for points
				symmErr_PtPair(pts1.col(i),pts2.col(i),*H);
			if (dist_symm < ptDistThresh)
				curInlierSet.push_back(i);
		}
		if(curInlierSet.size() > maxInlierSet.size()) {
			if (maxInlierSet.size()>0)
				scale = pow(double(maxInlierSet.size())/curInlierSet.size(),0.5);
			ptDistThresh = ptDistThresh * scale;
			prevSize = maxInlierSet.size();
			maxInlierSet = curInlierSet;			
		}
		else
			break;
		if (maxInlierSet.size()<4)
			break;
		//--- re-estimate ---
		cv::Mat inlier1(maxInlierSet.size(),1,CV_32FC2),
			inlier2(maxInlierSet.size(),1,CV_32FC2);
		for (int i=0; i<maxInlierSet.size(); ++i) {
			inlier1.at<cv::Vec2f>(i,0)[0] = pointMatches[maxInlierSet[i]][0].x;
			inlier1.at<cv::Vec2f>(i,0)[1] = pointMatches[maxInlierSet[i]][0].y;
			inlier2.at<cv::Vec2f>(i,0)[0] = pointMatches[maxInlierSet[i]][1].x;
			inlier2.at<cv::Vec2f>(i,0)[1] = pointMatches[maxInlierSet[i]][1].y;
		}		
		*H = cv::findHomography(inlier1,inlier2, 0);
#ifdef DEBUG
		cv::Mat canv1 = view1.mImg.clone(), canv2 = view2.mImg.clone();
		for (int i=0; i<maxInlierSet.size(); ++i)
		{
			cv::Scalar clr = cv::Scalar(rand()%255,rand()%255,rand()%255,0);
			cv::circle(canv1, Mfg2dPoint(pts1.col(maxInlierSet[i])).cvPt(), 
				2, clr, 2);
			cv::circle(canv2, Mfg2dPoint(pts2.col(maxInlierSet[i])).cvPt(), 
				2, clr, 2);
		}
		showImage("Point inliers-view1: "+num2str(maxInlierSet.size()), &canv1);
		showImage("Point inliers-view2", &canv2);
//		cv::waitKey();
		cv::destroyWindow("Point inliers-view1: "+num2str(maxInlierSet.size()));
		cv::destroyWindow("Point inliers-view2");
#endif

	}

	return maxInlierSet.size();
}

void MfgTwoView::findPlane_4Pts ()
{
	int nPts = pointMatches.size();
	cv::Mat pts1(nPts,1,CV_32FC2), pts2(nPts,1,CV_32FC2),
		p1(3,nPts,CV_64F), p2(3,nPts,CV_64F);

	for (int i=0; i<pointMatches.size(); ++i) {
		pts1.at<cv::Vec2f>(i,0)[0] = pointMatches[i][0].x;
		pts1.at<cv::Vec2f>(i,0)[1] = pointMatches[i][0].y;
		pts2.at<cv::Vec2f>(i,0)[0] = pointMatches[i][1].x;
		pts2.at<cv::Vec2f>(i,0)[1] = pointMatches[i][1].y;									
	}	

	// normalize pts before computing fundmental matrix
	cv::Mat zPts1(3,nPts,CV_64F), zPts2(3,nPts,CV_64F);
	for (int i=0; i<pointMatches.size(); ++i) {
		cv::Mat tmp1 = (cv::Mat_<double>(3,1)<<pointMatches[i][0].x,
			pointMatches[i][0].y, 1),
			tmp2 = (cv::Mat_<double>(3,1)<<pointMatches[i][1].x,
			pointMatches[i][1].y, 1);
		cv::Mat pt1 = view1.camMat.inv() * tmp1,
			pt2 = view1.camMat.inv() * tmp2;
		pt1.copyTo(zPts1.col(i));
		pt2.copyTo(zPts2.col(i));
	}

	cv::Mat inSetH, inSetF;
	cv::Mat H = cv::findHomography(pts1,pts2,cv::RANSAC, 1, inSetH);
	// --- recover extrinsic parameters---
	cv::Mat F,E;
//	F = cv::findFundamentalMat(pts1, pts2, inSetF, cv::FM_RANSAC, 1.5);
//	E = view1.camMat.t()*F*view1.camMat;
//	cout<<"cv_E="<<E<<cv::determinant(E)<<endl;
//	cout<<cv::norm(inSetF)*cv::norm(inSetF)
//		<<'\t'<<cv::norm(inSetH)*cv::norm(inSetH) <<endl;

	
	while (!essn_ransac (&zPts1, &zPts2, &E, view1.camMat,	&inSetF, 
			false, rotMat_vpt, IDEAL_IMAGE_WIDTH))
			;

	// *** Display ****
	cv::Mat cvs1 = view1.mImg.clone(), cvs2 = view2.mImg.clone();
	for (int i=0; i<nPts; ++i) {	// plot coplanar points	 
		cv::Scalar clr = cv::Scalar(rand()%255,rand()%255,rand()%255,0);
	}
	cv::Mat ztest1, ztest2;
	// --- export inlier matches to text, and delete outliers from pointMatches
	ofstream file("ptmatch.txt");
	for (int i=pointMatches.size()-1; i>=0; --i) {
		if (inSetF.at<uchar>(i) >= 1) {
			ztest1 = zPts1.col(i);
			ztest2 = zPts2.col(i);
			file << pointMatches[i][0].x <<'\t'<<pointMatches[i][0].y <<'\t'
				<<pointMatches[i][1].x <<'\t'<<pointMatches[i][1].y <<'\n'
				<<flush;
		} 	else	{ // eliminate false matches 		
			pointMatches.erase(pointMatches.begin()+i);
			pointMatcheIdx.erase(pointMatcheIdx.begin()+i);
		}
	}
	file.close();

	cv::Mat R1, R2, trans;
	decEssential (&E, &R1, &R2, &trans); 
	//--- check Cheirality ---
	ztest1 = ztest1/ztest1.at<double>(2);
	ztest2 = ztest2/ztest2.at<double>(2);
	cv::Mat Rt = findTrueRt(R1,R2,trans,
		cv::Point2d(ztest1.at<double>(0), ztest1.at<double>(1)),
		cv::Point2d(ztest2.at<double>(0), ztest2.at<double>(1)));	
	rotMat_epi = Rt.colRange(0,3);
	cout<<"R_epi="<<rotMat_epi<<endl;	
	cout<<"t="<<Rt.col(3)<<endl;
	t_epi = Rt.col(3);

	cv::circle(cvs1, pointMatches[0][0], 16,
		cv::Scalar(0,0,0,0), 3);
	cv::circle(cvs2, pointMatches[0][1], 16,
		cv::Scalar(0,0,0,0), 3);
	//	showImage("view1_4pt="+num2str(cv::norm(inSetH)*cv::norm(inSetH)),&cvs1);
	//	showImage("view2_4pt",&cvs2);
	//	cv::waitKey(0);
	//	cv::destroyWindow("view1_4pt="+num2str(cv::norm(inSetH)*cv::norm(inSetH)));
	//	cv::destroyWindow("view2_4pt");	
}

void MfgTwoView::matchKeyPoints ()
{
	vector<vector<cv::DMatch>> knnMatches;
	if (1 && view1.keyPointPoses.size() * view2.keyPointPoses.size() > 	3e6) {
		cv::FlannBasedMatcher matcher;	// this gives fast inconsistent output
		matcher.knnMatch(view1.keyPointFeatures, view2.keyPointFeatures,
			knnMatches,2);
		
		cout<<"flann match"<<endl;
	}
	else { // BF is slow but result is consistent
		cv::BruteForceMatcher<cv::L2<float>> matcher; // for opencv2.3.0
		//		cv::BFMatcher matcher( cv::NORM_L2, false ); // for opencv2.4.2
		matcher.knnMatch(view1.keyPointFeatures, view2.keyPointFeatures,
			knnMatches,2);		
		cout<<"bf match"<<endl;
	}	
	for (int i=0; i<knnMatches.size(); ++i) { 
		double ratio = knnMatches[i][0].distance/knnMatches[i][1].distance;
		if ( ratio < THRESH_POINT_MATCH_RATIO) 		{
			pointMatcheIdx.push_back(knnMatches[i][0]);
			vector<cv::Point2d> ptPair;
			ptPair.push_back(view1.keyPointPoses[knnMatches[i][0].queryIdx].pt);
			ptPair.push_back(view2.keyPointPoses[knnMatches[i][0].trainIdx].pt);
			pointMatches.push_back(ptPair);
		}
	}	
	cout<<"Keypoint Matches: "<<pointMatches.size()<<endl;
}

void MfgTwoView::matchLines()
{	
	for (int i=0; i<view1.idealLineGroups.size(); ++i) {
		vector<vector<int>> lineMatchGroup;
		matchLinesByPointPairs(&view1.mImgGray, &view2.mImgGray, 
			view1.idealLineGroups[i], view2.idealLineGroups[i],
			pointMatches, lineMatchGroup);
		lineMatchesIdx.push_back(lineMatchGroup);
		cout<<"Line matches found: "<<lineMatchGroup.size()<<endl;
	}
}

void MfgTwoView::matchLineSegments()
{
	for (int i=0; i<view1.lineSegmentGroups.size(); ++i) {
		vector<vector<int>> matchGroup;
		vector<MfgLineSegment> segments1,segments2;
		for (int j=0; j<view1.lineSegmentGroups[i].size(); ++j)
			segments1.push_back(view1.lineSegments
			[view1.lineSegmentGroups[i][j]]);
		for (int j=0; j<view2.lineSegmentGroups[i].size(); ++j)
			segments2.push_back(view2.lineSegments
			[view2.lineSegmentGroups[i][j]]);
		matchLinesByPointPairs(&view1.mImgGray, &view2.mImgGray, 
			segments1, segments2,pointMatches, matchGroup);
		//matchLinesByPointPairsCounting(&view1.mImgGray, &view2.mImgGray, 
		//	segments1, segments1,pointMatches, matchGroup);
		segmentMatchesIdx.push_back(matchGroup);
		cout<<"Segments matches found: "<<matchGroup.size()<<endl;
	}
}

void MfgTwoView::drawPointMatches ()
{
	cv::Mat img_2nn;
	cv::drawMatches(view1.mImg, view1.keyPointPoses, 
		view2.mImg, view2.keyPointPoses,
		pointMatcheIdx, img_2nn, cv::Scalar::all(-1), 
		cv::Scalar::all(-1), vector<char>(),
		cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	showImage( "Key Point Matches", &img_2nn);
	//	cv::waitKey(0);
}

void MfgTwoView::drawIdealLineMatches()
{
	cv::Mat canvas1 = view1.mImg.clone(), canvas2 = view2.mImg.clone();
	for (int i=0; i<lineMatchesIdx.size(); ++i) {
		for (int j=0; j<lineMatchesIdx[i].size(); ++j) {

			cv::Scalar color(rand()%255,rand()%255,rand()%255,0);
			int idx1, idx2;
			idx1 = lineMatchesIdx[i][j][0];
			idx2 = lineMatchesIdx[i][j][1];
			if (view1.idealLineGroups[i][idx1].lowPalx) {
				cv::line(canvas1,view1.idealLineGroups[i][idx1].ptA.cvPt(),
				view1.idealLineGroups[i][idx1].ptB.cvPt(), color, 1);
			cv::line(canvas2,view2.idealLineGroups[i][idx2].ptA.cvPt(),
				view2.idealLineGroups[i][idx2].ptB.cvPt(), color, 1);
			} else {				
			cv::line(canvas1,view1.idealLineGroups[i][idx1].ptA.cvPt(),
				view1.idealLineGroups[i][idx1].ptB.cvPt(), color, 2);
			cv::line(canvas2,view2.idealLineGroups[i][idx2].ptA.cvPt(),
				view2.idealLineGroups[i][idx2].ptB.cvPt(), color, 2);
			}
			showImage("Line Matches - view1", &canvas1);
			showImage("Line Matches - view2", &canvas2);

		}	
	}
	//	cv::waitKey(0);

}

void MfgTwoView::drawLineSegmentMatches()
{	
	cv::Mat canvas1 = view1.mImg.clone(), canvas2 = view2.mImg.clone();
	for (int i=0; i<segmentMatchesIdx.size(); ++i) {
		for (int j=0; j<segmentMatchesIdx[i].size(); ++j) {
			cv::Scalar color(rand()%255,rand()%255,rand()%255,0);
			int idx1, idx2;
			idx1 = segmentMatchesIdx[i][j][0];
			idx2 = segmentMatchesIdx[i][j][1];
			cv::line(canvas1,
				view1.lineSegments[view1.lineSegmentGroups[i][idx1]].ptA.cvPt(),
				view1.lineSegments[view1.lineSegmentGroups[i][idx1]].ptB.cvPt(),
				color, 2);
			cv::line(canvas2,
				view2.lineSegments[view2.lineSegmentGroups[i][idx2]].ptA.cvPt(),
				view2.lineSegments[view2.lineSegmentGroups[i][idx2]].ptB.cvPt(),
				color, 2);
			showImage("Line Segments Matches - view1", &canvas1);
			showImage("Line Segments Matches - view2", &canvas2);
			cv::waitKey(0);
		}	
	}
}

void MfgTwoView::drawMappedLines(cv::Mat* H)
{	
	for (int i = 0; i< lineMatchesIdx.size(); ++i) {
		for (int j=0; j<lineMatchesIdx[i].size(); ++j) {
			cv::Mat canv1 = view1.mImg.clone(),canv2 = view2.mImg.clone();
			MfgLineSegment l1, l2;
			l1 = view1.idealLineGroups[i][lineMatchesIdx[i][j][0]];
			l2 = view2.idealLineGroups[i][lineMatchesIdx[i][j][1]];
			cv::Scalar clr = cv::Scalar(rand()%255,rand()%255,rand()%255,0);
			cv::line(canv1,l1.ptA.cvPt(), l1.ptB.cvPt(),clr,2);
			cv::line(canv2,l2.ptA.cvPt(), l2.ptB.cvPt(),clr,2);
			clr = cv::Scalar(rand()%255,rand()%255,rand()%255,0);
			cv::line(canv1, Mfg2dPoint(H->inv()*l2.ptA.matHomo()).cvPt(),
				Mfg2dPoint(H->inv()*l2.ptB.matHomo()).cvPt(), clr, 2);
			cv::line(canv2, Mfg2dPoint(*H*l1.ptA.matHomo()).cvPt(),
				Mfg2dPoint(*H*l1.ptB.matHomo()).cvPt(), clr, 2);

			double d = symmErr_LinePair(l1,l2,*H);
			showImage("mapped lines: view1, dist="+num2str(d), &canv1);
			showImage("mapped lines: view2", &canv2);
			cv::waitKey();
			cv::destroyAllWindows();
		}
	}
}

void MfgTwoView::draw3dGl()
{
	// plot first camera, big
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

	// plot second camera
	cv::Mat c = -rotMat_epi.t()*t_epi;
	cv::Mat xw = (cv::Mat_<double>(3,1)<< 1,0,0)/2,
		yw = (cv::Mat_<double>(3,1)<< 0,1,0)/2,
		zw = (cv::Mat_<double>(3,1)<< 0,0,1)/2;
	cv::Mat x_ = rotMat_epi.t() * (xw-t_epi),
		y_ = rotMat_epi.t() * (yw-t_epi),
		z_ = rotMat_epi.t() * (zw-t_epi);
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
	for(int i=0; i<primaryPlanes.size(); ++i)	{
		glColor3f((rand()%100)/100.0,(rand()%100)/100.0,(rand()%100)/100.0);
		//		cout<<primaryPlanes[i].matPlane ()<<"\n";		
		for (int j=0; j<primaryPlanes[i].mem2dSegments.size(); ++j) {
			cv::Mat edp1, edp2;
			projectImgPt2Plane(primaryPlanes[i].mem2dSegments[j].ptA.matHomo(),
				primaryPlanes[i],view1.camMat, edp1);
			projectImgPt2Plane(primaryPlanes[i].mem2dSegments[j].ptB.matHomo(),
				primaryPlanes[i],view1.camMat, edp2);

			glVertex3f(edp1.at<double>(0),edp1.at<double>(1),edp1.at<double>(2));
			glVertex3f(edp2.at<double>(0),edp2.at<double>(1),edp2.at<double>(2));
		}		
	}
	glEnd();	


	cv::Mat P1 = (cv::Mat_<double>(3,4)<<1,0,0,0,
		0,1,0,0,
		0,0,1,0);
	cv::Mat P2(3,4,CV_64F);
	rotMat_epi.copyTo(P2.colRange(0,3)); 	 t_epi.copyTo(P2.col(3));


	//	cout<<"P2"<<P2<<endl;
	glPointSize(2.0);
	//	glColor3f((rand()%100)/100.0,(rand()%100)/100.0,(rand()%100)/100.0);
	glBegin(GL_POINTS);
	for (int i=0; i<pointMatches.size(); ++i) {
		cv::Mat p1 = view1.camMat.inv()*
			Mfg2dPoint(pointMatches[i][0].x,pointMatches[i][0].y).matHomo(),
			p2 = view1.camMat.inv()*
			Mfg2dPoint(pointMatches[i][1].x,pointMatches[i][1].y).matHomo();
		cv::Mat Pt = linearTriangulate (P1,P2,Mfg2dPoint(p1).cvPt(),
			Mfg2dPoint(p2).cvPt());

		//		glColor3f(0,0,1);
		//		glVertex3f(Pt.at<double>(0)/Pt.at<double>(3),
		//			Pt.at<double>(1)/Pt.at<double>(3),Pt.at<double>(2)/Pt.at<double>(3));

		p1 = p1/p1.at<double>(2);
		p2 = p2/p2.at<double>(2);
		Pt = Pt*0;

		CvMat cP1=P1, cP2 = P2, 
			cp1=p1.rowRange(0,2), cp2=p2.rowRange(0,2), 
			cPt= Pt; 

		cvTriangulatePoints(&cP1, &cP2,	&cp1, &cp2, &cPt);
		cv::Mat Ptc(&cPt);
		Ptc = Ptc/Ptc.at<double>(3);
		glColor3f(1,0,0);
		glVertex3f(Ptc.at<double>(0),Ptc.at<double>(1),Ptc.at<double>(2));


	}
	glEnd();
}
