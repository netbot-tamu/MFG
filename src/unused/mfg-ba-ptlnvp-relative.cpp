#include "mfg.h"
#include "mfg_utils.h"

#define LNERR_SAMPLE  // error metric: area/length
//#define USE_VERT_LINE_ONLY
#define USE_2DVP
extern int IDEAL_IMAGE_WIDTH;
struct Data_BA_PTLNVP_REL
{
	int numView, frontPosIdx, frontFrmIdx;
	cv::Mat					K;
	vector<KeyPoint3d>&		kp;    // key points in map
	vector<int>				kptIdx2Opt;    // idx of key points to optimize
	vector<int>				kptIdx2Rpj_notOpt; 
	vector<cv::Mat>			prevPs;		// projection matrices of previous frames
	vector<View>&			views;
	int numVP;
	vector<IdealLine3d>&	il;
	vector<int>				idlIdx2Opt;
	vector<int>				idlIdx2Rpj_notOpt;
	vector<int>				vpIdx2Opt;
	vector<VanishPnt3d>		mfgVPs;
	vector<double>			ref, ref_t;
	int						vp2dObsNum;

	int						vperr_func; // choose which function to use
	// for debugging puropse
	double* ms; 
	double err_pt, err_ln, err_vp, err_all;
	double err_pt_mean, err_ln_mean, err_vp_mean;

	Data_BA_PTLNVP_REL(vector<KeyPoint3d>& kp_, vector<IdealLine3d>& il_, vector<View>& views_) :
			kp(kp_), il(il_), views(views_) {}

};

void costFun_BA_PtLnVp_Rel(double *p, double *error, int numPara, int numMeas, void *adata)
{
	

	struct Data_BA_PTLNVP_REL * dp = (struct Data_BA_PTLNVP_REL *) adata;
	double kernelPt = 2;
	double kernelLn = 5;
	double kernelVpLn = 5;
	vector<double> ref = dp->ref;
	vector<double> ref_t = dp->ref_t;
	int vp2dObsNum = dp->vp2dObsNum;

	// ----- recover parameters for each view and landmark -----
	// ---- pose para ----
	int pidx = 0;
#ifdef USE_2DVP
	vector<cv::Mat>  Ps(dp->views.size());
	for (int i=0, j=0; i < dp->frontPosIdx; ++i, ++j)	{
		Ps[i] = dp->prevPs[j];
	}
#else
	vector<cv::Mat>  Ps(dp->numView); // projection matrices
	for (int i = dp->frontFrmIdx, j=0; i < dp->frontPosIdx; ++i, ++j)	{
		Ps[i] = dp->prevPs[j];
	}
#endif
	Ps[0] = dp->K*(cv::Mat_<double>(3,4)<< 1, 0, 0, 0,
									       0, 1, 0, 0,
									       0, 0, 1, 0);
	for (int i = dp->frontPosIdx; i < dp->numView; ++i) {
		Quaterniond q( p[pidx], p[pidx+1], p[pidx+2], p[pidx+3]);
		q.normalize();
		cv::Mat Ri = (cv::Mat_<double>(3,3) 
			<< q.matrix()(0,0), q.matrix()(0,1), q.matrix()(0,2),
			q.matrix()(1,0), q.matrix()(1,1), q.matrix()(1,2),
			q.matrix()(2,0), q.matrix()(2,1), q.matrix()(2,2));
		cv::Mat ti;
		if (i==1) {
			ti = angle2unitVec (p[pidx+4],p[pidx+5]);		
			pidx = pidx + 6;
		} else {
			if (dp->frontPosIdx<=1) {
			ti = (cv::Mat_<double>(3,1)<<p[pidx+4],p[pidx+5],p[pidx+6]);
			} else {
				ti = (cv::Mat_<double>(3,1)<<p[pidx+4] + ref_t[0],
					p[pidx+5] + ref_t[1], p[pidx+6] + ref_t[2]);
			}
			pidx = pidx + 7;
		}	
		cv::Mat P(3,4,CV_64F);
		Ri.copyTo(P.colRange(0,3));
		ti.copyTo(P.col(3));
		P = dp->K * P;
		Ps[i] = P.clone();
	}

	
	vector<double> err_pt_vec, err_ln_vec, err_vp_vec;

	// ----- reproject points -----		
	int eidx = 0;
	for(int i=0; i < dp->kptIdx2Opt.size(); ++i) {
		int idx =  dp->kptIdx2Opt[i];
		for(int j=0; j < dp->kp[idx].viewId_ptLid.size(); ++j) {
			int vid = dp->kp[idx].viewId_ptLid[j][0];
			int lid = dp->kp[idx].viewId_ptLid[j][1];
			if (vid >= dp->frontFrmIdx) {
				cv::Mat impt = Ps[vid] *(cv::Mat_<double>(4,1)<<p[pidx]+ref[0],p[pidx+1]+ref[1],p[pidx+2]+ref[2],1);
				error[eidx] = mat2cvpt(impt).x - dp->views[vid].featurePoints[lid].x;
				error[eidx+1] = mat2cvpt(impt).y - dp->views[vid].featurePoints[lid].y;
				error[eidx] = sqrt(pesudoHuber(error[eidx], kernelPt));
				error[eidx+1] = sqrt(pesudoHuber(error[eidx+1], kernelPt));

				err_pt_vec.push_back(error[eidx]*error[eidx] + error[eidx+1]*error[eidx+1]);
				
				eidx = eidx + 2;
			}
		}
		pidx = pidx + 3;
	}
	for(int i=0; i < dp->kptIdx2Rpj_notOpt.size(); ++i) {
		int idx =  dp->kptIdx2Rpj_notOpt[i];
		for(int j=0; j < dp->kp[idx].viewId_ptLid.size(); ++j) {
			int vid = dp->kp[idx].viewId_ptLid[j][0];
			int lid = dp->kp[idx].viewId_ptLid[j][1];
			if (vid >= dp->frontFrmIdx) {
				cv::Mat impt = Ps[vid] *  dp->kp[idx].mat(); // just provide reprojection
				error[eidx] = mat2cvpt(impt).x - dp->views[vid].featurePoints[lid].x;
				error[eidx+1] = mat2cvpt(impt).y - dp->views[vid].featurePoints[lid].y;
				error[eidx] = sqrt(pesudoHuber(error[eidx], kernelPt));
				error[eidx+1] = sqrt(pesudoHuber(error[eidx+1], kernelPt));

				err_pt_vec.push_back(error[eidx]*error[eidx] + error[eidx+1]*error[eidx+1]);				
				eidx = eidx + 2;
			}
		}
	}
	
	vector<cv::Mat> VPs;
	for(int i=0; i < dp->mfgVPs.size(); ++i) {
		VPs.push_back(dp->mfgVPs[i].mat());
	}
	for(int i=0; i < dp->vpIdx2Opt.size(); ++i) {
		cv::Mat vp = (cv::Mat_<double>(4,1)<< p[pidx],p[pidx+1],p[pidx+2],0);
		VPs[dp->vpIdx2Opt[i]] = vp;
		pidx = pidx + 3;	
	}
	
	double lineElemLen = 100 * IDEAL_IMAGE_WIDTH/640.0;  // basic length
	// ----- reproject lines -----
	for(int i=0; i < dp->idlIdx2Opt.size(); ++i) {
		int idx = dp->idlIdx2Opt[i];
		cv::Mat E1 = (cv::Mat_<double>(4,1)<< p[pidx]+ref[0], p[pidx+1]+ref[1], p[pidx+2]+ref[2], 1);   // a point on line
		cv::Mat E2 = VPs[dp->il[idx].vpGid];  // use vp direction as line's
		pidx = pidx + 3;
		for (int j=0; j < dp->il[idx].viewId_lnLid.size(); ++j) {
			int vid = dp->il[idx].viewId_lnLid[j][0];
			if (vid >= dp->frontFrmIdx) {
				cv::Mat pe1 = Ps[vid] * E1;
				cv::Mat pe2 = Ps[vid] * E2;
				cv::Mat lneq = (pe1).cross(pe2);
				int lid = dp->il[idx].viewId_lnLid[j][1];	
#ifdef LNERR_SAMPLE
				for(int k=0; k < dp->views[vid].idealLines[lid].lsEndpoints.size(); k=k+2) {
					cv::Point2d ep1 = dp->views[vid].idealLines[lid].lsEndpoints[k];
					cv::Point2d ep2 = dp->views[vid].idealLines[lid].lsEndpoints[k+1];
					error[eidx] = ls2lnArea(lneq, ep1, ep2)/cv::norm(ep1-ep2)/2
						* min(1.0, lineElemLen/cv::norm(ep1-ep2));   // normalized by ls length
					error[eidx] = sqrt(pesudoHuber(error[eidx], kernelLn));
					error[eidx+1] = error[eidx];

					err_ln_vec.push_back(error[eidx]*error[eidx]);  
					err_ln_vec.push_back(error[eidx+1]*error[eidx+1]);// for each line segment
										
					eidx = eidx + 2;
					
				}
#else
				for(int k=0; k < dp->views[vid].idealLines[lid].lsEndpoints.size(); k=k+1) {
					// endpoint to line(projected) distance
					error[eidx] = point2LineDist(lneq, dp->views[vid].idealLines[lid].lsEndpoints[k]);
					error[eidx] = sqrt(pesudoHuber(error[eidx], kernelLn));
					
					err_ln_vec.push_back(error[eidx]*error[eidx]); 

					++eidx;
				}
#endif
			}
		}
	}

	for(int i=0; i < dp->idlIdx2Rpj_notOpt.size(); ++i) {
		int idx = dp->idlIdx2Rpj_notOpt[i];
		for (int j=0; j < dp->il[idx].viewId_lnLid.size(); ++j) {
			int vid = dp->il[idx].viewId_lnLid[j][0];
			if (vid >= dp->frontFrmIdx) {
				cv::Mat lneq = projectLine(dp->il[idx],Ps[vid]);
				int lid = dp->il[idx].viewId_lnLid[j][1];
#ifdef LNERR_SAMPLE
				for(int k=0; k < dp->views[vid].idealLines[lid].lsEndpoints.size(); k=k+2) {
					cv::Point2d ep1 = dp->views[vid].idealLines[lid].lsEndpoints[k];
					cv::Point2d ep2 = dp->views[vid].idealLines[lid].lsEndpoints[k+1];
					error[eidx] = ls2lnArea(lneq, ep1, ep2)/cv::norm(ep1-ep2)/2
						* min(1.0, lineElemLen/cv::norm(ep1-ep2));   // normalized by ls length
					error[eidx] = sqrt(pesudoHuber(error[eidx], kernelLn));
					error[eidx+1] = error[eidx];

					err_ln_vec.push_back(error[eidx]*error[eidx]);  // for each linesegment
					err_ln_vec.push_back(error[eidx+1]*error[eidx+1]);

					eidx = eidx + 2;
				}
#else
				for(int k=0; k < dp->views[vid].idealLines[lid].lsEndpoints.size(); k=k+1) {
					// endpoint to line(projected) distance
					error[eidx] = point2LineDist(lneq, dp->views[vid].idealLines[lid].lsEndpoints[k]);
					error[eidx] = sqrt(pesudoHuber(error[eidx], kernelLn));
					
					err_ln_vec.push_back(error[eidx]*error[eidx]); 

					++eidx;
				}
#endif
			}
		}
	}

#ifdef USE_2DVP
	// ---- reproject vanishing point ----
	for(int i=0; i < dp->vpIdx2Opt.size(); ++i) {
		int vpGid = dp->vpIdx2Opt[i];		
		int start = max((int)0, (int)dp->mfgVPs[vpGid].viewId_vpLid.size() - vp2dObsNum);
		for(int j=start; j < dp->mfgVPs[vpGid].viewId_vpLid.size(); ++j) {
			int vid = dp->mfgVPs[vpGid].viewId_vpLid[j][0];
			int vpid = dp->mfgVPs[vpGid].viewId_vpLid[j][1];
			if(dp->views[vid].vanishPoints[vpid].gid != vpGid)
				cerr<<"vp gid not right!"<<endl;
			cv::Mat vpproj = dp->K.inv() * Ps[vid] * VPs[vpGid];
			vpproj = vpproj/cv::norm(vpproj);
			cv::Mat vpobs = dp->K.inv() * dp->views[vid].vanishPoints[vpid].mat();
			vpobs = vpobs/cv::norm(vpobs);
			double vp2derror_weight = 60;
			cv::Mat cov_ptobs = dp->views[vid].vanishPoints[vpid].cov_pt(dp->K);
	//		vp2derror_weight = 3/sqrt(cov_ptobs.at<double>(0,0)+cov_ptobs.at<double>(1,1));// inverse of angle std
			if(dp->vperr_func == 1) { // angle
				error[eidx] = asin(cv::norm(vpproj.cross(vpobs))) * vp2derror_weight; // angle between two direction
			}
			cv::Mat tmp=Ps[vid] * VPs[vpGid];
			cv::Mat vpproj2d = (cv::Mat_<double>(2,1)<<tmp.at<double>(0)/tmp.at<double>(2),
								tmp.at<double>(1)/tmp.at<double>(2));
			cv::Mat vpobs2d = dp->views[vid].vanishPoints[vpid].mat(0);
			cv::Mat dist = (vpproj2d-vpobs2d).t()*dp->views[vid].vanishPoints[vpid].cov.inv()*(vpproj2d-vpobs2d);
			if(dp->vperr_func == 2) { // vp2d dist
			error[eidx] = sqrt(cv::norm(dist));
			}
//			cout << error[eidx] <<"\t";


/*			cv::Mat pt_prj = pantilt(vpproj2d, dp->K);
			cv::Mat pt_obs = dp->views[vid].vanishPoints[vpid].pantilt(dp->K);
			cv::Mat cov_ptobs = dp->views[vid].vanishPoints[vpid].cov_pt(dp->K);
			cv::Mat pt_dif1 = 
			(pt_prj - pt_obs).t() * cov_ptobs.inv() * (pt_prj - pt_obs);
			cv::Mat pt_prj_revserse = reversePanTilt(pt_prj) ; 
			cv::Mat pt_dif2 = 
			(pt_prj_revserse - pt_obs).t() * cov_ptobs.inv() * (pt_prj_revserse - pt_obs);
			error[eidx] = sqrt(min(cv::norm(pt_dif1),cv::norm(pt_dif2)));

			if (cv::norm(pt_dif1) > cv::norm(pt_dif2)) {
				cout<<"pan-tilt:"<<pt_prj<<pt_obs<<endl;
				cout<<"pt-cov"<<cov_ptobs<<"\t"<<sqrt(cv::norm(pt_dif1))<<"\t"<<sqrt(cv::norm(pt_dif2))<<endl;
				cout<<"vp 2d dist="<<sqrt(cv::norm(dist))<<endl;
				cout<<"vp angle="<<asin((cv::norm(vpproj.cross(vpobs)))) * 180/PI <<"\t"
					<<sqrt(cov_ptobs.at<double>(0,0)+cov_ptobs.at<double>(1,1))*180/PI<<endl;
			}
//			cout << error[eidx] <<"\n";
*/		
			err_vp_vec.push_back(error[eidx]*error[eidx]);
			++eidx;
		}
	}
#endif


	// ----- compute error -----
	double cost_pt = 0, cost_ln = 0, cost_vp = 0, cost;
	for(int i=0; i < err_pt_vec.size(); ++i) {
		cost_pt = cost_pt + err_pt_vec[i];
	}
	for(int i=0; i < err_ln_vec.size(); ++i) {
		cost_ln = cost_ln + err_ln_vec[i];
	}
#ifdef USE_2DVP
	for(int i=0; i < err_vp_vec.size(); ++i) {
		cost_vp += err_vp_vec[i];
	}
#endif
	dp->err_pt = cost_pt;
	dp->err_ln = cost_ln;
	dp->err_pt_mean = cost_pt/err_pt_vec.size();
	dp->err_ln_mean = cost_ln/(err_ln_vec.size()/2); // average for each linesegment(consisting of 2 endpoints)

	cost = cost_pt + cost_ln + cost_vp;
	dp->err_all = cost;
	
	static int count = 0;
	count++;
	if(!(count%200)) {
		cout << cost <<"("<<cost_vp<<")\t";	
	}
}

void Mfg::adjustBundle_PtLnVp_Rel (int numPos, int numFrm)
// local bundle adjustment: points
{
	// ----- BA setting -----
//	int numPos = 3;		// number of camera poses to optimize 
//	int numFrm = 5;	// number of frames to provide measurements (for reprojection error)
//	Note: numFrm should be larger or equal to numPos+2, to fix scale
	int vp2dObsNum, vperr_func=1;
	if(rotateMode()) {
		vp2dObsNum = 5;  // # the past 2d vp for ba
	} else {
		vp2dObsNum = 10;  // # the past 2d vp for ba
	}
	int vpObsThresh = 500;
	int frontPosIdx = max(1, (int)views.size() - numPos);
	int frontFrmIdx = max(0, (int)views.size() - numFrm);
	
	// ----- Relative frame -----
	cv::Mat refmat = -views[frontFrmIdx].R.t()*views[frontFrmIdx].t;
	vector<double> ref(3);
	ref[0] = refmat.at<double>(0);
	ref[1] = refmat.at<double>(1);
	ref[2] = refmat.at<double>(2);
	vector<double> ref_t(3);
	ref_t[0] = views[frontFrmIdx].t.at<double>(0);
	ref_t[1] = views[frontFrmIdx].t.at<double>(1);
	ref_t[2] = views[frontFrmIdx].t.at<double>(2);

	// ----- LM parameter setting -----
	double opts[LM_OPTS_SZ], info[LM_INFO_SZ];
	opts[0] = LM_INIT_MU; //
	opts[1] = 1E-10; // gradient threshold, original 1e-15
	opts[2] = 1E-50; // relative para change threshold? original 1e-50
	opts[3] = 1E-20; // error threshold (below it, stop)
	opts[4] = LM_DIFF_DELTA;
	int maxIter = 500;

	// ----- optimization parameters -----
	vector<double> paraVec; 
	// ---- camera pose parameters ----
	for(int i = frontPosIdx; i < views.size(); ++i) {	
		Quaterniond qi = r2q(views[i].R);
		paraVec.push_back(qi.w());
		paraVec.push_back(qi.x());
		paraVec.push_back(qi.y());
		paraVec.push_back(qi.z());
		if (i==1) {
			double alpha, beta;
			unitVec2angle(views[i].t, &alpha, &beta);
			paraVec.push_back(alpha);
			paraVec.push_back(beta);		
		} else {		
			if ( frontPosIdx <= 1) {
			paraVec.push_back(views[i].t.at<double>(0));
			paraVec.push_back(views[i].t.at<double>(1));
			paraVec.push_back(views[i].t.at<double>(2));
			} else {
				paraVec.push_back(views[i].t.at<double>(0) - ref_t[0]); //....................
			paraVec.push_back(views[i].t.at<double>(1) - ref_t[1]);     // ...................
			paraVec.push_back(views[i].t.at<double>(2) - ref_t[2]);
			}
		}
	}
	// ---- structure parameters ----
	// --- points ---
	vector<int> kptIdx2Opt; // keyPoints idx to optimize 
	vector<int> kptIdx2Rpj_notOpt; // keypoint idx to reproject but not optimize
	for(int i=0; i < keyPoints.size(); ++i) {
		if(!keyPoints[i].is3D) continue;
		for(int j=0; j < keyPoints[i].viewId_ptLid.size(); ++j) {
			if (keyPoints[i].viewId_ptLid[j][0] >= frontPosIdx) { // point to optimize
		// don't optimize too-old (established before frontFrmIdx) points, 
		// but still use their recent observations/reprojections after frontPosIdx		
				if(keyPoints[i].viewId_ptLid[0][0] < frontFrmIdx
					&& keyPoints[i].estViewId < views.back().id) {
					kptIdx2Rpj_notOpt.push_back(i);
				} else {
					paraVec.push_back(keyPoints[i].x -ref[0]);
					paraVec.push_back(keyPoints[i].y -ref[1]);
					paraVec.push_back(keyPoints[i].z -ref[2]);
					kptIdx2Opt.push_back(i);
				}
				break;
			}
		}		
	}
	// --- vanishing points ---	
	vector<int> vpIdx2Opt;
	// vp that is observed in current window
	for(int i=0; i < vanishingPoints.size(); ++i) {
		for(int j=0; j<vanishingPoints[i].viewId_vpLid.size(); ++j) {
			if(vanishingPoints[i].viewId_vpLid[j][0] >= frontPosIdx) { // observed recently
				vpIdx2Opt.push_back(i);
				paraVec.push_back(vanishingPoints[i].x);
				paraVec.push_back(vanishingPoints[i].y);
				paraVec.push_back(vanishingPoints[i].z);
				break;
			}
		}
	}
/*	for(int i=0; i < vanishingPoints.size(); ++i) {
		if(vanishingPoints[i].viewId_vpLid.size() < vpObsThresh) {
			vpIdx2Opt.push_back(i);
			paraVec.push_back(vanishingPoints[i].x);
			paraVec.push_back(vanishingPoints[i].y);
			paraVec.push_back(vanishingPoints[i].z);
		}
	}
*/	cout<<"vp="<<vanishingPoints.size()<<",vpIdx2Opt="<<vpIdx2Opt.size()<<endl;

	// --- lines ---
	vector<int> idlIdx2Opt;
	vector<int> idlIdx2Rpj_notOpt;
	for(int i=0; i < idealLines.size(); ++i) {
		if(!idealLines[i].is3D) continue;
#ifdef USE_VERT_LINE_ONLY
		if(idealLines[i].vpGid != 0) continue;
#endif
		for(int j=0; j < idealLines[i].viewId_lnLid.size(); ++j) {
			if(idealLines[i].viewId_lnLid[j][0] >= frontPosIdx) {
				if(idealLines[i].viewId_lnLid[0][0] < frontFrmIdx
					&& idealLines[i].estViewId < views.back().id ) {
					idlIdx2Rpj_notOpt.push_back(i);
				} else {
					paraVec.push_back(idealLines[i].midpt.x -ref[0]);
					paraVec.push_back(idealLines[i].midpt.y -ref[1]);
					paraVec.push_back(idealLines[i].midpt.z -ref[2]);
					idlIdx2Opt.push_back(i);
				}
				break;
			}
		}
	}

	double pn = 0, pmax = 0;
	int numPara = paraVec.size();
	double* para = new double[numPara];
	for (int i=0; i<numPara; ++i) {
		para[i] = paraVec[i];
		pn = pn + para[i]*para[i];
		if (pmax < abs(para[i]))
			pmax = abs(para[i]);
	}
	cout<<"para vector norm = "<< sqrt(pn) <<"\t max="<<pmax<<endl;


//	idlIdx2Rpj_notOpt.clear(); idlIdx2Opt.clear(); // not use lines in ba
	// ----- optimization measurements -----
	vector<double> measVec;
	for(int i=0; i < kptIdx2Opt.size(); ++i) {
		for(int j=0; j < keyPoints[kptIdx2Opt[i]].viewId_ptLid.size(); ++j) {
			if(keyPoints[kptIdx2Opt[i]].viewId_ptLid[j][0] >= frontFrmIdx) {
				measVec.push_back(0);
				measVec.push_back(0);
			}
		}
	}
	for(int i=0; i < kptIdx2Rpj_notOpt.size(); ++i) {
		for(int j=0; j < keyPoints[kptIdx2Rpj_notOpt[i]].viewId_ptLid.size(); ++j) {
			if(keyPoints[kptIdx2Rpj_notOpt[i]].viewId_ptLid[j][0] >= frontFrmIdx) {
				measVec.push_back(0);
				measVec.push_back(0);
			}
		}
	}

	for(int i=0; i < idlIdx2Opt.size(); ++i) {
		for(int j=0; j < idealLines[idlIdx2Opt[i]].viewId_lnLid.size(); ++j) {
			if(idealLines[idlIdx2Opt[i]].viewId_lnLid[j][0] >= frontFrmIdx) {
				int vid = idealLines[idlIdx2Opt[i]].viewId_lnLid[j][0];
				int lid = idealLines[idlIdx2Opt[i]].viewId_lnLid[j][1];
				for(int k=0; k < views[vid].idealLines[lid].lsEndpoints.size(); ++k) {
					measVec.push_back(0);
				}
			}
		}
	}
	for(int i=0; i < idlIdx2Rpj_notOpt.size(); ++i) {
		for(int j=0; j < idealLines[idlIdx2Rpj_notOpt[i]].viewId_lnLid.size(); ++j) {
			if(idealLines[idlIdx2Rpj_notOpt[i]].viewId_lnLid[j][0] >= frontFrmIdx) {
				int vid = idealLines[idlIdx2Rpj_notOpt[i]].viewId_lnLid[j][0];
				int lid = idealLines[idlIdx2Rpj_notOpt[i]].viewId_lnLid[j][1];
				for(int k=0; k < views[vid].idealLines[lid].lsEndpoints.size(); ++k) {
					measVec.push_back(0);
				}
			}
		}
	}
#ifdef USE_2DVP
	for(int i=0; i< vpIdx2Opt.size(); ++i) {
		int vpGid = vpIdx2Opt[i];
		int start = max((int)0, (int)vanishingPoints[vpGid].viewId_vpLid.size() - vp2dObsNum);
		for(int j = start; j < vanishingPoints[vpGid].viewId_vpLid.size(); ++j) {
			measVec.push_back(0);
		}
	}
#endif

	int numMeas = measVec.size();
	double* meas = new double[numMeas];
	for ( int i=0; i<numMeas; ++i) {
		meas[i] = measVec[i];
	}

	// ----- pass additional data -----
	Data_BA_PTLNVP_REL data(keyPoints, idealLines, views);
	data.kptIdx2Opt = kptIdx2Opt;
	data.kptIdx2Rpj_notOpt = kptIdx2Rpj_notOpt;
	data.numView = views.size();
	data.frontPosIdx = frontPosIdx;
	data.frontFrmIdx = frontFrmIdx;
	data.K = K;
	data.ms = meas;
	data.numVP = vpIdx2Opt.size();
	data.ref = ref;
	data.ref_t = ref_t;
	data.vpIdx2Opt = vpIdx2Opt;
	data.mfgVPs = vanishingPoints;

	data.idlIdx2Opt = idlIdx2Opt;
	data.idlIdx2Rpj_notOpt = idlIdx2Rpj_notOpt;
#ifdef USE_2DVP
	data.vp2dObsNum = vp2dObsNum;
	data.vperr_func = vperr_func;
	for(int i=0; i<frontPosIdx; ++i) {
		cv::Mat P(3,4,CV_64F);
		views[i].R.copyTo(P.colRange(0,3));
		views[i].t.copyTo(P.col(3));
		P = K * P;
		data.prevPs.push_back(P);
	}
#else
	for(int i=frontFrmIdx; i<frontPosIdx; ++i) {
		cv::Mat P(3,4,CV_64F);
		views[i].R.copyTo(P.colRange(0,3));
		views[i].t.copyTo(P.col(3));
		P = K * P;
		data.prevPs.push_back(P);
	}
#endif
	
	// ----- start LM solver -----
	MyTimer timer;
	timer.start();
	cout<<"View "+num2str(views.back().id)<<", paraDim="<<numPara<<", measDim="<<numMeas<<endl;
	int ret = dlevmar_dif(costFun_BA_PtLnVp_Rel, para, meas, numPara, numMeas,
						maxIter, opts, info, NULL, NULL, (void*)&data);
	timer.end();
	cout<<" Time used: "<<timer.time_s<<" sec. ";
	termReason((int)info[6]);
	delete[] meas;	

	// ----- update camera and structure parameters -----
	int pidx = 0;
	for(int i = frontPosIdx; i < views.size(); ++i) {	
		Quaterniond q( para[pidx], para[pidx+1], para[pidx+2], para[pidx+3]);
		q.normalize();
		views[i].R = (cv::Mat_<double>(3,3) 
			<< q.matrix()(0,0), q.matrix()(0,1), q.matrix()(0,2),
			q.matrix()(1,0), q.matrix()(1,1), q.matrix()(1,2),
			q.matrix()(2,0), q.matrix()(2,1), q.matrix()(2,2));
		if (i==1) {
			views[i].t = angle2unitVec(para[pidx+4],para[pidx+5]);
			pidx = pidx + 6;
		} else {
			if(frontPosIdx <=1 ) {
			views[i].t = (cv::Mat_<double>(3,1)<<para[pidx+4],para[pidx+5],para[pidx+6]);
			} else	{
				views[i].t = (cv::Mat_<double>(3,1)<<para[pidx+4] + ref_t[0],
					para[pidx+5] + ref_t[1], para[pidx+6] + ref_t[2]);
			}
			pidx = pidx + 7;
		}
	}
	// ---- structure parameters ----
	for(int i=0; i < kptIdx2Opt.size(); ++i) {
		int idx = kptIdx2Opt[i];
		keyPoints[idx].x = para[pidx]  +ref[0];
		keyPoints[idx].y = para[pidx+1]+ref[1];
		keyPoints[idx].z = para[pidx+2]+ref[2];
		pidx = pidx + 3;
	}
	for(int i=0; i < vpIdx2Opt.size(); ++i) {
		vanishingPoints[vpIdx2Opt[i]].x = para[pidx];
		vanishingPoints[vpIdx2Opt[i]].y = para[pidx+1];
		vanishingPoints[vpIdx2Opt[i]].z = para[pidx+2];
		pidx = pidx + 3;
	}

	for ( int i=0; i < idlIdx2Opt.size(); ++i) {
		int idx = idlIdx2Opt[i];
		cv::Point3d oldMidPt = idealLines[idx].midpt;
		idealLines[idx].midpt.x = para[pidx]  + ref[0];
		idealLines[idx].midpt.y = para[pidx+1]+ ref[1];
		idealLines[idx].midpt.z = para[pidx+2]+ ref[2];

		//update line direction
		idealLines[idx].direct = vanishingPoints[idealLines[idx].vpGid].mat(0);
		pidx = pidx + 3;
		
		// keep old midpt
		idealLines[idx].midpt = projectPt3d2Ln3d (idealLines[idx], oldMidPt);
	}

	views.back().errAll = data.err_all;
	views.back().errPt = data.err_pt;
	views.back().errLn = data.err_ln;
	views.back().errPtMean = data.err_pt_mean;
	views.back().errLnMean = data.err_ln_mean;
	
}



