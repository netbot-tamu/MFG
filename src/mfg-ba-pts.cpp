#include "mfg.h"
#include "mfg_utils.h"

struct Data_BA_PTS
{
	int						numView, frontPosIdx, frontFrmIdx;
	cv::Mat					K;
	vector<KeyPoint3d>&		kp;    // key points in map
	vector<int>				kptIdx2Opt;    // idx of key points to optimize
	vector<int>				kptIdx2Rpj_notOpt; 
	vector<cv::Mat>			prevPs;		// projection matrices of previous frames
	vector<View>&			views;

	double* ms; // for debugging puropse
	double err_pt, err_all;
	double err_pt_median, err_pt_mean;

	Data_BA_PTS(vector<KeyPoint3d>& kp_, vector<View>& views_) : kp(kp_), views(views_) {}

};

void costFun_BA_Pts(double *p, double *error, int numPara, int numMeas, void *adata)
{
	struct Data_BA_PTS* dp = (struct Data_BA_PTS *) adata;
	double kernelPt = 2;
	double kernelLn = 5;
	double kernelVpLn = 5;

	// ----- recover parameters for each view and landmark -----
	// ---- pose para ----
	int pidx = 0;
	vector<cv::Mat>  Ps(dp->numView); // projection matrices
	for (int i = dp->frontFrmIdx, j=0; i < dp->frontPosIdx; ++i, ++j)	{
		Ps[i] = dp->prevPs[j];
	}
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
		//	cout<<ti<<endl;
			pidx = pidx + 6;
		} else {
			ti = (cv::Mat_<double>(3,1)<<p[pidx+4],p[pidx+5],p[pidx+6]);
			pidx = pidx + 7;
		}	
		cv::Mat P(3,4,CV_64F);
		Ri.copyTo(P.colRange(0,3));
		ti.copyTo(P.col(3));
		P = dp->K * P;
		Ps[i] = P.clone();

	}
		
	// ----- reproject points -----		
	int eidx = 0;
	for(int i=0; i < dp->kptIdx2Opt.size(); ++i) {
		int idx =  dp->kptIdx2Opt[i];
		for(int j=0; j < dp->kp[idx].viewId_ptLid.size(); ++j) {
			int vid = dp->kp[idx].viewId_ptLid[j][0];
			int lid = dp->kp[idx].viewId_ptLid[j][1];
			if (vid >= dp->frontFrmIdx) {
				cv::Mat impt = Ps[vid] *(cv::Mat_<double>(4,1)<<p[pidx],p[pidx+1],p[pidx+2],1);
				error[eidx] = mat2cvpt(impt).x - dp->views[vid].featurePoints[lid].x;
				error[eidx+1] = mat2cvpt(impt).y - dp->views[vid].featurePoints[lid].y;
				error[eidx] = sqrt(pesudoHuber(error[eidx], kernelPt));
				error[eidx+1] = sqrt(pesudoHuber(error[eidx+1], kernelPt));
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

				eidx = eidx + 2;
			}
		}
	}
	
	// compute error stat, for testing
	double cost_pt = 0;
	double maxerrpt = 0;
//	vector<double> cost_pt_vec;
	for(int i=0; i < numMeas; i=i+2) {
		cost_pt = cost_pt + error[i]*error[i] + error[i+1]*error[i+1];
//		cost_pt_vec.push_back(error[i]*error[i]+ error[i+1]*error[i+1]);
//		if(maxerrpt < cost_pt_vec.back())
//			maxerrpt = cost_pt_vec.back();
	}
//	size_t n = cost_pt_vec.size() / 2;		 
//	nth_element(cost_pt_vec.begin(), cost_pt_vec.begin()+n, cost_pt_vec.end());
//	dp->err_pt_median = cost_pt_vec[n]; // median
	dp->err_pt_mean = cost_pt/(numMeas/2);
	dp->err_pt = cost_pt;
	dp->err_all = cost_pt;

	static int count = 0;
	count++;
	if(!(count % 200)) {
	//	cout << cost_pt<<"("<<dp->err_pt_mean<<","<<maxerrpt<< ")\t"; //total, mean, max
		cout << cost_pt<<"("<<dp->err_pt_mean<< ")\t"; //total, mean
	//	cout << cost_pt <<"\t";
	}
}

void Mfg::adjustBundle_Pt ()
// local bundle adjustment: points
{
	// ----- BA setting -----
	int numPos = 3;		// number of camera poses to optimize 
	int numFrm = 5;	// number of frames to provide measurements (for reprojection error)
	// Note: numFrm should be larger or equal to numPos+2, to fix scale

	if (angVel > 5) {
		cout<<"************ need more for BA ************ \n";
		numPos = 4;
		numFrm = 7;
	}

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
	int frontPosIdx = max(1, (int)views.size() - numPos);
	int frontFrmIdx = max(0, (int)views.size() - numFrm);
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
			paraVec.push_back(views[i].t.at<double>(0));
			paraVec.push_back(views[i].t.at<double>(1));
			paraVec.push_back(views[i].t.at<double>(2));
		}
	}
	// ---- structure parameters ----
	vector<int> kptIdx2Opt; // keyPoint idx to optimize 
	vector<int> kptIdx2Rpj_notOpt; // keypoint idx to reproject but not optimize
	// points-to-optimize contains those first appearing after frontFrmIdx and still being observed after frontPosIdx
	for(int i=0; i < keyPoints.size(); ++i) {
		if(!keyPoints[i].is3D) continue;
		for(int j=0; j < keyPoints[i].viewId_ptLid.size(); ++j) {
			if (keyPoints[i].viewId_ptLid[j][0] >= frontPosIdx) {
		// don't optimize too-old (established before frontFrmIdx) points, 
		// but still use their recent observations/reprojections after frontPosIdx		
				if(keyPoints[i].viewId_ptLid[0][0] < frontFrmIdx) {
					kptIdx2Rpj_notOpt.push_back(i);
				} else {
					paraVec.push_back(keyPoints[i].x);
					paraVec.push_back(keyPoints[i].y);
					paraVec.push_back(keyPoints[i].z);
					kptIdx2Opt.push_back(i);
				}
				break;
			}
		}		
	}

	int numPara = paraVec.size();
	double* para = new double[numPara];
	for (int i=0; i<numPara; ++i) {
		para[i] = paraVec[i];
	}

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
	int numMeas = measVec.size();
	double* meas = new double[numMeas];
	for ( int i=0; i<numMeas; ++i) {
		meas[i] = measVec[i];
	}

	// ----- pass additional data -----
	Data_BA_PTS data(keyPoints, views);
	data.kptIdx2Opt = kptIdx2Opt;
	data.kptIdx2Rpj_notOpt = kptIdx2Rpj_notOpt;
	data.numView = views.size();
	data.frontPosIdx = frontPosIdx;
	data.frontFrmIdx = frontFrmIdx;
	data.K = K;
	data.ms = meas;
	
	for(int i=frontFrmIdx; i<frontPosIdx; ++i) {
		cv::Mat P(3,4,CV_64F);
		views[i].R.copyTo(P.colRange(0,3));
		views[i].t.copyTo(P.col(3));
		P = K * P;
		data.prevPs.push_back(P);
	}
	
	// ----- start LM solver -----
	MyTimer timer;
	timer.start();
	cout<<"View "+num2str(views.back().id)<<", paraDim="<<numPara<<", measDim="<<numMeas<<endl;
	int ret = dlevmar_dif(costFun_BA_Pts, para, meas, numPara, numMeas,
						maxIter, opts, info, NULL, NULL, (void*)&data);
	timer.end();
	cout<<"\n Time used: "<<timer.time_s<<" sec. ";
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
			views[i].t = (cv::Mat_<double>(3,1)<<para[pidx+4],para[pidx+5],para[pidx+6]);
			pidx = pidx + 7;
		}
	}
	// ---- structure parameters ----
	for(int i=0; i < kptIdx2Opt.size(); ++i) {
		int idx = kptIdx2Opt[i];
		keyPoints[idx].x = para[pidx];
		keyPoints[idx].y = para[pidx+1];
		keyPoints[idx].z = para[pidx+2];
		pidx = pidx + 3;
	}

	// ----- write to file -----
	views.back().errAll = data.err_all;
	views.back().errPt = data.err_pt;
	views.back().errPtMean = data.err_pt_mean;
//	exportCamPose (*this, num2str(views.back().id)+".txt");
	exportCamPose (*this, "latest.txt");
}