
#include "optimize.h"

#include <iostream>
#include <eigen3/Eigen/Geometry>
#include "levmar.h"
#include "utils.h"

using namespace std;

////////////////////////////////////////////////////////////////////////////////
// Optimization of R^T with Vanishing Points
////////////////////////////////////////////////////////////////////////////////
// M measurement size
// N para size
void costFun_optimizeRt_withVP (double *p, double *error, int N, int M, void *adata)
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

// optimize relative pose (from 5-point alg) using vanishing point correspondences
// input: K, vppairs, featPtMatches
// output: R, t
void optimizeRt_withVP (cv::Mat K, VPointPairs vppairs, double weightVP,
						FeaturePointPairs& featPtMatches, cv::Mat R, cv::Mat t)
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
}

////////////////////////////////////////////////////////////////////////////////
// Optimization of T Given R
////////////////////////////////////////////////////////////////////////////////
// M measurement size
// N para size
void costFun_optimize_t_givenR (double *p, double *error, int N, int M, void *adata)
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

// optimize relative pose (from 5-point alg) using vanishing point correspondences
// input: K, R, vppairs, featPtMatches
// output: t
void optimize_t_givenR (cv::Mat K, cv::Mat R, FeaturePointPairs& featPtMatches,
						cv::Mat t)
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

