#include "utils.h"
#include "settings.h"

#include <Eigen/StdVector>

#include <vector>
#include <stdint.h>

#if defined(_MSC_VER)
//#include <unordered_set>
#include <unordered_map>
//#ifdef __APPLE__
#elif defined(__APPLE__)
#include <unordered_map>
#else
// TODO: FIXME
//#include <tr1/unordered_set>
#include <tr1/unordered_map>
#endif

#include "g2o/config.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/icp/types_icp.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"

#if defined G2O_HAVE_CHOLMOD
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#endif
#if defined G2O_HAVE_CSPARSE
#include "g2o/solvers/csparse/linear_solver_csparse.h"
#endif

#include "g2o/types/sba/sbacam.h"
#include "levmar.h"

using namespace Eigen;
using namespace std;

struct Data_EST3D
{
	vector<cv::Mat> Rs,  ts;
	cv::Mat K;
	vector<cv::Point2d>  pt;
	vector<vector<double> > KRs, Kts;

};

void costFun_EST3D(double *p, double *error, int numPara, int numMeas, void *adata)
{
	struct Data_EST3D* dp = (struct Data_EST3D *) adata;
	double cost=0;
	cv::Mat X = (cv::Mat_<double>(3,1)<<p[0], p[1], p[2]);
	for(int i=0; i < dp->pt.size(); ++i) {
		cv::Point2d pi = mat2cvpt(dp->K *(dp->Rs[i] * X + dp->ts[i]));
		error[i*2] = pi.x - dp->pt[i].x;
		error[i*2+1] = pi.y - dp->pt[i].y;
		cost = cost + error[i*2]*error[i*2] + error[i*2+1]*error[i*2+1];

	}

}


void costFun_EST3D2(double *p, double *error, int numPara, int numMeas, void *adata)
{
	struct Data_EST3D* dp = (struct Data_EST3D *) adata;
	double cost=0;
	cv::Mat X = (cv::Mat_<double>(3,1)<<p[0], p[1], p[2]);
	for(int i=0; i < dp->pt.size(); ++i) {
		cv::Point2d pi = mat2cvpt(dp->K *(dp->Rs[i] * X + dp->ts[i]));
		double xh = dp->KRs[i][0]*p[0] + dp->KRs[i][1]*p[1] + dp->KRs[i][2]*p[2] + dp->Kts[i][0];
		double yh = dp->KRs[i][3]*p[0] + dp->KRs[i][4]*p[1] + dp->KRs[i][5]*p[2] + dp->Kts[i][1];
		double zh = dp->KRs[i][6]*p[0] + dp->KRs[i][7]*p[1] + dp->KRs[i][8]*p[2] + dp->Kts[i][2];
		error[i*2] = (xh/zh) - dp->pt[i].x;
		error[i*2+1] = (yh/zh) - dp->pt[i].y;

//		cost = cost + error[i*2]*error[i*2] + error[i*2+1]*error[i*2+1];

	}

}

void est3dpt (vector<cv::Mat> Rs, vector<cv::Mat> ts, cv::Mat K, vector<cv::Point2d> pt, cv::Mat& X, int maxIter)
// input: Rs, ts, pt
// output: X
{
	// ----- LM parameter setting -----
	double opts[LM_OPTS_SZ], info[LM_INFO_SZ];
	opts[0] = LM_INIT_MU; //
	opts[1] = 1E-50; // gradient threshold, original 1e-15
	opts[2] = 1E-20; // relative para change threshold? original 1e-50
	opts[3] = 1E-20; // error threshold (below it, stop)
	opts[4] = LM_DIFF_DELTA;


	int numPara = 3, numMeas = pt.size()*2;
	double* para = new double[numPara];
	para[0] = X.at<double>(0);
	para[1] = X.at<double>(1);
	para[2] = X.at<double>(2);
	double* meas = new double[numMeas];
	for(int i=0; i<numMeas; ++i) meas[i] = 0;

	Data_EST3D data;
	data.pt = pt;
	data.Rs = Rs;
	data.ts = ts;
	data.K = K;
	data.KRs.resize(Rs.size());
	data.Kts.resize(ts.size());
	for (int i=0; i<Rs.size(); ++i) {
		cv::Mat KR = K * Rs[i];
		cv::Mat Kt = K * ts[i];
		data.KRs[i].resize(9);
		data.Kts[i].resize(3);
		data.KRs[i][0] = KR.at<double>(0,0); data.KRs[i][1] = KR.at<double>(0,1); data.KRs[i][2] = KR.at<double>(0,2);
		data.KRs[i][3] = KR.at<double>(1,0); data.KRs[i][4] = KR.at<double>(1,1); data.KRs[i][5] = KR.at<double>(1,2);
		data.KRs[i][6] = KR.at<double>(2,0); data.KRs[i][7] = KR.at<double>(2,1); data.KRs[i][8] = KR.at<double>(2,2);
		data.Kts[i][0] = Kt.at<double>(0);   data.Kts[i][1] = Kt.at<double>(1);   data.Kts[i][2] = Kt.at<double>(2);
	}

	double * para2 = new double[3]; para2[0] = para[0]; para2[1] = para[1]; para2[2] = para[2];

	// ----- start LM solver -----
	int ret = dlevmar_dif(costFun_EST3D2, para, meas, numPara, numMeas,
						maxIter, opts, info, NULL, NULL, (void*)&data);
/*	cout<<endl;
	 ret = dlevmar_dif(costFun_EST3D2, para2, meas, numPara, numMeas,
						maxIter, opts, info, NULL, NULL, (void*)&data);
	 cout<<endl<<endl;
*/	X = (cv::Mat_<double>(3,1)<<para[0], para[1], para[2]);

	delete[] meas;
	delete[] para;
	delete[] para2;


}

void est3dpt_g2o (vector<cv::Mat> Rs, vector<cv::Mat> ts, cv::Mat K, vector<cv::Point2d> pts2d, cv::Mat& X)
{
	// ----- G2O parameter setting -----
	int maxIters = 15;
	// some handy typedefs
	typedef g2o::BlockSolver< g2o::BlockSolverTraits<Eigen::Dynamic, Eigen::Dynamic> >  MyBlockSolver;
	typedef g2o::LinearSolverCSparse<MyBlockSolver::PoseMatrixType> MyLinearSolver;

	// setup the solver
	g2o::SparseOptimizer optimizer;
	optimizer.setVerbose(false);
	MyLinearSolver* linearSolver = new MyLinearSolver();

	MyBlockSolver* solver_ptr = new MyBlockSolver(linearSolver);
	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);

	// -- add the parameter representing the sensor offset  !!!!
	g2o::ParameterSE3Offset* sensorOffset = new g2o::ParameterSE3Offset;
	sensorOffset->setOffset(Eigen::Isometry3d::Identity());
	sensorOffset->setId(0);
	optimizer.addParameter(sensorOffset);

	// ----- set g2o vertices ------
	int vertex_id = 0;
#ifdef __APPLE__
	unordered_map<int,int> camfid2vid;
#else
	tr1::unordered_map<int,int> camfid2vid;
#endif

	int n = Rs.size(); //number of observations
	// ----- optimization parameters -----
	vector<g2o::VertexCam*> camvertVec;
	camvertVec.reserve(n);
	// ---- camera pose parameters ----
	for(int i = 0; i < n; ++i) {
		Eigen:: Quaterniond q = r2q(Rs[i]);
		Eigen::Isometry3d pose;
		pose = q;
		pose.translation() = Eigen::Vector3d(ts[i].at<double>(0),ts[i].at<double>(1),ts[i].at<double>(2));
		g2o::VertexCam * v_cam = new g2o::VertexCam();
		v_cam->setId(vertex_id);
		g2o::SBACam sc(q.inverse(), pose.inverse().translation());
		sc.setKcam(K.at<double>(0,0),K.at<double>(1,1),K.at<double>(0,2),K.at<double>(1,2),0); ///??????
		v_cam->setEstimate(sc);
		v_cam->setFixed(true);

		optimizer.addVertex(v_cam);
		camfid2vid[i] = vertex_id;
		++vertex_id;
		camvertVec.push_back(v_cam);
  	}

	// ---- structure parameters ----
	g2o::VertexSBAPointXYZ * v_p = new g2o::VertexSBAPointXYZ();
	v_p->setId(vertex_id);
	Eigen::Vector3d pt(X.at<double>(0),X.at<double>(1),X.at<double>(2));
	v_p->setEstimate(pt);

	v_p->setFixed(false);

	optimizer.addVertex(v_p);
	int	pt_vertext_id = vertex_id;

	// ----- optimization measurements -----
	for(int i=0; i < n; ++i) {
		g2o::EdgeProjectP2MC * e = new g2o::EdgeProjectP2MC();
		e->vertices()[0] = dynamic_cast<g2o::OptimizableGraph::Vertex*>
			(optimizer.vertices().find(pt_vertext_id)->second);
		if(e->vertices()[0]==0) {
			cerr<<"no pt vert found ... terminated \n";
			exit(0);
		}
		e->vertices()[1] = dynamic_cast<g2o::OptimizableGraph::Vertex*>
			(optimizer.vertices().find(camfid2vid[i])->second);
		if(e->vertices()[1]==0) {
			cerr<<"no cam vert found ... terminated \n";
			exit(0);
		}
		Eigen::Vector2d meas(pts2d[i].x, pts2d[i].y);
		e->setMeasurement(meas);
		e->information() = Matrix2d::Identity();
		/*				g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
		rk->setDelta(mfgSettings->getBaKernelDeltaPoint());
		e->setRobustKernel(rk);

		*/
		optimizer.addEdge(e);
	}
	// ----- start g2o -----
//	MyTimer timer; 	timer.start();
	optimizer.initializeOptimization();
	optimizer.computeActiveErrors(); double baerr=optimizer.activeChi2();
	optimizer.optimize(maxIters);
//	timer.end(); cout<<"est3dpt_g2o time:"<<timer.time_ms<<"ms\t";
//	optimizer.computeActiveErrors(); cout<<"ba errore: "<<baerr<<"-->"<<optimizer.activeChi2()<<endl;

	X.at<double>(0) = v_p->estimate()(0);
	X.at<double>(1) = v_p->estimate()(1);
	X.at<double>(2) = v_p->estimate()(2);
	optimizer.clear();
}
cv::Mat triangulatePoint_nonlin (const cv::Mat& R1, const cv::Mat& t1, const cv::Mat& R2,
			const cv::Mat& t2,const cv::Mat& K,	cv::Point2d pt1, cv::Point2d pt2)
// nonlinear triangulation of 3d pt
{
	vector<cv::Mat> Rs, ts;
	Rs.push_back(R1);
	Rs.push_back(R2);
	ts.push_back(t1);
	ts.push_back(t2);
	vector<cv::Point2d> pts;
	pts.push_back(pt1);
	pts.push_back(pt2);
	cv::Mat X = triangulatePoint (R1, t1, R2, t2, K, pt1, pt2); // linear triangulation

//	est3dpt_g2o (Rs, ts, K, pts, X);
	MyTimer tm; tm.start();
	est3dpt (Rs, ts, K, pts, X, 80);
//	tm.end(); cout<<"est3dpt "<<tm.time_ms<<" ms"<<endl;
	return X;
}

// ***********************************************************************************
//************************************************************************************

struct Data_SCALE
{
	cv::Mat Rn, Rtn_1, t;
	vector<cv::Point3d>  X;
	vector<cv::Point2d>  x;
	cv::Mat K;
};

void costFun_Scale(double *p, double *error, int numPara, int numMeas, void *adata)
{
	struct Data_SCALE* dp = (struct Data_SCALE *) adata;

	cv::Mat tn = dp->Rtn_1 + p[0] * dp->t  ;
	double cost = 0;
	for(int i=0; i< dp->X.size(); ++i) {
		cv::Point2d xi = mat2cvpt(dp->K * (dp->Rn * cvpt2mat(dp->X[i],0) + tn));
		error[i*2] = xi.x - dp->x[i].x;
		error[i*2+1] = xi.y - dp->x[i].y;

		cost += error[i*2]*error[i*2] + error[i*2+1]*error[i*2+1];
	}

}

double optimizeScale (vector<cv::Point3d> X, vector<cv::Point2d> x, cv::Mat K,
					cv::Mat Rn, cv::Mat Rtn_1, cv::Mat t, double s)
{
	cv::Mat tn = Rtn_1 + t * s;

	// ----- LM parameter setting -----
	double opts[LM_OPTS_SZ], info[LM_INFO_SZ];
	opts[0] = LM_INIT_MU; //
	opts[1] = 1E-50; // gradient threshold, original 1e-15
	opts[2] = 1E-200; // relative para change threshold? original 1e-50
	opts[3] = 1E-20; // error threshold (below it, stop)
	opts[4] = LM_DIFF_DELTA;
	int maxIter = 500;


	int numPara = 1, numMeas = x.size() * 2;
	double* para = new double[numPara];
	para[0] = s;
	double* meas = new double[numMeas];
	for(int i=0; i < numMeas; ++i) meas[i] = 0;

	Data_SCALE data;
	data.Rn = Rn;
	data.Rtn_1 = Rtn_1;
	data.t = t;
	data.X = X;
	data.x = x;
	data.K = K;

	// ----- start LM solver -----
	int ret = dlevmar_dif(costFun_Scale, para, meas, numPara, numMeas,
						maxIter, opts, info, NULL, NULL, (void*)&data);
	delete[] meas;
//	termReason((int)info[6]);

	return para[0];
}
