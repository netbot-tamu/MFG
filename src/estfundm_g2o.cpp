#include "mfg.h"
#include "mfg_utils.h"
#include <Eigen/StdVector>
#include <stdint.h>

#ifdef _MSC_VER
#include <unordered_set>
#else
#include <tr1/unordered_set>
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
#elif defined G2O_HAVE_CSPARSE
#include "g2o/solvers/csparse/linear_solver_csparse.h"
#endif

#include "g2o/types/sba/sbacam.h"
//#include "edge_se3_lineendpts.h"
#include <fstream>
//#include "levmar-2.6/lm.h"
#include "levmar-2.6/levmar.h"
//#include <Windows.h>

void optimize_E_g2o (cv::Mat p1, cv::Mat p2, cv::Mat K, cv::Mat *E)
{
// ----- G2O parameter setting -----
	int maxIters = 40;
	// some handy typedefs
	typedef g2o::BlockSolver< g2o::BlockSolverTraits<Eigen::Dynamic, Eigen::Dynamic> >  MyBlockSolver;
	typedef g2o::LinearSolverCSparse<MyBlockSolver::PoseMatrixType> MyLinearSolver;
	typedef g2o::LinearSolverDense<MyBlockSolver::PoseMatrixType> MyDenseLinearSolver;

	// setup the solver
	g2o::SparseOptimizer optimizer;
	optimizer.setVerbose(0);
	MyLinearSolver* linearSolver = new MyLinearSolver();

	MyBlockSolver* solver_ptr = new MyBlockSolver(linearSolver);
	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);

	// -- add the parameter representing the sensor offset  !!!!
	g2o::ParameterSE3Offset* sensorOffset = new g2o::ParameterSE3Offset;
	sensorOffset->setOffset(Eigen::Isometry3d::Identity());
	sensorOffset->setId(0);
	optimizer.addParameter(sensorOffset);

	int n = p1.cols;
	
	cv::Mat R1, R2, t;	
	decEssential (E, &R1, &R2, &t);
	
	cv::Mat Rt = // find true R and t
		findTrueRt(R1,R2,t,mat2cvpt(p1.col(0)),mat2cvpt(p2.col(0)));
	
	if(Rt.cols < 3) return;

	cv::Mat R =  Rt.colRange(0,3);
	t = Rt.col(3);

	int vertex_id = 0;
	// === first camera ===
		Eigen::Quaterniond q0;
		q0.setIdentity();
		g2o::VertexCam * v_cam0 = new g2o::VertexCam();
		v_cam0->setId(vertex_id);
		g2o::SBACam sc0(q0, Vector3d(0,0,0));
		sc0.setKcam(K.at<double>(0,0),K.at<double>(1,1),K.at<double>(0,2),K.at<double>(1,2),0); ///??????
		v_cam0->setEstimate(sc0);
		v_cam0->setFixed(true);
		optimizer.addVertex(v_cam0);
		++vertex_id;

	// === second camera ===
	g2o::VertexCam* cam1;
		Eigen::Quaterniond q1 = r2q(R);
		Eigen::Isometry3d pose;
		pose = q1;
		pose.translation() = Eigen::Vector3d(t.at<double>(0),t.at<double>(1),t.at<double>(2));
		g2o::VertexCam * v_cam1 = new g2o::VertexCam();
		v_cam1->setId(vertex_id);
		g2o::SBACam sc1(q1.inverse(), pose.inverse().translation());
		sc1.setKcam(K.at<double>(0,0),K.at<double>(1,1),K.at<double>(0,2),K.at<double>(1,2),0); ///??????
		v_cam1->setEstimate(sc1);
		v_cam1->setFixed(false);
		optimizer.addVertex(v_cam1);
		++vertex_id;
		cam1 = v_cam1;
	// === temporary 3d points ===
	for(int i=0; i<n; ++i) {
		cv::Mat p3d = triangulatePoint(cv::Mat::eye(3,3,CV_64F), cv::Mat::zeros(3,1,CV_64F), R, t,  
			cv::Mat::eye(3,3,CV_64F), mat2cvpt(p1.col(i)),  mat2cvpt(p2.col(i)));
		g2o::VertexSBAPointXYZ * v_p = new g2o::VertexSBAPointXYZ();
		v_p->setId(vertex_id);
		Eigen::Vector3d pt(p3d.at<double>(0),p3d.at<double>(1),p3d.at<double>(2));
		v_p->setEstimate(pt);
		v_p->setFixed(false);
		optimizer.addVertex(v_p);
		++vertex_id;
	}

	// === optimization measurments ===
	for (int i=0; i<n; ++i) {
		g2o::EdgeProjectP2MC * e0 = new g2o::EdgeProjectP2MC();
		e0->vertices()[0] = dynamic_cast<g2o::OptimizableGraph::Vertex*>
			(optimizer.vertices().find(i+2)->second);
		e0->vertices()[1] = dynamic_cast<g2o::OptimizableGraph::Vertex*>
			(optimizer.vertices().find(0)->second);
		cv::Point2d impt0 = mat2cvpt(K * p1.col(i));

		Eigen::Vector2d meas0 (impt0.x, impt0.y);
		e0->setMeasurement(meas0);
		e0->information() = Matrix2d::Identity();
		g2o::RobustKernelHuber* rk0 = new g2o::RobustKernelHuber;
		rk0->setDelta(1);
		e0->setRobustKernel(rk0);
		optimizer.addEdge(e0);
		
		g2o::EdgeProjectP2MC * e1 = new g2o::EdgeProjectP2MC();
		e1->vertices()[0] = dynamic_cast<g2o::OptimizableGraph::Vertex*>
			(optimizer.vertices().find(i+2)->second);
		e1->vertices()[1] = dynamic_cast<g2o::OptimizableGraph::Vertex*>
			(optimizer.vertices().find(1)->second);
		cv::Point2d impt1 = mat2cvpt(K * p2.col(i));

		Eigen::Vector2d meas1 (impt1.x, impt1.y);
		e1->setMeasurement(meas1);
		e1->information() = Matrix2d::Identity();
		g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
		rk1->setDelta(1);
		e1->setRobustKernel(rk1);
		optimizer.addEdge(e1);
		
	}

	// === optimize ===
	optimizer.initializeOptimization();
	optimizer.optimize(maxIters);
	Quaterniond q(cam1->estimate().inverse().rotation());
	double qd[] = {q.w(), q.x(), q.y(), q.z()};
	R = q2r(qd);
	t = (cv::Mat_<double>(3,1)<<cam1->estimate().inverse().translation()[0],
			cam1->estimate().inverse().translation()[1],
			cam1->estimate().inverse().translation()[2]);
	t = t/cv::norm(t);
	
	*E = vec2SkewMat(t) * R;
	optimizer.clear();
}
