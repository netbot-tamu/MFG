#include "mfg.h"
#include "mfg_utils.h"
#include <Eigen/StdVector>
#include <stdint.h>

#ifdef _MSC_VER
//#include <unordered_set>
#include <unordered_map>
#else
// TODO: FIXME
//#include <tr1/unordered_set>
#include <unordered_map>
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
//#include "levmar-2.6\lm.h"
#include "levmar-2.6/levmar.h"
//#include <Windows.h>

using namespace Eigen;
extern SysPara syspara;

void Mfg::adjustBundle_Pt_G2O (int numPos, int numFrm)
// local bundle adjustment: points
{	
	// ----- BA setting -----
	// Note: numFrm should be larger or equal to numPos+2, to fix scale

	// ----- G2O parameter setting -----
	int maxIters = 15;
	// some handy typedefs
	typedef g2o::BlockSolver< g2o::BlockSolverTraits<Eigen::Dynamic, Eigen::Dynamic> >  MyBlockSolver;
	typedef g2o::LinearSolverCSparse<MyBlockSolver::PoseMatrixType> MyLinearSolver;
	typedef g2o::LinearSolverDense<MyBlockSolver::PoseMatrixType> MyDenseLinearSolver;

	// setup the solver
	g2o::SparseOptimizer optimizer;
	optimizer.setVerbose(false);
	MyLinearSolver* linearSolver = new MyLinearSolver();
	MyDenseLinearSolver* linearDenseSolver= new MyDenseLinearSolver();

	MyBlockSolver* solver_ptr = new MyBlockSolver(linearSolver);
//	MyBlockSolver* solver_ptr = new MyBlockSolver(linearDenseSolver);
	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
//	g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
	optimizer.setAlgorithm(solver);

	// -- add the parameter representing the sensor offset  !!!!
	g2o::ParameterSE3Offset* sensorOffset = new g2o::ParameterSE3Offset;
	sensorOffset->setOffset(Eigen::Isometry3d::Identity());
	sensorOffset->setId(0);
	optimizer.addParameter(sensorOffset);

	// ----- set g2o vertices ------
	int vertex_id = 0;  
   // TODO: needs tr1 namespace???
	unordered_map<int,int> camvid2fid, camfid2vid, ptvid2gid, ptgid2vid;
  
	int frontPosIdx = max(1, (int)views.size() - numPos);
	int frontFrmIdx = max(0, (int)views.size() - numFrm);
	
	// ----- optimization parameters -----
	vector<g2o::VertexCam*> camvertVec;
	camvertVec.reserve(numPos);
	// ---- camera pose parameters ----
	for(int i = frontFrmIdx; i < views.size(); ++i) {	
		Eigen:: Quaterniond q = r2q(views[i].R);
		Eigen::Isometry3d pose;
		pose = q;
		pose.translation() = Eigen::Vector3d(views[i].t.at<double>(0),views[i].t.at<double>(1),views[i].t.at<double>(2));
		g2o::VertexCam * v_cam = new g2o::VertexCam();
		v_cam->setId(vertex_id);
		g2o::SBACam sc(q.inverse(), pose.inverse().translation());
	//	((g2o::SBACam*)v_cam)->setKcam(K.at<double>(0,0),K.at<double>(1,1),K.at<double>(0,2),K.at<double>(1,2),0);//???
		sc.setKcam(K.at<double>(0,0),K.at<double>(1,1),K.at<double>(0,2),K.at<double>(1,2),0); ///??????
		v_cam->setEstimate(sc);
		
		if (i<1 || i<frontPosIdx) {			
			v_cam->setFixed(true);
		}
		optimizer.addVertex(v_cam);
		camvid2fid[vertex_id] = i;
		camfid2vid[i] = vertex_id;
		++vertex_id;
		camvertVec.push_back(v_cam);
  	}
	
	// ---- structure parameters ----
	vector<g2o::VertexSBAPointXYZ*> ptvertVec;
	vector<int> kptIdx2Opt; // keyPoint idx to optimize 
	vector<int> kptIdx2Rpj_notOpt; // keypoint idx to reproject but not optimize
	// points-to-optimize contains those first appearing after frontFrmIdx and still being observed after frontPosIdx
	for(int i=0; i < keyPoints.size(); ++i) {
		if(!keyPoints[i].is3D) continue;
		for(int j=0; j < keyPoints[i].viewId_ptLid.size(); ++j) {
			if (keyPoints[i].viewId_ptLid[j][0] >= frontPosIdx) {
		// don't optimize too-old (established before frontFrmIdx) points, 
		// but still use their recent observations/reprojections after frontPosIdx	
				g2o::VertexSBAPointXYZ * v_p = new g2o::VertexSBAPointXYZ();
					v_p->setId(vertex_id);
			//		v_p->setMarginalized(true);
					Eigen::Vector3d pt;
					pt<<keyPoints[i].x,keyPoints[i].y,keyPoints[i].z;
					v_p->setEstimate(pt);

				if(keyPoints[i].viewId_ptLid[0][0] < frontFrmIdx && keyPoints[i].estViewId < views.back().id) {
					kptIdx2Rpj_notOpt.push_back(i);
					v_p->setFixed(true);
				} else {
					v_p->setFixed(false);
					kptIdx2Opt.push_back(i);
					ptvertVec.push_back(v_p);
				}
				optimizer.addVertex(v_p);
				ptgid2vid[keyPoints[i].gid] = vertex_id;
				ptvid2gid[vertex_id] = keyPoints[i].gid;
				++vertex_id;				
				break;
			}
		}		
	}
	// ----- optimization measurements -----
	for(int i=0; i < kptIdx2Opt.size(); ++i) {
		for(int j=0; j < keyPoints[kptIdx2Opt[i]].viewId_ptLid.size(); ++j) {
			int fid = keyPoints[kptIdx2Opt[i]].viewId_ptLid[j][0]; //fram(view) id
			int lid = keyPoints[kptIdx2Opt[i]].viewId_ptLid[j][1];//local id
			if(fid >= frontFrmIdx) {
				g2o::EdgeProjectP2MC * e = new g2o::EdgeProjectP2MC();
				e->vertices()[0] = dynamic_cast<g2o::OptimizableGraph::Vertex*>
						(optimizer.vertices().find(ptgid2vid[kptIdx2Opt[i]])->second);
				if(e->vertices()[0]==0) {
					cerr<<"no pt vert found ... terminated \n";
						exit(0);
				}
				e->vertices()[1] = dynamic_cast<g2o::OptimizableGraph::Vertex*>
						(optimizer.vertices().find(camfid2vid[fid])->second);
				if(e->vertices()[1]==0) {
					cerr<<"no cam vert found ... terminated \n";
						exit(0);
				}
				Eigen::Vector2d meas;
				meas<<views[fid].featurePoints[lid].x,views[fid].featurePoints[lid].y;
				e->setMeasurement(meas);
                e->information() = Matrix2d::Identity();
				if(syspara.ba_use_kernel) {
					g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
					rk->setDelta(syspara.ba_kernel_delta_pt);
					e->setRobustKernel(rk);
				}
				optimizer.addEdge(e);
			
			}
		}
	}
	for(int i=0; i < kptIdx2Rpj_notOpt.size(); ++i) {
		for(int j=0; j < keyPoints[kptIdx2Rpj_notOpt[i]].viewId_ptLid.size(); ++j) {
			int fid = keyPoints[kptIdx2Rpj_notOpt[i]].viewId_ptLid[j][0] ;
			int lid = keyPoints[kptIdx2Rpj_notOpt[i]].viewId_ptLid[j][1] ;
			if(fid >= frontFrmIdx) {
				g2o::EdgeProjectP2MC * e = new g2o::EdgeProjectP2MC();

				e->vertices()[0] = dynamic_cast<g2o::OptimizableGraph::Vertex*>
						(optimizer.vertices().find(ptgid2vid[kptIdx2Rpj_notOpt[i]])->second);
				if(e->vertices()[0]==0) {
					cerr<<"no pt vert found ... terminated \n";
						exit(0);
				}
				e->vertices()[1] = dynamic_cast<g2o::OptimizableGraph::Vertex*>
						(optimizer.vertices().find(camfid2vid[fid])->second);
				if(e->vertices()[1]==0) {
					cerr<<"no cam vert found ... terminated \n";
						exit(0);
				}
				Eigen::Vector2d meas;
				meas<<views[fid].featurePoints[lid].x,views[fid].featurePoints[lid].y;
				e->setMeasurement(meas);
                e->information() = Matrix2d::Identity();
				if(syspara.ba_use_kernel) {
					g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
					rk->setDelta(syspara.ba_kernel_delta_pt);
					e->setRobustKernel(rk);
				}
				optimizer.addEdge(e);
			}
		}
	}
	// ----- start g2o -----
	MyTimer timer;
	timer.start();
	optimizer.initializeOptimization();
	optimizer.computeActiveErrors(); double baerr=optimizer.activeChi2();
	optimizer.optimize(maxIters);
	timer.end(); cout<<"ba time:"<<timer.time_ms<<"ms\t";
	optimizer.computeActiveErrors(); cout<<"ba errore: "<<baerr<<"-->"<<optimizer.activeChi2()<<endl;
	// ----- update camera and structure parameters -----
	int idx = 0;
		double scale = 1; 
	bool   toScale = false;
	assert(camvertVec.size() == views.size()-frontFrmIdx);
	for(int i = frontFrmIdx; i < views.size(); ++i,++idx) {	
		if(camvertVec[idx]->fixed()) continue;
		if(camvid2fid.find(camvertVec[idx]->id())==camvid2fid.end())
			cout<<camvertVec[idx]->id()<<" vert not found ...\n";
		Vector3d t = camvertVec[idx]->estimate().inverse().translation();
		Quaterniond q(camvertVec[idx]->estimate().inverse().rotation());
		double qd[] = {q.w(), q.x(), q.y(), q.z()};
		views[i].R = q2r(qd);
		views[i].t = (cv::Mat_<double>(3,1)<<t(0),t(1),t(2));
		if (i == 1) { // second keyframe
			toScale = true;
			scale = 1/t.norm();
		}
	}
	// ---- structure parameters ----
	assert(kptIdx2Opt.size() == ptvertVec.size());
	for(int i=0; i < kptIdx2Opt.size(); ++i) {
		if(! ptvertVec[i]->fixed()){		
			Vector3d kpt = ptvertVec[i]->estimate();
			keyPoints[kptIdx2Opt[i]].x = kpt(0);  
			keyPoints[kptIdx2Opt[i]].y = kpt(1);
			keyPoints[kptIdx2Opt[i]].z = kpt(2);
		}
	}
	optimizer.clear();
	if(toScale) {
		for(int i=0; i<views.size();++i) {
			views[i].t = views[i].t * scale;
		}
		for(int i=0; i<keyPoints.size();++i) {
			if (keyPoints[i].is3D) {
			keyPoints[i].x = keyPoints[i].x * scale;
			keyPoints[i].y = keyPoints[i].y * scale;
			keyPoints[i].z = keyPoints[i].z * scale;
			}
		}
		for(int i=0; i<idealLines.size();++i) {
			if(idealLines[i].is3D) {
			idealLines[i].length = idealLines[i].length * scale;
			idealLines[i].midpt = idealLines[i].midpt * scale;
			}
		}
	
	}
}
