#include "mfg_utils.h"

#ifdef G2O_HAVE_OPENGL
#include "g2o/stuff/opengl_wrapper.h"
#endif

#include <iostream>

#ifdef G2O_HAVE_OPENGL
#include "g2o/stuff/opengl_wrapper.h"
#endif

#include "Eigen/src/SVD/JacobiSVD.h"
#include "edge_line_vp_plane.h"
#include "vertex_vnpt.h"
#include "vertex_plane.h"

namespace g2o {

	bool EdgeLineVpPlane::read(std::istream& is) {
		return true;
	}

	bool EdgeLineVpPlane::write(std::ostream& os) const 
	{
		return true;
	}

	void EdgeLineVpPlane::computeError() {
		VertexSBAPointXYZ* v_lnpt = static_cast<VertexSBAPointXYZ*>(_vertices[0]);
		VertexVanishPoint* v_vp = static_cast<VertexVanishPoint*>(_vertices[1]);
		const VertexPlane3d *v_plane = static_cast<const VertexPlane3d*>(_vertices[2]);

		// represent the line by two points
		cv::Point3d lnpt0 (v_lnpt->estimate()(0), v_lnpt->estimate()(1),v_lnpt->estimate()(2));
		cv::Point3d lnpt1 = lnpt0 + cv::Point3d(v_vp->estimate()(0),v_vp->estimate()(1),v_vp->estimate()(2));

		// project real endpoints onto the line
		cv::Point3d A = projectPt3d2Ln3d (lnpt0, lnpt1, endptA);
		cv::Point3d B = projectPt3d2Ln3d (lnpt0, lnpt1, endptB);
		
		Vector3d vA(A.x, A.y, A.z);
		Vector3d vB(B.x, B.y, B.z);
		// compute line (endpoints) to plane distance
		double d = 1/v_plane->estimate().norm();    // plane depth
		Vector3d n = v_plane->estimate() * d; // plane unit normal
		_error(0) = sqrt (pow(vA.dot(n) + d, 2) + pow(vB.dot(n) + d, 2) );
	}
}
