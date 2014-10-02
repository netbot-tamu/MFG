#include "mfg_utils.h"
#include "g2o/types/slam3d/parameter_se3_offset.h"

#ifdef G2O_HAVE_OPENGL
#include "g2o/stuff/opengl_wrapper.h"
#endif

#include <iostream>

#ifdef G2O_HAVE_OPENGL
#include "g2o/stuff/opengl_wrapper.h"
#endif

#include "Eigen/src/SVD/JacobiSVD.h"
#include "edge_line_vp_cam.h"
#include "vertex_vnpt.h"

namespace g2o {

	bool EdgeLineVpCam::read(std::istream& is) {
		return true;
	}

	bool EdgeLineVpCam::write(std::ostream& os) const 
	{
		return true;
	}

	void EdgeLineVpCam::computeError() {
		VertexSBAPointXYZ* v_lnpt = static_cast<VertexSBAPointXYZ*>(_vertices[0]);
		VertexVanishPoint* v_vp = static_cast<VertexVanishPoint*>(_vertices[1]);
		const VertexCam* cam = static_cast<const VertexCam*>(_vertices[2]);

		const Vector3d &vp = v_vp->estimate();
		Vector4d hvp(vp(0),vp(1),vp(2),0);  // homogeneous, last element = 0 for vanishing point
		Vector3d hvp_im = cam->estimate().w2i * hvp; // homogenous, in image coordinates

		const Vector3d &lnpt = v_lnpt->estimate();
		Vector4d h_lnpt(lnpt(0), lnpt(1), lnpt(2), 1);
		Vector3d h_lnpt_im = cam->estimate().w2i * h_lnpt;

		Vector3d leq_im = hvp_im.cross(h_lnpt_im); // line equation in image coords
		double leq[3] = {leq_im(0), leq_im(1), leq_im(2)};
		_error(0) = 0;
		for (int i=0; i<segpts.size(); ++i) 
		{
			_error(0) += pow(point2LineDist(leq, segpts[i]),2);
		}
		_error(0) = sqrt(_error(0));
	
	}
}
