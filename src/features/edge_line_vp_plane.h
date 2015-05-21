#ifndef G2O_EDGE_LINE_VP_PLANE_H_
#define G2O_EDGE_LINE_VP_PLANE_H_

#include <iostream>
#include <opencv2/core/core.hpp>

#include "vertex_vnpt.h"

#include "g2o/core/base_multi_edge.h"
#include "g2o/types/sba/types_sba.h"
#include "g2o/types/slam3d/parameter_se3_offset.h"


namespace g2o {

	class EdgeLineVpPlane : public BaseMultiEdge<1, double>
	{
	public:
		//      EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		EdgeLineVpPlane() {
			information().setIdentity();
			resize(3);
		}
		virtual bool read(std::istream& is);
		virtual bool write(std::ostream& os) const;
		void computeError();

		void setMeasurement(const double& m){
			_measurement = m;
		}

//	protected:
		cv::Point3d endptA, endptB; // line segment endpoints
	};


} // end namespace


#endif
