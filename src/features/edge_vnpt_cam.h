#ifndef G2O_EDGE_VNTP_CAM_H_
#define G2O_EDGE_VNTP_CAM_H_

#include <iostream>

#include "vertex_vnpt.h"
#include "g2o/core/base_binary_edge.h"
#include "g2o/types/sba/types_sba.h"
#include "g2o/types/slam3d/parameter_se3_offset.h"

namespace g2o {
  // first two args are the measurement type, second two the connection classes
  class EdgeVnptCam : public BaseBinaryEdge<2, Eigen::Vector2d, VertexVanishPoint, VertexCam> {
  public:
//    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeVnptCam();
    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;

    // return the error estimate as a 2-vector
    void computeError();
    // jacobian
 //   virtual void linearizeOplus();

    virtual void setMeasurement(const Eigen::VectorXd& m){
      _measurement = m;
    }

    virtual bool setMeasurementData(const double* d){
      Eigen::Map<const Eigen::Vector2d> v(d,2);
      _measurement = v;
      return true;
    }

    virtual bool getMeasurementData(double* d) const{
      Eigen::Map<Eigen::Vector2d> v(d, 2);
      v=_measurement;
      return true;
    }

    virtual int measurementDimension() const {return 2;}

    virtual double initialEstimatePossible(const OptimizableGraph::VertexSet& from,
             OptimizableGraph::Vertex* to) {
      (void) to;
      return (from.count(_vertices[0]) == 1 ? 1.0 : -1.0);
    }

	virtual void initialEstimate(const OptimizableGraph::VertexSet& from, OptimizableGraph::Vertex* to) {}

	Eigen::Matrix<double,2,2> vptCov;

  private:
    Eigen::Matrix<double,2,3+6> J; // jacobian before projection ????????
  };

}
#endif
